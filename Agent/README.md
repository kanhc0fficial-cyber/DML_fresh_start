# Agent 模块说明

`Agent/` 是仓库里面向无人机俯视图空间问答的轻量级 ReAct 智能体实现。它不依赖 LangChain 一类外层框架，而是直接用 Python 手写了：

1. **提示词协议层**：约束模型只能按 `Thought -> Action -> Observation -> Final Answer` 工作；
2. **工具执行层**：把工具名路由到视觉定位与 3D 几何计算模块；
3. **状态机推理层**：循环调用多模态大模型、解析动作、执行工具、注入 Observation；
4. **批量运行层**：面向竞赛数据集做断点续传、并发、重试和 JSONL 提交文件输出。

---

## 目录结构

```text
Agent/
├── __init__.py
├── prompt_template.py
├── tool_executor.py
├── react_agent.py
├── pipeline_runner.py
├── tests/
│   └── test_agent_mock.py
└── README.md
```

各文件职责如下：

| 文件 | 作用 |
|---|---|
| `prompt_template.py` | 定义系统提示词、工具名常量、用户问题与 Observation 文本模板 |
| `tool_executor.py` | 将 Agent 输出的工具调用映射到 Grounding 与 3D 数学引擎 |
| `react_agent.py` | 核心 ReAct 状态机，负责 VLM 调用、解析、循环控制、兜底 |
| `pipeline_runner.py` | 批量推理主循环，负责数据加载、路径解析、重试、并发、检查点与提交文件 |
| `tests/test_agent_mock.py` | Mock 模式下的导入、工具、状态机、Pipeline、Checkpoint 测试 |

---

## 代码级工作流

```text
图像 + 问题
   │
   ▼
ReactAgent.run()
   │
   ├─ 构造 system/user messages
   ├─ 调用 VLM
   ├─ 解析 Final Answer ?
   │      └─ 是 -> 直接返回
   └─ 解析 Action + Action Input
          │
          ▼
      ToolExecutor.execute()
          │
          ├─ get_bounding_box -> 视觉定位_Grounding/grounding_tool.py
          └─ 距离/方向/尺寸 -> 3D数学引擎/spatial_reasoner.py
                    │
                    ▼
          Observation 回写到对话历史
                    │
                    ▼
               继续下一轮

达到最大轮数后：
ReactAgent._fallback() 强制模型基于已有 Observation 输出 Final Answer
```

批量场景下，`CompetitionPipeline.run()` 会在外层包住 `ReactAgent.run()`，负责：

- 读取 JSON / JSONL 数据集；
- 解析图像相对路径；
- 失败重试与指数退避；
- 多线程并发；
- 检查点续跑；
- 输出提交文件与运行日志。

---

## 核心模块详解

### 1. `prompt_template.py`

这个文件定义了 Agent 的“语言协议”。

- `TOOL_NAMES` / `ALL_TOOL_NAMES`：工具名常量，避免状态机里写魔法字符串；
- `AGENT_SYSTEM_PROMPT`：约束模型只能使用 6 个工具之一，并强制输出两种格式之一：
  - `Thought + Action + Action Input`
  - `Thought + Final Answer`
- `format_user_question(question)`：把自然语言问题包装成标准用户文本；
- `format_observation(tool_name, result)`：把工具结果包装成 `Observation: ...` 文本。

当前 Agent 暴露给模型的工具只有 6 个：

1. `get_bounding_box`
2. `calculate_3d_distance`
3. `calculate_horizontal_distance`
4. `calculate_vertical_distance`
5. `get_direction`
6. `get_object_size`

---

### 2. `tool_executor.py`

`ToolExecutor` 是 Agent 与底层感知/几何模块之间的统一路由层。

#### 初始化参数

- `image_path`
- `grounding_backend`
- `depth_backend`
- `camera_intrinsics`
- `depth_scale_factor`
- `grounding_kwargs`
- `depth_kwargs`

#### 关键实现点

- **懒加载 `SpatialReasoner`**：只有第一次用到距离/方向/尺寸工具时才实例化；
- **统一异常包装**：所有工具调用都走 `_safe_call()`，异常不会打断 Agent 主循环，而是被转成字符串；
- **统一入口**：外部只需要调用 `execute(tool_name, params)`。

#### 实际路由关系

| Agent 工具 | 目标实现 |
|---|---|
| `get_bounding_box` | `视觉定位_Grounding/grounding_tool.py` |
| `calculate_3d_distance` | `SpatialReasoner.measure_distance(..., mode="absolute")` |
| `calculate_horizontal_distance` | `SpatialReasoner.measure_distance(..., mode="horizontal")` |
| `calculate_vertical_distance` | `SpatialReasoner.measure_distance(..., mode="vertical")` |
| `get_direction` | `SpatialReasoner.get_direction(...)` |
| `get_object_size` | `SpatialReasoner.get_object_size(...)` |

---

### 3. `react_agent.py`

`ReactAgent` 是整个 Agent 的核心控制器。

#### 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `model` | `qwen-vl-max` | 多模态模型名称 |
| `api_key` | 环境变量读取 | DashScope API Key |
| `base_url` | DashScope OpenAI 兼容地址 | API 基地址 |
| `max_iterations` | `8` | 最多循环多少轮 |
| `max_tokens` | `512` | 单轮输出 token 上限 |
| `temperature` | `0.0` | 推理温度 |
| `grounding_backend` | `qwen3vl_api` | 目标定位后端 |
| `depth_backend` | `depth_anything_v3` | 深度估计后端 |
| `verbose` | `False` | 是否打印详细中间过程 |

#### 代码行为

1. 读取图像并编码为 base64 data URL；
2. 构造 `system` + `user` 消息；
3. 调用大模型；
4. 先解析 `Final Answer`；
5. 若没有最终答案，再解析 `Action` 与 `Action Input`；
6. 执行工具并把结果写回成 `Observation`；
7. 继续下一轮；
8. 超过最大轮数时进入 `_fallback()`。

#### 关键保护机制

- **停止序列**：调用 VLM 时设置 `stop=["\\nObservation:", "Observation:"]`，减少模型伪造 Observation 的机会；
- **格式恢复**：若既没有合法 `Action` 也没有 `Final Answer`，会注入格式错误提示并继续；
- **JSON 容错**：`Action Input` 先走 `json.loads()`，失败后回退到 `ast.literal_eval()`；
- **显式兜底**：达到上限后强制模型基于已有 Observation 输出最终答案。

#### 公开入口

- `ReactAgent.run(image_path, question, ...)`
- `run_agent(image_path, question, ...)`

---

### 4. `pipeline_runner.py`

`CompetitionPipeline` 是面向批量数据集的外层运行器。

#### 支持的数据格式

输入支持两种：

- `.json`：JSON 数组
- `.jsonl`：每行一条 JSON

每条记录至少包含：

```json
{
  "id": "q1",
  "image": "scene.png",
  "question": "红色车和路灯距离多远？"
}
```

如果没有 `id`，代码会自动按行号补齐。

#### 路径解析规则

`_resolve_image()` 的优先级是：

1. 数据里已经给的是绝对路径；
2. 若设置了 `image_base_dir`，优先拼到这个目录下；
3. 否则拼到数据文件所在目录下。

#### 关键能力

- `Checkpoint`：JSONL 持久化检查点，线程安全；
- `_process_one()`：单条样本重试，失败时返回错误记录，不拖垮整批任务；
- `_get_agent()`：每个线程独立持有一个 `ReactAgent` 实例；
- `_write_submission()`：按 `id` 排序，写出标准提交 JSONL；
- 同时生成 `*.run_log.json` 记录总量、错误数、吞吐量等统计信息。

#### 公开入口

- `CompetitionPipeline.run(data_path, output_path, resume=True)`
- `run_competition(data_path, output_path, ...)`

---

## 后端与依赖关系

Agent 本身不直接实现视觉定位和 3D 几何，而是调用兄弟目录中的模块：

- `视觉定位_Grounding/`
- `3D数学引擎/`

### Grounding 后端

代码里明确支持：

- `qwen3vl_api`
- `dino`
- `qwen3vl_local`
- `mock`

其中 `mock` 会返回合成边界框，方便测试与联调。

### Depth 后端

批量运行 CLI 中支持：

- `depth_anything_v3`
- `depth_anything_v2`
- `depth_anything`
- `midas`
- `mock`

---

## 使用方式

### 1. 作为包导入

从仓库根目录可以直接导入：

```python
from Agent import ReactAgent, run_agent, CompetitionPipeline, run_competition
```

---

### 2. 单条推理

```python
from Agent import run_agent

answer = run_agent(
    image_path="/absolute/path/to/scene.jpg",
    question="图中红色轿车和路灯相距多远？",
    model="qwen-vl-max",
    grounding_backend="qwen3vl_api",
    depth_backend="depth_anything_v3",
)

print(answer)
```

---

### 3. 批量推理

```python
from Agent import run_competition

summary = run_competition(
    data_path="/absolute/path/to/competition.json",
    output_path="/absolute/path/to/submission.jsonl",
    model="qwen-vl-max",
    grounding_backend="qwen3vl_api",
    depth_backend="depth_anything_v3",
    max_workers=4,
    checkpoint_dir="/absolute/path/to/checkpoints",
)

print(summary)
```

---

### 4. CLI

从仓库根目录运行：

```bash
python Agent/pipeline_runner.py \
  --data /absolute/path/to/competition.json \
  --output /absolute/path/to/submission.jsonl \
  --model qwen-vl-max \
  --grounding qwen3vl_api \
  --depth depth_anything_v3 \
  --workers 4
```

Mock 联调：

```bash
python Agent/pipeline_runner.py \
  --mock \
  --data /absolute/path/to/sample.json \
  --output /absolute/path/to/submission.jsonl
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--workers` | 并发线程数 |
| `--max-iterations` | 单题 ReAct 最大轮数 |
| `--retry` | 单题最大重试次数 |
| `--interval` | 同线程连续请求最小间隔 |
| `--image-dir` | 图像根目录 |
| `--checkpoint-dir` | 检查点目录 |
| `--no-resume` | 禁用断点续跑 |
| `--verbose` | 打印详细推理过程 |

---

## 输出文件

### 提交文件

`submission.jsonl`

每行只保留两个字段：

```json
{"id": "q1", "answer": "8.5 meters"}
```

### 运行日志

`submission.run_log.json`

包含：

- `data_path`
- `output_path`
- `grounding_backend`
- `depth_backend`
- `model`
- `max_workers`
- `total`
- `done_new`
- `skipped`
- `error_cnt`
- `elapsed_s`
- `throughput_per_min`

---

## 测试

测试文件：`Agent/tests/test_agent_mock.py`

覆盖点包括：

- 包与模块导入；
- `ToolExecutor` mock 工具调用；
- `ReactAgent` 的状态机控制流；
- `Action` / `Final Answer` 解析；
- `CompetitionPipeline` 的输入加载、断点续跑、错误隔离；
- `Checkpoint` 持久化。

运行命令：

```bash
python Agent/tests/test_agent_mock.py --verbose
```

如果环境缺少 `Pillow`、`numpy` 或底层依赖，导入与 mock 测试也会失败；因此建议先补齐最小依赖再运行。

---

## 最小依赖建议

若只想跑 Agent 主流程与 mock 测试，至少建议安装：

```bash
pip install pillow numpy
```

若要跑真实 API / 深度模型，再补充：

```bash
pip install openai transformers torch torchvision tqdm
```

---

## 适合继续扩展的位置

如果后续要继续演进 Agent，通常从下面几处下手：

- **新增工具**：改 `prompt_template.py` + `tool_executor.py`
- **调整推理协议**：改 `AGENT_SYSTEM_PROMPT` 与 `ReactAgent` 解析逻辑
- **更换批量调度策略**：改 `CompetitionPipeline`
- **替换感知后端**：保持 `ToolExecutor` 接口不变，扩展底层 Grounding / Depth 模块

---

## 总结

从代码实现上看，`Agent/` 已经形成了一条完整链路：

- 前端是受约束的 ReAct 协议；
- 中间是统一工具路由与异常隔离；
- 后端是视觉定位与 3D 几何推理；
- 外层有批量竞赛运行器负责稳定性与工程化输出。

如果你要读代码，建议按这个顺序看：

1. `prompt_template.py`
2. `tool_executor.py`
3. `react_agent.py`
4. `pipeline_runner.py`
5. `tests/test_agent_mock.py`
