# Agent — 轻量级 ReAct 空间推理智能体

> **定位**：面向无人机俯视图空间推理竞赛（如 Open3D-VQA / DML 打榜）的轻量级手写 ReAct 状态机框架。  
> 拒绝 LangChain 黑盒，100% 自主控制循环、格式、兜底策略。

---

## 一、整体架构

```
Agent/
├── prompt_template.py   ← 第一段：大脑与"语言协议"（System Prompt + 格式协议）
├── tool_executor.py     ← 第二段：工具执行器（路由到 Grounding + 3D 引擎）
├── react_agent.py       ← 第二段：ReAct 状态机主循环（Thought → Action → Observation）
├── pipeline_runner.py   ← 第三段：竞赛打榜批量推理主循环（断点续传 + 并发 + 提交文件）
├── tests/
│   └── test_agent_mock.py  ← 端到端 Mock 测试（35 条，无需 API Key / GPU）
└── README.md            ← 本文档
```

### 数据流

```
问题 + 图像
    │
    ▼
[System Prompt]          ← prompt_template.py
    │                       定义角色、工具列表、输出格式
    ▼
[VLM API 调用]           ← react_agent.py
    │
    ├─ 解析 Action?
    │    └─ [ToolExecutor.execute()]  ← tool_executor.py
    │          ├─ get_bounding_box        → 视觉定位_Grounding/
    │          ├─ calculate_3d_distance   → 3D数学引擎/spatial_reasoner.py
    │          ├─ calculate_horizontal_distance
    │          ├─ calculate_vertical_distance
    │          ├─ get_direction
    │          └─ get_object_size
    │
    ├─ 解析 Final Answer? → 返回答案
    └─ 超出最大迭代?      → 兜底策略（Fallback）
         │
         ▼
[CompetitionPipeline]    ← pipeline_runner.py
    ├─ 断点续传（Checkpoint）
    ├─ 指数退避重试
    ├─ 多线程并发
    └─ 输出 JSONL 提交文件
```

---

## 二、快速使用

### 2.1 单条推理（一行代码）

```python
from Agent.react_agent import run_agent
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxx"

answer = run_agent(
    image_path="scene.jpg",
    question="图中红色轿车和路灯相距多远？",
)
print(answer)   # 例如: "8.5 meters"
```

### 2.2 批量竞赛推理（9k 数据）

```python
from Agent.pipeline_runner import run_competition

run_competition(
    data_path="competition_9k.json",
    output_path="submission.jsonl",
    model="qwen-vl-max",
    grounding_backend="qwen3vl_api",
    depth_backend="depth_anything_v3",
    max_workers=4,          # 并发线程数
    checkpoint_dir="./checkpoints",  # 断点续传目录
)
```

### 2.3 命令行运行（竞赛场景）

```bash
# 真实模式
export DASHSCOPE_API_KEY=sk-xxxxxxxx
python Agent/pipeline_runner.py \
    --data competition_9k.json \
    --output submission.jsonl \
    --model qwen-vl-max \
    --grounding qwen3vl_api \
    --depth depth_anything_v3 \
    --workers 4

# Mock 模式（验证流程，无需 API Key 和 GPU）
python Agent/pipeline_runner.py \
    --mock \
    --data sample.json \
    --output submission.jsonl
```

### 2.4 运行测试

```bash
# 全部 35 条测试（无需 API Key 或 GPU，约 1 秒完成）
python Agent/tests/test_agent_mock.py --verbose
```

---

## 三、模块详解

### 3.1 第一段：`prompt_template.py` — 大脑与"语言协议"

**核心设计**：将对大模型的"思维方式要求"写死在系统提示词里，使模型无法绕过工具链直接猜测答案。

系统提示词 `AGENT_SYSTEM_PROMPT` 包含三大要素：

| 要素 | 内容 |
|------|------|
| **角色设定** | 精确的无人机 3D 空间推理智能体，禁止猜测物理量 |
| **可用工具列表** | 6 个工具，含精确的参数格式和返回值说明 |
| **强制输出格式** | Thought/Action/Action Input 或 Final Answer，格式违反时系统会给出纠错 Observation |

**可用工具一览**：

| 工具名 | 功能 | 适用题型 |
|--------|------|---------|
| `get_bounding_box` | 定位目标，返回 2D 边界框 | 确认目标存在；辅助定位 |
| `calculate_3d_distance` | 三维直线距离（含高度差） | "A 和 B 距离多远？" |
| `calculate_horizontal_distance` | 水平面距离（忽略高度差） | "水平距离是多少？" |
| `calculate_vertical_distance` | 高度差 | "A 比 B 高多少？" |
| `get_direction` | 方向（钟表/罗盘/完整） | "B 在 A 的什么方向？" |
| `get_object_size` | 物体三维尺寸（宽/高/深） | "这辆车有多宽？" |

### 3.2 第二段：`tool_executor.py` — 工具执行器

**核心设计**：所有工具方法都用 `_safe_call()` 包裹，任何异常均转为字符串返回，
确保 Agent 主循环永远不会因工具崩溃而中断。

```python
from Agent.tool_executor import ToolExecutor

executor = ToolExecutor(
    image_path="scene.jpg",
    grounding_backend="qwen3vl_api",   # 或 "mock" 用于测试
    depth_backend="depth_anything_v3", # 或 "mock" 用于测试
)

# 直接调用工具
result = executor.execute("get_bounding_box", {"object_name": "红色轿车"})
result = executor.execute("calculate_3d_distance", {"obj_1": "红色轿车", "obj_2": "路灯"})
result = executor.execute("get_direction", {
    "from_object": "无人机", "to_object": "建筑物", "mode": "clock"
})
```

### 3.3 第二段：`react_agent.py` — ReAct 状态机主循环

**核心设计**：纯 Python 手写状态机，完全控制每一步的格式解析和异常处理。

关键机制：

| 机制 | 说明 |
|------|------|
| **停止序列** | API 调用时传入 `stop=["Observation:"]`，阻止模型伪造工具返回值 |
| **格式纠错** | 模型输出不符合格式时，注入"格式错误"Observation 并继续循环 |
| **JSON 容错** | 单引号 JSON / 格式略有偏差时，`ast.literal_eval` 兜底解析 |
| **兜底策略** | 达到最大迭代上限后，追加强制指令让模型基于已有数据给出答案 |

### 3.4 第三段：`pipeline_runner.py` — 竞赛打榜主循环

**核心设计**：针对 9k 量级数据的高稳定性批量推理引擎。

| 功能 | 说明 |
|------|------|
| **断点续传** | 中途崩溃后重启，自动跳过已完成条目（Checkpoint JSONL） |
| **指数退避重试** | API 限速/网络波动时自动等待重试，最多 3 次 |
| **并发控制** | ThreadPoolExecutor，可配置线程数（默认 1，稳定优先） |
| **进度跟踪** | tqdm 进度条（自动回退到日志打印） |
| **格式严格** | 输出 JSONL，每行 `{"id": "...", "answer": "..."}`，按 id 排序 |
| **运行日志** | 自动生成 `.run_log.json`，记录总条数/错误数/吞吐量 |

```python
from Agent.pipeline_runner import CompetitionPipeline

pipeline = CompetitionPipeline(
    grounding_backend="qwen3vl_api",
    depth_backend="depth_anything_v3",
    max_workers=4,
    retry_times=3,
    checkpoint_dir="./checkpoints",
)
summary = pipeline.run(
    data_path="competition_9k.json",
    output_path="submission.jsonl",
    resume=True,   # 断点续传（默认）
)
print(summary)
# {"total": 9000, "done_new": 9000, "skipped": 0, "error_cnt": 12, "elapsed_s": 3600.5}
```

---

## 四、后端一览

| 类型 | 后端名 | 说明 | 依赖 |
|------|--------|------|------|
| **Grounding** | `qwen3vl_api`（默认） | Qwen3-VL API，无需 GPU | `openai`, `DASHSCOPE_API_KEY` |
| | `dino` | GroundingDINO 开源 | `groundingdino-py`, `torch` |
| | `qwen3vl_local` | 本地 Qwen3-VL，离线部署 | `transformers`, `torch` |
| | `mock` | 合成结果，用于测试 | 仅需 `pillow` |
| **Depth** | `depth_anything_v3`（默认） | 最新深度估算 | `transformers`, `torch` |
| | `depth_anything_v2` | 稳定版 | 同上 |
| | `midas` | 经典方案 | `torch.hub` |
| | `mock` | 合成深度图，用于测试 | 仅需 `numpy` |

---

## 五、参数参考

### ReactAgent

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `"qwen-vl-max"` | VLM 模型名称 |
| `api_key` | 环境变量 `DASHSCOPE_API_KEY` | DashScope API Key |
| `max_iterations` | `8` | 最大工具调用轮数（竞赛推荐 6~10） |
| `max_tokens` | `512` | 每轮最大生成 token 数 |
| `temperature` | `0.0` | 采样温度（0 = 贪婪解码，竞赛推荐） |
| `grounding_backend` | `"qwen3vl_api"` | 视觉定位后端 |
| `depth_backend` | `"depth_anything_v3"` | 深度估算后端 |
| `verbose` | `False` | 是否打印每一步详细信息 |

### CompetitionPipeline

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_workers` | `1` | 并发线程数（先用 1 验证，再提高） |
| `retry_times` | `3` | 单条最大重试次数 |
| `retry_base_delay` | `2.0` | 退避基础延迟（秒） |
| `request_interval` | `0.5` | 同线程连续请求最小间隔（秒） |
| `checkpoint_dir` | `"./checkpoints"` | 断点目录 |

---

## 六、依赖安装

```bash
# 最小依赖（VLM API + Grounding API + Depth 模型）
pip install openai pillow numpy transformers torch torchvision

# 可选：进度条
pip install tqdm

# 设置 API Key
export DASHSCOPE_API_KEY=sk-xxxxxxxx
```

---

## 七、竞赛建议

1. **模型选择**：首选 `qwen3-vl-72b-instruct`（能力最强），API 限额不足时退而使用 `qwen-vl-max`。
2. **max_iterations**：设为 6~8，过少会导致工具调用不足，过多会超时。
3. **temperature=0**：竞赛中关闭随机性，确保相同输入得到相同输出（方便 debug）。
4. **并发策略**：先用 `max_workers=1` 跑 50 条验证正确性，再提高到 4~8（注意 API 限速）。
5. **断点续传**：始终保持 `resume=True`（默认），保证 9k 数据即使中途崩溃也能续传。
6. **错误监控**：检查 `run_log.json` 中的 `error_cnt`，若过高则降低并发数或加大重试延迟。


> **定位**：面向无人机俯视图空间推理竞赛（如 Open3D-VQA / DML 打榜）的轻量级手写 ReAct 状态机框架。  
> 拒绝 LangChain 黑盒，100% 自主控制循环、格式、兜底策略。

---

## 一、整体架构

```
Agent/
├── prompt_template.py   ← 第一段：大脑与"语言协议"（System Prompt + 格式协议）
├── tool_executor.py     ← 第二段：工具执行器（路由到 Grounding + 3D 引擎）
├── react_agent.py       ← 第二段：ReAct 状态机主循环（Thought → Action → Observation）
└── README.md            ← 本文档
```

### 数据流

```
问题 + 图像
    │
    ▼
[System Prompt]          ← prompt_template.py
    │                       定义角色、工具列表、输出格式
    ▼
[VLM API 调用]           ← react_agent.py
    │
    ├─ 解析 Action?
    │    └─ [ToolExecutor.execute()]  ← tool_executor.py
    │          ├─ get_bounding_box        → 视觉定位_Grounding/
    │          ├─ calculate_3d_distance   → 3D数学引擎/spatial_reasoner.py
    │          ├─ calculate_horizontal_distance
    │          ├─ calculate_vertical_distance
    │          ├─ get_direction
    │          └─ get_object_size
    │
    ├─ 解析 Final Answer? → 返回答案
    └─ 超出最大迭代?      → 兜底策略（Fallback）
```

---

## 二、快速使用

### 2.1 最简单调用（一行代码）

```python
from Agent.react_agent import run_agent
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxx"

answer = run_agent(
    image_path="scene.jpg",
    question="图中红色轿车和路灯相距多远？",
)
print(answer)   # 例如: "8.5 meters"
```

### 2.2 实例化 Agent（批量推理推荐）

```python
from Agent.react_agent import ReactAgent

agent = ReactAgent(
    model="qwen-vl-max",              # 或 "qwen3-vl-72b-instruct"
    api_key="sk-xxxxxxxx",            # 或通过环境变量 DASHSCOPE_API_KEY 传入
    grounding_backend="qwen3vl_api",  # 视觉定位后端
    depth_backend="depth_anything_v3",# 深度估算后端
    max_iterations=8,                 # 最多 8 轮工具调用
    verbose=True,                     # 打印每一步的 Thought/Action/Observation
)

# 对 9000 条数据批量推理
for item in dataset:
    answer = agent.run(item["image"], item["question"])
    print(answer)
```

### 2.3 Mock 模式（无需任何 API Key 或 GPU，验证框架流程）

```python
from Agent.react_agent import ReactAgent

# 注意：mock 模式下工具会返回合成结果，VLM 调用仍需 API Key。
# 如需完全离线测试，可继承 ReactAgent 并 override _call_vlm()。
agent = ReactAgent(
    grounding_backend="mock",
    depth_backend="mock",
    verbose=True,
)
```

---

## 三、模块详解

### 3.1 第一段：`prompt_template.py` — 大脑与"语言协议"

**核心设计**：将对大模型的"思维方式要求"写死在系统提示词里，使模型无法绕过工具链直接猜测答案。

系统提示词 `AGENT_SYSTEM_PROMPT` 包含三大要素：

| 要素 | 内容 |
|------|------|
| **角色设定** | 精确的无人机 3D 空间推理智能体，禁止猜测物理量 |
| **可用工具列表** | 6 个工具，含精确的参数格式和返回值说明 |
| **强制输出格式** | Thought/Action/Action Input 或 Final Answer，格式违反时系统会给出纠错 Observation |

**可用工具一览**：

| 工具名 | 功能 | 适用题型 |
|--------|------|---------|
| `get_bounding_box` | 定位目标，返回 2D 边界框 | 确认目标存在；辅助定位 |
| `calculate_3d_distance` | 三维直线距离（含高度差） | "A 和 B 距离多远？" |
| `calculate_horizontal_distance` | 水平面距离（忽略高度差） | "水平距离是多少？" |
| `calculate_vertical_distance` | 高度差 | "A 比 B 高多少？" |
| `get_direction` | 方向（钟表/罗盘/完整） | "B 在 A 的什么方向？" |
| `get_object_size` | 物体三维尺寸（宽/高/深） | "这辆车有多宽？" |

### 3.2 第二段：`tool_executor.py` — 工具执行器

**核心设计**：所有工具方法都用 `_safe_call()` 包裹，任何异常均转为字符串返回，
确保 Agent 主循环永远不会因工具崩溃而中断。

```python
from Agent.tool_executor import ToolExecutor

executor = ToolExecutor(
    image_path="scene.jpg",
    grounding_backend="qwen3vl_api",
    depth_backend="depth_anything_v3",
)

# 直接调用工具
result = executor.execute("get_bounding_box", {"object_name": "红色轿车"})
result = executor.execute("calculate_3d_distance", {"obj_1": "红色轿车", "obj_2": "路灯"})
result = executor.execute("get_direction", {
    "from_object": "无人机", "to_object": "建筑物", "mode": "clock"
})
```

### 3.3 第二段：`react_agent.py` — ReAct 状态机主循环

**核心设计**：纯 Python 手写状态机，完全控制每一步的格式解析和异常处理。

关键机制：

| 机制 | 说明 |
|------|------|
| **停止序列** | API 调用时传入 `stop=["Observation:"]`，阻止模型伪造工具返回值 |
| **格式纠错** | 模型输出不符合格式时，注入"格式错误"Observation 并继续循环 |
| **JSON 容错** | 单引号 JSON / 格式略有偏差时，`ast.literal_eval` 兜底解析 |
| **兜底策略** | 达到最大迭代上限后，追加强制指令让模型基于已有数据给出答案 |

---

## 四、参数参考

### ReactAgent

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `"qwen-vl-max"` | VLM 模型名称 |
| `api_key` | 环境变量 `DASHSCOPE_API_KEY` | DashScope API Key |
| `max_iterations` | `8` | 最大工具调用轮数（竞赛推荐 6~10） |
| `max_tokens` | `512` | 每轮最大生成 token 数 |
| `temperature` | `0.0` | 采样温度（0 = 贪婪解码，竞赛推荐） |
| `grounding_backend` | `"qwen3vl_api"` | 视觉定位后端 |
| `depth_backend` | `"depth_anything_v3"` | 深度估算后端 |
| `camera_intrinsics` | `None` | 相机内参，None 时自动估算（90° FOV） |
| `depth_scale_factor` | `10.0` | 深度缩放系数（近似米数上限） |
| `verbose` | `False` | 是否打印每一步详细信息 |

---

## 五、依赖安装

```bash
# 最小依赖（VLM API + Grounding API + Depth 模型）
pip install openai pillow numpy transformers torch torchvision

# 设置 API Key
export DASHSCOPE_API_KEY=sk-xxxxxxxx
```

---

## 六、竞赛建议

1. **模型选择**：首选 `qwen3-vl-72b-instruct`（能力最强），API 限额不足时退而使用 `qwen-vl-max`。
2. **max_iterations**：设为 6~8，过少会导致工具调用不足，过多会超时。
3. **temperature=0**：竞赛中关闭随机性，确保相同输入得到相同输出（方便 debug）。
4. **批量推理**：对 9k 数据建议多进程并行，每个进程持有独立 `ReactAgent` 实例。
5. **错误监控**：检查返回值中是否含 `ERROR:` 前缀，统计工具失败率以指导参数调整。
