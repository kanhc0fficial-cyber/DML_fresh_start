# 3D 数学引擎

无人机视觉空间推理的"小脑"模块。

将视觉定位（Day 4 Grounding）和深度估算（Day 5 Depth）输出组合，通过精确几何计算回答以下类型的基准测试题：

| 任务类型 | 示例问题 | 输出 |
|---------|----------|------|
| 绝对距离 | "红车和白面包车距离多远？" | `8.45 meters` |
| 水平距离 | "忽略高度差，两楼之间距离？" | `12.30 meters` |
| 垂直高度差 | "无人机比建筑物高多少？" | `5.20 meters` |
| 钟表方向 | "从 A 看，B 在几点钟方向？" | `3 o'clock` |
| 罗盘方向 | "建筑物在车的哪个方向？" | `North-East` |
| 物体尺寸 | "卡车有多宽？" | `width: 3.2m, height: 2.4m` |

---

## 文件结构

```
3D数学引擎/
├── geometry_tools.py     # 核心：逆投影 + 距离 + 方向计算
├── depth_estimator.py    # 深度估算包装（MiDaS / Depth-Anything / Mock）
├── spatial_reasoner.py   # 编排层：Grounding + Depth + Geometry 串联
├── benchmark_eval.py     # Benchmark 评测脚本（含 Mock 模式）
├── test_geometry.py      # 单元测试 + 集成测试（39 个测试，无需真实模型）
└── README.md
```

---

## 数据流

```
问题: "红车和白面包车距离多远？"
    │
    ▼
[视觉定位_Grounding]           # Day 4 工具
  → bbox_red_car  = [120, 80, 280, 200]
  → bbox_white_van = [350, 90, 520, 210]
    │
    ▼
[depth_estimator]              # Day 5 工具
  → depth_map: H×W float32（每像素近似米数）
    │
    ▼
[geometry_tools.unproject_to_3d]   # 核心数学
  → p_car = [X₁, Y₁, Z₁]
  → p_van = [X₂, Y₂, Z₂]
    │
    ▼
[geometry_tools.calculate_absolute_distance]
  → 8.45 meters
    │
    ▼
大模型最终输出: "The distance is 8.45 meters."
```

---

## 快速上手

### 安装依赖（最小集）

```bash
pip install numpy pillow
```

> 使用真实深度模型时额外安装（按需选择）：
> ```bash
> pip install torch torchvision                   # MiDaS
> pip install transformers torch torchvision      # Depth-Anything
> ```

---

### 方式一：直接调用核心数学函数

```python
import numpy as np
from geometry_tools import (
    unproject_to_3d,
    calculate_absolute_distance,
    calculate_clock_direction,
)

# 已有 2D 框和深度图
bbox_A = (120, 80, 280, 200)
bbox_B = (350, 90, 520, 210)
depth_map = np.full((480, 640), 10.0)   # 均一深度 10m（演示用）

p_A = unproject_to_3d(bbox_A, depth_map)
p_B = unproject_to_3d(bbox_B, depth_map)

dist = calculate_absolute_distance(p_A, p_B)
print(f"距离: {dist:.2f} m")

clock = calculate_clock_direction(p_A, p_B)
print(f"从 A 看 B 的方向: {clock}")
```

---

### 方式二：使用 SpatialReasoner（推荐，端到端）

```python
from spatial_reasoner import SpatialReasoner

# Mock 模式（无需任何模型，用于开发/测试）
reasoner = SpatialReasoner(grounding_backend="mock", depth_backend="mock")

# 真实模式（需要 API Key 和模型）
# reasoner = SpatialReasoner(grounding_backend="qwen3vl_api", depth_backend="midas")

# 距离计算
result = reasoner.measure_distance("scene.jpg", "red car", "white van")
print(result["distance_m"])  # 例如 8.45

# 方向计算（钟表方向）
result = reasoner.get_direction("scene.jpg", "drone", "building", mode="clock")
print(result["direction"])   # 例如 "3 o'clock"

# 方向计算（罗盘方向）
result = reasoner.get_direction("scene.jpg", "observer", "target", mode="cardinal")
print(result["direction"])   # 例如 "North-East"

# 物体尺寸
result = reasoner.get_object_size("scene.jpg", "truck")
print(result["width_m"], result["height_m"])  # 例如 3.2, 2.4
```

---

### 方式三：运行 Benchmark 评测

```bash
# Mock 模式（无需任何模型，验证流程）：
python benchmark_eval.py --mock --generate-sample

# 真实模型评测（需要 API Key）：
export DASHSCOPE_API_KEY=sk-xxxxxxxx
python benchmark_eval.py \
    --data your_benchmark.json \
    --grounding qwen3vl_api \
    --depth midas
```

Benchmark JSON 格式示例：
```json
[
  {
    "image": "scene_01.jpg",
    "question": "What is the distance between the red car and the white van?",
    "task_type": "distance",
    "query_objects": ["red car", "white van"],
    "gt_answer": "8.5 meters",
    "gt_distance_m": 8.5
  },
  {
    "image": "scene_02.jpg",
    "question": "From the drone's perspective, where is the building?",
    "task_type": "direction_clock",
    "query_objects": ["drone", "building"],
    "gt_answer": "3 o'clock"
  }
]
```

---

### 方式四：运行测试套件

```bash
python test_geometry.py          # 39 个测试（无需任何真实模型）
python test_geometry.py --verbose  # 详细输出
```

---

## 模块 API 参考

### `geometry_tools.py`

| 函数 | 说明 |
|------|------|
| `unproject_to_3d(bbox, depth_map, camera_intrinsics)` | 2D 框 → 3D 坐标 |
| `calculate_absolute_distance(p1, p2)` | 三维欧氏距离 |
| `calculate_horizontal_distance(p1, p2)` | XZ 平面距离（忽略高度）|
| `calculate_vertical_distance(p1, p2)` | Y 轴高度差 |
| `calculate_object_size(bbox, depth_map)` | 物体宽/高/深度估算 |
| `is_left_or_right_of_camera(p)` | 目标在镜头左/右/中 |
| `calculate_clock_direction(p_A, p_B)` | 站在 A 看 B 的钟表方向 |
| `calculate_cardinal_direction(p_A, p_B)` | 罗盘方向（N/S/E/W）|
| `calculate_relative_position(p_A, p_B)` | 综合相对位置描述 |
| `bbox_pair_distance(bbox_A, bbox_B, depth_map, mode)` | 一步计算两框距离 |
| `bbox_pair_direction(bbox_A, bbox_B, depth_map, mode)` | 一步计算两框方向 |

### `depth_estimator.py`

| 后端 | 说明 | 依赖 |
|------|------|------|
| `"midas"` | MiDaS DPT-Large（高精度）| `torch`, `torchvision` |
| `"depth_anything"` | Depth-Anything V2（轻量）| `transformers`, `torch` |
| `"mock"` | 合成深度图（测试用）| `numpy`, `pillow` |

### `spatial_reasoner.py`

| 方法 | 说明 |
|------|------|
| `SpatialReasoner.measure_distance(image, obj_A, obj_B, mode)` | 测量距离 |
| `SpatialReasoner.get_direction(image, from_obj, to_obj, mode)` | 计算方向 |
| `SpatialReasoner.get_object_size(image, obj)` | 估算物体尺寸 |

---

## 相机内参说明

如果官方数据集提供相机内参（焦距 `f_x`/`f_y` 和主点 `c_x`/`c_y`），在实例化时传入：

```python
reasoner = SpatialReasoner(
    camera_intrinsics=(f_x, f_y, c_x, c_y),
)
```

如未提供，引擎自动根据图像尺寸推算（假设水平视野角约 90°），适合无人机航拍等广角场景的近似估算。

---

## 与前期工作的衔接

| 组件 | 对应 | 接口 |
|------|------|------|
| `视觉定位_Grounding/grounding_tool.py` | Day 4 | `get_bounding_box(image, text)` → `BBox2D` |
| `3D数学引擎/depth_estimator.py` | Day 5 | `get_depth_map(image)` → `np.ndarray` |
| `3D数学引擎/geometry_tools.py` | Day 7-8 | 核心数学引擎 |
| `3D数学引擎/spatial_reasoner.py` | Agent 工具调用层 | `SpatialReasoner` |
| `3D数学引擎/benchmark_eval.py` | 评测框架 | `run_benchmark(samples)` |
