# 视觉空间推理流水线

> **适用场景**：无人机俯视图 VQA（视觉问答），例如 Open3D-VQA 竞赛基准。  
> **三工具联动**：视觉定位（Grounding） → 深度估算（Depth Anything V3） → 3D 数学引擎

---

## 一、整体数据流

```
题目示例："图中红车与白色面包车距离多远？"
          │
          ▼
┌─────────────────────────────────────┐
│  视觉定位_Grounding/grounding_tool  │  ← Day 4 工具
│  输入：图像 + 目标描述               │
│  输出：bbox_A = [x1,y1,x2,y2]       │
│        bbox_B = [x3,y3,x4,y4]       │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  3D数学引擎/depth_estimator         │  ← Day 5 工具
│  后端：Depth Anything V3            │
│  输出：depth_map  H×W float32 (米)  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  3D数学引擎/geometry_tools          │  ← Day 7-8 工具
│  逆投影 → P_A [X,Y,Z], P_B [X,Y,Z] │
│  欧氏距离 → "15.3 meters"           │
└─────────────────────────────────────┘
                   │
                   ▼
        大模型最终答案："15.3 meters"
```

---

## 二、工具一：视觉定位（Grounding）

### 2.1 功能

接收一张图像和目标描述（自然语言），返回目标物体在图像中的像素坐标边界框  
`[x_min, y_min, x_max, y_max]`。

### 2.2 支持后端

| 后端 | 文件 | 特点 | 依赖 |
|------|------|------|------|
| `qwen3vl_api`（默认） | `grounding_qwen3vl.py` | 调用阿里云 DashScope API，无需 GPU | `openai`, `pillow` |
| `qwen3vl_local` | `grounding_qwen3vl.py` | 本地加载 Qwen3-VL 模型，离线部署 | `transformers`, `torch` |
| `dino` | `grounding_dino.py` | GroundingDINO 开源零样本检测 | `groundingdino-py`, `torch` |

### 2.3 接口说明

**统一入口（推荐）**：

```python
# 视觉定位_Grounding/grounding_tool.py
from grounding_tool import get_bounding_box

bbox = get_bounding_box(image, "红色车", backend="qwen3vl_api")
# 返回: [x_min, y_min, x_max, y_max]  像素坐标，失败返回 None
```

**Qwen3-VL API（推荐，无需本地 GPU）**：

```python
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxx"

from grounding_qwen3vl import Qwen3VLAPIGrounder

grounder = Qwen3VLAPIGrounder(model="qwen-vl-max")
bbox = grounder.get_bounding_box("scene.jpg", "白色面包车")
# 返回: [120.5, 80.0, 280.3, 200.1]
```

**Qwen3-VL 本地推理**：

```python
from grounding_qwen3vl import Qwen3VLGrounder

grounder = Qwen3VLGrounder(
    model_name_or_path="Qwen/Qwen3-VL-7B-Instruct",
    device="cuda",
)
bbox = grounder.get_bounding_box("scene.jpg", "蓝色卡车")
```

**GroundingDINO（纯开源，英文提示效果更好）**：

```python
from grounding_dino import GroundingDINOGrounder

grounder = GroundingDINOGrounder(box_threshold=0.35, text_threshold=0.25)
bbox = grounder.get_bounding_box("scene.jpg", "blue truck")
```

### 2.4 Qwen3-VL 坐标格式说明

Qwen3-VL 输出归一化坐标（0~1000 整数），`grounding_qwen3vl.py` 已自动转换为像素坐标。
边界框格式与 GroundingDINO 输出保持一致，均为 `[x_min, y_min, x_max, y_max]`。

### 2.5 安装

```bash
# Qwen3-VL API 模式（最简单）
pip install openai pillow

# GroundingDINO 模式
pip install groundingdino-py torch torchvision pillow
```

---

## 三、工具二：深度估算（Depth Anything V3）

### 3.1 功能

对输入图像进行单目深度估算，输出与原图等大的深度图（H×W float32，近似米数）。
深度图中每个像素值代表该位置到相机的估计距离。

### 3.2 支持后端

| 后端标识 | 模型 | 特点 | 速度 |
|---------|------|------|------|
| `depth_anything_v3`（**默认，推荐**） | Depth-Anything-V3-Small-hf | 最新版，精度最高 | 快 |
| `depth_anything_v2` | Depth-Anything-V2-Small-hf | 稳定版 | 快 |
| `depth_anything` | 同 V2（兼容别名） | 向下兼容 | 快 |
| `midas` | MiDaS DPT-Large | 经典方案，torch.hub 加载 | 中 |
| `mock` | 合成深度图 | 无需模型，用于开发测试 | 极快 |

> **为什么推荐 V3？**  
> Depth Anything V3 在室外场景（无人机俯视）的深度精度显著优于 V2 和 MiDaS，
> 且通过 HuggingFace transformers pipeline 加载，无需额外编译环境。

### 3.3 接口说明

```python
from depth_estimator import get_depth_map

# 推荐：Depth Anything V3
depth_map = get_depth_map("scene.jpg", backend="depth_anything_v3")
# 返回: np.ndarray  shape=(H, W)  dtype=float32，单位：近似米

# 指定更大的模型（更高精度，更慢）
depth_map = get_depth_map(
    "scene.jpg",
    backend="depth_anything_v3",
    model_name="depth-anything/Depth-Anything-V3-Large-hf",
)

# 自定义深度缩放（默认 10.0m 为最大深度范围）
depth_map = get_depth_map("scene.jpg", scale_factor=30.0)

# Mock 模式（无需任何模型）
depth_map = get_depth_map("scene.jpg", backend="mock",
                           mock_near_depth=5.0, mock_far_depth=20.0)
```

### 3.4 深度值说明

⚠️ **重要**：Depth Anything（及 MiDaS）输出的是**相对深度**，而非绝对米数。
`scale_factor` 参数将相对深度线性映射到 `[0, scale_factor]` 米的范围。

- 如果已知相机内参和真实场景尺寸，可通过已知距离的参照物对 `scale_factor` 进行标定。
- 对于没有绝对标定的场景，相对深度已足够区分"远近"关系，可用于方向推理任务；
  距离类任务建议结合场景先验知识对 `scale_factor` 进行调整。

### 3.5 安装

```bash
pip install transformers torch torchvision pillow numpy
```

---

## 四、工具三：3D 数学引擎（geometry_tools）

### 4.1 功能概述

核心数学库，将 Grounding 输出的 2D 边界框与 Depth 输出的深度图转换为三维坐标，
并提供完整的空间推理计算函数：距离、方向、物体尺寸。

这是让 Agent 能够回答精确物理量的关键——大模型（大脑）懂语言，3D 数学引擎（小脑）
负责把视觉信息转化为精确数字。

### 4.2 核心原理：逆投影（Unprojection）

```
相机坐标系：Z 轴朝前，X 轴向右，Y 轴向下，原点在相机光心

给定：
  - 像素坐标 (u, v)  ← bbox 中心
  - 深度值 Z         ← depth_map[v, u]（取中心 10×10 区域中位数，防止玻璃/遮挡噪声）
  - 相机内参 f_x, f_y, c_x, c_y

计算：
  X = (u - c_x) × Z / f_x     ← 相似三角形原理
  Y = (v - c_y) × Z / f_y
  Z = Z（直接来自深度图）
```

若未提供相机内参，自动按水平视野角 90° 估算（适合无人机广角镜头）：
`f_x = f_y = 图像宽度 / 2`

### 4.3 完整 API 参考

#### 逆投影
```python
from geometry_tools import unproject_to_3d

p = unproject_to_3d(bbox, depth_map, camera_intrinsics=None)
# bbox: (x_min, y_min, x_max, y_max)  像素坐标
# 返回: np.ndarray [X, Y, Z]  相机坐标系 3D 点
```

#### 距离计算
```python
from geometry_tools import (
    calculate_absolute_distance,   # 三维欧氏距离（含高度差）
    calculate_horizontal_distance, # XZ 平面水平距离（忽略高度差）
    calculate_vertical_distance,   # 绝对高度差（Y 轴方向）
    calculate_object_size,         # 物体三维尺寸（宽/高/深）
    bbox_pair_distance,            # 一步计算两框距离（推荐）
)

# 一步计算（推荐）
dist = bbox_pair_distance(bbox_A, bbox_B, depth_map, mode="absolute")
# mode: "absolute" | "horizontal" | "vertical"
```

#### 方向推理
```python
from geometry_tools import (
    is_left_or_right_of_camera,  # 目标在镜头左/右/中
    calculate_clock_direction,    # 站在 A 看 B 的钟表方向（如 "3 o'clock"）
    calculate_cardinal_direction, # 罗盘方向（如 "North-East"）
    calculate_relative_position,  # 综合相对位置描述
    bbox_pair_direction,          # 一步计算两框方向（推荐）
)

# 钟表方向（最难的 VQA 任务，数学方法理论准确率 ~100%）
clock = calculate_clock_direction(p_A, p_B)   # 如 "3 o'clock"

# 一步计算（推荐）
direction = bbox_pair_direction(bbox_A, bbox_B, depth_map, mode="clock")
```

### 4.4 任务类型对照

| 题目类型 | 推荐函数 | 说明 |
|---------|---------|------|
| "A 和 B 距离多远？" | `bbox_pair_distance(..., mode="absolute")` | 三维直线距离 |
| "A 和 B 水平距离多远？" | `bbox_pair_distance(..., mode="horizontal")` | 忽略高度差 |
| "A 比 B 高多少？" | `bbox_pair_distance(..., mode="vertical")` | 仅高度差 |
| "从 A 看，B 在几点方向？" | `bbox_pair_direction(..., mode="clock")` | 钟表方位 |
| "B 在 A 的哪个方向？" | `bbox_pair_direction(..., mode="cardinal")` | 罗盘方向 |
| "A 有多宽/多高？" | `calculate_object_size(bbox_A, depth_map)` | 物体尺寸 |

---

## 五、高层编排：SpatialReasoner

`spatial_reasoner.py` 将三个工具串联成一个简单易用的接口。

```python
from spatial_reasoner import SpatialReasoner

# 完整真实模式（需要 API Key 和 GPU）
reasoner = SpatialReasoner(
    grounding_backend="qwen3vl_api",
    depth_backend="depth_anything_v3",
    # camera_intrinsics=(f_x, f_y, c_x, c_y),  # 可选，有则更精准
)

# 开发/测试模式（无需任何模型）
reasoner = SpatialReasoner(grounding_backend="mock", depth_backend="mock")
```

### 5.1 测量距离

```python
result = reasoner.measure_distance("scene.jpg", "红色车", "白色面包车", mode="absolute")

print(result["distance_m"])    # 例如: 8.45
print(result["bbox_A"])        # 目标 A 的 2D 框
print(result["point3d_A"])     # 目标 A 的 3D 坐标 [X, Y, Z]
print(result["error"])         # None 表示成功，否则为错误描述
```

### 5.2 判断方向

```python
# 钟表方向（站在 A 看 B）
result = reasoner.get_direction("scene.jpg", "无人机", "建筑物", mode="clock")
print(result["direction"])     # 例如: "3 o'clock"

# 罗盘方向
result = reasoner.get_direction("scene.jpg", "A", "B", mode="cardinal")
print(result["direction"])     # 例如: "North-East"

# 完整信息（同时返回钟表方向、罗盘方向、水平/垂直距离）
result = reasoner.get_direction("scene.jpg", "A", "B", mode="full")
print(result["direction"])
# {
#   "clock_direction": "3 o'clock",
#   "cardinal_direction": "East",
#   "horizontal_dist_m": 12.5,
#   "vertical_relation": "above",
#   "vertical_dist_m": 2.1,
# }
```

### 5.3 估算物体尺寸

```python
result = reasoner.get_object_size("scene.jpg", "卡车")
print(result["width_m"])   # 例如: 2.8
print(result["height_m"])  # 例如: 3.1
```

---

## 六、Benchmark 评测

### 6.1 数据格式

基准测试数据为 JSON 数组，支持 4 种任务类型：

```json
[
  {
    "image": "images/scene_01.jpg",
    "question": "What is the distance between the red car and the white van?",
    "task_type": "distance",
    "query_objects": ["red car", "white van"],
    "gt_answer": "8.5 meters",
    "gt_distance_m": 8.5
  },
  {
    "image": "images/scene_02.jpg",
    "question": "From the drone, in which direction is the building?",
    "task_type": "direction_clock",
    "query_objects": ["drone", "building"],
    "gt_answer": "3 o'clock"
  },
  {
    "image": "images/scene_03.jpg",
    "question": "Which compass direction is the car from the tree?",
    "task_type": "direction_cardinal",
    "query_objects": ["tree", "car"],
    "gt_answer": "North-East"
  },
  {
    "image": "images/scene_04.jpg",
    "question": "How wide is the truck?",
    "task_type": "size",
    "query_objects": ["truck"],
    "gt_answer": "approximately 3 meters",
    "gt_width_m": 3.0
  }
]
```

> 图像路径支持相对路径（相对于 JSON 文件目录）和绝对路径。

### 6.2 运行评测

```bash
# Mock 模式（验证流程，无需任何模型）
python benchmark_eval.py --mock --generate-sample

# 真实模型评测
export DASHSCOPE_API_KEY=sk-xxxxxxxx
python benchmark_eval.py \
    --data your_benchmark.json \
    --grounding qwen3vl_api \
    --depth depth_anything_v3 \
    --distance-threshold 30 \
    --clock-threshold 2

# 仅使用 GroundingDINO + Depth Anything V3（无需 API Key）
python benchmark_eval.py \
    --data your_benchmark.json \
    --grounding dino \
    --depth depth_anything_v3
```

### 6.3 评测输出

```
=============================================================
Benchmark 汇总 | backend=(qwen3vl_api, depth_anything_v3)
=============================================================
  distance               | pass=28/30 | acc=93.3%
  direction_clock        | pass=27/30 | acc=90.0%
  direction_cardinal     | pass=29/30 | acc=96.7%
  size                   | pass=22/30 | acc=73.3%
总耗时: 45.6s
```

---

## 七、从零到运行：快速启动

### 最小安装（Mock 模式，仅需 5 秒）

```bash
pip install numpy pillow
cd 3D数学引擎
python benchmark_eval.py --mock --generate-sample
```

### 完整安装（真实模型）

```bash
pip install openai pillow numpy transformers torch torchvision

# 设置 API Key（用于 Grounding）
export DASHSCOPE_API_KEY=sk-xxxxxxxx

# 运行测试（验证所有模块正常）
cd 3D数学引擎
python test_geometry.py   # 41 个测试，全部通过

# 运行真实评测
python benchmark_eval.py \
    --data benchmark.json \
    --grounding qwen3vl_api \
    --depth depth_anything_v3
```

---

## 八、与数据集的接口对接

### 8.1 Open3D-VQA 格式适配

如果使用的是 Open3D-VQA 或类似数据集，需将原始格式转换为本框架的 JSON 格式：

```python
import json

def convert_open3dvqa_to_benchmark(raw_data: list[dict]) -> list[dict]:
    """将 Open3D-VQA 格式转换为本框架评测格式。"""
    converted = []
    for item in raw_data:
        task_type_map = {
            "distance": "distance",
            "direction": "direction_clock",
            "counting": None,   # 本框架暂不支持
        }
        task = task_type_map.get(item.get("question_type"), "distance")
        if task is None:
            continue

        converted.append({
            "image": item["image_path"],
            "question": item["question"],
            "task_type": task,
            "query_objects": item["referred_objects"],   # 或解析自问题文本
            "gt_answer": item["answer"],
            "gt_distance_m": item.get("gt_distance"),   # 距离任务的真实值
        })
    return converted

with open("open3dvqa_val.json") as f:
    raw = json.load(f)

converted = convert_open3dvqa_to_benchmark(raw)
with open("benchmark_converted.json", "w") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)
```

### 8.2 相机内参接入

如果数据集提供相机内参（常见于 KITTI、nuScenes 等格式）：

```python
# 从数据集 JSON 中读取内参
cam_params = dataset_meta["camera_intrinsics"]
intrinsics = (
    cam_params["fx"],  # 水平焦距（像素）
    cam_params["fy"],  # 垂直焦距（像素）
    cam_params["cx"],  # 主点 x（通常约为图像宽度的一半）
    cam_params["cy"],  # 主点 y（通常约为图像高度的一半）
)

reasoner = SpatialReasoner(
    grounding_backend="qwen3vl_api",
    depth_backend="depth_anything_v3",
    camera_intrinsics=intrinsics,   # ← 传入真实内参，显著提升距离精度
)
```

---

## 九、文件结构

```
视觉定位_Grounding/
├── grounding_tool.py      # 统一入口（自动选择后端）
├── grounding_qwen3vl.py   # Qwen3-VL Grounding（API + 本地两种方式）
├── grounding_dino.py      # GroundingDINO 后端
└── test_grounding.py      # Grounding 模块测试

3D数学引擎/
├── geometry_tools.py      # 核心数学：逆投影、距离、方向
├── depth_estimator.py     # 深度估算：MiDaS / Depth-Anything V2/V3 / Mock
├── spatial_reasoner.py    # 高层编排：Grounding + Depth + Geometry 串联
├── benchmark_eval.py      # Benchmark 评测脚本
├── test_geometry.py       # 41 个单元 + 集成测试（无需真实模型）
└── README.md              # 本文档
```

---

## 十、已知局限与改进方向

| 局限 | 说明 | 改进方向 |
|------|------|---------|
| 深度为相对值 | Depth Anything 输出相对深度，非绝对米数 | 用已知尺寸物体（如车辆）做运行时标定 |
| 单目深度误差 | 无 LiDAR/双目，深度误差 10~30% | 引入双目相机或 LiDAR 融合 |
| Grounding 定位精度 | 目标框不准时，3D 坐标误差放大 | 多次检测取置信度最高框；加 SAM 分割 mask |
| 相机内参缺失 | 90° FOV 假设对部分镜头误差较大 | 从数据集 EXIF 或标定文件读取真实内参 |
| 遮挡 | bbox 中心点可能落在遮挡区域 | 取中心 10×10 区域深度中位数（已实现） |
