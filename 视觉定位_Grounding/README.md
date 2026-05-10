# 视觉定位_Grounding

本模块实现了**两种零样本视觉目标定位（Grounding）**方法，提供统一接口 `get_bounding_box(image, object_text)`，返回目标的像素坐标边界框 `[x_min, y_min, x_max, y_max]`。

---

## 文件结构

```
视觉定位_Grounding/
├── grounding_tool.py         # 统一接口（推荐直接使用）
├── grounding_qwen3vl.py      # 方法一：Qwen3-VL（API + 本地推理）
├── grounding_dino.py         # 方法二：Grounding DINO（开源零样本检测标杆）
├── generate_test_images.py   # 生成 10 张合成测试图（含 ground truth）
├── test_grounding.py         # 测试脚本（含 Mock 模式，无需真实模型）
├── test_images/              # 生成的测试图像及 ground_truth.json
├── visualization_results/    # 测试后的可视化输出
├── model_weights/            # 模型权重（自动下载，gitignore）
└── README.md
```

---

## 三种后端说明

### 方式一（推荐）：Qwen3-VL API

通过阿里云 **DashScope OpenAI 兼容接口**调用 Qwen3-VL，无需本地 GPU 或模型权重。

- API 端点：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 模型名：`qwen-vl-max`（默认）或 `qwen3-vl-72b-instruct` 等
- 优点：即用即调，中文理解能力强，无需本地资源
- 要求：需要 DashScope API Key（[申请地址](https://bailian.console.aliyun.com/)）

**安装依赖：**
```bash
pip install openai pillow
```

**设置 API Key：**
```bash
export DASHSCOPE_API_KEY=sk-xxxxxxxx
```

### 方式二：Grounding DINO

开源界零样本（Zero-shot）目标检测的标杆模型，支持任意自然语言描述直接定位，无需微调和 API Key。

- 项目地址：https://github.com/IDEA-Research/GroundingDINO
- 模型：`GroundingDINO-SwinT-OGC`（~172 MB，首次运行自动下载）
- 优点：纯本地运行，无需任何网络依赖，速度快
- 注意：对英文描述效果更佳（如 `"red car"` 优于 `"红色车"`）

**安装依赖：**
```bash
pip install groundingdino-py pillow torch torchvision
# 或从源码编译（GPU 加速）：
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### 方式三：Qwen3-VL 本地推理（离线部署）

本地加载 Qwen3-VL 模型权重进行推理，无需 API Key 和网络，适合离线或私有化部署场景。

- 模型：`Qwen/Qwen3-VL-7B-Instruct`（~14 GB）
- 优点：数据不出本地，适合对数据安全有要求的场景
- 要求：GPU 显存约 16 GB
- 说明：当前 `transformers` 使用 `Qwen2_5_VLForConditionalGeneration` 类加载（Qwen3-VL 与 Qwen2.5-VL 共享架构）

**安装依赖：**
```bash
pip install transformers accelerate torch torchvision pillow
```

---

## 快速使用

### 统一接口

```python
from grounding_tool import get_bounding_box

# 方式一：Qwen3-VL API（默认，需要 DASHSCOPE_API_KEY）
box = get_bounding_box("car.jpg", "红色车")
box = get_bounding_box("car.jpg", "红色车", backend="qwen3vl_api")
print(box)  # [x_min, y_min, x_max, y_max]（像素坐标）

# 方式二：Grounding DINO
box = get_bounding_box("car.jpg", "red car", backend="dino")

# 方式三：Qwen3-VL 本地推理
box = get_bounding_box("car.jpg", "红色车", backend="qwen3vl_local")

# 返回所有检测框（多目标）
boxes = get_bounding_box("scene.jpg", "红色车", return_all=True)
```

### 直接使用各后端

```python
# Qwen3-VL API
from grounding_qwen3vl import Qwen3VLAPIGrounder

grounder = Qwen3VLAPIGrounder(api_key="sk-xxx", model="qwen-vl-max")
box = grounder.get_bounding_box("car.jpg", "红色车")

# Grounding DINO
from grounding_dino import GroundingDINOGrounder

grounder = GroundingDINOGrounder(device="cuda", box_threshold=0.35)
box = grounder.get_bounding_box("car.jpg", "red car")

# Qwen3-VL 本地推理
from grounding_qwen3vl import Qwen3VLGrounder

grounder = Qwen3VLGrounder(model_name_or_path="Qwen/Qwen3-VL-7B-Instruct")
box = grounder.get_bounding_box("car.jpg", "红色车")
```

---

## 测试

### Mock 模式（无需任何模型）

验证测试流程和 IoU 计算逻辑：

```bash
python test_grounding.py --mock
```

### 真实模型测试

```bash
# 使用 Qwen3-VL API（推荐）
export DASHSCOPE_API_KEY=sk-xxxxxxxx
python test_grounding.py --backend qwen3vl_api

# 使用 Grounding DINO
python test_grounding.py --backend dino

# 使用 Qwen3-VL 本地推理
python test_grounding.py --backend qwen3vl_local
```

测试脚本会：
1. 自动生成 10 张合成测试图（含"红色车"和"白房子"，附 ground truth 坐标）
2. 调用模型进行定位预测
3. 计算 IoU（≥ 0.5 视为通过）
4. 输出结果表格，保存可视化图像到 `visualization_results/`
5. 将详细结果保存为 JSON 文件

---

## 测试图像说明

10 张合成图涵盖以下场景：

| # | 文件名 | 目标 | 说明 |
|---|--------|------|------|
| 1 | test_01_red_car_only | 红色车 | 单一红色车 |
| 2 | test_02_white_house_only | 白房子 | 单一白色房子 |
| 3 | test_03_car_and_house | 红色车 | 车+房子混合 |
| 4 | test_04_car_and_house_2 | 白房子 | 车+房子混合 |
| 5 | test_05_two_cars | 红色车 | 两辆红色车 |
| 6 | test_06_two_houses | 白房子 | 两栋白房子 |
| 7 | test_07_car_large | 红色车 | 大尺寸红色车 |
| 8 | test_08_house_large | 白房子 | 大尺寸白房子 |
| 9 | test_09_car_corner | 红色车 | 角落位置红色车 |
| 10 | test_10_house_corner | 白房子 | 角落位置白房子 |

---

## 参数参考

### Qwen3VLAPIGrounder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `api_key` | 环境变量 `DASHSCOPE_API_KEY` | DashScope API Key |
| `model` | `"qwen-vl-max"` | API 模型名称 |
| `base_url` | DashScope 地址 | API 端点 |
| `max_tokens` | 256 | 最大生成 token 数 |
| `image_format` | `"JPEG"` | 图像编码格式 |

### GroundingDINOGrounder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `box_threshold` | 0.35 | 框置信度阈值，提高可减少误检 |
| `text_threshold` | 0.25 | 文本匹配阈值 |
| `device` | `"cuda"` | 推理设备，无 GPU 时设为 `"cpu"` |
| `lazy_load` | True | 首次调用时才加载模型 |

### Qwen3VLGrounder（本地）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name_or_path` | `"Qwen/Qwen3-VL-7B-Instruct"` | 模型路径或 Hub 名称 |
| `device` | `"cuda"` | 推理设备 |
| `lazy_load` | True | 首次调用时才加载模型 |
