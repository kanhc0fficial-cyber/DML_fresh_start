# 视觉定位_Grounding

本模块实现了两种**零样本视觉目标定位（Grounding）**方法，提供统一接口 `get_bounding_box(image, object_text)`，返回目标的像素坐标边界框 `[x_min, y_min, x_max, y_max]`。

---

## 文件结构

```
视觉定位_Grounding/
├── grounding_tool.py         # 统一接口（推荐直接使用）
├── grounding_dino.py         # 方法一：Grounding DINO（零样本检测标杆）
├── grounding_qwen3vl.py      # 方法二：Qwen3-VL（视觉语言大模型 Grounding）
├── generate_test_images.py   # 生成 10 张合成测试图（含 ground truth）
├── test_grounding.py         # 测试脚本（含 Mock 模式，无需真实模型）
├── test_images/              # 生成的测试图像及 ground_truth.json
├── visualization_results/    # 测试后的可视化输出
├── model_weights/            # 模型权重（自动下载，gitignore）
└── README.md
```

---

## 两种方法说明

### 方法一：Grounding DINO（推荐，开源无 API Key）

**Grounding DINO** 是开源界零样本（Zero-shot）目标检测的标杆模型，支持任意自然语言描述直接定位目标，无需微调。

- 项目地址：https://github.com/IDEA-Research/GroundingDINO
- 模型：`GroundingDINO-SwinT-OGC`（~172 MB，首次运行自动下载）
- 优点：纯本地运行，无需 API Key，速度快
- 注意：对英文描述效果更好（如 `"red car"` 优于 `"红色车"`）

**安装依赖：**
```bash
pip install groundingdino-py pillow torch torchvision
# 或从源码编译（GPU 加速）：
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### 方法二：Qwen3-VL（视觉语言大模型）

**Qwen3-VL** 是阿里通义千问系列的视觉语言模型，内置 Grounding 能力，只需在提示词中要求输出边界框坐标即可，无需外接检测工具。

输出格式：`<|object_ref_start|>目标<|object_ref_end|><|box_start|>(x_min,y_min),(x_max,y_max)<|box_end|>`（坐标为 0~1000 范围内的归一化整数）

- 模型：`Qwen/Qwen3-VL-7B-Instruct`（~14 GB，需要本地 GPU）
- 优点：理解中文描述能力强，多目标同时定位
- 注意：需要下载模型权重，推理显存要求约 16 GB
- 说明：当前 `transformers` 使用 `Qwen2_5_VLForConditionalGeneration` 类加载（Qwen3-VL 与 Qwen2.5-VL 共享架构），未来 transformers 独立支持 Qwen3-VL 后会自动更新

**安装依赖：**
```bash
pip install transformers accelerate torch torchvision pillow
```

---

## 快速使用

### 统一接口

```python
from grounding_tool import get_bounding_box

# 使用 Grounding DINO（默认，推荐）
box = get_bounding_box("car.jpg", "red car", backend="dino")
print(box)  # [x_min, y_min, x_max, y_max]（像素坐标）

# 使用 Qwen3-VL
box = get_bounding_box("house.jpg", "白房子", backend="qwen3vl")

# 返回所有检测框（多目标）
boxes = get_bounding_box("scene.jpg", "red car", backend="dino", return_all=True)
```

### 直接使用各后端

```python
# Grounding DINO
from grounding_dino import GroundingDINOGrounder

grounder = GroundingDINOGrounder(device="cuda", box_threshold=0.35)
box = grounder.get_bounding_box("car.jpg", "red car")

# Qwen3-VL
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
# 使用 Grounding DINO 测试 10 张图
python test_grounding.py --backend dino

# 使用 Qwen3-VL 测试
python test_grounding.py --backend qwen3vl
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

## 参数调优

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `box_threshold` | 0.35 | GroundingDINO 框置信度阈值，提高可减少误检 |
| `text_threshold` | 0.25 | GroundingDINO 文本匹配阈值 |
| `device` | "cuda" | 推理设备，无 GPU 时设为 "cpu" |
| `lazy_load` | True | 首次调用时才加载模型，节省内存 |
