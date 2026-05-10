"""
生成 10 张合成测试图像，用于测试 get_bounding_box 函数。
每张图像包含明确的"红色车"或"白房子"，并附带真实边界框坐标。

运行：
    python generate_test_images.py
输出：test_images/ 目录下的 PNG 文件 + test_images/ground_truth.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent / "test_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# 固定随机种子，确保可复现
random.seed(42)

# ──────────────────────────────────────────────
# 绘制辅助函数
# ──────────────────────────────────────────────

def _draw_red_car(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int) -> None:
    """在给定位置绘制一辆简单的红色车。"""
    # 车身
    car_body_y = y + h // 3
    draw.rectangle([x, car_body_y, x + w, y + h], fill=(200, 30, 30))
    # 车顶
    roof_margin_x = w // 5
    draw.rectangle(
        [x + roof_margin_x, y, x + w - roof_margin_x, car_body_y],
        fill=(220, 50, 50),
    )
    # 车轮（黑色圆）
    wheel_r = h // 8
    draw.ellipse(
        [x + w // 6 - wheel_r, y + h - wheel_r * 2,
         x + w // 6 + wheel_r, y + h + wheel_r * 2 - h // 4],
        fill=(30, 30, 30),
    )
    draw.ellipse(
        [x + w * 5 // 6 - wheel_r, y + h - wheel_r * 2,
         x + w * 5 // 6 + wheel_r, y + h + wheel_r * 2 - h // 4],
        fill=(30, 30, 30),
    )
    # 车窗（浅蓝色）
    win_margin = w // 10
    draw.rectangle(
        [x + roof_margin_x + win_margin, y + 4,
         x + w - roof_margin_x - win_margin, car_body_y - 4],
        fill=(173, 216, 230),
    )


def _draw_white_house(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int) -> None:
    """在给定位置绘制一栋简单的白色房子。"""
    roof_h = h // 3
    wall_h = h - roof_h

    # 墙体（白色矩形）
    draw.rectangle([x, y + roof_h, x + w, y + h], fill=(245, 245, 245), outline=(180, 180, 180))

    # 屋顶（灰色三角形）
    draw.polygon(
        [(x, y + roof_h), (x + w // 2, y), (x + w, y + roof_h)],
        fill=(160, 160, 160),
        outline=(120, 120, 120),
    )

    # 门（棕色矩形）
    door_w = w // 5
    door_h = wall_h // 2
    door_x = x + w // 2 - door_w // 2
    door_y = y + h - door_h
    draw.rectangle([door_x, door_y, door_x + door_w, y + h], fill=(139, 90, 43))

    # 窗户（浅蓝色矩形）
    win_size = w // 6
    draw.rectangle(
        [x + w // 6, y + roof_h + 10, x + w // 6 + win_size, y + roof_h + 10 + win_size],
        fill=(173, 216, 230), outline=(100, 100, 100),
    )
    draw.rectangle(
        [x + w * 4 // 6, y + roof_h + 10, x + w * 4 // 6 + win_size, y + roof_h + 10 + win_size],
        fill=(173, 216, 230), outline=(100, 100, 100),
    )


def _add_label(draw: ImageDraw.Draw, text: str, x: int, y: int) -> None:
    """在图像左上角添加文字标签。"""
    draw.text((x, y), text, fill=(255, 255, 0))


# ──────────────────────────────────────────────
# 生成 10 张测试图像
# ──────────────────────────────────────────────

IMG_W, IMG_H = 640, 480
BG_COLORS = [
    (135, 206, 235),  # 天蓝（晴天）
    (200, 220, 200),  # 浅绿（草地）
    (180, 180, 180),  # 灰色（阴天）
    (240, 230, 210),  # 米黄（沙漠/土路）
    (100, 140, 180),  # 深蓝（黄昏）
]

TEST_CASES = [
    # (描述, target_object, 主目标生成函数, 次目标生成函数或None)
    ("red_car_only",         "红色车",  "car",   None),
    ("white_house_only",     "白房子",  "house", None),
    ("car_and_house",        "红色车",  "car",   "house"),
    ("car_and_house_2",      "白房子",  "car",   "house"),
    ("two_cars",             "红色车",  "car",   "car"),
    ("two_houses",           "白房子",  "house", "house"),
    ("car_large",            "红色车",  "car",   None),
    ("house_large",          "白房子",  "house", None),
    ("car_corner",           "红色车",  "car",   None),
    ("house_corner",         "白房子",  "house", None),
]

ground_truth: dict[str, dict] = {}


def _random_box(img_w: int, img_h: int, min_size: int = 80, max_size: int = 200) -> tuple[int, int, int, int]:
    """在图像范围内随机生成一个矩形 (x, y, w, h)，确保不越界。"""
    w = random.randint(min_size, max_size)
    h = random.randint(min_size, max_size)
    x = random.randint(10, img_w - w - 10)
    y = random.randint(10, img_h - h - 10)
    return x, y, w, h


for idx, (name, target_obj, primary, secondary) in enumerate(TEST_CASES):
    img_num = idx + 1
    filename = f"test_{img_num:02d}_{name}.png"
    filepath = OUTPUT_DIR / filename

    # 背景
    bg = BG_COLORS[idx % len(BG_COLORS)]
    img = Image.new("RGB", (IMG_W, IMG_H), color=bg)
    draw = ImageDraw.Draw(img)

    # 决定主目标位置
    if name.endswith("corner"):
        # 放角落
        x, y = 20, 20
        w, h = random.randint(100, 160), random.randint(80, 130)
    elif name.startswith("car_large") or name.startswith("house_large"):
        # 大一点
        w, h = random.randint(200, 280), random.randint(160, 220)
        x = random.randint(50, IMG_W - w - 50)
        y = random.randint(50, IMG_H - h - 50)
    else:
        x, y, w, h = _random_box(IMG_W, IMG_H)

    primary_box = [x, y, x + w, y + h]

    # 绘制主目标
    if primary == "car":
        _draw_red_car(draw, x, y, w, h)
    else:
        _draw_white_house(draw, x, y, w, h)

    # 绘制次目标（不重叠）
    secondary_box = None
    if secondary is not None:
        for _ in range(30):  # 最多尝试 30 次，避免死循环
            x2, y2, w2, h2 = _random_box(IMG_W, IMG_H)
            # 简单检测：中心距足够远
            cx1, cy1 = x + w / 2, y + h / 2
            cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
            if abs(cx1 - cx2) > (w + w2) / 2 + 20 and abs(cy1 - cy2) > (h + h2) / 2 + 20:
                secondary_box = [x2, y2, x2 + w2, y2 + h2]
                if secondary == "car":
                    _draw_red_car(draw, x2, y2, w2, h2)
                else:
                    _draw_white_house(draw, x2, y2, w2, h2)
                break

    # 标注文字（帮助肉眼核查）
    _add_label(draw, f"#{img_num} target: {target_obj}", 5, 5)

    img.save(str(filepath))

    # 记录真实值
    entry: dict = {
        "file": filename,
        "target": target_obj,
        "gt_primary_box": primary_box,  # [x_min, y_min, x_max, y_max]
    }
    if secondary_box:
        entry["gt_secondary_box"] = secondary_box
    ground_truth[filename] = entry

# 保存 ground truth JSON
gt_path = OUTPUT_DIR / "ground_truth.json"
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)

print(f"✅ 已生成 {len(TEST_CASES)} 张测试图像 -> {OUTPUT_DIR}")
print(f"✅ Ground truth 已保存 -> {gt_path}")
