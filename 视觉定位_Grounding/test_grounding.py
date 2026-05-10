"""
测试脚本：测试 get_bounding_box 函数在 10 张图上的定位效果。

功能：
1. 自动生成（或复用）10 张测试图像（含"红色车"和"白房子"）
2. 分别用两种方法（Qwen3-VL、Grounding DINO）进行定位
3. 计算 IoU，评估定位准确性
4. 输出结果表格并保存可视化图像

注意：
- 实际调用大模型（Qwen3-VL 或 GroundingDINO）需要相应的环境和权重。
- 本脚本支持 --mock 模式，可在无模型时验证接口逻辑（返回模拟预测框）。

用法：
    # 真实推理（需要已安装 groundingdino-py 或 transformers）
    python test_grounding.py --backend dino
    python test_grounding.py --backend qwen3vl

    # Mock 模式（无需任何模型，测试流程与指标计算）
    python test_grounding.py --mock
    python test_grounding.py --mock --backend qwen3vl
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
GT_JSON = TEST_IMAGES_DIR / "ground_truth.json"
VIZ_DIR = SCRIPT_DIR / "visualization_results"

# IoU 通过阈值
IOU_PASS_THRESHOLD = 0.5


# ──────────────────────────────────────────────
# IoU 计算
# ──────────────────────────────────────────────

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """计算两个 [x_min, y_min, x_max, y_max] 边界框的 IoU。"""
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_max - x_min)
    inter_h = max(0.0, y_max - y_min)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


# ──────────────────────────────────────────────
# Mock 预测（用于无模型测试）
# ──────────────────────────────────────────────

def _mock_predict(gt_box: list[float], noise: float = 0.10) -> list[float]:
    """在真实框基础上加入随机噪声，模拟模型预测（IoU ≈ 0.7~0.9）。"""
    import random
    random.seed(sum(int(v) for v in gt_box))
    x_min, y_min, x_max, y_max = gt_box
    w = x_max - x_min
    h = y_max - y_min
    dx = noise * w * random.uniform(-1, 1)
    dy = noise * h * random.uniform(-1, 1)
    dw = noise * w * random.uniform(-0.5, 0.5)
    dh = noise * h * random.uniform(-0.5, 0.5)
    return [
        max(0.0, x_min + dx),
        max(0.0, y_min + dy),
        x_max + dx + dw,
        y_max + dy + dh,
    ]


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────

def _visualize(
    image_path: Path,
    gt_box: list[float],
    pred_box: Optional[list[float]],
    iou: float,
    save_path: Path,
) -> None:
    """在图像上绘制真实框（绿色）和预测框（红色），保存结果。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("Pillow 未安装，跳过可视化")
        return

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 真实框（绿色）
    draw.rectangle(gt_box, outline=(0, 255, 0), width=3)
    draw.text((gt_box[0], max(0, gt_box[1] - 16)), "GT", fill=(0, 255, 0))

    # 预测框（红色）
    if pred_box is not None:
        draw.rectangle(pred_box, outline=(255, 80, 80), width=3)
        draw.text(
            (pred_box[0], max(0, pred_box[1] - 16)),
            f"Pred IoU={iou:.2f}",
            fill=(255, 80, 80),
        )
    else:
        draw.text((10, 30), "No detection", fill=(255, 0, 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(save_path))


# ──────────────────────────────────────────────
# 主测试流程
# ──────────────────────────────────────────────

def run_tests(backend: str = "dino", mock: bool = False) -> None:
    # Step 0: 生成测试图像（若不存在）
    if not GT_JSON.exists():
        logger.info("测试图像不存在，正在生成...")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "generate_test_images.py")],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error("生成图像失败:\n%s", result.stderr)
            sys.exit(1)
        logger.info(result.stdout.strip())

    # Step 1: 加载 ground truth
    with open(GT_JSON, encoding="utf-8") as f:
        ground_truth: dict = json.load(f)

    logger.info("=" * 60)
    logger.info("测试开始 | backend=%s | mock=%s", backend, mock)
    logger.info("=" * 60)

    # Step 2: 初始化 grounder（非 mock 时）
    grounder = None
    if not mock:
        try:
            if backend == "dino":
                from grounding_dino import GroundingDINOGrounder
                grounder = GroundingDINOGrounder(lazy_load=True)
            elif backend == "qwen3vl":
                from grounding_qwen3vl import Qwen3VLGrounder
                # lazy_load=True 延迟至首张图推理时才加载模型，节省无图时的内存占用
                grounder = Qwen3VLGrounder(lazy_load=True)
            else:
                raise ValueError(f"未知 backend: {backend}")
        except ImportError as e:
            logger.error("导入失败: %s\n请安装依赖或使用 --mock 模式运行", e)
            sys.exit(1)

    # Step 3: 逐图测试
    results = []
    for filename, meta in ground_truth.items():
        img_path = TEST_IMAGES_DIR / filename
        gt_box = meta["gt_primary_box"]
        target = meta["target"]

        t0 = time.time()

        if mock:
            pred_box = _mock_predict(gt_box)
        else:
            try:
                pred_box = grounder.get_bounding_box(img_path, target)
            except Exception as e:
                logger.warning("图像 %s 推理异常: %s", filename, e)
                pred_box = None

        elapsed = time.time() - t0

        iou = compute_iou(gt_box, pred_box) if pred_box is not None else 0.0
        passed = iou >= IOU_PASS_THRESHOLD

        results.append({
            "image": filename,
            "target": target,
            "gt_box": gt_box,
            "pred_box": pred_box,
            "iou": iou,
            "passed": passed,
            "elapsed_s": elapsed,
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            "[%s] %s | target='%s' | IoU=%.3f | time=%.2fs",
            status, filename, target, iou, elapsed,
        )

        # 可视化
        viz_name = f"{Path(filename).stem}_viz.png"
        _visualize(img_path, gt_box, pred_box, iou, VIZ_DIR / viz_name)

    # Step 4: 汇总报告
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    avg_iou = sum(r["iou"] for r in results) / total if total > 0 else 0.0
    avg_time = sum(r["elapsed_s"] for r in results) / total if total > 0 else 0.0

    # 分 target 统计
    for tgt in sorted({r["target"] for r in results}):
        subset = [r for r in results if r["target"] == tgt]
        tgt_pass = sum(1 for r in subset if r["passed"])
        tgt_iou = sum(r["iou"] for r in subset) / len(subset)
        logger.info(
            "  目标='%s' | %d/%d pass | avg IoU=%.3f",
            tgt, tgt_pass, len(subset), tgt_iou,
        )

    logger.info("=" * 60)
    logger.info(
        "总结 | backend=%s | pass=%d/%d | avg_IoU=%.3f | avg_time=%.2fs",
        backend, passed_count, total, avg_iou, avg_time,
    )
    logger.info("可视化结果已保存至: %s", VIZ_DIR)
    logger.info("=" * 60)

    # Step 5: 保存 JSON 结果
    result_json = SCRIPT_DIR / f"test_results_{backend}{'_mock' if mock else ''}.json"
    serializable = []
    for r in results:
        serializable.append({
            k: (v if not hasattr(v, "tolist") else v.tolist())
            for k, v in r.items()
        })
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info("详细结果已保存至: %s", result_json)

    # 如果有 FAIL，以非零退出码退出
    if passed_count < total:
        sys.exit(1)


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="测试 get_bounding_box 函数（Qwen3-VL / Grounding DINO）"
    )
    parser.add_argument(
        "--backend",
        choices=["dino", "qwen3vl"],
        default="dino",
        help="使用的后端模型（默认: dino）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 Mock 预测（无需真实模型，用于验证测试流程）",
    )
    args = parser.parse_args()
    run_tests(backend=args.backend, mock=args.mock)
