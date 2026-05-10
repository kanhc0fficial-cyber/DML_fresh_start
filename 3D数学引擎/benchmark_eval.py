"""
Benchmark 评测脚本

用于在 Open3D-VQA 风格的空间推理题目上评测完整 Agent 流水线。

题目格式（JSON）示例：
    [
        {
            "image": "scene_01.jpg",
            "question": "What is the distance between the red car and the white van?",
            "task_type": "distance",            # distance / direction / size
            "query_objects": ["red car", "white van"],
            "gt_answer": "8.5 meters",
            "gt_distance_m": 8.5                # 距离任务的真实值
        },
        {
            "image": "scene_02.jpg",
            "question": "From the perspective of the drone, where is the building relative to the car?",
            "task_type": "direction",
            "query_objects": ["drone", "building"],
            "gt_answer": "3 o'clock"
        }
    ]

运行方式：
    # Mock 模式（无需任何模型）：
    python benchmark_eval.py --mock

    # 真实模型（需要 API Key 和本地环境）：
    export DASHSCOPE_API_KEY=sk-xxx
    python benchmark_eval.py --data benchmark_data.json --grounding qwen3vl_api --depth depth_anything_v3

    # 自动生成示例数据并运行 Mock 评测：
    python benchmark_eval.py --mock --generate-sample
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# 评测指标
# ─────────────────────────────────────────────────────────────────────────────

def distance_relative_error(pred: float, gt: float) -> float:
    """计算距离预测的相对误差（%）。"""
    if gt == 0:
        return float("inf")
    return abs(pred - gt) / gt * 100.0


def direction_exact_match(pred: str, gt: str) -> bool:
    """方向预测精确匹配（忽略大小写和前后空格）。"""
    return pred.strip().lower() == gt.strip().lower()


def direction_clock_error(pred: str, gt: str) -> int:
    """钟表方向预测的最小跳数误差（0~6）。"""
    def parse_clock(s: str) -> int:
        s = s.strip().lower().replace(" o'clock", "").replace("o'clock", "")
        try:
            v = int(round(float(s)))
        except ValueError:
            return -1
        return v % 12

    p = parse_clock(pred)
    g = parse_clock(gt)
    if p < 0 or g < 0:
        return 12  # 无法解析时返回最大误差
    diff = abs(p - g)
    return min(diff, 12 - diff)


# ─────────────────────────────────────────────────────────────────────────────
# 示例数据生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_benchmark(out_path: Path, n_samples: int = 10) -> list[dict]:
    """生成 Mock 评测数据（纯合成，不需要真实图像）。"""
    from PIL import Image, ImageDraw
    import random

    rng = random.Random(2025)
    samples = []
    img_dir = SCRIPT_DIR / "benchmark_images"
    img_dir.mkdir(exist_ok=True)

    objects = [
        ("red car", (220, 50, 50)),
        ("blue truck", (50, 80, 220)),
        ("white van", (230, 230, 230)),
        ("yellow bus", (240, 200, 30)),
        ("green tree", (40, 160, 60)),
    ]

    task_types = ["distance", "direction_clock", "direction_cardinal", "size"]

    for i in range(n_samples):
        w, h = 640, 480
        img = Image.new("RGB", (w, h), color=(100, 130, 160))
        draw = ImageDraw.Draw(img)

        # 随机放置两个物体
        obj_A_name, obj_A_color = objects[i % len(objects)]
        obj_B_name, obj_B_color = objects[(i + 2) % len(objects)]

        ax = rng.randint(50, 300)
        ay = rng.randint(100, 380)
        bx = rng.randint(300, 580)
        by = rng.randint(100, 380)

        draw.rectangle([ax - 30, ay - 20, ax + 30, ay + 20], fill=obj_A_color)
        draw.rectangle([bx - 30, by - 20, bx + 30, by + 20], fill=obj_B_color)

        img_name = f"sample_{i + 1:02d}.png"
        img.save(str(img_dir / img_name))
        # Store path relative to the JSON file for portability
        rel_img = str(Path("benchmark_images") / img_name)

        task_type = task_types[i % len(task_types)]

        if task_type == "distance":
            # 根据像素距离估算"合理"的 GT 距离（仅供评测逻辑演示）
            px_dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            gt_dist = round(px_dist / 30.0, 1)  # 简单映射：30px ≈ 1m
            samples.append({
                "image": rel_img,
                "question": f"What is the distance between the {obj_A_name} and the {obj_B_name}?",
                "task_type": "distance",
                "query_objects": [obj_A_name, obj_B_name],
                "gt_answer": f"{gt_dist} meters",
                "gt_distance_m": gt_dist,
            })

        elif task_type == "direction_clock":
            dx = bx - ax
            dz_fake = (by - ay)  # 用 y 像素差模拟 z
            angle = math.degrees(math.atan2(dx, dz_fake if dz_fake != 0 else 1))
            if angle < 0:
                angle += 360
            clock = int(round(angle / 30)) % 12
            if clock == 0:
                clock = 12
            samples.append({
                "image": rel_img,
                "question": (
                    f"From the perspective of the {obj_A_name}, "
                    f"where is the {obj_B_name}?"
                ),
                "task_type": "direction_clock",
                "query_objects": [obj_A_name, obj_B_name],
                "gt_answer": f"{clock} o'clock",
            })

        elif task_type == "direction_cardinal":
            dx = bx - ax
            dz_fake = -(by - ay)  # 像素 Y 轴朝下，Z 轴朝前相反
            angle = math.degrees(math.atan2(dx, dz_fake if dz_fake != 0 else 1))
            if angle < 0:
                angle += 360
            dirs = ["North", "North-East", "East", "South-East",
                    "South", "South-West", "West", "North-West"]
            gt_dir = dirs[int(round(angle / 45)) % 8]
            samples.append({
                "image": rel_img,
                "question": (
                    f"In which compass direction is the {obj_B_name} from the {obj_A_name}?"
                ),
                "task_type": "direction_cardinal",
                "query_objects": [obj_A_name, obj_B_name],
                "gt_answer": gt_dir,
            })

        else:  # size
            samples.append({
                "image": rel_img,
                "question": f"How wide is the {obj_A_name}?",
                "task_type": "size",
                "query_objects": [obj_A_name],
                "gt_answer": "approximately 2 meters",
                "gt_width_m": 2.0,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info("已生成 %d 条评测样本 → %s", len(samples), out_path)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 主评测循环
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    samples: list[dict],
    grounding_backend: str = "mock",
    depth_backend: str = "mock",
    distance_error_threshold_pct: float = 30.0,
    clock_error_threshold: int = 2,
    verbose: bool = True,
    base_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """在给定样本列表上运行评测，返回汇总指标。

    Parameters
    ----------
    base_dir : Path or None
        用于解析样本中相对图像路径的基准目录。
        None 时使用脚本所在目录（SCRIPT_DIR）。
    """
    from spatial_reasoner import SpatialReasoner

    resolve_base = base_dir or SCRIPT_DIR

    reasoner = SpatialReasoner(
        grounding_backend=grounding_backend,
        depth_backend=depth_backend,
    )

    task_results: dict[str, list] = {
        "distance": [],
        "direction_clock": [],
        "direction_cardinal": [],
        "size": [],
    }

    total_start = time.time()

    for idx, sample in enumerate(samples):
        image_raw = sample["image"]
        # 解析相对路径（相对于 base_dir）
        image_path = Path(image_raw)
        if not image_path.is_absolute():
            image_path = resolve_base / image_path
        image = str(image_path)
        task_type = sample.get("task_type", "unknown")
        objects = sample.get("query_objects", [])
        gt_answer = sample.get("gt_answer", "")

        logger.info("[%d/%d] task=%s | %s", idx + 1, len(samples), task_type, sample.get("question", ""))

        try:
            if task_type == "distance":
                if len(objects) < 2:
                    raise ValueError("distance 任务需要 2 个 query_objects")
                result = reasoner.measure_distance(image, objects[0], objects[1], mode="absolute")
                pred_dist = result.get("distance_m")
                gt_dist = sample.get("gt_distance_m")

                if pred_dist is not None and gt_dist is not None:
                    err_pct = distance_relative_error(pred_dist, gt_dist)
                    passed = err_pct <= distance_error_threshold_pct
                else:
                    err_pct = None
                    passed = False

                rec = {
                    "sample_idx": idx,
                    "pred": pred_dist,
                    "gt": gt_dist,
                    "err_pct": err_pct,
                    "passed": passed,
                    "error_msg": result.get("error"),
                }
                task_results["distance"].append(rec)
                status = "✅" if passed else "❌"
                logger.info(
                    "  %s pred=%.2f m  gt=%.2f m  err=%.1f%%",
                    status,
                    pred_dist or 0,
                    gt_dist or 0,
                    err_pct or 0,
                )

            elif task_type == "direction_clock":
                if len(objects) < 2:
                    raise ValueError("direction_clock 任务需要 2 个 query_objects")
                result = reasoner.get_direction(image, objects[0], objects[1], mode="clock")
                pred_dir = result.get("direction", "")
                err_clock = direction_clock_error(str(pred_dir), gt_answer)
                passed = err_clock <= clock_error_threshold

                rec = {
                    "sample_idx": idx,
                    "pred": pred_dir,
                    "gt": gt_answer,
                    "clock_err": err_clock,
                    "passed": passed,
                    "error_msg": result.get("error"),
                }
                task_results["direction_clock"].append(rec)
                status = "✅" if passed else "❌"
                logger.info(
                    "  %s pred='%s'  gt='%s'  clock_err=%d",
                    status, pred_dir, gt_answer, err_clock,
                )

            elif task_type == "direction_cardinal":
                if len(objects) < 2:
                    raise ValueError("direction_cardinal 任务需要 2 个 query_objects")
                result = reasoner.get_direction(image, objects[0], objects[1], mode="cardinal")
                pred_dir = result.get("direction", "")
                passed = direction_exact_match(str(pred_dir), gt_answer)

                rec = {
                    "sample_idx": idx,
                    "pred": pred_dir,
                    "gt": gt_answer,
                    "passed": passed,
                    "error_msg": result.get("error"),
                }
                task_results["direction_cardinal"].append(rec)
                status = "✅" if passed else "❌"
                logger.info("  %s pred='%s'  gt='%s'", status, pred_dir, gt_answer)

            elif task_type == "size":
                if len(objects) < 1:
                    raise ValueError("size 任务需要至少 1 个 query_objects")
                result = reasoner.get_object_size(image, objects[0])
                pred_w = result.get("width_m")
                gt_w = sample.get("gt_width_m")

                if pred_w is not None and gt_w is not None:
                    err_pct = distance_relative_error(pred_w, gt_w)
                    passed = err_pct <= distance_error_threshold_pct
                else:
                    err_pct = None
                    passed = False

                rec = {
                    "sample_idx": idx,
                    "pred_width_m": pred_w,
                    "gt_width_m": gt_w,
                    "err_pct": err_pct,
                    "passed": passed,
                    "error_msg": result.get("error"),
                }
                task_results["size"].append(rec)
                status = "✅" if passed else "❌"
                logger.info(
                    "  %s pred_w=%.2f m  gt_w=%.2f m  err=%.1f%%",
                    status,
                    pred_w or 0,
                    gt_w or 0,
                    err_pct or 0,
                )

            else:
                logger.warning("未知任务类型: %s，跳过", task_type)

        except Exception as exc:
            logger.error("  ❌ 异常: %s", exc, exc_info=True)

    # ── 汇总指标 ──────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "grounding_backend": grounding_backend,
        "depth_backend": depth_backend,
        "total_samples": len(samples),
        "elapsed_s": round(time.time() - total_start, 2),
        "tasks": {},
    }

    for task, records in task_results.items():
        if not records:
            continue
        passed_cnt = sum(1 for r in records if r["passed"])
        acc = passed_cnt / len(records) * 100 if records else 0.0
        summary["tasks"][task] = {
            "n": len(records),
            "passed": passed_cnt,
            "accuracy_pct": round(acc, 1),
        }
        if task in ("distance", "size"):
            errs = [r.get("err_pct") for r in records if r.get("err_pct") is not None]
            summary["tasks"][task]["mean_err_pct"] = round(float(np.mean(errs)), 1) if errs else None
        if task == "direction_clock":
            errs = [r.get("clock_err") for r in records if r.get("clock_err") is not None]
            summary["tasks"][task]["mean_clock_err"] = round(float(np.mean(errs)), 2) if errs else None

    logger.info("=" * 60)
    logger.info("Benchmark 汇总 | backend=(%s, %s)", grounding_backend, depth_backend)
    logger.info("=" * 60)
    for task, stats in summary["tasks"].items():
        logger.info(
            "  %-22s | pass=%d/%d | acc=%.1f%%",
            task, stats["passed"], stats["n"], stats["accuracy_pct"],
        )
    logger.info("总耗时: %.2fs", summary["elapsed_s"])

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D 空间推理 Benchmark 评测脚本")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="评测数据 JSON 路径（若省略则自动生成示例数据）",
    )
    parser.add_argument(
        "--grounding",
        choices=["qwen3vl_api", "dino", "qwen3vl_local", "mock"],
        default="mock",
        help="Grounding 后端（默认: mock）",
    )
    parser.add_argument(
        "--depth",
        choices=["midas", "depth_anything_v2", "depth_anything_v3", "depth_anything", "mock"],
        default="mock",
        help="深度估算后端（默认: mock）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="强制使用 mock 后端（等价于 --grounding mock --depth mock）",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        dest="generate_sample",
        help="自动生成示例评测数据并运行",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="评测结果保存路径（JSON，默认: benchmark_results_<backend>.json）",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=30.0,
        dest="distance_threshold",
        help="距离任务通过的最大相对误差（%%，默认 30）",
    )
    parser.add_argument(
        "--clock-threshold",
        type=int,
        default=2,
        dest="clock_threshold",
        help="钟表方向任务通过的最大跳数误差（默认 2）",
    )
    args = parser.parse_args()

    if args.mock:
        args.grounding = "mock"
        args.depth = "mock"

    # 加载或生成评测数据
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            samples = json.load(f)
        logger.info("加载评测数据：%s（%d 条）", args.data, len(samples))
    else:
        sample_json = SCRIPT_DIR / "benchmark_sample_data.json"
        samples = generate_sample_benchmark(sample_json, n_samples=10)

    # 运行评测
    summary = run_benchmark(
        samples,
        grounding_backend=args.grounding,
        depth_backend=args.depth,
        distance_error_threshold_pct=args.distance_threshold,
        clock_error_threshold=args.clock_threshold,
    )

    # 保存结果
    out_path = args.output or str(
        SCRIPT_DIR / f"benchmark_results_{args.grounding}_{args.depth}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("评测结果已保存至: %s", out_path)
