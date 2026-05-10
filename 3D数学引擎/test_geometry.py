"""
单元测试 + 集成测试：3D 数学引擎

涵盖：
  1. geometry_tools 核心数学函数的纯数值正确性测试
  2. depth_estimator Mock 模式测试
  3. spatial_reasoner 端到端集成测试（Mock 模式，无需任何真实模型）
  4. benchmark_eval Mock 评测流程测试

运行方式：
    python test_geometry.py              # 运行所有测试
    python test_geometry.py --verbose    # 详细日志输出
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# 确保当前目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助工具
# ─────────────────────────────────────────────────────────────────────────────

def _make_constant_depth_map(h: int, w: int, depth: float = 10.0) -> np.ndarray:
    """生成均一深度图。"""
    return np.full((h, w), depth, dtype=np.float32)


def _make_gradient_depth_map(h: int, w: int, near: float = 5.0, far: float = 20.0) -> np.ndarray:
    """生成从上到下线性增加的深度图。"""
    row = np.linspace(near, far, h, dtype=np.float32)
    return np.tile(row.reshape(-1, 1), (1, w))


def _make_blank_image(w: int = 640, h: int = 480, color=(100, 130, 160)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


# ─────────────────────────────────────────────────────────────────────────────
# 1. geometry_tools 测试
# ─────────────────────────────────────────────────────────────────────────────

class TestUnprojection(unittest.TestCase):
    """测试 unproject_to_3d：从 2D 像素 + 深度还原 3D 坐标。"""

    def setUp(self):
        from geometry_tools import unproject_to_3d
        self.unproject = unproject_to_3d

    def test_center_pixel_returns_zero_xy(self):
        """图像中心点在给定内参下，X 和 Y 应为 0。"""
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=10.0)
        # 中心框
        bbox = (w / 2 - 1, h / 2 - 1, w / 2 + 1, h / 2 + 1)
        intrinsics = (w / 2, h / 2, w / 2, h / 2)  # f_x=f_y=cx, cy=h/2
        p = self.unproject(bbox, depth_map, intrinsics)

        self.assertAlmostEqual(p[0], 0.0, places=1, msg="中心点 X 应为 0")
        self.assertAlmostEqual(p[1], 0.0, places=1, msg="中心点 Y 应为 0")
        self.assertAlmostEqual(p[2], 10.0, places=1, msg="Z 应等于深度图值")

    def test_depth_propagation(self):
        """反投影的 Z 值应与深度图一致。"""
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=15.0)
        bbox = (100, 100, 200, 200)
        p = self.unproject(bbox, depth_map)
        self.assertAlmostEqual(p[2], 15.0, places=0)

    def test_right_of_center_positive_x(self):
        """中心右侧的目标，X 坐标应为正值。"""
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=10.0)
        # 右半部分
        bbox = (400, 200, 500, 300)
        p = self.unproject(bbox, depth_map)
        self.assertGreater(p[0], 0, "右侧目标 X 应为正")

    def test_left_of_center_negative_x(self):
        """中心左侧的目标，X 坐标应为负值。"""
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=10.0)
        bbox = (50, 200, 150, 300)
        p = self.unproject(bbox, depth_map)
        self.assertLess(p[0], 0, "左侧目标 X 应为负")

    def test_default_intrinsics_used_when_none(self):
        """不提供内参时应使用默认内参，不抛出异常。"""
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=8.0)
        bbox = (200, 200, 300, 300)
        p = self.unproject(bbox, depth_map, camera_intrinsics=None)
        self.assertEqual(len(p), 3)
        self.assertFalse(np.any(np.isnan(p)), "结果不应包含 NaN")


class TestDistanceCalculators(unittest.TestCase):
    """测试距离计算函数的数学正确性。"""

    def setUp(self):
        from geometry_tools import (
            calculate_absolute_distance,
            calculate_horizontal_distance,
            calculate_vertical_distance,
        )
        self.abs_dist = calculate_absolute_distance
        self.horiz_dist = calculate_horizontal_distance
        self.vert_dist = calculate_vertical_distance

    def test_same_point_distance_is_zero(self):
        p = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(self.abs_dist(p, p), 0.0, places=6)
        self.assertAlmostEqual(self.horiz_dist(p, p), 0.0, places=6)
        self.assertAlmostEqual(self.vert_dist(p, p), 0.0, places=6)

    def test_absolute_distance_known_value(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        self.assertAlmostEqual(self.abs_dist(p1, p2), 5.0, places=4)

    def test_horizontal_ignores_y(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 100.0, 4.0])  # Y 很大但应被忽略
        horiz = self.horiz_dist(p1, p2)
        self.assertAlmostEqual(horiz, 5.0, places=4, msg="水平距离不应包含 Y 轴")

    def test_vertical_only_y(self):
        p1 = np.array([0.0, 3.0, 0.0])
        p2 = np.array([100.0, 7.0, 100.0])  # X, Z 很大但应被忽略
        vert = self.vert_dist(p1, p2)
        self.assertAlmostEqual(vert, 4.0, places=4, msg="垂直距离只应包含 Y 轴差值")

    def test_symmetry(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 6.0, 3.0])
        self.assertAlmostEqual(self.abs_dist(p1, p2), self.abs_dist(p2, p1), places=6)


class TestObjectSize(unittest.TestCase):
    """测试 calculate_object_size 的基本行为。"""

    def test_returns_positive_dimensions(self):
        from geometry_tools import calculate_object_size
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=10.0)
        bbox = (100, 100, 300, 300)
        size = calculate_object_size(bbox, depth_map)
        self.assertIn("width_m", size)
        self.assertIn("height_m", size)
        self.assertIn("depth_m", size)
        self.assertGreater(size["width_m"], 0, "宽度应为正值")
        self.assertGreater(size["height_m"], 0, "高度应为正值")
        self.assertAlmostEqual(size["depth_m"], 10.0, places=0)


class TestDirectionCalculators(unittest.TestCase):
    """测试方向计算函数的语义正确性。"""

    def setUp(self):
        from geometry_tools import (
            is_left_or_right_of_camera,
            calculate_clock_direction,
            calculate_cardinal_direction,
        )
        self.lr = is_left_or_right_of_camera
        self.clock = calculate_clock_direction
        self.cardinal = calculate_cardinal_direction

    def test_left_right_of_camera(self):
        self.assertEqual(self.lr(np.array([-2.0, 0.0, 5.0])), "left")
        self.assertEqual(self.lr(np.array([2.0, 0.0, 5.0])), "right")
        self.assertEqual(self.lr(np.array([0.1, 0.0, 5.0])), "center")

    def test_clock_12_oclock_straight_ahead(self):
        p_A = np.array([0.0, 0.0, 0.0])
        p_B = np.array([0.0, 0.0, 5.0])  # 正前方
        self.assertEqual(self.clock(p_A, p_B), "12 o'clock")

    def test_clock_3_oclock_right(self):
        p_A = np.array([0.0, 0.0, 0.0])
        p_B = np.array([5.0, 0.0, 0.0])  # 正右方
        self.assertEqual(self.clock(p_A, p_B), "3 o'clock")

    def test_clock_6_oclock_behind(self):
        p_A = np.array([0.0, 0.0, 0.0])
        p_B = np.array([0.0, 0.0, -5.0])  # 正后方
        self.assertEqual(self.clock(p_A, p_B), "6 o'clock")

    def test_clock_9_oclock_left(self):
        p_A = np.array([0.0, 0.0, 0.0])
        p_B = np.array([-5.0, 0.0, 0.0])  # 正左方
        self.assertEqual(self.clock(p_A, p_B), "9 o'clock")

    def test_cardinal_north(self):
        p_A = np.array([0.0, 0.0, 0.0])
        p_B = np.array([0.0, 0.0, 10.0])  # +Z 为北
        self.assertEqual(self.cardinal(p_A, p_B), "North")

    def test_cardinal_east(self):
        p_A = np.array([0.0, 0.0, 0.0])
        p_B = np.array([10.0, 0.0, 0.0])  # +X 为东
        self.assertEqual(self.cardinal(p_A, p_B), "East")

    def test_cardinal_south(self):
        p_A = np.array([0.0, 0.0, 0.0])
        p_B = np.array([0.0, 0.0, -10.0])  # -Z 为南
        self.assertEqual(self.cardinal(p_A, p_B), "South")


class TestBboxPairHelpers(unittest.TestCase):
    """测试 bbox_pair_distance 和 bbox_pair_direction 便捷函数。"""

    def setUp(self):
        from geometry_tools import bbox_pair_distance, bbox_pair_direction
        self.bbox_dist = bbox_pair_distance
        self.bbox_dir = bbox_pair_direction

    def test_distance_nonnegative(self):
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=10.0)
        bbox_A = (50, 100, 150, 200)
        bbox_B = (400, 100, 500, 200)
        dist = self.bbox_dist(bbox_A, bbox_B, depth_map)
        self.assertGreaterEqual(dist, 0.0)

    def test_direction_is_string(self):
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=10.0)
        bbox_A = (50, 200, 150, 300)
        bbox_B = (450, 200, 550, 300)
        direction = self.bbox_dir(bbox_A, bbox_B, depth_map, mode="clock")
        self.assertIsInstance(direction, str)
        self.assertIn("o'clock", direction)


# ─────────────────────────────────────────────────────────────────────────────
# 2. depth_estimator 测试
# ─────────────────────────────────────────────────────────────────────────────

class TestDepthEstimatorMock(unittest.TestCase):
    """测试 depth_estimator 的 Mock 模式。"""

    def setUp(self):
        from depth_estimator import get_depth_map, clear_estimator_cache
        self.get_depth = get_depth_map
        clear_estimator_cache()

    def test_mock_returns_correct_shape(self):
        img = _make_blank_image(640, 480)
        depth = self.get_depth(img, backend="mock")
        self.assertEqual(depth.shape, (480, 640))

    def test_mock_depth_in_expected_range(self):
        img = _make_blank_image(640, 480)
        near, far = 3.0, 15.0
        depth = self.get_depth(img, backend="mock", mock_near_depth=near, mock_far_depth=far)
        # 中心深度应接近 near_depth
        self.assertGreaterEqual(depth.min(), near - 1.0, "最小深度不应远小于 near_depth")
        # 角落距离中心约 √2，深度可超过 far_depth；允许 50% 余量
        self.assertLessEqual(depth.max(), far * 1.5 + 1.0, "最大深度不应远超 far_depth")

    def test_mock_center_shallower_than_edge(self):
        """中心区域深度应浅于边缘（默认 Mock 逻辑）。"""
        img = _make_blank_image(640, 480)
        depth = self.get_depth(img, backend="mock", mock_near_depth=5.0, mock_far_depth=20.0)
        center_depth = depth[240, 320]
        corner_depth = depth[0, 0]
        self.assertLess(center_depth, corner_depth, "中心应浅于角落")

    def test_mock_accepts_path(self):
        """Mock 后端应能接受 PIL.Image 而不抛出异常。"""
        img = _make_blank_image(320, 240)
        depth = self.get_depth(img, backend="mock")
        self.assertEqual(depth.dtype, np.float32)

    def test_unknown_backend_raises(self):
        img = _make_blank_image()
        with self.assertRaises(ValueError):
            self.get_depth(img, backend="nonexistent_backend")


# ─────────────────────────────────────────────────────────────────────────────
# 3. spatial_reasoner 集成测试（纯 Mock）
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialReasonerMock(unittest.TestCase):
    """端到端集成测试：使用 Mock grounding + Mock depth。"""

    def setUp(self):
        from spatial_reasoner import SpatialReasoner
        self.reasoner = SpatialReasoner(
            grounding_backend="mock",
            depth_backend="mock",
        )
        self.image = _make_blank_image()

    def test_measure_distance_returns_float(self):
        result = self.reasoner.measure_distance(self.image, "red car", "white van")
        self.assertIsNone(result.get("error"), f"不应有错误: {result.get('error')}")
        self.assertIsNotNone(result["distance_m"])
        self.assertGreater(result["distance_m"], 0.0)

    def test_measure_distance_horizontal_mode(self):
        result = self.reasoner.measure_distance(self.image, "car", "building", mode="horizontal")
        self.assertIsNone(result.get("error"))
        self.assertGreaterEqual(result["distance_m"], 0.0)

    def test_measure_distance_vertical_mode(self):
        result = self.reasoner.measure_distance(self.image, "drone", "ground", mode="vertical")
        self.assertIsNone(result.get("error"))
        self.assertGreaterEqual(result["distance_m"], 0.0)

    def test_get_direction_clock(self):
        result = self.reasoner.get_direction(self.image, "drone", "car", mode="clock")
        self.assertIsNone(result.get("error"))
        direction = result["direction"]
        self.assertIsInstance(direction, str)
        self.assertIn("o'clock", direction)

    def test_get_direction_cardinal(self):
        result = self.reasoner.get_direction(self.image, "observer", "target", mode="cardinal")
        self.assertIsNone(result.get("error"))
        valid_directions = [
            "North", "South", "East", "West",
            "North-East", "North-West", "South-East", "South-West",
        ]
        self.assertIn(result["direction"], valid_directions)

    def test_get_direction_full(self):
        result = self.reasoner.get_direction(self.image, "A", "B", mode="full")
        self.assertIsNone(result.get("error"))
        info = result["direction"]
        self.assertIn("clock_direction", info)
        self.assertIn("horizontal_dist_m", info)
        self.assertIn("vertical_relation", info)

    def test_get_object_size(self):
        result = self.reasoner.get_object_size(self.image, "truck")
        self.assertIsNone(result.get("error"))
        self.assertGreater(result["width_m"], 0)
        self.assertGreater(result["height_m"], 0)

    def test_result_includes_bboxes_and_3d_points(self):
        result = self.reasoner.measure_distance(self.image, "car", "tree")
        self.assertIn("bbox_A", result)
        self.assertIn("bbox_B", result)
        self.assertIn("point3d_A", result)
        self.assertIn("point3d_B", result)
        self.assertEqual(len(result["point3d_A"]), 3)
        self.assertEqual(len(result["point3d_B"]), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. benchmark_eval 流程测试
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkEval(unittest.TestCase):
    """测试 benchmark_eval 的 Mock 模式是否能正常运行完整流程。"""

    def test_generate_and_run_mock_benchmark(self):
        import tempfile, os
        from benchmark_eval import generate_sample_benchmark, run_benchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_json = Path(tmpdir) / "samples.json"
            # 使用 monkeypatch：将 benchmark_images 目录临时设为 tmpdir
            import benchmark_eval
            orig_dir = benchmark_eval.SCRIPT_DIR
            benchmark_eval.SCRIPT_DIR = Path(tmpdir)

            try:
                samples = generate_sample_benchmark(sample_json, n_samples=4)
                self.assertEqual(len(samples), 4)

                summary = run_benchmark(
                    samples,
                    grounding_backend="mock",
                    depth_backend="mock",
                    base_dir=Path(tmpdir),
                )
            finally:
                benchmark_eval.SCRIPT_DIR = orig_dir

        self.assertIn("total_samples", summary)
        self.assertEqual(summary["total_samples"], 4)
        self.assertIn("tasks", summary)
        for task, stats in summary["tasks"].items():
            self.assertIn("accuracy_pct", stats)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 边界情况测试
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):
    """边界情况：极端输入、零深度、边缘框等。"""

    def test_zero_depth_does_not_crash(self):
        """深度为 0 时不应崩溃（物体在摄像机处）。"""
        from geometry_tools import unproject_to_3d
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=0.0)
        bbox = (200, 200, 300, 300)
        p = unproject_to_3d(bbox, depth_map)
        self.assertFalse(np.any(np.isnan(p)), "零深度时不应产生 NaN")

    def test_edge_bbox_does_not_crash(self):
        """位于图像边缘的框不应崩溃。"""
        from geometry_tools import unproject_to_3d
        h, w = 480, 640
        depth_map = _make_constant_depth_map(h, w, depth=5.0)
        edge_bboxes = [
            (0, 0, 50, 50),           # 左上角
            (590, 0, 640, 50),        # 右上角
            (0, 430, 50, 480),        # 左下角
            (590, 430, 640, 480),     # 右下角
        ]
        for bbox in edge_bboxes:
            p = unproject_to_3d(bbox, depth_map)
            self.assertFalse(np.any(np.isnan(p)), f"边缘框 {bbox} 不应产生 NaN")

    def test_clock_direction_consistency(self):
        """钟表方向应在 1~12 之间，且结果字符串格式正确。"""
        from geometry_tools import calculate_clock_direction
        rng = np.random.default_rng(42)
        for _ in range(20):
            p_A = rng.uniform(-10, 10, size=3)
            p_B = rng.uniform(-10, 10, size=3)
            result = calculate_clock_direction(p_A, p_B)
            self.assertRegex(result, r"^(1[0-2]|[1-9]) o'clock$")

    def test_depth_map_3d_channel_handled(self):
        """深度图若为 (H, W, 1) 格式，应能正确处理。"""
        from geometry_tools import unproject_to_3d
        h, w = 240, 320
        depth_map_3d = np.full((h, w, 1), 8.0, dtype=np.float32)
        bbox = (100, 80, 200, 160)
        p = unproject_to_3d(bbox, depth_map_3d)
        self.assertAlmostEqual(p[2], 8.0, places=0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D 数学引擎测试套件")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志输出")
    args, remaining = parser.parse_known_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2 if args.verbose else 1)
