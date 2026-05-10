"""
空间推理器（Spatial Reasoner）

高层编排模块，将以下三个子工具串联为完整的空间推理流水线：
  1. 视觉定位_Grounding — 从图像中定位目标，返回 2D 边界框
  2. depth_estimator   — 估算深度图（相对米数）
  3. geometry_tools    — 反投影 + 几何计算（距离、方向、尺寸）

公共接口：
    from spatial_reasoner import SpatialReasoner

    reasoner = SpatialReasoner(grounding_backend="mock", depth_backend="mock")
    result = reasoner.measure_distance(image, "red car", "white van")
    result = reasoner.get_direction(image, "drone", "building")
    result = reasoner.get_object_size(image, "truck")

也可以直接调用独立函数（无需实例化）：
    from spatial_reasoner import measure_distance, get_direction
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from PIL import Image

# 内部模块
from geometry_tools import (
    BBox2D,
    CameraIntrinsics,
    bbox_pair_direction,
    bbox_pair_distance,
    calculate_object_size,
    calculate_relative_position,
    unproject_to_3d,
)
from depth_estimator import get_depth_map

logger = logging.getLogger(__name__)

GroundingBackend = Literal["qwen3vl_api", "dino", "qwen3vl_local", "mock"]
DepthBackend = Literal["midas", "depth_anything", "mock"]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def _get_bbox_grounding(
    image: Image.Image,
    object_text: str,
    backend: GroundingBackend,
    **kwargs,
) -> Optional[BBox2D]:
    """调用 grounding 后端，返回 (x_min, y_min, x_max, y_max) 或 None。"""
    if backend == "mock":
        # Mock：根据描述哈希值在图中放置一个假框
        w, h = image.size
        import hashlib
        seed = int(hashlib.md5(object_text.encode()).hexdigest(), 16) % (2 ** 31)
        rng = np.random.default_rng(seed)
        cx = rng.uniform(0.2, 0.8) * w
        cy = rng.uniform(0.2, 0.8) * h
        half_w = rng.uniform(0.05, 0.15) * w
        half_h = rng.uniform(0.05, 0.15) * h
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    # 真实 grounding：需要 grounding_tool 在 Python 路径中
    _ensure_grounding_in_path()
    from grounding_tool import get_bounding_box  # type: ignore[import]

    box = get_bounding_box(image, object_text, backend=backend, **kwargs)
    if box is None:
        return None
    return tuple(box)  # type: ignore[return-value]


def _ensure_grounding_in_path() -> None:
    """将视觉定位_Grounding 目录加入 sys.path，使 grounding_tool 可导入。"""
    grounding_dir = Path(__file__).parent.parent / "视觉定位_Grounding"
    if str(grounding_dir) not in sys.path:
        sys.path.insert(0, str(grounding_dir))


# ─────────────────────────────────────────────────────────────────────────────
# 核心类：SpatialReasoner
# ─────────────────────────────────────────────────────────────────────────────

class SpatialReasoner:
    """端到端空间推理器。

    Parameters
    ----------
    grounding_backend : GroundingBackend
        视觉定位后端。"mock" 不需要任何模型，用于开发/测试。
    depth_backend : DepthBackend
        深度估算后端。"mock" 不需要任何模型。
    camera_intrinsics : (f_x, f_y, c_x, c_y) or None
        相机内参。None 时按 90° FOV 估算。
    depth_scale_factor : float
        深度图缩放因子（将相对深度映射为大约多少米）。
    grounding_kwargs : dict
        传递给 grounding 后端的额外参数（如 api_key）。
    depth_kwargs : dict
        传递给 depth 后端的额外参数（如 model_type）。
    """

    def __init__(
        self,
        grounding_backend: GroundingBackend = "qwen3vl_api",
        depth_backend: DepthBackend = "midas",
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        depth_scale_factor: float = 10.0,
        grounding_kwargs: Optional[dict] = None,
        depth_kwargs: Optional[dict] = None,
    ):
        self.grounding_backend = grounding_backend
        self.depth_backend = depth_backend
        self.camera_intrinsics = camera_intrinsics
        self.depth_scale_factor = depth_scale_factor
        self.grounding_kwargs = grounding_kwargs or {}
        self.depth_kwargs = depth_kwargs or {}

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_depth(self, image: Image.Image) -> np.ndarray:
        return get_depth_map(
            image,
            backend=self.depth_backend,
            scale_factor=self.depth_scale_factor,
            **self.depth_kwargs,
        )

    def _get_bbox(self, image: Image.Image, description: str) -> Optional[BBox2D]:
        bbox = _get_bbox_grounding(
            image, description, self.grounding_backend, **self.grounding_kwargs
        )
        if bbox is None:
            logger.warning("未能定位目标: '%s'", description)
        return bbox

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def measure_distance(
        self,
        image: str | Path | Image.Image,
        object_A: str,
        object_B: str,
        mode: str = "absolute",
    ) -> dict[str, Any]:
        """测量两个目标之间的距离。

        Parameters
        ----------
        object_A, object_B : str
            目标描述（自然语言），如 "红色车"、"white van"。
        mode : "absolute" | "horizontal" | "vertical"
            距离类型：绝对三维距离 / 水平面距离 / 垂直高度差。

        Returns
        -------
        dict:
            "distance_m"  : float  距离值（米）
            "mode"        : str    距离类型
            "bbox_A"      : 目标 A 的 2D 边界框
            "bbox_B"      : 目标 B 的 2D 边界框
            "point3d_A"   : 目标 A 的 3D 坐标
            "point3d_B"   : 目标 B 的 3D 坐标
            "error"       : str or None  错误信息（若定位失败）
        """
        t0 = time.time()
        pil_image = _load_image(image)

        bbox_A = self._get_bbox(pil_image, object_A)
        bbox_B = self._get_bbox(pil_image, object_B)

        if bbox_A is None or bbox_B is None:
            missing = [n for n, b in [(object_A, bbox_A), (object_B, bbox_B)] if b is None]
            return {
                "distance_m": None,
                "mode": mode,
                "bbox_A": bbox_A,
                "bbox_B": bbox_B,
                "point3d_A": None,
                "point3d_B": None,
                "error": f"无法定位: {missing}",
                "elapsed_s": time.time() - t0,
            }

        depth_map = self._get_depth(pil_image)
        p_A = unproject_to_3d(bbox_A, depth_map, self.camera_intrinsics)
        p_B = unproject_to_3d(bbox_B, depth_map, self.camera_intrinsics)
        dist = bbox_pair_distance(bbox_A, bbox_B, depth_map, self.camera_intrinsics, mode=mode)

        logger.info(
            "measure_distance | A='%s' B='%s' | mode=%s | dist=%.3f m | time=%.2fs",
            object_A, object_B, mode, dist, time.time() - t0,
        )

        return {
            "distance_m": round(dist, 3),
            "mode": mode,
            "bbox_A": list(bbox_A),
            "bbox_B": list(bbox_B),
            "point3d_A": p_A.tolist(),
            "point3d_B": p_B.tolist(),
            "error": None,
            "elapsed_s": round(time.time() - t0, 3),
        }

    def get_direction(
        self,
        image: str | Path | Image.Image,
        from_object: str,
        to_object: str,
        mode: str = "clock",
    ) -> dict[str, Any]:
        """计算从 from_object 看向 to_object 的方向。

        Parameters
        ----------
        from_object : str  观察者目标描述
        to_object   : str  被观察目标描述
        mode : "clock" | "cardinal" | "lr" | "full"
            "clock"    : 钟表方位（如 "3 o'clock"）
            "cardinal" : 罗盘方向（如 "North-East"）
            "lr"       : 相对镜头的左/右（如 "left"）
            "full"     : 返回完整相对位置信息 dict

        Returns
        -------
        dict 含 "direction"、"bbox_from"、"bbox_to"、"point3d_from"、"point3d_to"
        """
        t0 = time.time()
        pil_image = _load_image(image)

        bbox_from = self._get_bbox(pil_image, from_object)
        bbox_to = self._get_bbox(pil_image, to_object)

        if bbox_from is None or bbox_to is None:
            missing = [n for n, b in [(from_object, bbox_from), (to_object, bbox_to)] if b is None]
            return {
                "direction": None,
                "mode": mode,
                "bbox_from": bbox_from,
                "bbox_to": bbox_to,
                "point3d_from": None,
                "point3d_to": None,
                "error": f"无法定位: {missing}",
                "elapsed_s": time.time() - t0,
            }

        depth_map = self._get_depth(pil_image)
        p_from = unproject_to_3d(bbox_from, depth_map, self.camera_intrinsics)
        p_to = unproject_to_3d(bbox_to, depth_map, self.camera_intrinsics)

        if mode == "full":
            direction = calculate_relative_position(p_from, p_to)
        else:
            direction = bbox_pair_direction(bbox_from, bbox_to, depth_map, self.camera_intrinsics, mode=mode)

        logger.info(
            "get_direction | from='%s' to='%s' | mode=%s | direction=%s | time=%.2fs",
            from_object, to_object, mode, direction, time.time() - t0,
        )

        return {
            "direction": direction,
            "mode": mode,
            "bbox_from": list(bbox_from),
            "bbox_to": list(bbox_to),
            "point3d_from": p_from.tolist(),
            "point3d_to": p_to.tolist(),
            "error": None,
            "elapsed_s": round(time.time() - t0, 3),
        }

    def get_object_size(
        self,
        image: str | Path | Image.Image,
        object_desc: str,
    ) -> dict[str, Any]:
        """估算目标物体的三维尺寸。

        Returns
        -------
        dict 含 "width_m"、"height_m"、"depth_m"、"bbox"
        """
        t0 = time.time()
        pil_image = _load_image(image)

        bbox = self._get_bbox(pil_image, object_desc)
        if bbox is None:
            return {
                "width_m": None,
                "height_m": None,
                "depth_m": None,
                "bbox": None,
                "error": f"无法定位: '{object_desc}'",
                "elapsed_s": time.time() - t0,
            }

        depth_map = self._get_depth(pil_image)
        size = calculate_object_size(bbox, depth_map, self.camera_intrinsics)

        logger.info(
            "get_object_size | obj='%s' | w=%.2fm h=%.2fm d=%.2fm | time=%.2fs",
            object_desc, size["width_m"], size["height_m"], size["depth_m"],
            time.time() - t0,
        )

        return {
            **size,
            "bbox": list(bbox),
            "error": None,
            "elapsed_s": round(time.time() - t0, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 模块级便捷函数（无需实例化 SpatialReasoner）
# ─────────────────────────────────────────────────────────────────────────────

def measure_distance(
    image: str | Path | Image.Image,
    object_A: str,
    object_B: str,
    mode: str = "absolute",
    grounding_backend: GroundingBackend = "mock",
    depth_backend: DepthBackend = "mock",
    **kwargs,
) -> dict[str, Any]:
    """便捷入口：测量两目标之间的距离（默认使用 mock 后端）。"""
    return SpatialReasoner(
        grounding_backend=grounding_backend,
        depth_backend=depth_backend,
        **kwargs,
    ).measure_distance(image, object_A, object_B, mode=mode)


def get_direction(
    image: str | Path | Image.Image,
    from_object: str,
    to_object: str,
    mode: str = "clock",
    grounding_backend: GroundingBackend = "mock",
    depth_backend: DepthBackend = "mock",
    **kwargs,
) -> dict[str, Any]:
    """便捷入口：获取两目标之间的方向关系（默认使用 mock 后端）。"""
    return SpatialReasoner(
        grounding_backend=grounding_backend,
        depth_backend=depth_backend,
        **kwargs,
    ).get_direction(image, from_object, to_object, mode=mode)
