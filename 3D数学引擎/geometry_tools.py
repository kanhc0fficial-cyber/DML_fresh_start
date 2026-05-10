"""
3D 数学引擎 — 核心几何工具库

将深度图 + 2D 边界框转换为 3D 坐标，并提供：
  - 逆投影算子（Unprojection）：2D 像素 + 深度 → 3D 坐标
  - 距离计算器（Distance Calculator）：绝对/水平/垂直距离、物体尺寸
  - 方位推断器（Direction Calculator）：镜头视角方向、钟表方位、N/S/E/W 方向

公共接口（推荐入口）：
    from geometry_tools import (
        unproject_to_3d,
        calculate_absolute_distance,
        calculate_horizontal_distance,
        calculate_vertical_distance,
        calculate_object_size,
        is_left_or_right_of_camera,
        calculate_clock_direction,
        calculate_cardinal_direction,
    )
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 辅助类型
# ─────────────────────────────────────────────────────────────────────────────

Point3D = np.ndarray  # shape (3,)  对应 [X, Y, Z]
BBox2D = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) 像素坐标
CameraIntrinsics = tuple[float, float, float, float]  # (f_x, f_y, c_x, c_y)


# ─────────────────────────────────────────────────────────────────────────────
# 核心模块 1：逆投影算子 (Unprojection)
# ─────────────────────────────────────────────────────────────────────────────

def _default_intrinsics(depth_map: np.ndarray) -> CameraIntrinsics:
    """根据图像尺寸推算默认相机内参（假设水平视野角约 90°）。"""
    h, w = depth_map.shape[:2]
    c_x = w / 2.0
    c_y = h / 2.0
    # 水平 FOV ≈ 90° → f_x = w / (2 * tan(45°)) = w / 2
    f_x = f_y = w / 2.0
    return f_x, f_y, c_x, c_y


def _robust_depth(
    depth_map: np.ndarray,
    u: int,
    v: int,
    patch: int = 5,
) -> float:
    """从深度图中稳健地提取一个深度值。

    取 (u, v) 为中心的 patch×patch 区域的中位数，
    防止单点噪声（例如玻璃穿透、遮挡边缘）影响结果。
    """
    h, w = depth_map.shape[:2]
    v0 = max(0, v - patch)
    v1 = min(h, v + patch)
    u0 = max(0, u - patch)
    u1 = min(w, u + patch)
    region = depth_map[v0:v1, u0:u1]
    if region.size == 0:
        return float(depth_map[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)])
    return float(np.median(region))


def unproject_to_3d(
    bbox: BBox2D,
    depth_map: np.ndarray,
    camera_intrinsics: Optional[CameraIntrinsics] = None,
    depth_patch: int = 5,
) -> Point3D:
    """将 2D 边界框反投影为相机坐标系中的 3D 点。

    Parameters
    ----------
    bbox : (x_min, y_min, x_max, y_max)
        像素坐标边界框（与 grounding_tool.get_bounding_box 输出格式相同）。
    depth_map : np.ndarray  shape (H, W) 或 (H, W, 1)
        深度图，每像素值为该点到相机的估计距离（米）。
    camera_intrinsics : (f_x, f_y, c_x, c_y) or None
        相机内参。None 时自动从图像尺寸估算（假设 90° 水平 FOV）。
    depth_patch : int
        取中心点附近 patch×patch 像素的深度中位数以提高鲁棒性。

    Returns
    -------
    np.ndarray  shape (3,)
        3D 坐标 [X, Y, Z]（相机坐标系，Z 轴朝前，X 轴朝右，Y 轴朝下）。
    """
    x_min, y_min, x_max, y_max = bbox
    u = int((x_min + x_max) / 2)
    v = int((y_min + y_max) / 2)

    if depth_map.ndim == 3:
        depth_map = depth_map[:, :, 0]

    Z = _robust_depth(depth_map, u, v, patch=depth_patch)

    if camera_intrinsics is None:
        f_x, f_y, c_x, c_y = _default_intrinsics(depth_map)
    else:
        f_x, f_y, c_x, c_y = camera_intrinsics

    X = (u - c_x) * Z / f_x
    Y = (v - c_y) * Z / f_y

    logger.debug(
        "unproject_to_3d | bbox=%s → uv=(%d,%d) | Z=%.3f | XYZ=(%.3f, %.3f, %.3f)",
        bbox, u, v, Z, X, Y, Z,
    )
    return np.array([X, Y, Z], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 核心模块 2：距离计算器 (Distance Calculator)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_absolute_distance(p1: Point3D, p2: Point3D) -> float:
    """计算两点之间的三维欧氏距离（直线距离）。

    对应 "Distance reasoning" 任务（含高度差）。
    """
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def calculate_horizontal_distance(p1: Point3D, p2: Point3D) -> float:
    """计算两点在水平面（XZ 平面）上的距离，忽略高度差（Y 轴）。

    对应 "Horizontal distance" 任务（无人机俯视平面距离估计）。
    """
    p1, p2 = np.asarray(p1), np.asarray(p2)
    return float(math.dist([p1[0], p1[2]], [p2[0], p2[2]]))


def calculate_vertical_distance(p1: Point3D, p2: Point3D) -> float:
    """计算两点之间的绝对高度差（Y 轴方向）。

    对应 "Vertical distance / height difference" 任务。
    注意：在相机坐标系中 Y 轴朝下，值越小越高。
    """
    return float(abs(np.asarray(p1)[1] - np.asarray(p2)[1]))


def calculate_object_size(
    bbox: BBox2D,
    depth_map: np.ndarray,
    camera_intrinsics: Optional[CameraIntrinsics] = None,
) -> dict[str, float]:
    """估算物体的三维尺寸（宽度、高度、深度）。

    通过对边界框的四个角点分别反投影，计算其在 3D 空间的跨度。

    Returns
    -------
    dict 包含 "width_m"（水平宽度）、"height_m"（垂直高度）、
    "depth_m"（沿 Z 轴的估计深度，等于中心点 Z 值）。
    """
    if depth_map.ndim == 3:
        depth_map = depth_map[:, :, 0]

    x_min, y_min, x_max, y_max = bbox

    bbox_left = (x_min, y_min, x_min + 1, y_max)
    bbox_right = (x_max - 1, y_min, x_max, y_max)
    bbox_top = (x_min, y_min, x_max, y_min + 1)
    bbox_bottom = (x_min, y_max - 1, x_max, y_max)

    p_left = unproject_to_3d(bbox_left, depth_map, camera_intrinsics)
    p_right = unproject_to_3d(bbox_right, depth_map, camera_intrinsics)
    p_top = unproject_to_3d(bbox_top, depth_map, camera_intrinsics)
    p_bottom = unproject_to_3d(bbox_bottom, depth_map, camera_intrinsics)
    p_center = unproject_to_3d(bbox, depth_map, camera_intrinsics)

    width_m = calculate_absolute_distance(p_left, p_right)
    height_m = calculate_absolute_distance(p_top, p_bottom)
    depth_m = float(p_center[2])

    return {
        "width_m": width_m,
        "height_m": height_m,
        "depth_m": depth_m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 核心模块 3：方位推断器 (Direction Calculator)
# ─────────────────────────────────────────────────────────────────────────────

def is_left_or_right_of_camera(
    p_target: Point3D,
    threshold: float = 0.5,
) -> str:
    """判断目标相对于镜头（相机）的左右方位。

    在相机坐标系中，X < 0 为左，X > 0 为右。

    Parameters
    ----------
    p_target : np.ndarray
        目标的 3D 坐标 [X, Y, Z]。
    threshold : float
        判断"居中"的 X 轴宽容范围（米）。

    Returns
    -------
    "left" | "right" | "center"
    """
    x = float(np.asarray(p_target)[0])
    if x < -threshold:
        return "left"
    elif x > threshold:
        return "right"
    return "center"


def is_above_or_below_camera(
    p_target: Point3D,
    threshold: float = 0.5,
) -> str:
    """判断目标相对于镜头的上下方位。

    在相机坐标系中，Y < 0 为上（相机上方），Y > 0 为下。
    """
    y = float(np.asarray(p_target)[1])
    if y < -threshold:
        return "above"
    elif y > threshold:
        return "below"
    return "level"


def calculate_clock_direction(p_A: Point3D, p_B: Point3D) -> str:
    """计算站在 A 点向前看时，B 点在几点钟方向。

    此函数处理的是最难的"视角转换方向推理"任务（Allocentric Direction）：
    以 A 为原点，使用 atan2 将 B 的相对位置映射为钟表方位。

    约定：
        - 正前方 (delta_Z > 0) → 12 点
        - 右侧   (delta_X > 0) → 3 点
        - 正后方 (delta_Z < 0) → 6 点
        - 左侧   (delta_X < 0) → 9 点

    Returns
    -------
    str  例如 "3 o'clock"
    """
    p_A, p_B = np.asarray(p_A), np.asarray(p_B)
    delta_X = float(p_B[0] - p_A[0])
    delta_Z = float(p_B[2] - p_A[2])

    angle_deg = math.degrees(math.atan2(delta_X, delta_Z))
    if angle_deg < 0:
        angle_deg += 360.0

    clock_float = angle_deg / 30.0
    clock = int(round(clock_float)) % 12
    if clock == 0:
        clock = 12

    return f"{clock} o'clock"


def calculate_cardinal_direction(p_A: Point3D, p_B: Point3D) -> str:
    """计算从 A 看向 B 的罗盘方向（N/S/E/W 及其组合）。

    约定：+Z 轴对应"北（North）"，+X 轴对应"东（East）"。

    Returns
    -------
    str  例如 "North-East"
    """
    p_A, p_B = np.asarray(p_A), np.asarray(p_B)
    delta_X = float(p_B[0] - p_A[0])
    delta_Z = float(p_B[2] - p_A[2])

    angle_deg = math.degrees(math.atan2(delta_X, delta_Z))
    if angle_deg < 0:
        angle_deg += 360.0

    # 8 个方向，每个 45°
    directions = ["North", "North-East", "East", "South-East",
                  "South", "South-West", "West", "North-West"]
    idx = int(round(angle_deg / 45.0)) % 8
    return directions[idx]


def calculate_relative_position(p_A: Point3D, p_B: Point3D) -> dict[str, str | float]:
    """综合计算 B 相对于 A 的三维相对位置描述。

    Returns
    -------
    dict 包含：
        "clock_direction"   : 钟表方位（如 "3 o'clock"）
        "cardinal_direction": 罗盘方向（如 "North-East"）
        "horizontal_dist_m" : 水平距离（米）
        "vertical_relation" : "above" | "same level" | "below"（B 相对于 A 的高度）
        "vertical_dist_m"   : 高度差（米）
    """
    p_A, p_B = np.asarray(p_A), np.asarray(p_B)
    h_dist = calculate_horizontal_distance(p_A, p_B)
    v_dist = calculate_vertical_distance(p_A, p_B)
    dy = float(p_B[1] - p_A[1])

    if abs(dy) < 0.3:
        v_rel = "same level"
    elif dy < 0:
        # B 的 Y 更小 → B 更高（Y 轴朝下）
        v_rel = "above"
    else:
        v_rel = "below"

    return {
        "clock_direction": calculate_clock_direction(p_A, p_B),
        "cardinal_direction": calculate_cardinal_direction(p_A, p_B),
        "horizontal_dist_m": round(h_dist, 3),
        "vertical_relation": v_rel,
        "vertical_dist_m": round(v_dist, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 便捷包装：直接接受 bbox + depth_map
# ─────────────────────────────────────────────────────────────────────────────

def bbox_pair_distance(
    bbox_A: BBox2D,
    bbox_B: BBox2D,
    depth_map: np.ndarray,
    camera_intrinsics: Optional[CameraIntrinsics] = None,
    mode: str = "absolute",
) -> float:
    """一步计算两个边界框之间的距离。

    Parameters
    ----------
    bbox_A, bbox_B : (x_min, y_min, x_max, y_max)
    depth_map : np.ndarray  深度图
    camera_intrinsics : 相机内参（可选）
    mode : "absolute" | "horizontal" | "vertical"

    Returns
    -------
    float  距离（米）
    """
    p_A = unproject_to_3d(bbox_A, depth_map, camera_intrinsics)
    p_B = unproject_to_3d(bbox_B, depth_map, camera_intrinsics)

    if mode == "horizontal":
        return calculate_horizontal_distance(p_A, p_B)
    elif mode == "vertical":
        return calculate_vertical_distance(p_A, p_B)
    return calculate_absolute_distance(p_A, p_B)


def bbox_pair_direction(
    bbox_A: BBox2D,
    bbox_B: BBox2D,
    depth_map: np.ndarray,
    camera_intrinsics: Optional[CameraIntrinsics] = None,
    mode: str = "clock",
) -> str:
    """一步计算两个边界框之间的方向关系。

    Parameters
    ----------
    mode : "clock" | "cardinal" | "lr"
        "clock"    : 钟表方位（"3 o'clock"）
        "cardinal" : 罗盘方向（"North-East"）
        "lr"       : 左右方向（以 B 相对于镜头的位置判断）
    """
    p_A = unproject_to_3d(bbox_A, depth_map, camera_intrinsics)
    p_B = unproject_to_3d(bbox_B, depth_map, camera_intrinsics)

    if mode == "cardinal":
        return calculate_cardinal_direction(p_A, p_B)
    elif mode == "lr":
        return is_left_or_right_of_camera(p_B)
    return calculate_clock_direction(p_A, p_B)
