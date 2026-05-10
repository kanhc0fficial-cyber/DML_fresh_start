"""
工具执行器（Tool Executor）

将 ReAct Agent 发出的工具调用指令路由到实际的后端模块：
  - 视觉定位_Grounding/grounding_tool.py  → get_bounding_box
  - 3D数学引擎/spatial_reasoner.py        → 距离 / 方向 / 尺寸计算

设计原则：
  - 每个工具方法都有完整的异常捕获，任何错误均以字符串形式返回，
    而非抛出异常，保证 Agent 主循环永远不会因工具崩溃而中断。
  - 支持 "mock" 后端，无需任何模型即可在开发/CI 环境中运行。

使用方式：
    from tool_executor import ToolExecutor

    executor = ToolExecutor(
        image_path="scene.jpg",
        grounding_backend="qwen3vl_api",
        depth_backend="depth_anything_v3",
    )

    result = executor.execute("get_bounding_box", {"object_name": "红色轿车"})
    result = executor.execute("calculate_3d_distance", {"obj_1": "红色轿车", "obj_2": "路灯"})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 将兄弟目录加入 sys.path，使内部模块可导入
_REPO_ROOT = Path(__file__).parent.parent
_GROUNDING_DIR = _REPO_ROOT / "视觉定位_Grounding"
_ENGINE_DIR = _REPO_ROOT / "3D数学引擎"

for _p in [str(_GROUNDING_DIR), str(_ENGINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# 工具执行器
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    """将 Agent 的工具调用请求路由到实际计算后端。

    Parameters
    ----------
    image_path : str | Path
        当前推理所用的无人机图像路径。
    grounding_backend : str
        视觉定位后端，可选 "qwen3vl_api" | "dino" | "qwen3vl_local" | "mock"。
    depth_backend : str
        深度估算后端，可选 "depth_anything_v3" | "depth_anything_v2" | "midas" | "mock"。
    camera_intrinsics : tuple or None
        相机内参 (f_x, f_y, c_x, c_y)，None 时按 90° FOV 自动估算。
    depth_scale_factor : float
        深度图缩放系数：将相对深度（归一化到 [0,1]）映射为近似米数的最大范围。
        默认 10.0 表示场景深度范围约为 0~10 米，可根据实际场景调整。
    grounding_kwargs : dict
        传递给 grounding 后端的额外参数（例如 api_key）。
    depth_kwargs : dict
        传递给 depth 后端的额外参数。
    """

    def __init__(
        self,
        image_path: str | Path,
        grounding_backend: str = "qwen3vl_api",
        depth_backend: str = "depth_anything_v3",
        camera_intrinsics: Optional[tuple] = None,
        depth_scale_factor: float = 10.0,
        grounding_kwargs: Optional[dict] = None,
        depth_kwargs: Optional[dict] = None,
    ) -> None:
        self.image_path = Path(image_path)
        self.grounding_backend = grounding_backend
        self.depth_backend = depth_backend
        self.camera_intrinsics = camera_intrinsics
        self.depth_scale_factor = depth_scale_factor
        self.grounding_kwargs = grounding_kwargs or {}
        self.depth_kwargs = depth_kwargs or {}

        # 延迟初始化 SpatialReasoner（避免导入时就触发模型加载）
        self._reasoner: Any = None

    # ------------------------------------------------------------------
    # 内部：懒加载 SpatialReasoner
    # ------------------------------------------------------------------

    @property
    def reasoner(self):
        """首次访问时才实例化 SpatialReasoner，避免不必要的模型加载。"""
        if self._reasoner is None:
            from spatial_reasoner import SpatialReasoner  # type: ignore[import]
            self._reasoner = SpatialReasoner(
                grounding_backend=self.grounding_backend,
                depth_backend=self.depth_backend,
                camera_intrinsics=self.camera_intrinsics,
                depth_scale_factor=self.depth_scale_factor,
                grounding_kwargs=self.grounding_kwargs,
                depth_kwargs=self.depth_kwargs,
            )
        return self._reasoner

    # ------------------------------------------------------------------
    # 内部：统一异常包装
    # ------------------------------------------------------------------

    def _safe_call(self, func, *args, **kwargs) -> Any:
        """执行 func(*args, **kwargs)，捕获所有异常并返回错误字符串。"""
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning("工具执行异常: %s: %s", type(exc).__name__, exc)
            return f"ERROR: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_bounding_box(self, object_name: str) -> str:
        """定位图像中目标物体，返回像素坐标边界框。

        Parameters
        ----------
        object_name : str
            目标物体的自然语言描述，例如 "红色轿车"。

        Returns
        -------
        str
            "[x_min, y_min, x_max, y_max]"（像素坐标）或 "null"（未找到）或错误信息。
        """
        def _call():
            from grounding_tool import get_bounding_box  # type: ignore[import]
            box = get_bounding_box(
                self.image_path,
                object_name,
                backend=self.grounding_backend,
                **self.grounding_kwargs,
            )
            if box is None:
                return "null（图像中未找到该目标，请尝试调整描述词）"
            return str([round(v, 1) for v in box])

        return self._safe_call(_call)

    def calculate_3d_distance(self, obj_1: str, obj_2: str) -> str:
        """计算两目标的三维直线距离（含高度差），单位：米。"""
        def _call():
            result = self.reasoner.measure_distance(
                self.image_path, obj_1, obj_2, mode="absolute"
            )
            if result.get("error"):
                return f"ERROR: {result['error']}"
            dist = result["distance_m"]
            return f"{dist:.2f} meters（三维直线距离）"

        return self._safe_call(_call)

    def calculate_horizontal_distance(self, obj_1: str, obj_2: str) -> str:
        """计算两目标的水平面投影距离（忽略高度差），单位：米。"""
        def _call():
            result = self.reasoner.measure_distance(
                self.image_path, obj_1, obj_2, mode="horizontal"
            )
            if result.get("error"):
                return f"ERROR: {result['error']}"
            dist = result["distance_m"]
            return f"{dist:.2f} meters（水平距离，已忽略高度差）"

        return self._safe_call(_call)

    def calculate_vertical_distance(self, obj_1: str, obj_2: str) -> str:
        """计算两目标之间的高度差，单位：米。"""
        def _call():
            result = self.reasoner.measure_distance(
                self.image_path, obj_1, obj_2, mode="vertical"
            )
            if result.get("error"):
                return f"ERROR: {result['error']}"
            dist = result["distance_m"]
            return f"{dist:.2f} meters（高度差）"

        return self._safe_call(_call)

    def get_direction(
        self,
        from_object: str,
        to_object: str,
        mode: str = "clock",
    ) -> str:
        """计算从 from_object 看向 to_object 的方向。

        Parameters
        ----------
        from_object : str  参考物体（观察者）
        to_object   : str  目标物体（被观察者）
        mode        : str  "clock" | "cardinal" | "full"
        """
        _valid_modes = {"clock", "cardinal", "full"}
        if mode not in _valid_modes:
            mode = "clock"

        def _call():
            result = self.reasoner.get_direction(
                self.image_path, from_object, to_object, mode=mode
            )
            if result.get("error"):
                return f"ERROR: {result['error']}"
            direction = result["direction"]
            if mode == "full":
                return str(direction)
            return str(direction)

        return self._safe_call(_call)

    def get_object_size(self, object_name: str) -> str:
        """估算单个目标物体的三维尺寸（宽、高、深），单位：米。"""
        def _call():
            result = self.reasoner.get_object_size(self.image_path, object_name)
            if result.get("error"):
                return f"ERROR: {result['error']}"
            w = result.get("width_m", 0.0)
            h = result.get("height_m", 0.0)
            d = result.get("depth_m", 0.0)
            return (
                f"宽度: {w:.2f}m，高度: {h:.2f}m，深度（沿视线方向）: {d:.2f}m"
            )

        return self._safe_call(_call)

    # ------------------------------------------------------------------
    # 统一路由入口
    # ------------------------------------------------------------------

    # 工具名称 → 方法映射
    _TOOL_DISPATCH: dict[str, str] = {
        "get_bounding_box":           "get_bounding_box",
        "calculate_3d_distance":      "calculate_3d_distance",
        "calculate_horizontal_distance": "calculate_horizontal_distance",
        "calculate_vertical_distance": "calculate_vertical_distance",
        "get_direction":              "get_direction",
        "get_object_size":            "get_object_size",
    }

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """执行指定工具，返回结果字符串。

        Parameters
        ----------
        tool_name : str
            工具名称，必须是 AGENT_SYSTEM_PROMPT 中定义的之一。
        params : dict
            工具参数字典，键名与工具定义中的 Parameters 一致。

        Returns
        -------
        str
            工具执行结果（成功）或错误描述（失败），始终返回字符串，不抛出异常。
        """
        method_name = self._TOOL_DISPATCH.get(tool_name)
        if method_name is None:
            known = ", ".join(self._TOOL_DISPATCH.keys())
            return f"ERROR: 未知工具 '{tool_name}'。已知工具：{known}"

        method = getattr(self, method_name)
        logger.info("执行工具 [%s] | 参数: %s", tool_name, params)

        # 根据每个工具的参数名分发
        try:
            if tool_name == "get_bounding_box":
                return method(params.get("object_name", ""))
            elif tool_name in ("calculate_3d_distance",
                               "calculate_horizontal_distance",
                               "calculate_vertical_distance"):
                return method(
                    params.get("obj_1", ""),
                    params.get("obj_2", ""),
                )
            elif tool_name == "get_direction":
                return method(
                    params.get("from_object", ""),
                    params.get("to_object", ""),
                    params.get("mode", "clock"),
                )
            elif tool_name == "get_object_size":
                return method(params.get("object_name", ""))
            else:
                return f"ERROR: 工具 '{tool_name}' 的参数路由未实现。"
        except Exception as exc:
            logger.error("工具路由异常 [%s]: %s", tool_name, exc)
            return f"ERROR: 工具执行失败：{exc}"
