"""
统一 Grounding 接口
支持两种后端：Qwen3-VL 和 Grounding DINO

用法示例：
    from grounding_tool import get_bounding_box

    # 使用 Grounding DINO（推荐，无需 API Key）
    box = get_bounding_box("car.jpg", "red car", backend="dino")

    # 使用 Qwen3-VL（需要模型权重）
    box = get_bounding_box("car.jpg", "红色车", backend="qwen3vl")

    print(box)  # [x_min, y_min, x_max, y_max]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

logger = logging.getLogger(__name__)

BackendType = Literal["dino", "qwen3vl"]


def get_bounding_box(
    image: str | Path | Image.Image,
    object_text: str,
    backend: BackendType = "dino",
    return_all: bool = False,
    **backend_kwargs,
) -> Optional[list[float] | list[list[float]]]:
    """通用目标定位接口，返回 [x_min, y_min, x_max, y_max]（像素坐标）。

    Parameters
    ----------
    image : str | Path | PIL.Image.Image
        图像路径或 PIL 图像。
    object_text : str
        目标描述，如 "红色车"、"white house"。
    backend : "dino" | "qwen3vl"
        使用的后端模型：
        - "dino"     : Grounding DINO，开源零样本检测标杆，无需 API Key
        - "qwen3vl"  : Qwen3-VL，视觉语言大模型，需要模型权重
    return_all : bool
        True 返回所有检测框（list of lists）；False 只返回最高置信度框。
    **backend_kwargs :
        传递给具体后端的额外参数（如 box_threshold、device 等）。

    Returns
    -------
    list[float] | list[list[float]] | None
        [x_min, y_min, x_max, y_max] 或多个框的列表；未找到返回 None。

    Raises
    ------
    ValueError
        未知的 backend 名称。
    """
    if backend == "dino":
        from grounding_dino import GroundingDINOGrounder
        grounder = GroundingDINOGrounder(**backend_kwargs)
        return grounder.get_bounding_box(image, object_text, return_all=return_all)

    elif backend == "qwen3vl":
        from grounding_qwen3vl import Qwen3VLGrounder
        grounder = Qwen3VLGrounder(**backend_kwargs)
        return grounder.get_bounding_box(image, object_text, return_all=return_all)

    else:
        raise ValueError(
            f"未知的 backend: '{backend}'。支持的选项: 'dino', 'qwen3vl'。"
        )
