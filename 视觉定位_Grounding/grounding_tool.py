"""
统一 Grounding 接口
支持三种后端：Qwen3-VL API、Qwen3-VL 本地推理、Grounding DINO

用法示例：
    from grounding_tool import get_bounding_box

    # 使用 Qwen3-VL API（推荐，需要 DASHSCOPE_API_KEY）
    box = get_bounding_box("car.jpg", "红色车", backend="qwen3vl_api")

    # 使用 Grounding DINO（开源，无需 API Key，需本地安装）
    box = get_bounding_box("car.jpg", "red car", backend="dino")

    # 使用 Qwen3-VL 本地推理（需要模型权重和 GPU）
    box = get_bounding_box("car.jpg", "红色车", backend="qwen3vl_local")

    print(box)  # [x_min, y_min, x_max, y_max]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

logger = logging.getLogger(__name__)

BackendType = Literal["dino", "qwen3vl_api", "qwen3vl_local"]


def get_bounding_box(
    image: str | Path | Image.Image,
    object_text: str,
    backend: BackendType = "qwen3vl_api",
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
    backend : "qwen3vl_api" | "dino" | "qwen3vl_local"
        使用的后端模型：
        - "qwen3vl_api"   : Qwen3-VL API（推荐）。需要 DASHSCOPE_API_KEY 环境变量，
                            无需本地 GPU。可通过 api_key、model 等参数自定义。
        - "dino"          : Grounding DINO，开源零样本检测标杆，无需 API Key，
                            需本地安装 groundingdino-py。
        - "qwen3vl_local" : Qwen3-VL 本地推理，需要模型权重和 GPU，无需 API Key。
    return_all : bool
        True 返回所有检测框（list of lists）；False 只返回第一个框。
    **backend_kwargs :
        传递给具体后端构造函数的额外参数。
        qwen3vl_api  支持：api_key, model, base_url, max_tokens, image_format
        dino         支持：config_path, weights_path, device, box_threshold, text_threshold
        qwen3vl_local 支持：model_name_or_path, device, lazy_load

    Returns
    -------
    list[float] | list[list[float]] | None
        [x_min, y_min, x_max, y_max] 或多个框的列表；未找到返回 None。

    Raises
    ------
    ValueError
        未知的 backend 名称。
    """
    if backend == "qwen3vl_api":
        from grounding_qwen3vl import Qwen3VLAPIGrounder
        grounder = Qwen3VLAPIGrounder(**backend_kwargs)
        return grounder.get_bounding_box(image, object_text, return_all=return_all)

    elif backend == "dino":
        from grounding_dino import GroundingDINOGrounder
        grounder = GroundingDINOGrounder(**backend_kwargs)
        return grounder.get_bounding_box(image, object_text, return_all=return_all)

    elif backend == "qwen3vl_local":
        from grounding_qwen3vl import Qwen3VLGrounder
        grounder = Qwen3VLGrounder(**backend_kwargs)
        return grounder.get_bounding_box(image, object_text, return_all=return_all)

    else:
        raise ValueError(
            f"未知的 backend: '{backend}'。"
            "支持的选项: 'qwen3vl_api', 'dino', 'qwen3vl_local'。"
        )
