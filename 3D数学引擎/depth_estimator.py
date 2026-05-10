"""
深度估算器（Depth Estimator）包装层

支持后端：
  - "midas"              : MiDaS v3（DPT-Large），通过 torch.hub 加载
  - "depth_anything_v2"  : Depth Anything V2（轻量，效果优秀）
  - "depth_anything_v3"  : Depth Anything V3（最新，推荐首选）
  - "mock"               : 合成深度图，用于单元测试 / 无 GPU 环境

向下兼容别名：
  - "depth_anything" → 等价于 "depth_anything_v2"

公共接口：
    from depth_estimator import get_depth_map

    depth_map = get_depth_map("image.jpg")                  # 返回 np.ndarray  H×W (float32, 近似米)
    depth_map = get_depth_map("image.jpg", backend="depth_anything_v3")
    depth_map = get_depth_map("image.jpg", backend="mock", mock_near_depth=5.0, mock_far_depth=20.0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BackendType = Literal["midas", "depth_anything_v2", "depth_anything_v3", "depth_anything", "mock"]

# 向下兼容：旧名称 "depth_anything" → 自动映射为 "depth_anything_v2"
_BACKEND_ALIASES: dict[str, str] = {
    "depth_anything": "depth_anything_v2",
}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def _scale_depth_to_metric(
    raw: np.ndarray,
    scale_factor: float = 10.0,
) -> np.ndarray:
    """将相对深度归一化到 [0, scale_factor] 米的近似范围。

    相对深度模型（如 MiDaS）输出的是视差（inverse depth）或相对深度，
    不是绝对米数。这里做简单线性映射，使结果在数量级上合理。
    如已有真实相机标定数据，应改为绝对深度估算流程。
    """
    d_min = raw.min()
    d_max = raw.max()
    if d_max - d_min < 1e-6:
        return np.full_like(raw, scale_factor / 2)
    return (raw - d_min) / (d_max - d_min) * scale_factor


# ─────────────────────────────────────────────────────────────────────────────
# 后端：MiDaS
# ─────────────────────────────────────────────────────────────────────────────

class _MiDaSEstimator:
    """通过 torch.hub 加载 MiDaS DPT-Large 进行深度估算。"""

    def __init__(self, model_type: str = "DPT_Large", device: Optional[str] = None):
        try:
            import torch
        except ImportError as e:
            raise ImportError("MiDaS 后端需要 PyTorch。请运行: pip install torch torchvision") from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("加载 MiDaS 模型 %s（设备: %s）…", model_type, self.device)

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate(self, image: Image.Image, scale_factor: float = 10.0) -> np.ndarray:
        import torch

        img_np = np.array(image)
        input_batch = self.transform(img_np).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_raw = prediction.cpu().numpy().astype(np.float32)
        return _scale_depth_to_metric(depth_raw, scale_factor)


# ─────────────────────────────────────────────────────────────────────────────
# 后端：Depth Anything v2
# ─────────────────────────────────────────────────────────────────────────────

class _DepthAnythingEstimator:
    """Depth Anything 封装（transformers HuggingFace pipeline 接口）。

    同时支持 V2 和 V3，通过 model_name 参数区分。
    """

    # 默认模型 ID（可通过 backend_kwargs 中的 model_name 覆盖）
    DEFAULT_V2_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
    DEFAULT_V3_MODEL = "depth-anything/Depth-Anything-V3-Small-hf"

    def __init__(self, model_name: Optional[str] = None,
                 version: str = "v2",
                 device: Optional[str] = None):
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError as e:
            raise ImportError(
                "Depth-Anything 后端需要 transformers 和 PyTorch。\n"
                "请运行: pip install transformers torch torchvision"
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_name is None:
            model_name = self.DEFAULT_V3_MODEL if version == "v3" else self.DEFAULT_V2_MODEL

        logger.info("加载 Depth-Anything %s 模型 %s（设备: %s）…", version.upper(), model_name, self.device)

        self.pipe = hf_pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
        )

    def estimate(self, image: Image.Image, scale_factor: float = 10.0) -> np.ndarray:
        result = self.pipe(image)
        depth_pil = result["depth"]
        depth_raw = np.array(depth_pil).astype(np.float32)
        return _scale_depth_to_metric(depth_raw, scale_factor)


# ─────────────────────────────────────────────────────────────────────────────
# 后端：Mock（合成深度图，仅用于测试）
# ─────────────────────────────────────────────────────────────────────────────

class _MockEstimator:
    """生成合成深度图用于单元测试，不需要任何模型。

    合成规则：
    - 图像中心区域（1/4 范围）深度较近（mock_near_depth）
    - 图像边缘深度较远（mock_far_depth）
    - 可叠加轻微高斯噪声以模拟真实传感器
    """

    def __init__(
        self,
        mock_near_depth: float = 5.0,
        mock_far_depth: float = 20.0,
        noise_std: float = 0.2,
    ):
        self.mock_near_depth = mock_near_depth
        self.mock_far_depth = mock_far_depth
        self.noise_std = noise_std

    def estimate(self, image: Image.Image, scale_factor: float = 10.0) -> np.ndarray:
        w, h = image.size
        # 生成从中心到边缘线性变化的深度
        cx, cy = w / 2, h / 2
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        dist_from_center = np.sqrt(((x_grid - cx) / cx) ** 2 + ((y_grid - cy) / cy) ** 2)
        depth = self.mock_near_depth + dist_from_center * (self.mock_far_depth - self.mock_near_depth)
        if self.noise_std > 0:
            rng = np.random.default_rng(42)
            depth += rng.normal(0, self.noise_std, depth.shape)
        return depth.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

_estimator_cache: dict[str, object] = {}


def get_depth_map(
    image: str | Path | Image.Image,
    backend: BackendType = "depth_anything_v3",
    scale_factor: float = 10.0,
    device: Optional[str] = None,
    mock_near_depth: float = 5.0,
    mock_far_depth: float = 20.0,
    **backend_kwargs,
) -> np.ndarray:
    """估算图像的深度图，返回 H×W float32 数组（单位：近似米）。

    Parameters
    ----------
    image : str | Path | PIL.Image.Image
        输入图像。
    backend : str
        使用的深度估算后端：
        - "depth_anything_v3" : Depth Anything V3（推荐，需要 transformers + torch）
        - "depth_anything_v2" : Depth Anything V2（需要 transformers + torch）
        - "depth_anything"    : 同 "depth_anything_v2"（向下兼容别名）
        - "midas"             : MiDaS DPT-Large（需要 torch，通过 torch.hub 加载）
        - "mock"              : 不需要任何模型，用于测试
    scale_factor : float
        将相对深度缩放到大约多少米（默认 10.0）。
        如果拥有真实相机标定，请忽略此参数并对结果做绝对标定。
    device : str or None
        计算设备（"cuda"/"cpu"）。None 时自动选择。
    mock_near_depth : float
        Mock 模式下图像中心的深度（米）。
    mock_far_depth : float
        Mock 模式下图像边缘的深度（米）。

    Returns
    -------
    np.ndarray  shape (H, W)  float32
    """
    # 处理向下兼容别名
    backend = _BACKEND_ALIASES.get(backend, backend)  # type: ignore[arg-type]

    pil_image = _load_image(image)

    # mock 的 cache key 必须包含 near/far，否则不同参数共用同一个 Estimator
    if backend == "mock":
        cache_key = f"mock_{mock_near_depth}_{mock_far_depth}"
    else:
        cache_key = f"{backend}_{device}"

    if cache_key not in _estimator_cache:
        if backend == "midas":
            _estimator_cache[cache_key] = _MiDaSEstimator(device=device, **backend_kwargs)
        elif backend == "depth_anything_v2":
            _estimator_cache[cache_key] = _DepthAnythingEstimator(
                version="v2", device=device, **backend_kwargs
            )
        elif backend == "depth_anything_v3":
            _estimator_cache[cache_key] = _DepthAnythingEstimator(
                version="v3", device=device, **backend_kwargs
            )
        elif backend == "mock":
            _estimator_cache[cache_key] = _MockEstimator(
                mock_near_depth=mock_near_depth,
                mock_far_depth=mock_far_depth,
            )
        else:
            raise ValueError(
                f"未知的 backend: '{backend}'。"
                "支持的选项: 'depth_anything_v3', 'depth_anything_v2', 'midas', 'mock'。"
            )

    estimator = _estimator_cache[cache_key]
    return estimator.estimate(pil_image, scale_factor=scale_factor)


def clear_estimator_cache() -> None:
    """清空后端模型缓存（用于测试或切换设备时释放内存）。"""
    _estimator_cache.clear()
