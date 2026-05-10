"""
Grounding 方法二：Grounding DINO（开源零样本目标检测）
GroundingDINO 是开源界零样本 (Zero-shot) 目标检测的标杆模型，
支持自然语言描述直接定位目标，无需微调。

项目地址：https://github.com/IDEA-Research/GroundingDINO

依赖安装：
    # 方式 A：通过 groundingdino-py 包（推荐，无需编译）
    pip install groundingdino-py pillow numpy torch torchvision

    # 方式 B：从源码安装（GPU 加速，需 CUDA 编译环境）
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git
    pip install pillow numpy torch torchvision

模型权重下载（首次运行自动下载，或手动指定路径）：
    - Config : https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
    - Weights: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 默认权重 & 配置路径
# ──────────────────────────────────────────────
_DEFAULT_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
_DEFAULT_CONFIG_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
_MODEL_DIR = Path(__file__).parent / "model_weights"
_DEFAULT_WEIGHTS_PATH = _MODEL_DIR / "groundingdino_swint_ogc.pth"
_DEFAULT_CONFIG_PATH = _MODEL_DIR / "GroundingDINO_SwinT_OGC.py"


def _download_file(url: str, dest: Path) -> None:
    """如果目标文件不存在，从 url 下载到 dest。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("文件已存在，跳过下载: %s", dest)
        return
    logger.info("正在下载 %s -> %s", url, dest)
    urllib.request.urlretrieve(url, str(dest))
    logger.info("下载完成: %s", dest)


class GroundingDINOGrounder:
    """使用 Grounding DINO 进行零样本视觉定位。

    Parameters
    ----------
    config_path : str | Path, optional
        GroundingDINO 配置文件路径。为 None 时自动下载到 model_weights/。
    weights_path : str | Path, optional
        模型权重路径。为 None 时自动下载到 model_weights/。
    device : str
        "cuda" 或 "cpu"。
    box_threshold : float
        边界框置信度阈值（0~1），越高越严格。
    text_threshold : float
        文本匹配阈值（0~1）。
    lazy_load : bool
        首次调用时才加载模型。
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        weights_path: Optional[str | Path] = None,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        lazy_load: bool = True,
    ) -> None:
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS_PATH
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._transform = None
        if not lazy_load:
            self._load_model()

    # ── 模型加载 ──────────────────────────────
    def _ensure_assets(self) -> None:
        """确保配置文件和权重文件存在，否则自动下载。"""
        if not self.config_path.exists():
            _download_file(_DEFAULT_CONFIG_URL, self.config_path)
        if not self.weights_path.exists():
            _download_file(_DEFAULT_WEIGHTS_URL, self.weights_path)

    def _load_model(self) -> None:
        """加载 GroundingDINO 模型。"""
        self._ensure_assets()
        try:
            from groundingdino.util.inference import load_model, load_image
            import torch

            logger.info("正在加载 GroundingDINO 模型...")
            self._model = load_model(
                str(self.config_path), str(self.weights_path)
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("GroundingDINO 模型加载完成。")
        except ImportError as e:
            raise ImportError(
                "请先安装 GroundingDINO：\n"
                "  pip install groundingdino-py\n"
                "或从源码安装：\n"
                "  pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
            ) from e

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    # ── 图像预处理 ────────────────────────────
    @staticmethod
    def _preprocess_image(image: str | Path | Image.Image):
        """返回 (PIL Image, transformed tensor) 供 GroundingDINO 使用。"""
        from groundingdino.util.inference import load_image
        import torchvision.transforms as T
        import torch

        if isinstance(image, (str, Path)):
            _, image_tensor = load_image(str(image))
            pil_img = Image.open(image).convert("RGB")
        else:
            # PIL Image → tensor
            # GroundingDINO 要求输入较长边 resize 到 800，保持纵横比
            transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            pil_img = image.convert("RGB")
            image_tensor = transform(pil_img)

        return pil_img, image_tensor

    # ── 核心接口 ─────────────────────────────
    def get_bounding_box(
        self,
        image: str | Path | Image.Image,
        object_text: str,
        return_all: bool = False,
    ) -> Optional[list[float] | list[list[float]]]:
        """定位图像中 object_text 对应目标，返回边界框。

        Parameters
        ----------
        image : str | Path | PIL.Image.Image
            图像路径或 PIL 图像。
        object_text : str
            目标文本描述（英文效果更佳，中文也支持）。
            例如："red car"、"white house"、"红色车"。
        return_all : bool
            True 时返回所有检测框（按置信度降序）；False 只返回最高置信度框。

        Returns
        -------
        list[float] | list[list[float]] | None
            [x_min, y_min, x_max, y_max]（像素坐标），未找到返回 None。
        """
        import torch

        pil_img, image_tensor = self._preprocess_image(image)
        img_w, img_h = pil_img.size

        image_tensor = image_tensor.to(self.device)

        # GroundingDINO 推理
        from groundingdino.util.inference import predict

        with torch.no_grad():
            boxes_cxcywh, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=object_text,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

        if boxes_cxcywh is None or len(boxes_cxcywh) == 0:
            logger.warning("GroundingDINO 未检测到目标: '%s'", object_text)
            return None

        # 按置信度降序排序
        order = logits.argsort(descending=True)
        boxes_cxcywh = boxes_cxcywh[order]
        logits = logits[order]
        phrases = [phrases[i] for i in order.tolist()]

        # cxcywh (归一化) → xyxy (像素)
        result_boxes = []
        for box in boxes_cxcywh:
            cx, cy, w, h = box.tolist()
            x_min = (cx - w / 2) * img_w
            y_min = (cy - h / 2) * img_h
            x_max = (cx + w / 2) * img_w
            y_max = (cy + h / 2) * img_h
            result_boxes.append([
                max(0.0, x_min),
                max(0.0, y_min),
                min(float(img_w), x_max),
                min(float(img_h), y_max),
            ])

        logger.debug(
            "GroundingDINO 检测到 %d 个 '%s': %s",
            len(result_boxes), object_text, phrases
        )
        return result_boxes if return_all else result_boxes[0]


# ──────────────────────────────────────────────
# 独立函数接口
# ──────────────────────────────────────────────
def get_bounding_box(
    image: str | Path | Image.Image,
    object_text: str,
    grounder: Optional[GroundingDINOGrounder] = None,
    **kwargs,
) -> Optional[list[float]]:
    """使用 Grounding DINO 定位目标，返回 [x_min, y_min, x_max, y_max]。

    Parameters
    ----------
    image : str | Path | PIL.Image.Image
        图像路径或 PIL 图像。
    object_text : str
        目标文本描述。GroundingDINO 对英文支持更好，
        中文查询会自动翻译为英文（可选，见 translate_query 参数）。
    grounder : GroundingDINOGrounder, optional
        已实例化的 grounder，为 None 时自动创建。
    **kwargs :
        传递给 GroundingDINOGrounder.get_bounding_box 的额外参数。

    Returns
    -------
    list[float] | None
        [x_min, y_min, x_max, y_max] 像素坐标，未找到返回 None。
    """
    if grounder is None:
        grounder = GroundingDINOGrounder()
    return grounder.get_bounding_box(image, object_text, **kwargs)
