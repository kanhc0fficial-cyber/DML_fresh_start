"""
Grounding 方法一：Qwen3-VL（视觉语言大模型）
利用 Qwen3-VL 的内置 Grounding 能力，通过提示词要求模型输出边界框坐标。

Qwen3-VL 支持在提示词中要求输出 <|object_ref_start|>...<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|> 格式。
坐标为归一化坐标（0~1000 范围内的整数），需按原图尺寸还原。

本模块提供两种调用方式：

1. API 调用（推荐，无需本地 GPU）：Qwen3VLAPIGrounder
   - 使用阿里云 DashScope OpenAI 兼容接口
   - 依赖：pip install openai pillow
   - 需要 DASHSCOPE_API_KEY 环境变量（或直接传入 api_key 参数）

2. 本地推理（离线部署）：Qwen3VLGrounder
   - 本地加载模型权重，无需网络和 API Key
   - 依赖：pip install transformers accelerate torch torchvision pillow
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 常量 & 正则
# ──────────────────────────────────────────────
# Qwen3-VL 坐标范围为 [0, 1000]
COORD_SCALE = 1000.0

# 匹配 <|box_start|>(x1,y1),(x2,y2)<|box_end|>
_BOX_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def _parse_qwen_boxes(
    text: str, img_w: int, img_h: int
) -> list[list[float]]:
    """从 Qwen3-VL 输出文本中解析所有边界框，转换为像素坐标。

    Returns:
        list of [x_min, y_min, x_max, y_max] in pixel coords.
    """
    boxes = []
    for m in _BOX_RE.finditer(text):
        x1_n, y1_n, x2_n, y2_n = (int(m.group(i)) for i in range(1, 5))
        x_min = x1_n / COORD_SCALE * img_w
        y_min = y1_n / COORD_SCALE * img_h
        x_max = x2_n / COORD_SCALE * img_w
        y_max = y2_n / COORD_SCALE * img_h
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes


def _build_prompt(object_text: str) -> str:
    """构造 Qwen3-VL Grounding 提示词。"""
    return (
        f"请在图像中找到所有「{object_text}」，"
        "并以 <|object_ref_start|>目标<|object_ref_end|>"
        "<|box_start|>(x_min,y_min),(x_max,y_max)<|box_end|> 格式输出每个目标的边界框。"
        "坐标为 0~1000 的归一化整数，左上角为 (0,0)，右下角为 (1000,1000)。"
        "若图中没有该目标，请回复「未找到目标」。"
    )


def _image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    """将 PIL Image 编码为 base64 data URL，用于 API 调用。"""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


# ──────────────────────────────────────────────
# 方式一（推荐）：API 调用
# ──────────────────────────────────────────────

class Qwen3VLAPIGrounder:
    """通过 DashScope OpenAI 兼容 API 调用 Qwen3-VL 进行视觉定位。

    无需本地 GPU 或模型权重，只需 API Key 即可使用。

    Parameters
    ----------
    api_key : str, optional
        DashScope API Key。为 None 时从环境变量 DASHSCOPE_API_KEY 读取。
    model : str
        API 模型名称，例如 "qwen-vl-max"、"qwen3-vl-72b-instruct"。
    base_url : str
        API 端点，默认使用阿里云 DashScope OpenAI 兼容地址。
    max_tokens : int
        最大生成 token 数。
    image_format : str
        发送给 API 的图像编码格式，"JPEG" 或 "PNG"。

    环境变量：
        DASHSCOPE_API_KEY : DashScope API Key（推荐通过环境变量传入）

    安装依赖：
        pip install openai pillow
    """

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-max",
        base_url: str = DASHSCOPE_BASE_URL,
        max_tokens: int = 256,
        image_format: str = "JPEG",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.image_format = image_format.upper()
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "未提供 API Key。请设置环境变量 DASHSCOPE_API_KEY，"
                "或在初始化时传入 api_key 参数。\n"
                "  export DASHSCOPE_API_KEY=sk-xxxxxxxx\n"
                "  grounder = Qwen3VLAPIGrounder(api_key='sk-xxxxxxxx')"
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "请先安装 openai 包：pip install openai"
                ) from e
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        return self._client

    def get_bounding_box(
        self,
        image: str | Path | Image.Image,
        object_text: str,
        return_all: bool = False,
    ) -> Optional[list[float] | list[list[float]]]:
        """通过 API 定位图像中 object_text 对应目标，返回边界框。

        Parameters
        ----------
        image : str | Path | PIL.Image.Image
            图像路径或 PIL 图像。
        object_text : str
            目标描述，例如 "红色车"、"白房子"。
        return_all : bool
            True 返回所有检测框；False 只返回第一个框。

        Returns
        -------
        list[float] | list[list[float]] | None
            [x_min, y_min, x_max, y_max]（像素坐标），未找到返回 None。
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        img_w, img_h = image.size

        data_url = _image_to_data_url(image, fmt=self.image_format)
        prompt_text = _build_prompt(object_text)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        logger.debug(
            "调用 Qwen3-VL API | model=%s | target='%s' | image=%dx%d",
            self.model, object_text, img_w, img_h,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        reply = response.choices[0].message.content or ""
        logger.debug("API 原始返回:\n%s", reply)

        boxes = _parse_qwen_boxes(reply, img_w, img_h)
        if not boxes:
            logger.warning("API 未返回边界框: %s", reply[:200])
            return None

        return boxes if return_all else boxes[0]


# ──────────────────────────────────────────────
# 方式二（离线部署）：本地推理
# ──────────────────────────────────────────────

class Qwen3VLGrounder:
    """使用 Qwen3-VL 进行零样本视觉定位（本地推理，无需 API Key）。

    Parameters
    ----------
    model_name_or_path : str
        本地路径或 HuggingFace Hub 上的模型名，例如 "Qwen/Qwen3-VL-7B-Instruct"。
    device : str
        "cuda" 或 "cpu"。
    lazy_load : bool
        若为 True，首次调用 get_bounding_box 时才加载模型（节省内存）。
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-7B-Instruct",
        device: str = "cuda",
        lazy_load: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._model = None
        self._processor = None
        if not lazy_load:
            self._load_model()

    # ── 模型加载 ──────────────────────────────
    def _load_model(self) -> None:
        """延迟加载 Qwen3-VL 模型和处理器。"""
        try:
            # Qwen3-VL 使用与 Qwen2.5-VL 相同的模型类（transformers 尚未独立 Qwen3-VL 类）
            # 若未来 transformers 发布 Qwen3VLForConditionalGeneration，请更新此处导入
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch

            logger.info("正在加载 Qwen3-VL 模型: %s", self.model_name_or_path)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._model.eval()
            logger.info("Qwen3-VL 模型加载完成。")
        except ImportError as e:
            raise ImportError(
                "请先安装依赖：pip install transformers accelerate torch torchvision"
            ) from e

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._load_model()
        return self._processor

    # ── 核心接口 ─────────────────────────────
    def get_bounding_box(
        self,
        image: str | Path | Image.Image,
        object_text: str,
        max_new_tokens: int = 256,
        return_all: bool = False,
    ) -> Optional[list[float]]:
        """定位图像中 object_text 对应目标，返回边界框。

        Parameters
        ----------
        image : str | Path | PIL.Image.Image
            图像路径或已加载的 PIL 图像。
        object_text : str
            目标描述，例如 "红色车"、"白房子"。
        max_new_tokens : int
            模型最大生成 token 数。
        return_all : bool
            若为 True，返回所有检测到的边界框列表；否则只返回置信度最高的第一个。

        Returns
        -------
        list[float] | list[list[float]] | None
            [x_min, y_min, x_max, y_max]（像素坐标），未找到时返回 None。
        """
        import torch

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        img_w, img_h = image.size

        prompt_text = _build_prompt(object_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # 构建输入
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        # 推理
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 解码（只取新生成部分）
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        response = self.processor.decode(generated_ids[0], skip_special_tokens=False)
        logger.debug("Qwen3-VL 原始输出:\n%s", response)

        boxes = _parse_qwen_boxes(response, img_w, img_h)
        if not boxes:
            logger.warning("未在输出中找到边界框: %s", response[:200])
            return None

        return boxes if return_all else boxes[0]


# ──────────────────────────────────────────────
# 独立函数接口（与 Grounding DINO 保持统一签名）
# ──────────────────────────────────────────────
def get_bounding_box(
    image: str | Path | Image.Image,
    object_text: str,
    grounder: Optional[Qwen3VLAPIGrounder | Qwen3VLGrounder] = None,
    **kwargs,
) -> Optional[list[float]]:
    """使用 Qwen3-VL 定位目标，返回 [x_min, y_min, x_max, y_max]。

    默认使用 API 方式（Qwen3VLAPIGrounder）。若已有实例化的 grounder，
    可直接传入（API 或本地均可）。

    Parameters
    ----------
    image : str | Path | PIL.Image.Image
        图像路径或 PIL 图像。
    object_text : str
        目标文本描述。
    grounder : Qwen3VLAPIGrounder | Qwen3VLGrounder, optional
        已实例化的 grounder，为 None 时自动创建 API 版本。
    **kwargs :
        传递给 grounder.get_bounding_box 的额外参数。

    Returns
    -------
    list[float] | None
        [x_min, y_min, x_max, y_max] 像素坐标，未找到返回 None。
    """
    if grounder is None:
        grounder = Qwen3VLAPIGrounder()
    return grounder.get_bounding_box(image, object_text, **kwargs)
