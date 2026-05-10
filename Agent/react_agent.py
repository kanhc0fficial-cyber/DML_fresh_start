"""
第二段：ReAct 状态机执行引擎
（Reasoning and Acting Loop）

这是整个 Agent 框架的核心控制器。它实现了一个纯 Python 轻量级状态机，
按照以下循环驱动大模型完成多步空间推理：

    ┌─────────────────────────────────────────────────────────────┐
    │  用户输入：图像路径 + 自然语言问题                          │
    └──────────────────────┬──────────────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  构造初始消息：[系统提示词] + [用户消息（图像 + 问题）]      │
    └──────────────────────┬──────────────────────────────────────┘
                           ▼
               ╔═══════════╧═══════════╗
               ║   ReAct 主循环         ║  ← 最多 MAX_ITERATIONS 轮
               ╠═══════════════════════╣
               ║ 1. 调用 VLM API        ║
               ║ 2. 解析响应            ║
               ║    ├─ Final Answer?   ║─── 找到 → 返回最终答案
               ║    └─ Action?         ║
               ║ 3. 执行工具           ║
               ║ 4. 追加 Observation   ║
               ╚═══════════╤═══════════╝
                           │ 达到上限
                           ▼
                    兜底策略（Fallback）

使用方式：
    from react_agent import ReactAgent

    agent = ReactAgent(
        model="qwen-vl-max",          # 或 "qwen3-vl-72b-instruct"
        grounding_backend="qwen3vl_api",
        depth_backend="depth_anything_v3",
    )
    answer = agent.run("scene.jpg", "图中红色轿车与路灯的距离是多少？")
    print(answer)   # 例如: "8.5 meters"
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from prompt_template import (
    AGENT_SYSTEM_PROMPT,
    ALL_TOOL_NAMES,
    format_observation,
    format_user_question,
)
from tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 正则：解析大模型输出
# ─────────────────────────────────────────────────────────────────────────────

# 匹配 "Action: <tool_name>"
_ACTION_RE = re.compile(
    r"Action\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)

# 匹配 "Action Input: <json_block>"（支持多行 JSON）
_ACTION_INPUT_RE = re.compile(
    r"Action\s+Input\s*:\s*(\{.*?\})",
    re.DOTALL | re.IGNORECASE,
)

# 匹配 "Final Answer: <answer>"（答案可能多行，取到字符串结尾）
_FINAL_ANSWER_RE = re.compile(
    r"Final\s+Answer\s*:\s*(.+)",
    re.DOTALL | re.IGNORECASE,
)

_FALLBACK_RESPONSE_MAX_LENGTH = 200  # 兜底响应截取最大字符数

# ─────────────────────────────────────────────────────────────────────────────
# 图像工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_data_url(image_path: str | Path, fmt: str = "JPEG") -> str:
    """将图像文件编码为 base64 data URL，用于 API 多模态消息。"""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# ReactAgent 主类
# ─────────────────────────────────────────────────────────────────────────────

class ReactAgent:
    """轻量级 ReAct（Reasoning and Acting）状态机 Agent。

    Parameters
    ----------
    model : str
        VLM 模型名称，例如 "qwen-vl-max" 或 "qwen3-vl-72b-instruct"。
    api_key : str, optional
        DashScope API Key。None 时从环境变量 DASHSCOPE_API_KEY 读取。
    base_url : str
        API 端点，默认使用阿里云 DashScope OpenAI 兼容地址。
    max_iterations : int
        ReAct 循环的最大轮数（防止无限循环）。
    max_tokens : int
        每次 VLM 调用允许生成的最大 token 数。
    temperature : float
        采样温度，0 时为贪婪解码（竞赛推荐）。
    grounding_backend : str
        视觉定位后端，"qwen3vl_api" | "dino" | "qwen3vl_local" | "mock"。
    depth_backend : str
        深度估算后端，"depth_anything_v3" | "depth_anything_v2" | "midas" | "mock"。
    camera_intrinsics : tuple or None
        相机内参 (f_x, f_y, c_x, c_y)，None 时自动估算。
    depth_scale_factor : float
        深度缩放系数（近似米数上限）。
    grounding_kwargs : dict
        传给 grounding 后端的额外参数。
    depth_kwargs : dict
        传给 depth 后端的额外参数。
    image_format : str
        发送给 API 的图像编码格式，"JPEG" 或 "PNG"。
    verbose : bool
        True 时将每一步的 Thought/Action/Observation 打印到 stderr。
    """

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        base_url: str = DASHSCOPE_BASE_URL,
        max_iterations: int = 8,
        max_tokens: int = 512,
        temperature: float = 0.0,
        grounding_backend: str = "qwen3vl_api",
        depth_backend: str = "depth_anything_v3",
        camera_intrinsics: Optional[tuple] = None,
        depth_scale_factor: float = 10.0,
        grounding_kwargs: Optional[dict] = None,
        depth_kwargs: Optional[dict] = None,
        image_format: str = "JPEG",
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.grounding_backend = grounding_backend
        self.depth_backend = depth_backend
        self.camera_intrinsics = camera_intrinsics
        self.depth_scale_factor = depth_scale_factor
        self.grounding_kwargs = grounding_kwargs or {}
        self.depth_kwargs = depth_kwargs or {}
        self.image_format = image_format.upper()
        self.verbose = verbose

        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self._client: Any = None

    # ------------------------------------------------------------------
    # VLM API 客户端（懒加载）
    # ------------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "未提供 API Key。请设置环境变量 DASHSCOPE_API_KEY，\n"
                    "或在初始化时传入 api_key 参数：\n"
                    "  ReactAgent(api_key='sk-xxxxxxxx')"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "请先安装 openai 包：pip install openai"
                ) from exc
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        return self._client

    # ------------------------------------------------------------------
    # 核心步骤：调用 VLM
    # ------------------------------------------------------------------

    def _call_vlm(self, messages: list[dict]) -> str:
        """向 VLM API 发送消息列表，返回模型回复文本。

        使用 "\nObservation:" 作为停止序列，阻止模型自行伪造工具返回值。

        Parameters
        ----------
        messages : list of dict
            符合 OpenAI Chat Completions 格式的消息列表。

        Returns
        -------
        str
            模型生成的文本，已去除首尾空白。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # 两条停止序列互补："\nObservation:" 捕获行中出现的情况，
        # "Observation:" 额外防止模型在响应首行就开始伪造观测值。
        stop=["\nObservation:", "Observation:"],
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            # 多段内容（部分 API 返回结构化 content）
            text_parts = [
                p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                for p in content
                if (isinstance(p, dict) and p.get("type") == "text")
                or (not isinstance(p, dict) and getattr(p, "type", "") == "text")
            ]
            return "\n".join(text_parts).strip()
        return (content or "").strip()

    # ------------------------------------------------------------------
    # 解析工具：提取 Action / Final Answer
    # ------------------------------------------------------------------

    def _parse_action(self, response: str) -> Optional[tuple[str, dict]]:
        """从模型响应中提取第一个工具调用。

        Returns
        -------
        (tool_name, params_dict) 或 None（未找到有效 Action）
        """
        action_match = _ACTION_RE.search(response)
        if not action_match:
            return None

        tool_name = action_match.group(1).strip()
        if tool_name not in ALL_TOOL_NAMES:
            logger.warning("模型调用了未知工具: '%s'", tool_name)
            return None

        input_match = _ACTION_INPUT_RE.search(response)
        if not input_match:
            logger.warning("找到 Action 但缺少 Action Input JSON")
            # 尝试用空参数继续（部分工具无参数时可能仍成功）
            return tool_name, {}

        raw_json = input_match.group(1).strip()
        try:
            params = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.warning("Action Input JSON 解析失败: %s | 原始文本: %s", exc, raw_json)
            # 尝试容错：用 ast.literal_eval 处理单引号 JSON
            # ast.literal_eval 作为容错回退，仅支持 Python 字面量语法（如单引号 JSON）。
            # 输入来自受控的 VLM API 响应，不存在任意代码执行风险。
            try:
                import ast
                params = ast.literal_eval(raw_json)
                if not isinstance(params, dict):
                    params = {}
            except Exception:
                params = {}

        return tool_name, params

    def _parse_final_answer(self, response: str) -> Optional[str]:
        """从模型响应中提取 Final Answer。

        Returns
        -------
        str（答案文本）或 None
        """
        match = _FINAL_ANSWER_RE.search(response)
        if match:
            return match.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # 日志 / 调试
    # ------------------------------------------------------------------

    def _log(self, tag: str, content: str) -> None:
        """在 verbose 模式下将步骤信息打印到标准错误流。"""
        if self.verbose:
            import sys
            print(f"\n[{tag}]\n{content}", file=sys.stderr, flush=True)
        logger.debug("[%s] %s", tag, content)

    # ------------------------------------------------------------------
    # 主入口：run()
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: str | Path,
        question: str,
        grounding_kwargs: Optional[dict] = None,
        depth_kwargs: Optional[dict] = None,
    ) -> str:
        """对给定图像和问题执行完整的 ReAct 推理循环。

        Parameters
        ----------
        image_path : str | Path
            无人机图像路径。
        question : str
            自然语言空间推理问题。
        grounding_kwargs : dict, optional
            覆盖实例级别的 grounding 后端参数（如 api_key）。
        depth_kwargs : dict, optional
            覆盖实例级别的 depth 后端参数。

        Returns
        -------
        str
            最终答案文本。若推理失败，返回带错误说明的兜底字符串。
        """
        image_path = Path(image_path)
        t_start = time.time()

        # 初始化工具执行器（每次 run 重新创建，确保 image_path 正确）
        executor = ToolExecutor(
            image_path=image_path,
            grounding_backend=self.grounding_backend,
            depth_backend=self.depth_backend,
            camera_intrinsics=self.camera_intrinsics,
            depth_scale_factor=self.depth_scale_factor,
            grounding_kwargs={**self.grounding_kwargs, **(grounding_kwargs or {})},
            depth_kwargs={**self.depth_kwargs, **(depth_kwargs or {})},
        )

        # ── 构造初始消息列表 ────────────────────────────────────────────
        data_url = _image_to_data_url(image_path, fmt=self.image_format)
        user_text = format_user_question(question)

        messages: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        self._log("QUESTION", question)

        # ── ReAct 主循环 ────────────────────────────────────────────────
        for iteration in range(1, self.max_iterations + 1):
            logger.info("ReAct 循环 第 %d/%d 轮", iteration, self.max_iterations)

            # 步骤 1：调用 VLM
            try:
                response = self._call_vlm(messages)
            except Exception as exc:
                logger.error("VLM API 调用失败: %s", exc)
                return f"[Agent 错误] VLM API 调用失败：{exc}"

            self._log(f"VLM 响应 (轮次 {iteration})", response)

            # 步骤 2：追加 assistant 消息到历史
            messages.append({"role": "assistant", "content": response})

            # 步骤 3：检查是否给出了最终答案
            final_answer = self._parse_final_answer(response)
            if final_answer:
                elapsed = time.time() - t_start
                logger.info("得到 Final Answer（%d 轮，%.2fs）: %s",
                            iteration, elapsed, final_answer)
                self._log("FINAL ANSWER", final_answer)
                return final_answer

            # 步骤 4：检查是否有工具调用
            action = self._parse_action(response)
            if action is None:
                # 模型没有按照格式输出 → 给出格式提醒再试
                hint = (
                    "Observation: [格式错误] 你的回复既没有包含合法的 'Action:' 工具调用，"
                    "也没有包含 'Final Answer:'。"
                    "请严格按照系统提示词中规定的格式重新输出。"
                )
                self._log("格式错误提示", hint)
                messages.append({"role": "user", "content": hint})
                continue

            tool_name, params = action
            self._log("ACTION", f"{tool_name} | 参数: {params}")

            # 步骤 5：执行工具
            tool_result = executor.execute(tool_name, params)
            self._log("TOOL RESULT", str(tool_result))

            # 步骤 6：将 Observation 追加到消息历史
            obs_text = format_observation(tool_name, tool_result)
            messages.append({"role": "user", "content": obs_text})

        # ── 超出最大迭代次数：兜底策略 ──────────────────────────────────
        elapsed = time.time() - t_start
        logger.warning(
            "ReAct 循环超过最大迭代次数 (%d)，启动兜底策略。耗时: %.2fs",
            self.max_iterations, elapsed,
        )

        fallback = self._fallback(messages, question)
        self._log("FALLBACK ANSWER", fallback)
        return fallback

    # ------------------------------------------------------------------
    # 兜底策略（Fallback）
    # ------------------------------------------------------------------

    def _fallback(self, messages: list[dict], question: str) -> str:
        """当 ReAct 循环耗尽迭代次数后，强制大模型给出一个答案。

        向模型追加一条强制指令，要求它立刻基于已有信息给出 Final Answer，
        不再调用更多工具。

        Parameters
        ----------
        messages : list[dict]
            当前完整的对话历史（含所有 Observation）。
        question : str
            原始问题（用于日志）。

        Returns
        -------
        str
            兜底答案文本。
        """
        fallback_prompt = (
            "Observation: [系统提示] 你已达到最大工具调用次数上限。"
            "请立刻根据目前所有 Observation 中的数据，"
            "给出一个你最有把握的 Final Answer。"
            "若数据不足，直接如实说明。"
            "严格按照格式：\nThought: <...>\nFinal Answer: <...>"
        )
        messages_fb = messages + [{"role": "user", "content": fallback_prompt}]

        try:
            response = self._call_vlm(messages_fb)
            self._log("FALLBACK VLM 响应", response)
            final = self._parse_final_answer(response)
            if final:
                return final
            # 若仍无 Final Answer，返回原始响应的前 200 字符
            return response[:_FALLBACK_RESPONSE_MAX_LENGTH].strip() or "无法确定答案。"
        except Exception as exc:
            logger.error("兜底策略 VLM 调用失败: %s", exc)
            return f"无法确定答案（兜底策略失败：{exc}）。"


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数：直接调用，无需实例化
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    image_path: str | Path,
    question: str,
    model: str = "qwen-vl-max",
    api_key: Optional[str] = None,
    grounding_backend: str = "qwen3vl_api",
    depth_backend: str = "depth_anything_v3",
    max_iterations: int = 8,
    verbose: bool = False,
    **agent_kwargs: Any,
) -> str:
    """便捷入口：一行代码完成 ReAct 推理。

    Parameters
    ----------
    image_path : str | Path
        无人机图像路径。
    question : str
        空间推理问题。
    model : str
        VLM 模型名称。
    api_key : str, optional
        DashScope API Key，None 时从环境变量读取。
    grounding_backend : str
        视觉定位后端。
    depth_backend : str
        深度估算后端。
    max_iterations : int
        最大工具调用轮数。
    verbose : bool
        是否打印每一步的详细信息。
    **agent_kwargs :
        透传给 ReactAgent.__init__() 的其他参数。

    Returns
    -------
    str
        最终答案文本。

    Examples
    --------
    >>> answer = run_agent("scene.jpg", "图中红色轿车和路灯相距多远？")
    >>> print(answer)
    8.5 meters
    """
    agent = ReactAgent(
        model=model,
        api_key=api_key,
        grounding_backend=grounding_backend,
        depth_backend=depth_backend,
        max_iterations=max_iterations,
        verbose=verbose,
        **agent_kwargs,
    )
    return agent.run(image_path, question)
