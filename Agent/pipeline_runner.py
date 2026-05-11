"""
第三段：主循环引擎（Competition Pipeline Runner）

面向打榜竞赛的批量推理主循环。核心设计目标：
  - 稳定性优先：任何单条数据失败不影响整体运行
  - 断点续传：中途崩溃后重启自动跳过已完成条目
  - 并发控制：可配置线程数和 API 请求速率限制
  - 格式严格：输出文件符合竞赛提交规范（JSONL）

使用方式（CLI）：
    # Mock 模式（验证流程，无需真实 API 和 GPU）
    python pipeline_runner.py --mock --data sample.json --output submission.jsonl

    # 真实竞赛模式
    export DASHSCOPE_API_KEY=sk-xxxxxxxx
    python pipeline_runner.py \\
        --data competition_data.json \\
        --output submission.jsonl \\
        --workers 4 \\
        --model qwen-vl-max \\
        --grounding qwen3vl_api \\
        --depth depth_anything_v3

使用方式（Python API）：
    from pipeline_runner import CompetitionPipeline

    pipeline = CompetitionPipeline(
        grounding_backend="qwen3vl_api",
        depth_backend="depth_anything_v3",
        max_workers=4,
        checkpoint_dir="./checkpoints",
    )
    pipeline.run(data_path="competition_data.json", output_path="submission.jsonl")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# 确保 Agent 目录在 sys.path 中（支持从任意目录调用本脚本）
_AGENT_DIR = Path(__file__).parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from react_agent import ReactAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 配置常量
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_WORKERS = 1          # 默认单线程（安全起步）
DEFAULT_RETRY_TIMES = 3          # API 失败最大重试次数
DEFAULT_RETRY_BASE_DELAY = 2.0   # 重试指数退避基础延迟（秒）
DEFAULT_REQUEST_INTERVAL = 0.5   # 同一线程内两次 API 调用最小间隔（秒）

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_path: str | Path) -> list[dict]:
    """从 JSON 或 JSONL 文件加载竞赛数据集。

    支持两种格式：
    - JSON 数组（.json）：[{"id": "...", "image": "...", "question": "..."}, ...]
    - JSON Lines（.jsonl）：每行一条 JSON 对象

    每条记录必须包含 "image" 和 "question" 字段。
    若缺少 "id" 字段，自动以行号（从 0 开始）填充。
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    suffix = data_path.suffix.lower()
    with open(data_path, encoding="utf-8") as f:
        if suffix == ".jsonl":
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)
            if not isinstance(records, list):
                raise ValueError(f"JSON 文件应为数组格式，实际类型: {type(records)}")

    # 补充缺失的 id 字段
    for i, rec in enumerate(records):
        if "id" not in rec:
            rec["id"] = str(i)

    logger.info("加载数据集: %s（%d 条）", data_path, len(records))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：检查点（Checkpoint）
# ─────────────────────────────────────────────────────────────────────────────

class Checkpoint:
    """线程安全的断点续传管理器。

    将已完成的结果持久化到磁盘（JSONL 格式），重启时自动加载已完成条目，
    跳过处理，确保每条数据恰好被处理一次（At-Most-Once 语义）。

    Parameters
    ----------
    checkpoint_path : Path
        检查点文件路径（JSONL，每行一条已完成结果）。
    """

    def __init__(self, checkpoint_path: Path) -> None:
        self._path = checkpoint_path
        self._lock = threading.Lock()
        self._done: dict[str, dict] = {}  # id → result record
        self._load()

    def _load(self) -> None:
        """启动时从磁盘加载已完成记录。"""
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        self._done[str(rec["id"])] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass
        logger.info("从检查点恢复 %d 条已完成记录 ← %s", len(self._done), self._path)

    def is_done(self, item_id: str) -> bool:
        """检查某条记录是否已处理完毕。"""
        with self._lock:
            return str(item_id) in self._done

    def save(self, result: dict) -> None:
        """追加一条完成记录到检查点文件并更新内存状态。"""
        item_id = str(result["id"])
        with self._lock:
            self._done[item_id] = result
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def all_results(self) -> list[dict]:
        """返回所有已完成的结果（按 id 排序）。"""
        with self._lock:
            return list(self._done.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._done)


# ─────────────────────────────────────────────────────────────────────────────
# 核心类：CompetitionPipeline
# ─────────────────────────────────────────────────────────────────────────────

class CompetitionPipeline:
    """面向竞赛打榜的批量推理主循环。

    Parameters
    ----------
    model : str
        VLM 模型名称（如 "qwen-vl-max"）。
    api_key : str, optional
        DashScope API Key，None 时从环境变量读取。
    grounding_backend : str
        视觉定位后端。"mock" 模式不需要 API Key 和 GPU。
    depth_backend : str
        深度估算后端。"mock" 模式不需要模型文件。
    max_workers : int
        并发线程数。1 = 单线程顺序处理（稳定，便于调试）。
        建议先用 1 验证流程，再逐步提高到 4~8（需考虑 API 限速）。
    retry_times : int
        单条数据失败后的最大重试次数（针对 API 限速或网络波动）。
    retry_base_delay : float
        重试指数退避的基础延迟（秒）。第 n 次重试等待 base * 2^(n-1) 秒。
    request_interval : float
        同一线程内连续两次 API 调用之间的最小间隔（秒），防止触发限速。
    max_iterations : int
        ReactAgent 的最大 ReAct 循环轮数。
    image_base_dir : str | Path | None
        图像路径基准目录。None 时使用数据文件所在目录。
    checkpoint_dir : str | Path
        检查点文件存储目录。
    verbose : bool
        是否打印每一步的 Thought/Action/Observation（调试用，生产环境建议关闭）。
    """

    def __init__(
        self,
        model: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        grounding_backend: str = "qwen3vl_api",
        depth_backend: str = "depth_anything_v3",
        max_workers: int = DEFAULT_MAX_WORKERS,
        retry_times: int = DEFAULT_RETRY_TIMES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        request_interval: float = DEFAULT_REQUEST_INTERVAL,
        max_iterations: int = 8,
        image_base_dir: Optional[str | Path] = None,
        checkpoint_dir: str | Path = "./checkpoints",
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.grounding_backend = grounding_backend
        self.depth_backend = depth_backend
        self.max_workers = max_workers
        self.retry_times = retry_times
        self.retry_base_delay = retry_base_delay
        self.request_interval = request_interval
        self.max_iterations = max_iterations
        self.image_base_dir = Path(image_base_dir) if image_base_dir else None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose

        # 线程局部存储：每个线程持有独立的 ReactAgent 实例，避免并发冲突
        self._thread_local = threading.local()

    # ------------------------------------------------------------------
    # 内部：懒加载每线程的 ReactAgent
    # ------------------------------------------------------------------

    def _get_agent(self) -> ReactAgent:
        """获取当前线程的 ReactAgent（首次访问时创建）。"""
        if not hasattr(self._thread_local, "agent"):
            self._thread_local.agent = ReactAgent(
                model=self.model,
                api_key=self.api_key,
                grounding_backend=self.grounding_backend,
                depth_backend=self.depth_backend,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )
        return self._thread_local.agent

    # ------------------------------------------------------------------
    # 内部：解析图像路径
    # ------------------------------------------------------------------

    def _resolve_image(self, raw_path: str, data_dir: Path) -> Path:
        """将数据集中的图像路径解析为绝对路径。

        解析优先级：
        1. 绝对路径 → 直接使用
        2. 相对路径 + image_base_dir（若指定）
        3. 相对路径 + data_dir（数据文件所在目录，默认）
        """
        p = Path(raw_path)
        if p.is_absolute():
            return p
        if self.image_base_dir is not None:
            candidate = self.image_base_dir / p
            if candidate.exists():
                return candidate
        return data_dir / p

    # ------------------------------------------------------------------
    # 内部：带重试的单条推理
    # ------------------------------------------------------------------

    def _process_one(self, item: dict, data_dir: Path) -> dict:
        """处理单条数据，带指数退避重试。

        Parameters
        ----------
        item : dict
            单条数据记录，必须含 "id"、"image"、"question" 字段。
        data_dir : Path
            数据文件所在目录，用于解析相对图像路径。

        Returns
        -------
        dict
            结果记录，必定含 "id"、"answer"、"status" 字段。
        """
        item_id = str(item["id"])
        question = item.get("question", "")
        image_raw = item.get("image", "")

        image_path = self._resolve_image(image_raw, data_dir)

        last_error = ""
        for attempt in range(self.retry_times + 1):
            if attempt > 0:
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[id=%s] 第 %d 次重试（等待 %.1fs）…",
                    item_id, attempt, delay,
                )
                time.sleep(delay)

            try:
                agent = self._get_agent()
                answer = agent.run(str(image_path), question)

                # 简单限速：避免连续调用过快触发 API 限流
                time.sleep(self.request_interval)

                return {
                    "id": item_id,
                    "answer": answer,
                    "status": "ok",
                    "attempts": attempt + 1,
                    "image": str(image_path),
                    "question": question,
                }

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.error("[id=%s] 推理异常（attempt=%d）: %s", item_id, attempt + 1, last_error)

        # 所有重试均失败 → 记录错误，继续处理下一条
        return {
            "id": item_id,
            "answer": f"[ERROR] {last_error}",
            "status": "error",
            "attempts": self.retry_times + 1,
            "image": str(image_path),
            "question": question,
        }

    # ------------------------------------------------------------------
    # 进度打印（tqdm 可选，回退到简单日志）
    # ------------------------------------------------------------------

    @staticmethod
    def _make_progress_bar(total: int):
        """尝试创建 tqdm 进度条，失败时返回 None。"""
        try:
            from tqdm import tqdm
            return tqdm(total=total, desc="推理进度", unit="条", dynamic_ncols=True)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # 主入口：run()
    # ------------------------------------------------------------------

    def run(
        self,
        data_path: str | Path,
        output_path: str | Path,
        resume: bool = True,
    ) -> dict[str, Any]:
        """对整个数据集执行批量推理，输出竞赛提交文件。

        Parameters
        ----------
        data_path : str | Path
            输入数据集路径（JSON 数组或 JSONL 格式）。
        output_path : str | Path
            最终提交文件路径（JSONL，每行一条 {"id": ..., "answer": ...}）。
        resume : bool
            True 时自动从检查点续传，跳过已完成条目（默认）。
            False 时忽略检查点，从头开始（覆盖现有结果，谨慎使用）。

        Returns
        -------
        dict
            运行汇总：total、done_new、skipped、error_cnt、elapsed_s。
        """
        data_path = Path(data_path)
        output_path = Path(output_path)
        data_dir = data_path.parent

        # ── 准备检查点 ───────────────────────────────────────────────────
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"ckpt_{data_path.stem}_{self.grounding_backend}_{self.depth_backend}.jsonl"
        ckpt_path = self.checkpoint_dir / ckpt_name
        ckpt = Checkpoint(ckpt_path) if resume else Checkpoint(Path(os.devnull))

        # ── 加载数据 ─────────────────────────────────────────────────────
        records = load_dataset(data_path)
        total = len(records)

        # 过滤掉已完成的条目
        pending = [r for r in records if not ckpt.is_done(str(r["id"]))]
        skipped = total - len(pending)
        if skipped:
            logger.info("断点续传：跳过 %d 条已完成记录，剩余 %d 条待处理", skipped, len(pending))

        if not pending:
            logger.info("所有数据已处理完毕，直接写出最终提交文件。")
            self._write_submission(ckpt.all_results(), output_path)
            return {
                "total": total, "done_new": 0, "skipped": skipped,
                "error_cnt": 0, "elapsed_s": 0.0,
            }

        # ── 批量推理 ─────────────────────────────────────────────────────
        t_start = time.time()
        done_new = 0
        error_cnt = 0
        pbar = self._make_progress_bar(len(pending))

        # 非续传模式下，在内存中收集本次处理的全部结果
        local_results: list[dict] = []

        def _task(item):
            result = self._process_one(item, data_dir)
            if resume:
                ckpt.save(result)
            return result

        if self.max_workers <= 1:
            # 单线程顺序处理（最稳定）
            for item in pending:
                result = _task(item)
                local_results.append(result)
                done_new += 1
                if result["status"] == "error":
                    error_cnt += 1
                status_sym = "✅" if result["status"] == "ok" else "❌"
                logger.info(
                    "[%d/%d] %s id=%s | %s",
                    done_new + skipped, total, status_sym,
                    result["id"], result["answer"][:60],
                )
                if pbar is not None:
                    pbar.update(1)
        else:
            # 多线程并发处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {executor.submit(_task, item): item for item in pending}
                for future in as_completed(future_map):
                    result = future.result()
                    local_results.append(result)
                    done_new += 1
                    if result["status"] == "error":
                        error_cnt += 1
                    status_sym = "✅" if result["status"] == "ok" else "❌"
                    logger.info(
                        "[%d/%d] %s id=%s | %s",
                        done_new + skipped, total, status_sym,
                        result["id"], result["answer"][:60],
                    )
                    if pbar is not None:
                        pbar.update(1)

        if pbar is not None:
            pbar.close()

        elapsed = time.time() - t_start

        # ── 写出最终提交文件 ─────────────────────────────────────────────
        if resume:
            all_results = ckpt.all_results()
        else:
            # 非续传模式：从本次内存结果中按原始数据顺序重建
            id_to_result = {str(r["id"]): r for r in local_results}
            all_results = [
                id_to_result.get(str(r["id"]), {"id": r["id"], "answer": "[MISSING]", "status": "missing"})
                for r in records
            ]
        self._write_submission(all_results, output_path)

        # ── 写出运行日志 ─────────────────────────────────────────────────
        run_summary = {
            "data_path": str(data_path),
            "output_path": str(output_path),
            "grounding_backend": self.grounding_backend,
            "depth_backend": self.depth_backend,
            "model": self.model,
            "max_workers": self.max_workers,
            "total": total,
            "done_new": done_new,
            "skipped": skipped,
            "error_cnt": error_cnt,
            "elapsed_s": round(elapsed, 2),
            "throughput_per_min": round(done_new / elapsed * 60, 1) if elapsed > 0 else 0,
        }
        log_path = output_path.with_suffix(".run_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

        logger.info(
            "=" * 60 + "\n"
            "运行完毕 | 总计=%d  新处理=%d  跳过=%d  错误=%d  耗时=%.2fs\n"
            "提交文件 → %s\n"
            "运行日志 → %s",
            total, done_new, skipped, error_cnt, elapsed,
            output_path, log_path,
        )
        return run_summary

    # ------------------------------------------------------------------
    # 写出竞赛提交文件
    # ------------------------------------------------------------------

    @staticmethod
    def _write_submission(results: list[dict], output_path: Path) -> None:
        """将结果列表写出为竞赛提交格式的 JSONL 文件。

        每行格式：{"id": "...", "answer": "..."}
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 按 id 排序（数字 id 按数值排，字符串 id 按字典序）
        def _sort_key(r):
            try:
                return (0, int(r["id"]))
            except (ValueError, KeyError):
                return (1, str(r.get("id", "")))

        sorted_results = sorted(results, key=_sort_key)

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in sorted_results:
                submission_rec = {"id": rec["id"], "answer": rec.get("answer", "")}
                f.write(json.dumps(submission_rec, ensure_ascii=False) + "\n")

        logger.info("提交文件已写出: %s（%d 条）", output_path, len(sorted_results))


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数：快速运行竞赛 Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_competition(
    data_path: str | Path,
    output_path: str | Path,
    model: str = "qwen-vl-max",
    api_key: Optional[str] = None,
    grounding_backend: str = "qwen3vl_api",
    depth_backend: str = "depth_anything_v3",
    max_workers: int = 1,
    checkpoint_dir: str = "./checkpoints",
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    """一行调用完整竞赛推理流水线。

    Parameters
    ----------
    data_path : str | Path
        输入数据集路径。
    output_path : str | Path
        提交文件输出路径。
    model : str
        VLM 模型名称。
    api_key : str, optional
        DashScope API Key（None 时读环境变量）。
    grounding_backend : str
        视觉定位后端。
    depth_backend : str
        深度估算后端。
    max_workers : int
        并发线程数。
    checkpoint_dir : str
        检查点目录。
    **pipeline_kwargs :
        透传给 CompetitionPipeline.__init__() 的其他参数。

    Returns
    -------
    dict  运行汇总。

    Examples
    --------
    >>> run_competition(
    ...     "data/competition_9k.json",
    ...     "submission.jsonl",
    ...     model="qwen-vl-max",
    ...     grounding_backend="qwen3vl_api",
    ...     depth_backend="depth_anything_v3",
    ...     max_workers=4,
    ... )
    """
    pipeline = CompetitionPipeline(
        model=model,
        api_key=api_key,
        grounding_backend=grounding_backend,
        depth_backend=depth_backend,
        max_workers=max_workers,
        checkpoint_dir=checkpoint_dir,
        **pipeline_kwargs,
    )
    return pipeline.run(data_path, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="竞赛打榜批量推理主循环（Agent Pipeline Runner）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, help="输入数据集路径（JSON 或 JSONL）")
    p.add_argument("--output", required=True, help="输出提交文件路径（JSONL）")
    p.add_argument("--model", default="qwen-vl-max", help="VLM 模型名称")
    p.add_argument(
        "--grounding",
        choices=["qwen3vl_api", "dino", "qwen3vl_local", "mock"],
        default="qwen3vl_api",
        help="视觉定位后端",
    )
    p.add_argument(
        "--depth",
        choices=["depth_anything_v3", "depth_anything_v2", "depth_anything", "midas", "mock"],
        default="depth_anything_v3",
        help="深度估算后端",
    )
    p.add_argument("--mock", action="store_true", help="强制使用 mock 后端（等价于 --grounding mock --depth mock）")
    p.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, dest="max_workers", help="并发线程数")
    p.add_argument("--max-iterations", type=int, default=8, dest="max_iterations", help="ReAct 最大循环轮数")
    p.add_argument("--retry", type=int, default=DEFAULT_RETRY_TIMES, dest="retry_times", help="单条最大重试次数")
    p.add_argument("--interval", type=float, default=DEFAULT_REQUEST_INTERVAL, dest="request_interval",
                   help="同一线程连续请求最小间隔（秒）")
    p.add_argument("--image-dir", default=None, dest="image_base_dir", help="图像根目录（覆盖数据文件目录）")
    p.add_argument("--checkpoint-dir", default="./checkpoints", dest="checkpoint_dir", help="检查点目录")
    p.add_argument("--no-resume", action="store_true", dest="no_resume", help="不使用断点续传，从头开始")
    p.add_argument("--verbose", action="store_true", help="打印每条推理的详细 Thought/Action/Observation")
    return p


if __name__ == "__main__":
    args = _build_cli().parse_args()

    if args.mock:
        args.grounding = "mock"
        args.depth = "mock"

    pipeline = CompetitionPipeline(
        model=args.model,
        grounding_backend=args.grounding,
        depth_backend=args.depth,
        max_workers=args.max_workers,
        max_iterations=args.max_iterations,
        retry_times=args.retry_times,
        request_interval=args.request_interval,
        image_base_dir=args.image_base_dir,
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose,
    )

    pipeline.run(
        data_path=args.data,
        output_path=args.output,
        resume=not args.no_resume,
    )
