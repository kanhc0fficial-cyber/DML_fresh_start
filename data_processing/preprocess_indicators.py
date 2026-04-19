"""
preprocess_indicators.py — 化验指标预处理脚本
================================================
从化验记录 Excel 中自动抽取各化验指标，生成标准化时间序列。

【设计原则】
  1. 自动扫描 Excel 中各小表块（不依赖固定行列索引），健壮性高。
  2. 8:30 单点化验指标（如班次品位）：仅做当天 8:30 → 次日 8:30 的 ffill
     （最多填充 1440 分钟，模拟"班次结果有效至下次化验"语义）。
  3. 其余多点化验指标（如每小时一次）：ffill 到下一个观测点为止
     （limit=None，即 pandas 默认 ffill，不做时间截断）。
  4. 不做插值、不做平滑，保持原始阶梯形信号，供下游建模自行处理。

【数据放置要求】
  data/
  └── 化验数据/
      ├── 化验记录_2023.xlsx    ← 每个 Excel 对应一段时间，可多个
      ├── 化验记录_2024.xlsx
      └── ...

  Excel 内每个 sheet 或表块的通用结构（支持两种）：
    A) 每列一个指标，第一行为指标名，第一列为时间（time/日期/时间 等）
    B) 多个小表块（每块：顶行=指标名，左列=时间），用空列分隔

  【关键约定】
    - 时间列名关键词：time / 时间 / 日期 / date（大小写不敏感）
    - 指标名与时间戳将被自动识别，不依赖固定行号
    - 若 sheet 中有多个小表块（以空列分隔），会自动逐块解析

【输出】
  data/indicators_final.parquet
  - index: DatetimeIndex (1min 频率，经 ffill 扩展)
  - columns: 各化验指标名（字符串）

【用法】
  python data_processing/preprocess_indicators.py

【依赖】
  pip install pandas openpyxl pyarrow
"""

import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─── 全局配置 ──────────────────────────────────────────────────────────────────
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 化验数据目录（可含多个 Excel 文件）
INDICATORS_DIR = BASE_DATA_DIR / "化验数据"

# 输出文件
OUTPUT_PARQUET = BASE_DATA_DIR / "indicators_final.parquet"

# 日志文件
LOG_FILE = BASE_DATA_DIR / "preprocess_indicators_log.txt"

# 重采样频率（最终宽表频率）
RESAMPLE_FREQ = "1min"

# 判断是否为"8:30 单点"的时间窗口（小时范围）
# 即：如果化验指标的每个值出现在 8:00~9:00 之间，则视为班次单点指标
SHIFT_HOUR_LO = 8   # 8:00
SHIFT_HOUR_HI = 9   # 9:00（不含）

# 8:30 单点指标的 ffill 最大时长（分钟，即到次日同时刻）
SHIFT_FFILL_LIMIT = 24 * 60  # 1440 分钟 = 24 小时

# 多点化验指标的 ffill 策略：填到下一个真实观测点（不限制分钟数）
# 若希望限制，将此值改为具体分钟数
MULTIPOINT_FFILL_LIMIT: Optional[int] = None  # None 表示不限制

# 物理范围（超范围视为异常，置 NaN）：可按指标名配置
# 示例：{"CONC_GRADE_XIN1": (0.0, 100.0), "CONC_GRADE_XIN2": (0.0, 100.0)}
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    # 精矿品位（铁品位）通常 55~72%
    # 根据实际情况调整
}


# ─── 日志 ──────────────────────────────────────────────────────────────────────
def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ─── 时间列识别 ────────────────────────────────────────────────────────────────
_TIME_KEYWORDS = {"time", "时间", "日期", "date", "datetime", "timestamp"}


def _is_time_col(col_name: str) -> bool:
    """判断列名是否为时间列（关键词匹配，大小写不敏感）。"""
    name = str(col_name).strip().lower()
    return any(kw in name for kw in _TIME_KEYWORDS)


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """
    将时间列解析为 pd.Timestamp。
    支持：datetime 对象、Excel 浮点数、字符串格式。
    """
    # 先尝试直接 pd.to_datetime
    parsed = pd.to_datetime(series, errors="coerce")
    # 对仍为 NaT 的，尝试把 Excel 序列数（浮点）转换
    excel_mask = parsed.isna() & series.notna()
    if excel_mask.any():
        try:
            excel_dates = pd.to_datetime(
                series[excel_mask].astype(float),
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )
            parsed[excel_mask] = excel_dates
        except Exception:
            pass
    return parsed


# ─── 单张 Sheet 解析 ────────────────────────────────────────────────────────────
def _parse_sheet_as_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    将一张 sheet 的原始 DataFrame 转换为以时间为 index 的指标 DataFrame。

    策略：
      1. 识别第一个时间列作为 index。
      2. 其余数值列为指标列。
      3. 若 sheet 中有多个小表块（以全 NaN 列分隔），逐块解析后 concat。

    返回：以 DatetimeIndex 为 index 的 DataFrame（可能为空）。
    """
    # ── 策略 A：寻找时间列 ────────────────────────────────────────────────────
    time_col = None
    for col in df_raw.columns:
        if _is_time_col(col):
            time_col = col
            break

    if time_col is not None:
        # 单表格式：时间列 + 指标列
        times = _parse_timestamps(df_raw[time_col])
        value_cols = [c for c in df_raw.columns if c != time_col]
        df_values = df_raw[value_cols].copy()
        df_values.index = times
        df_values = df_values[~df_values.index.isna()]
        df_values.index = pd.DatetimeIndex(df_values.index)
        df_values = df_values.apply(pd.to_numeric, errors="coerce")
        df_values.index.name = "time"
        return df_values

    # ── 策略 B：无时间列名，尝试用第一列作为时间列 ───────────────────────────
    # 判断第一列是否是时间
    first_col = df_raw.columns[0]
    parsed_first = _parse_timestamps(df_raw[first_col])
    if parsed_first.notna().sum() > len(df_raw) * 0.5:
        # 第一列确实是时间
        df_values = df_raw.iloc[:, 1:].copy()
        df_values.index = parsed_first
        df_values = df_values[~df_values.index.isna()]
        df_values.index = pd.DatetimeIndex(df_values.index)
        df_values = df_values.apply(pd.to_numeric, errors="coerce")
        df_values.index.name = "time"
        return df_values

    # ── 策略 C：检测多个小表块（以全 NaN 列分隔）────────────────────────────
    all_nan_cols = [c for c in df_raw.columns if df_raw[c].isna().all()]
    if all_nan_cols:
        blocks = []
        cols = list(df_raw.columns)
        start_idx = 0
        for nan_col in all_nan_cols:
            sep_idx = cols.index(nan_col)
            block_cols = cols[start_idx:sep_idx]
            if block_cols:
                block_df = _parse_sheet_as_dataframe(df_raw[block_cols])
                if not block_df.empty:
                    blocks.append(block_df)
            start_idx = sep_idx + 1
        # 最后一个块
        block_cols = cols[start_idx:]
        if block_cols:
            block_df = _parse_sheet_as_dataframe(df_raw[block_cols])
            if not block_df.empty:
                blocks.append(block_df)
        if blocks:
            return pd.concat(blocks, axis=1, sort=True)

    # 无法解析
    return pd.DataFrame()


# ─── 判断指标类型（8:30 单点 vs 多点）────────────────────────────────────────
def _is_shift_point_indicator(series: pd.Series) -> bool:
    """
    判断该指标是否为"8:30 单点"型：
    - 有效值的出现时间几乎全部落在 SHIFT_HOUR_LO ~ SHIFT_HOUR_HI 小时段内。
    - 阈值：>= 70% 的有效值落在该时间窗口内。
    """
    valid_idx = series.dropna().index
    if len(valid_idx) == 0:
        return False
    hours = pd.DatetimeIndex(valid_idx).hour
    in_window = ((hours >= SHIFT_HOUR_LO) & (hours < SHIFT_HOUR_HI)).sum()
    return (in_window / len(valid_idx)) >= 0.7


# ─── ffill 策略应用 ─────────────────────────────────────────────────────────
def _apply_ffill(series: pd.Series) -> pd.Series:
    """
    根据指标类型选择 ffill 策略：
      - 8:30 单点型：ffill limit = SHIFT_FFILL_LIMIT 分钟（1min 重采样后）
      - 多点型：ffill limit = MULTIPOINT_FFILL_LIMIT（None 表示到下一个真实观测点）
    """
    if _is_shift_point_indicator(series):
        _log(f"    [8:30单点] {series.name}  → ffill limit={SHIFT_FFILL_LIMIT} min")
        return series.ffill(limit=SHIFT_FFILL_LIMIT)
    else:
        _log(f"    [多点]     {series.name}  → ffill limit={MULTIPOINT_FFILL_LIMIT}")
        return series.ffill(limit=MULTIPOINT_FFILL_LIMIT)


# ─── 物理范围裁剪 ─────────────────────────────────────────────────────────────
def _apply_physical_clip(series: pd.Series) -> pd.Series:
    """对配置了物理界的指标进行裁剪，超范围置 NaN。"""
    name = str(series.name).strip()
    if name in PHYSICAL_BOUNDS:
        lo, hi = PHYSICAL_BOUNDS[name]
        return series.where((series >= lo) & (series <= hi))
    return series


# ─── 解析单个 Excel 文件 ────────────────────────────────────────────────────
def parse_indicators_excel(xlsx_path: Path) -> pd.DataFrame:
    """
    解析一个化验记录 Excel 文件，返回合并后的宽表 DataFrame。

    遍历所有 sheet，尝试解析后 concat 成按时间排序的宽表。
    """
    _log(f"  [解析] {xlsx_path.name}")
    frames = []

    try:
        xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        _log(f"  [跳过] 无法打开 {xlsx_path.name}: {e}")
        return pd.DataFrame()

    for sheet_name in xl.sheet_names:
        try:
            df_raw = xl.parse(sheet_name, header=0)
        except Exception as e:
            _log(f"    [跳过 sheet] {sheet_name}: {e}")
            continue

        if df_raw.empty:
            continue

        # 统一列名为字符串，去除前后空格
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        df_parsed = _parse_sheet_as_dataframe(df_raw)
        if df_parsed.empty:
            _log(f"    [跳过 sheet] {sheet_name}：无法解析时间列或数据列")
            continue

        # 去除列名为数字或空字符串的无意义列
        df_parsed = df_parsed[[c for c in df_parsed.columns
                                if str(c).strip() and not str(c).strip().startswith("Unnamed")]]

        if not df_parsed.empty:
            _log(f"    [解析] sheet={sheet_name}  行={len(df_parsed)}  列={list(df_parsed.columns)}")
            frames.append(df_parsed)

    if not frames:
        return pd.DataFrame()

    # 合并所有 sheet（外连接）
    result = pd.concat(frames, axis=0, sort=True)
    result = result[~result.index.duplicated(keep="first")].sort_index()
    return result


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def build_indicators(indicators_dir: Path, output_path: Path):
    """
    扫描 indicators_dir 下所有 xlsx，解析化验指标，
    合并 → 重采样 → ffill → 输出 Parquet。
    """
    xlsx_files = sorted(indicators_dir.glob("*.xlsx")) + sorted(indicators_dir.glob("*.xls"))
    if not xlsx_files:
        _log(f"[错误] 未找到任何 Excel 文件，目录: {indicators_dir}")
        return

    _log(f"[扫描] 共找到 {len(xlsx_files)} 个化验 Excel 文件")

    # 合并所有 Excel 的宽表
    all_frames = []
    for xlsx_path in xlsx_files:
        df = parse_indicators_excel(xlsx_path)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        _log("[错误] 所有 Excel 文件均无法解析！请检查文件格式。")
        return

    # 外连接合并，去重，排序
    combined = pd.concat(all_frames, axis=0, sort=True)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    _log(f"[合并] 原始合并后: {combined.shape}  时间范围: {combined.index.min()} ~ {combined.index.max()}")

    # 确保 index 是 DatetimeIndex
    if not isinstance(combined.index, pd.DatetimeIndex):
        combined.index = pd.to_datetime(combined.index, errors="coerce")
        combined = combined[~combined.index.isna()]

    # ── 重采样到 1min（保持稀疏 NaN，不 ffill）──────────────────────────────
    # 重采样时不自动 ffill，只在有数据的时间点放入值
    full_range = pd.date_range(
        start=combined.index.min().floor("min"),
        end=combined.index.max().ceil("min"),
        freq=RESAMPLE_FREQ,
    )
    # 先对原始数据 reindex 到分钟级（不改变值，只对齐时间轴到最近分钟）
    combined.index = combined.index.round("min")
    combined = combined[~combined.index.duplicated(keep="first")]
    result_df = combined.reindex(full_range)
    result_df.index.name = "time"

    # ── 物理裁剪 + 分类 ffill ────────────────────────────────────────────────
    _log("[ffill] 正在对各指标应用物理裁剪和 ffill 策略...")
    processed_cols = {}
    for col in result_df.columns:
        s = result_df[col].copy()
        s = _apply_physical_clip(s)
        s = _apply_ffill(s)
        processed_cols[col] = s

    result_df = pd.DataFrame(processed_cols, index=result_df.index)

    _log(f"[完成] 最终化验指标宽表: {result_df.shape}")
    _log(f"       时间范围: {result_df.index.min()} ~ {result_df.index.max()}")
    _log(f"       整体缺失率: {result_df.isnull().mean().mean():.2%}")
    _log(f"       指标列表: {list(result_df.columns)}")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    _log(f"[保存] 输出至 {output_path}  大小: {size_mb:.2f} MB")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    _log("=" * 60)
    _log("preprocess_indicators.py — 化验指标预处理脚本 启动")
    _log(f"化验数据目录: {INDICATORS_DIR}")
    _log(f"输出文件: {OUTPUT_PARQUET}")
    _log("=" * 60)

    if not INDICATORS_DIR.exists():
        _log(f"[错误] 化验数据目录不存在: {INDICATORS_DIR}")
        _log("       请在 data/化验数据/ 下放置化验记录 Excel 文件后重新运行。")
    else:
        build_indicators(INDICATORS_DIR, OUTPUT_PARQUET)
        _log("\n全部完成！")
