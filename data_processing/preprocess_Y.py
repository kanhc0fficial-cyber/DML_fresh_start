"""
preprocess_Y.py — Y（精矿品位）变量处理脚本
=============================================
从化验记录 Excel 中自动定位"新1#精矿"和"新2#精矿"两条生产线的精矿品位数据，
生成标准化的 Y 目标变量时间序列。

【设计原则】
  1. 自动寻找"新1#"和"新2#"精矿化验块：
     - 扫描所有 sheet，匹配关键词（新1、新2、xin1、xin2、精矿等）
     - 不依赖固定行列索引，健壮性高
  2. 时间戳对齐到分钟精度（floor 到分钟）
  3. 物理范围裁剪（0 ~ 100%）：超范围值直接赋 NaN，不做任何插值
  4. 不做多余过滤（不去除"异常班次"、不做平滑）
  5. 两条线独立输出到同一个 Parquet 的两列：y_fx_xin1 和 y_fx_xin2

【数据放置要求】
  data/
  └── 化验数据/
      ├── 化验记录_2023.xlsx    ← 每个文件对应一段时间，可多个
      └── ...

  Excel 内需包含包含精矿品位的表块，典型结构（示例）：
    |    时间     | 新1#精矿品位 | 新2#精矿品位 |  ...  |
    |-------------|-------------|-------------|-------|
    | 2023-10-13  |    65.2     |    64.8     |  ...  |

  或者按 sheet 分块，sheet 名含"精矿"、"新1"、"新2"等关键词。

  【关键约定】
    - 支持多种列名格式（大小写、全/半角不限）：
        新1#精矿品位、新1精矿品位、xin1精矿、新1品位、新1#、精矿品位(新1)等
    - 时间列名含 time/时间/日期/date 等关键词（大小写不敏感）
    - 若一个 sheet 中同时有两列可匹配新1和新2，会同时提取

【输出】
  data/y_target_final.parquet
  - index: DatetimeIndex (分钟精度，不规则间隔，即原始化验频率)
  - columns: ["y_fx_xin1", "y_fx_xin2"]  （其中一列可能全为 NaN 如果未找到）

【用法】
  python data_processing/preprocess_Y.py

【依赖】
  pip install pandas openpyxl pyarrow
"""

import re
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─── 全局配置 ──────────────────────────────────────────────────────────────────
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 化验数据目录（与 preprocess_indicators.py 共用）
INDICATORS_DIR = BASE_DATA_DIR / "化验数据"

# 输出文件
OUTPUT_PARQUET = BASE_DATA_DIR / "y_target_final.parquet"

# 日志文件
LOG_FILE = BASE_DATA_DIR / "preprocess_Y_log.txt"

# Y 列名映射（输出列名固定，便于下游脚本统一引用）
Y_COL_XIN1 = "y_fx_xin1"
Y_COL_XIN2 = "y_fx_xin2"

# 精矿品位的物理合理范围（单位：%）
# 铁精矿品位通常在 50~72% 之间；超出范围视为录入错误，赋 NaN
Y_PHYSICAL_LO = 0.0
Y_PHYSICAL_HI = 100.0

# 严格物理界（可选，更严格的工艺范围；None 表示不使用）
# 例如铁精矿一般不低于 50%，不高于 72%
# 若数据确实存在此范围外的正常值，请改为 None
Y_STRICT_LO: Optional[float] = None   # e.g. 50.0
Y_STRICT_HI: Optional[float] = None   # e.g. 72.0

# 新1# 产线关键词（用于匹配列名/sheet名，正则表达式）
_PATTERN_XIN1 = re.compile(
    r"(新\s*1\s*[##]?|xin\s*1|新一|line\s*1|1\s*[##]\s*精矿)",
    re.IGNORECASE,
)

# 新2# 产线关键词
_PATTERN_XIN2 = re.compile(
    r"(新\s*2\s*[##]?|xin\s*2|新二|line\s*2|2\s*[##]\s*精矿)",
    re.IGNORECASE,
)

# 品位关键词（辅助匹配，防止把非品位列误识别为 Y）
_PATTERN_GRADE = re.compile(
    r"(品位|grade|品级|铁品位|精矿|concentrate)",
    re.IGNORECASE,
)

# 时间列关键词
_TIME_KEYWORDS = {"time", "时间", "日期", "date", "datetime", "timestamp"}


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


# ─── 时间列识别与解析 ─────────────────────────────────────────────────────────
def _is_time_col(col_name: str) -> bool:
    name = str(col_name).strip().lower()
    return any(kw in name for kw in _TIME_KEYWORDS)


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """将时间列解析为 pd.Timestamp，支持 datetime/字符串/Excel浮点数。"""
    parsed = pd.to_datetime(series, errors="coerce")
    # 尝试 Excel 序列日期格式
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


# ─── 判断列是否为新1#/新2#精矿品位列 ─────────────────────────────────────────
def _classify_col(col_name: str) -> Optional[str]:
    """
    返回 'xin1'、'xin2' 或 None（无法识别）。

    同时匹配产线关键词和品位关键词，减少误识别。
    若列名只含产线关键词而没有品位信息，但 sheet 本身已是精矿品位表，
    则由调用方额外传入上下文判断（见 _extract_y_from_df）。
    """
    name = str(col_name).strip()
    has_xin1 = bool(_PATTERN_XIN1.search(name))
    has_xin2 = bool(_PATTERN_XIN2.search(name))
    has_grade = bool(_PATTERN_GRADE.search(name))

    if has_xin1 and (has_grade or has_xin1):
        return "xin1"
    if has_xin2 and (has_grade or has_xin2):
        return "xin2"
    return None


def _classify_sheet(sheet_name: str) -> Optional[str]:
    """判断 sheet 名对应哪条产线（若无法判断返回 None）。"""
    if _PATTERN_XIN1.search(sheet_name):
        return "xin1"
    if _PATTERN_XIN2.search(sheet_name):
        return "xin2"
    return None


# ─── 从 DataFrame 提取 Y 列 ──────────────────────────────────────────────────
def _extract_y_from_df(
    df_raw: pd.DataFrame,
    sheet_hint: Optional[str] = None,
) -> dict[str, pd.Series]:
    """
    从一个 DataFrame 中提取新1#/新2#精矿品位列。

    参数：
      df_raw:     原始 DataFrame（从 excel.parse 读出）
      sheet_hint: sheet 名提供的产线提示（'xin1'/'xin2'/None）

    返回：
      {'xin1': pd.Series, 'xin2': pd.Series}  中的部分或全部
    """
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    results: dict[str, pd.Series] = {}

    # 识别时间列
    time_col = None
    for col in df_raw.columns:
        if _is_time_col(col):
            time_col = col
            break

    # 回退：用第一列作为时间列（如果它像时间）
    if time_col is None:
        first_col = df_raw.columns[0]
        trial = _parse_timestamps(df_raw[first_col])
        if trial.notna().sum() > len(df_raw) * 0.5:
            time_col = first_col

    if time_col is None:
        return results  # 没有时间列，跳过

    times = _parse_timestamps(df_raw[time_col])
    valid_time_mask = times.notna()
    if valid_time_mask.sum() == 0:
        return results

    # 逐列判断是否为品位列
    value_cols = [c for c in df_raw.columns if c != time_col]

    for col in value_cols:
        line = _classify_col(col)

        # 若列名无法判断产线，但 sheet 只有一条线，则用 sheet_hint
        if line is None and sheet_hint is not None:
            # 进一步判断：列名是否含"品位"等关键词，或数值在合理范围内
            if _PATTERN_GRADE.search(col) or _looks_like_grade_column(df_raw[col]):
                line = sheet_hint

        if line is None:
            continue

        # 提取数值
        vals = pd.to_numeric(df_raw[col], errors="coerce")
        s = pd.Series(
            vals.values,
            index=pd.DatetimeIndex(times.values),
            name=f"y_fx_{line}",
        )
        s = s[valid_time_mask.values]
        s = s.dropna()

        if s.empty:
            continue

        # 时间对齐到分钟精度
        s.index = s.index.floor("min")
        s = s[~s.index.duplicated(keep="first")].sort_index()

        # 物理范围裁剪
        s = _apply_physical_clip(s)

        if line in results:
            # 若已有该产线的数据，合并（取均值处理重复时间戳）
            combined = pd.concat([results[line], s])
            combined = combined.groupby(combined.index).mean()
            results[line] = combined
        else:
            results[line] = s

    return results


def _looks_like_grade_column(series: pd.Series) -> bool:
    """
    启发式判断：一列数值是否像品位列（0~100 之间的浮点数，均值在合理范围）。
    用于当列名不含明确品位关键词时的辅助判断。
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 3:
        return False
    # 品位通常在 30~85% 之间，且标准差不为 0
    mean_val = numeric.mean()
    std_val = numeric.std()
    return (30.0 <= mean_val <= 85.0) and (std_val > 0.01)


# ─── 物理范围裁剪 ─────────────────────────────────────────────────────────────
def _apply_physical_clip(series: pd.Series) -> pd.Series:
    """
    物理范围裁剪：
      1. 全局安全界 [0, 100]（品位不可能超出此范围）
      2. 如果配置了严格物理界，进一步裁剪
    """
    s = series.where((series >= Y_PHYSICAL_LO) & (series <= Y_PHYSICAL_HI))
    if Y_STRICT_LO is not None and Y_STRICT_HI is not None:
        s = s.where((s >= Y_STRICT_LO) & (s <= Y_STRICT_HI))
    return s


# ─── 解析单个 Excel 文件 ────────────────────────────────────────────────────
def parse_y_from_excel(xlsx_path: Path) -> dict[str, pd.Series]:
    """
    解析一个化验记录 Excel 文件，提取新1#/新2#精矿品位数据。

    返回：{'xin1': pd.Series, 'xin2': pd.Series}（可能只有其中一个）
    """
    _log(f"  [解析] {xlsx_path.name}")
    results: dict[str, list] = {"xin1": [], "xin2": []}

    try:
        xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        _log(f"  [跳过] 无法打开 {xlsx_path.name}: {e}")
        return {}

    for sheet_name in xl.sheet_names:
        # 判断 sheet 名对应的产线提示
        sheet_line_hint = _classify_sheet(sheet_name)

        try:
            df_raw = xl.parse(sheet_name, header=0)
        except Exception as e:
            _log(f"    [跳过 sheet] {sheet_name}: {e}")
            continue

        if df_raw.empty:
            continue

        extracted = _extract_y_from_df(df_raw, sheet_hint=sheet_line_hint)

        for line, s in extracted.items():
            if not s.empty:
                results[line].append(s)
                _log(f"    [找到] sheet={sheet_name}  产线={line}  有效点数={len(s)}")

    # 合并各产线的所有 Series
    final: dict[str, pd.Series] = {}
    for line, series_list in results.items():
        if series_list:
            combined = pd.concat(series_list)
            # 同一时间点多次录入取均值
            combined = combined.groupby(combined.index).mean()
            combined = combined.sort_index()
            final[line] = combined

    return final


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def build_y(indicators_dir: Path, output_path: Path):
    """
    扫描化验数据目录，提取两条产线的精矿品位，输出 Parquet。
    """
    xlsx_files = sorted(indicators_dir.glob("*.xlsx")) + sorted(indicators_dir.glob("*.xls"))
    if not xlsx_files:
        _log(f"[错误] 未找到任何 Excel 文件，目录: {indicators_dir}")
        return

    _log(f"[扫描] 共找到 {len(xlsx_files)} 个化验 Excel 文件")

    all_xin1: list[pd.Series] = []
    all_xin2: list[pd.Series] = []

    for xlsx_path in xlsx_files:
        y_data = parse_y_from_excel(xlsx_path)
        if "xin1" in y_data:
            all_xin1.append(y_data["xin1"])
        if "xin2" in y_data:
            all_xin2.append(y_data["xin2"])

    # 合并
    xin1_series = None
    xin2_series = None

    if all_xin1:
        s = pd.concat(all_xin1)
        s = s.groupby(s.index).mean().sort_index()
        s.name = Y_COL_XIN1
        xin1_series = s
        _log(f"[新1#] 共 {len(s)} 个化验点  "
             f"品位范围: [{s.min():.2f}, {s.max():.2f}]  "
             f"均值: {s.mean():.2f}")
    else:
        _log("[警告] 未找到新1#精矿品位数据！")

    if all_xin2:
        s = pd.concat(all_xin2)
        s = s.groupby(s.index).mean().sort_index()
        s.name = Y_COL_XIN2
        xin2_series = s
        _log(f"[新2#] 共 {len(s)} 个化验点  "
             f"品位范围: [{s.min():.2f}, {s.max():.2f}]  "
             f"均值: {s.mean():.2f}")
    else:
        _log("[警告] 未找到新2#精矿品位数据！")

    if xin1_series is None and xin2_series is None:
        _log("[错误] 两条产线均未找到品位数据！")
        _log("       请检查：")
        _log("       1. 化验 Excel 中是否包含列名含'新1'/'新2'/'品位'的列？")
        _log("       2. 或 sheet 名是否含'新1'/'新2'等关键词？")
        _log("       3. 如需自定义匹配规则，修改脚本顶部的 _PATTERN_XIN1 / _PATTERN_XIN2。")
        return

    # 拼宽表
    parts = []
    if xin1_series is not None:
        parts.append(xin1_series)
    if xin2_series is not None:
        parts.append(xin2_series)

    y_df = pd.concat(parts, axis=1, sort=True)
    y_df.index.name = "time"

    # 确保两列都存在（即使全 NaN）
    if Y_COL_XIN1 not in y_df.columns:
        y_df[Y_COL_XIN1] = np.nan
    if Y_COL_XIN2 not in y_df.columns:
        y_df[Y_COL_XIN2] = np.nan
    y_df = y_df[[Y_COL_XIN1, Y_COL_XIN2]]

    _log(f"\n[完成] Y 目标宽表: {y_df.shape}")
    _log(f"       时间范围: {y_df.index.min()} ~ {y_df.index.max()}")
    _log(f"       新1# 有效点数: {y_df[Y_COL_XIN1].notna().sum()}")
    _log(f"       新2# 有效点数: {y_df[Y_COL_XIN2].notna().sum()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_df.to_parquet(output_path, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    _log(f"[保存] 输出至 {output_path}  大小: {size_mb:.2f} MB")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    _log("=" * 60)
    _log("preprocess_Y.py — Y（精矿品位）变量处理脚本 启动")
    _log(f"化验数据目录: {INDICATORS_DIR}")
    _log(f"输出文件: {OUTPUT_PARQUET}")
    _log("=" * 60)

    if not INDICATORS_DIR.exists():
        _log(f"[错误] 化验数据目录不存在: {INDICATORS_DIR}")
        _log("       请在 data/化验数据/ 下放置化验记录 Excel 文件后重新运行。")
    else:
        build_y(INDICATORS_DIR, OUTPUT_PARQUET)
        _log("\n全部完成！")
