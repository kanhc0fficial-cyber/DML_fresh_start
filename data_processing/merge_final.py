"""
merge_final.py — 最终 X/Y/指标对齐宽表生成脚本
================================================
将预处理好的 X 特征、Y 目标变量和化验指标合并为两类建模用宽表：

  1. 时序宽表（timeseries_dataset_*.parquet）：锚点为 X 全时间轴，保留完整 1min
     频率连续时序，Y 列稀疏（仅化验时间点有值，其余为 NaN）。
     → 用于时序因果发现算法（TCDF / NTS-NOTEARS / MultiScale-NTS 等），
       这些算法依赖连续等时间间隔数据构建滑动窗口，必须保留完整时序。

  2. 监督宽表（modeling_dataset_*.parquet）：从时序宽表中过滤出 Y 非 NaN 的行，
     仅保留有化验标签的时间点。
     → 用于双重机器学习（DML）等有监督建模，Y 必须已知。

【设计原则】
  1. 锚点为 X 全时间轴：保留所有 X 有效时间点（1min 连续频率），不压缩到稀疏
     的 Y 化验点，保护时序连续性。
  2. ±1 分钟容差对齐：Y 的时间戳通常在化验记录的整点或半点，X 是 1min 频率；
     用 merge_asof（容差 1min）将 Y 对齐到 X 的时间戳（而非反过来）。
  3. 合并化验指标（indicators）列：同样以 X 为锚点，±1min 容差 merge_asof。
  4. X 完全为空的行剔除：该时间点所有 X 特征均缺失，代表真实停产/无数据，
     在时序宽表和监督宽表中均剔除。
  5. 监督宽表由时序宽表派生：对时序宽表按 Y 非 NaN 过滤即可得到监督宽表，
     无需二次对齐，保证两个输出的 X/指标值完全一致。

【前置条件（需先运行以下脚本）】
  python data_processing/preprocess_X.py          → data/X_features_final.parquet
  python data_processing/preprocess_Y.py          → data/y_target_final.parquet
  python data_processing/preprocess_indicators.py → data/indicators_final.parquet

【数据文件说明】
  data/
  ├── X_features_final.parquet    ← 1min 频率 X 特征宽表（index=time）
  ├── y_target_final.parquet      ← Y 品位（稀疏时间点，columns=y_fx_xin1/y_fx_xin2）
  ├── indicators_final.parquet    ← 化验指标宽表（1min 频率，ffill 后）
  ├── timeseries_dataset_final.parquet    ← 本脚本输出（全时序，供因果发现）
  └── modeling_dataset_final.parquet      ← 本脚本输出（Y 锚点子集，供 DML）

【输出说明】
  data/timeseries_dataset_final.parquet（主输出，时序因果发现用）
  - index: time（X 的全时间轴，1min 频率）
  - columns: 所有 X 特征列 + 所有化验指标列 + y_fx_xin1 + y_fx_xin2
  - 行：全部有效 X 时间点；X 完全为空（全 NaN）的行已剔除；Y 列稀疏

  data/modeling_dataset_final.parquet（派生输出，DML 监督学习用，向后兼容）
  - 与 timeseries_dataset 结构相同，但只保留 Y 非 NaN 的行

【用法】
  python data_processing/merge_final.py

  可选：按产线拆分建模数据集：
    python data_processing/merge_final.py --line xin1
    python data_processing/merge_final.py --line xin2

【依赖】
  pip install pandas pyarrow numpy
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─── 全局配置 ──────────────────────────────────────────────────────────────────
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 输入文件（由前三个脚本生成）
X_PARQUET = BASE_DATA_DIR / "X_features_final.parquet"
Y_PARQUET = BASE_DATA_DIR / "y_target_final.parquet"
IND_PARQUET = BASE_DATA_DIR / "indicators_final.parquet"

# 输出文件 —— 时序宽表（全时间轴，供时序因果发现算法使用）
TS_OUTPUT_PARQUET = BASE_DATA_DIR / "timeseries_dataset_final.parquet"

# 输出文件 —— 监督宽表（仅 Y 非 NaN 行，供 DML 等有监督建模使用；向后兼容）
OUTPUT_PARQUET = BASE_DATA_DIR / "modeling_dataset_final.parquet"

# 日志文件
LOG_FILE = BASE_DATA_DIR / "merge_final_log.txt"

# Y 列名
Y_COL_XIN1 = "y_fx_xin1"
Y_COL_XIN2 = "y_fx_xin2"

# ±1 分钟对齐容差（单位：ns，供 pd.merge_asof 使用）
MERGE_TOLERANCE = pd.Timedelta("1min")

# X 特征行过滤：只剔除所有 X 特征均为 NaN 的行（真实停产/全盲点），不按缺失率阈值剔除。
# PLC exception-based recording 下，大部分"缺失"已由 preprocess_X.py 的活跃掩码 ffill 填充；
# 若此处仍有大量缺失，说明该时间点系统根本没有采集到任何数据，属于真实停产空白。

# 是否保存分产线的子数据集（便于分别建模）
SAVE_PER_LINE = True


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


# ─── 加载数据 ──────────────────────────────────────────────────────────────────
def load_x(x_path: Path) -> pd.DataFrame:
    """加载 X 特征宽表，确保 index 为 DatetimeIndex。"""
    _log(f"[加载] X 特征: {x_path}")
    X = pd.read_parquet(x_path)
    X.index = pd.to_datetime(X.index).tz_localize(None)
    X.index.name = "time"
    X = X.sort_index()
    _log(f"  X 形状: {X.shape}  时间范围: {X.index.min()} ~ {X.index.max()}")
    return X


def load_y(y_path: Path) -> pd.DataFrame:
    """加载 Y 目标宽表，去除两列均为 NaN 的行。"""
    _log(f"[加载] Y 目标: {y_path}")
    y = pd.read_parquet(y_path)
    y.index = pd.to_datetime(y.index).tz_localize(None)
    y.index.name = "time"
    y = y.sort_index()

    # 确保两列存在
    for col in [Y_COL_XIN1, Y_COL_XIN2]:
        if col not in y.columns:
            _log(f"  [警告] Y 文件中没有列 '{col}'，将填充全 NaN")
            y[col] = np.nan

    y = y[[Y_COL_XIN1, Y_COL_XIN2]]
    _log(f"  Y 形状: {y.shape}")
    _log(f"  新1# 有效点数: {y[Y_COL_XIN1].notna().sum()}")
    _log(f"  新2# 有效点数: {y[Y_COL_XIN2].notna().sum()}")
    return y


def load_indicators(ind_path: Path) -> Optional[pd.DataFrame]:
    """加载化验指标宽表（可选，不存在时返回 None）。"""
    if not ind_path.exists():
        _log(f"[提示] 未找到化验指标文件 {ind_path}，跳过合并化验指标。")
        return None
    _log(f"[加载] 化验指标: {ind_path}")
    ind = pd.read_parquet(ind_path)
    ind.index = pd.to_datetime(ind.index).tz_localize(None)
    ind.index.name = "time"
    ind = ind.sort_index()
    _log(f"  指标形状: {ind.shape}  列: {list(ind.columns[:10])}{'...' if len(ind.columns)>10 else ''}")
    return ind


# ─── 核心对齐函数 ──────────────────────────────────────────────────────────────
def merge_asof_with_tolerance(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
    right_suffix: str = "_r",
) -> pd.DataFrame:
    """
    用 merge_asof 将 right 的列对齐到 left 的时间戳，容差 ±tolerance。

    策略：
      - 先用 direction='nearest'（最近邻）做一次 merge_asof
      - 再过滤掉时间差 > tolerance 的匹配（置 NaN）

    参数：
      left:         锚点 DataFrame（通常是连续 1min 频率的 X；也可以是稀疏时序）
      right:        被对齐的 DataFrame（稀疏或连续时序均可）
      tolerance:    最大允许时间差
      right_suffix: 用于避免列名冲突的后缀

    返回：
      合并后的 DataFrame，index 来自 left，包含 left 和 right 的列。
    """
    # reset_index 以便 merge_asof 使用 'time' 列
    left_reset = left.reset_index()   # 含 'time' 列
    right_reset = right.reset_index()  # 含 'time' 列

    # 重命名 right 的 time 列以便后续计算时间差
    right_reset = right_reset.rename(columns={"time": "_time_r"})

    merged = pd.merge_asof(
        left_reset.sort_values("time"),
        right_reset.sort_values("_time_r"),
        left_on="time",
        right_on="_time_r",
        direction="nearest",
    )

    # 过滤超过容差的匹配：超出容差的 right 列置 NaN
    if "_time_r" in merged.columns:
        time_diff = (merged["time"] - merged["_time_r"]).abs()
        out_of_tolerance = time_diff > tolerance
        right_cols = [c for c in merged.columns if c not in left_reset.columns and c != "_time_r"]
        if out_of_tolerance.any():
            _log(f"  [容差过滤] 超出 ±{tolerance} 的匹配: {out_of_tolerance.sum()} 个，对应右侧列置 NaN")
            merged.loc[out_of_tolerance, right_cols] = np.nan
        merged = merged.drop(columns=["_time_r"])

    merged = merged.set_index("time")
    return merged


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def build_modeling_dataset(
    line: Optional[str] = None,
    output_path: Optional[Path] = None,
    ts_output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    构建最终宽表，输出两个 Parquet 文件：

      1. 时序宽表（ts_output_path）：以 X 全时间轴为锚点，保留完整 1min 连续时序。
         Y 列稀疏（仅化验时间点有值，其余 NaN）。供时序因果发现算法使用。

      2. 监督宽表（output_path）：从时序宽表过滤出 Y 非 NaN 的行。供 DML 使用。

    参数：
      line:           'xin1'、'xin2' 或 None（None 表示保留两条线的所有 Y 列）
      output_path:    监督宽表 Parquet 路径（None 时使用默认 OUTPUT_PARQUET）
      ts_output_path: 时序宽表 Parquet 路径（None 时使用默认 TS_OUTPUT_PARQUET）

    返回：
      时序宽表 DataFrame（全时间轴，Y 列稀疏）
    """
    if output_path is None:
        if line is not None:
            output_path = BASE_DATA_DIR / f"modeling_dataset_{line}_final.parquet"
        else:
            output_path = OUTPUT_PARQUET

    if ts_output_path is None:
        if line is not None:
            ts_output_path = BASE_DATA_DIR / f"timeseries_dataset_{line}_final.parquet"
        else:
            ts_output_path = TS_OUTPUT_PARQUET

    # ── 检查必需输入文件 ──────────────────────────────────────────────────────
    missing_inputs = []
    for p, name in [(X_PARQUET, "X特征"), (Y_PARQUET, "Y目标")]:
        if not p.exists():
            missing_inputs.append(f"{name}: {p}")
    if missing_inputs:
        _log("[错误] 以下输入文件不存在，请先运行对应预处理脚本：")
        for m in missing_inputs:
            _log(f"  {m}")
        return pd.DataFrame()

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    X = load_x(X_PARQUET)
    y = load_y(Y_PARQUET)
    ind = load_indicators(IND_PARQUET)

    # ── 确定本次使用的 Y 列 ────────────────────────────────────────────────────
    if line == "xin1":
        y_sub = y[[Y_COL_XIN1]]
        y_cols_to_include = [Y_COL_XIN1]
    elif line == "xin2":
        y_sub = y[[Y_COL_XIN2]]
        y_cols_to_include = [Y_COL_XIN2]
    else:
        y_sub = y
        y_cols_to_include = [Y_COL_XIN1, Y_COL_XIN2]

    _log(f"\n[对齐] 锚点为 X 全时间轴（{len(X)} 行，1min 频率），将 Y 对齐到 X 时间戳")
    _log(f"       产线过滤: {line or '全部'}  Y 有效点数: "
         f"{y_sub.notna().any(axis=1).sum()}")

    # ── 以 X 为锚点：将 Y 对齐到 X 的时间戳 ─────────────────────────────────
    # 注意方向：左表=X（锚点），右表=Y（稀疏），结果行数 = X 行数。
    # 每个 X 时间点找最近的 Y 时间点（容差 ±1min）；超出容差的置 NaN，
    # 因此大多数 X 行的 Y 列为 NaN，保留了完整时序连续性。
    _log(f"[对齐] 将 Y（{y_sub.shape[1]} 列）对齐到 X 时间戳（±{MERGE_TOLERANCE}）...")
    merged = merge_asof_with_tolerance(X, y_sub, tolerance=MERGE_TOLERANCE)
    _log(f"  对齐后（X 锚点）: {merged.shape}  Y 有效行数: "
         f"{merged[y_cols_to_include].notna().any(axis=1).sum()}")

    # ── 合并化验指标 ──────────────────────────────────────────────────────────
    if ind is not None and not ind.empty:
        _log(f"[对齐] 将化验指标（{ind.shape[1]} 列）对齐到 X 时间戳（±{MERGE_TOLERANCE}）...")
        # 避免与已有列名冲突
        ind_cols = [c for c in ind.columns if c not in merged.columns]
        if ind_cols:
            ind_sub = ind[ind_cols]
            merged = merge_asof_with_tolerance(merged, ind_sub, tolerance=MERGE_TOLERANCE)
            _log(f"  合并化验指标后: {merged.shape}")
        else:
            _log("  [提示] 化验指标列与 X/Y 列名完全重合，跳过合并（可能需要检查列名冲突）")

    # ── X 完全为空的行剔除 ────────────────────────────────────────────────────
    # 只统计 X 特征列（不含 Y 列和指标列）；只过滤所有 X 特征均为 NaN 的行，
    # 这类行代表真实停产/该时间点系统根本未采集到任何数据，属于真实停产空白。
    x_cols_in_merged = [c for c in X.columns if c in merged.columns]
    y_ind_cols = set(y_cols_to_include)
    if ind is not None:
        y_ind_cols.update(ind.columns)

    if x_cols_in_merged:
        mask_all_nan = merged[x_cols_in_merged].isnull().all(axis=1)
        n_removed = mask_all_nan.sum()
        if n_removed > 0:
            _log(f"[过滤] X 完全为空的行（真实停产）: {n_removed} 个 → 已剔除")
            merged = merged[~mask_all_nan]
        _log(f"  过滤后剩余: {len(merged)} 行")
    else:
        _log("[警告] 在合并结果中未找到任何 X 特征列，请检查列名格式。")

    # ── 列排列：Y 列放最后 ────────────────────────────────────────────────────
    non_y_cols = [c for c in merged.columns if c not in (Y_COL_XIN1, Y_COL_XIN2)]
    y_present = [c for c in [Y_COL_XIN1, Y_COL_XIN2] if c in merged.columns]
    merged = merged[non_y_cols + y_present]

    # ── 统计（时序宽表） ──────────────────────────────────────────────────────
    _log(f"\n[完成] 时序宽表（全时间轴）: {merged.shape}")
    _log(f"       时间范围: {merged.index.min()} ~ {merged.index.max()}")
    _log(f"       X 特征列数: {len(x_cols_in_merged)}")
    _log(f"       化验指标列数: {len(ind.columns) if ind is not None else 0}")
    for ycol in y_present:
        _log(f"       {ycol} 有效点数（Y 稀疏）: {merged[ycol].notna().sum()}")
    if x_cols_in_merged:
        _log(f"       X 整体缺失率: {merged[x_cols_in_merged].isnull().mean().mean():.2%}")

    # ── 保存时序宽表（全时间轴，供时序因果发现）─────────────────────────────
    ts_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(ts_output_path, compression="snappy")
    size_mb = ts_output_path.stat().st_size / 1024 / 1024
    _log(f"[保存] 时序宽表 → {ts_output_path}  大小: {size_mb:.2f} MB")

    # ── 派生监督宽表（Y 非 NaN 行，供 DML 等有监督学习；向后兼容）───────────
    modeling_df = merged.dropna(subset=y_present, how="all")
    _log(f"\n[完成] 监督宽表（Y 非 NaN 子集）: {modeling_df.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modeling_df.to_parquet(output_path, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    _log(f"[保存] 监督宽表 → {output_path}  大小: {size_mb:.2f} MB")

    return merged


# ─── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="merge_final.py — 生成 X/Y/指标对齐时序宽表与监督宽表"
    )
    parser.add_argument(
        "--line",
        choices=["xin1", "xin2"],
        default=None,
        help="按产线过滤（xin1=新1#, xin2=新2#）；不指定则保留两条线所有 Y 列",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="自定义监督宽表输出路径（默认为 data/modeling_dataset_final.parquet）",
    )
    parser.add_argument(
        "--ts-output",
        type=str,
        default=None,
        help="自定义时序宽表输出路径（默认为 data/timeseries_dataset_final.parquet）",
    )
    args = parser.parse_args()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    _log("=" * 60)
    _log("merge_final.py — 最终宽表对齐合并脚本 启动")
    _log(f"产线过滤: {args.line or '全部（xin1 + xin2）'}")
    _log(f"锚点策略: X 全时间轴（保留完整 1min 连续时序）")
    _log(f"X 过滤策略: 仅剔除 X 完全为空（全 NaN）的行")
    _log(f"时间对齐容差: ±{MERGE_TOLERANCE}")
    _log("=" * 60)

    out_path = Path(args.output) if args.output else None
    ts_out_path = Path(args.ts_output) if args.ts_output else None
    df = build_modeling_dataset(
        line=args.line, output_path=out_path, ts_output_path=ts_out_path
    )

    # 如果未指定产线，且 SAVE_PER_LINE=True，额外保存两条线的分别数据集
    if args.line is None and SAVE_PER_LINE and not df.empty:
        _log("\n[分产线] 额外保存各产线独立数据集...")
        for line_name, y_col in [("xin1", Y_COL_XIN1), ("xin2", Y_COL_XIN2)]:
            if y_col not in df.columns:
                continue

            # 时序宽表：全时间轴（Y 稀疏）
            ts_line_out = BASE_DATA_DIR / f"timeseries_dataset_{line_name}_final.parquet"
            df.to_parquet(ts_line_out, compression="snappy")
            _log(f"  [保存] {line_name} 时序宽表: {df.shape}  → {ts_line_out}")

            # 监督宽表：仅 Y 非 NaN 行（DML 用）
            modeling_line_df = df.dropna(subset=[y_col])
            if modeling_line_df.empty:
                _log(f"  [跳过] {line_name} 无有效 Y 点，跳过监督宽表")
                continue
            line_out = BASE_DATA_DIR / f"modeling_dataset_{line_name}_final.parquet"
            modeling_line_df.to_parquet(line_out, compression="snappy")
            _log(f"  [保存] {line_name} 监督宽表: {modeling_line_df.shape}  → {line_out}")

    _log("\n全部完成！")
