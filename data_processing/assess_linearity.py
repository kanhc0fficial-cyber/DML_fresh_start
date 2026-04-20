"""
assess_linearity.py — 线性工作点假设检验脚本
=============================================
判定选矿车间操作变量与结果变量之间的关系是否可以认为处在一个
静态的线性工作点（即在当前数据波动范围内，D→Y 关系近似线性）。

═══════════════════════════════════════════════════════════════════
  背景：
    当前 DML 管线假设操作变量 D 与结果变量 Y 的关系是线性的（PLM 框架），
    理由是选矿车间稳定运行时数据波动小，可认为处在一个工作点，
    在此小范围内关系近似线性。
    本脚本通过多种统计检验来验证这一假设是否成立。

  检验方法：
    1. Ramsey RESET 检验：在 Y ~ D 的线性模型上追加 Ŷ² 和 Ŷ³，
       检验非线性项是否显著（H₀: 线性足够）
    2. 多项式拟合对比：比较 Y ~ D（线性）与 Y ~ D + D² + D³ 的调整 R²，
       如果高阶项带来显著改善，说明非线性显著
    3. 局部线性检验（分段斜率一致性）：将 D 值按分位数分为若干段，
       在每段内拟合线性模型，如果斜率差异大（CV > 阈值），说明全局非线性
    4. 残差自相关检验（Durbin-Watson）：检验 Y ~ D 线性回归残差的自相关性
    5. 滞后非线性检验：考虑工业数据滞后效应，检验 Y(t) ~ D(t-lag) 的关系

  输出：
    - data_processing/线性工作点检验结果/linearity_assessment_xin2.csv
        每个操作变量的各项检验结果
    - data_processing/线性工作点检验结果/linearity_summary_xin2.md
        文字汇总报告

  参考：
    数据加载逻辑参照 data_processing/merge_final.py 的接口设计，
    变量筛选逻辑参照 反驳性实验/run_refutation_xin2_baseline_v1.py 的 build_xin2_data()。

═══════════════════════════════════════════════════════════════════

用法：
  python data_processing/assess_linearity.py
  python data_processing/assess_linearity.py --line xin2
  python data_processing/assess_linearity.py --line xin1
  python data_processing/assess_linearity.py --alpha 0.01
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
#  路径配置（与 data_processing/ 中其他脚本一致）
# ═══════════════════════════════════════════════════════════════════
BASE_DIR  = Path(__file__).resolve().parent          # data_processing/
REPO_ROOT = BASE_DIR.parent                          # 仓库根目录
DATA_DIR  = REPO_ROOT / "data"

# 输入文件
MODELING_DATASET_XIN1 = DATA_DIR / "modeling_dataset_xin1_final.parquet"
MODELING_DATASET_XIN2 = DATA_DIR / "modeling_dataset_xin2_final.parquet"
X_PARQUET             = DATA_DIR / "X_features_final.parquet"
Y_PARQUET             = DATA_DIR / "y_target_final.parquet"

# 操作性分类表
DEFAULT_OPERABILITY_CSV = REPO_ROOT / "数据预处理" / \
    "数据与处理结果-分阶段-去共线性后" / \
    "non_collinear_representative_vars_operability.csv"

# 输出目录
OUTPUT_DIR = BASE_DIR / "线性工作点检验结果"


# ═══════════════════════════════════════════════════════════════════
#  超参 / 阈值
# ═══════════════════════════════════════════════════════════════════
ALPHA_RESET       = 0.05   # RESET 检验显著性水平
ALPHA_POLY        = 0.05   # 多项式 F 检验显著性水平
N_SEGMENTS        = 4      # 分段线性检验中的分段数
SLOPE_CV_THRESH   = 0.50   # 分段斜率 CV 阈值（超过则认为非线性）
DW_LOWER          = 1.5    # Durbin-Watson 下限（< 此值认为正自相关显著）
DW_UPPER          = 2.5    # Durbin-Watson 上限（> 此值认为负自相关显著）
MAX_LAG           = 15     # 滞后扫描最大步数
MIN_SAMPLES       = 30     # 每段/每次回归最少样本量


# ═══════════════════════════════════════════════════════════════════
#  数据加载（参考 merge_final.py + baseline_v1 的 build_xin2_data）
# ═══════════════════════════════════════════════════════════════════
def load_modeling_data(line: str = "xin2"):
    """
    加载指定产线的建模数据集。

    参数：
      line: "xin1" 或 "xin2"

    返回：
      (df, operable_set, observable_set)
    """
    y_col_name = f"y_fx_{line}"
    modeling_path = MODELING_DATASET_XIN1 if line == "xin1" else MODELING_DATASET_XIN2

    # ── 加载操作性分类 ──────────────────────────────────────────────
    if not DEFAULT_OPERABILITY_CSV.exists():
        raise FileNotFoundError(
            f"操作性分类文件不存在：{DEFAULT_OPERABILITY_CSV}\n"
            f"请先运行 数据预处理/classify_operability.py 生成该文件。"
        )
    op_df = pd.read_csv(DEFAULT_OPERABILITY_CSV, encoding="utf-8-sig")
    op_df["Group"] = op_df["Group"].str.strip().str.upper()

    if line == "xin2":
        line_df = op_df[op_df["Group"].isin(["B", "C"])].copy()
    else:
        line_df = op_df[op_df["Group"].isin(["A", "C"])].copy()

    operable_set = set(
        line_df[line_df["Operability"].str.strip() == "operable"]["Variable_Name"].str.strip()
    )
    observable_set = set(
        line_df[line_df["Operability"].str.strip() == "observable"]["Variable_Name"].str.strip()
    )
    print(f"[数据准备] 产线 {line}: operable={len(operable_set)}, observable={len(observable_set)}")

    # ── 加载建模宽表 ───────────────────────────────────────────────
    if modeling_path.exists():
        print(f"[数据准备] 读取已对齐建模宽表：{modeling_path}")
        df = pd.read_parquet(modeling_path)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "time"
        if y_col_name in df.columns:
            df = df.rename(columns={y_col_name: "Y_grade"})
        elif "Y_grade" not in df.columns:
            raise KeyError(f"建模宽表中未找到 '{y_col_name}' 或 'Y_grade' 列")
        # 删除其他 Y 列
        other_y = [c for c in df.columns if c.startswith("y_fx_") and c != "Y_grade"]
        if other_y:
            df = df.drop(columns=other_y)
        df = df.dropna(subset=["Y_grade"])

    elif X_PARQUET.exists() and Y_PARQUET.exists():
        print(f"[数据准备] 未找到已对齐宽表，回退到分别读取 X + Y")
        X = pd.read_parquet(X_PARQUET)
        X.index = pd.to_datetime(X.index).tz_localize(None)
        X.index.name = "time"
        X = X.sort_index()

        y = pd.read_parquet(Y_PARQUET)
        y.index = pd.to_datetime(y.index).tz_localize(None)
        y.index.name = "time"
        y = y.sort_index()

        if y_col_name not in y.columns:
            raise KeyError(f"Y 文件中未找到 '{y_col_name}' 列")

        y_sub = y[[y_col_name]].dropna()
        y_reset = y_sub.reset_index().sort_values("time")
        X_reset = X.reset_index().rename(columns={"time": "_time_x"}).sort_values("_time_x")
        merged = pd.merge_asof(
            y_reset, X_reset,
            left_on="time", right_on="_time_x",
            direction="nearest", tolerance=pd.Timedelta("1min"),
        )
        merged = merged.drop(columns=["_time_x"])
        merged = merged.set_index("time")
        merged = merged.rename(columns={y_col_name: "Y_grade"})
        merged = merged.dropna(subset=["Y_grade"])
        df = merged
    else:
        raise FileNotFoundError(
            f"未找到数据文件。请先运行 data_processing/ 下的预处理脚本。\n"
            f"  已对齐宽表（推荐）: {modeling_path}\n"
            f"  或分别: {X_PARQUET} + {Y_PARQUET}"
        )

    # ── 过滤 ────────────────────────────────────────────────────────
    df = df.loc[:, df.std() > 1e-4]
    all_known_vars = operable_set | observable_set
    valid_cols = [c for c in df.columns if c in all_known_vars or c == "Y_grade"]
    df_filtered = df[valid_cols]

    cols_in_df = set(df_filtered.columns) - {"Y_grade"}
    operable_in_df = operable_set & cols_in_df
    observable_in_df = observable_set & cols_in_df
    print(f"[数据准备] 最终 DataFrame：{df_filtered.shape}，"
          f"operable={len(operable_in_df)}，observable={len(observable_in_df)}")
    return df_filtered, operable_in_df, observable_in_df


# ═══════════════════════════════════════════════════════════════════
#  检验 1：Ramsey RESET 检验
# ═══════════════════════════════════════════════════════════════════
def ramsey_reset_test(D: np.ndarray, Y: np.ndarray, alpha: float = ALPHA_RESET):
    """
    Ramsey RESET 检验：在线性模型 Y = a + b*D 上追加 Ŷ² 和 Ŷ³。
    如果非线性项联合显著（F 检验 p < alpha），拒绝线性假设。

    返回：(f_stat, p_value, reject_linear)
    """
    n = len(D)
    if n < MIN_SAMPLES:
        return np.nan, np.nan, None

    # 拟合线性模型 Y = a + b*D
    X_lin = np.column_stack([np.ones(n), D])
    beta_lin = np.linalg.lstsq(X_lin, Y, rcond=None)[0]
    Y_hat = X_lin @ beta_lin
    RSS_restricted = np.sum((Y - Y_hat) ** 2)

    # 增广模型：Y = a + b*D + c*Ŷ² + d*Ŷ³
    X_aug = np.column_stack([X_lin, Y_hat ** 2, Y_hat ** 3])
    beta_aug = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
    Y_hat_aug = X_aug @ beta_aug
    RSS_unrestricted = np.sum((Y - Y_hat_aug) ** 2)

    # F 统计量
    df1 = 2  # 新增参数数
    df2 = n - X_aug.shape[1]
    if df2 <= 0 or RSS_unrestricted < 1e-12:
        return np.nan, np.nan, None

    f_stat = ((RSS_restricted - RSS_unrestricted) / df1) / (RSS_unrestricted / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    reject_linear = bool(p_value < alpha)

    return float(f_stat), float(p_value), reject_linear


# ═══════════════════════════════════════════════════════════════════
#  检验 2：多项式拟合对比（线性 vs 三次）
# ═══════════════════════════════════════════════════════════════════
def polynomial_comparison(D: np.ndarray, Y: np.ndarray, alpha: float = ALPHA_POLY):
    """
    比较线性模型和三次多项式模型的拟合优度。
    用 F 检验判断高阶项是否显著。

    返回：(r2_linear, r2_cubic, f_stat, p_value, reject_linear)
    """
    n = len(D)
    if n < MIN_SAMPLES:
        return np.nan, np.nan, np.nan, np.nan, None

    # 线性拟合
    X_lin = np.column_stack([np.ones(n), D])
    beta_lin = np.linalg.lstsq(X_lin, Y, rcond=None)[0]
    Y_hat_lin = X_lin @ beta_lin
    SS_res_lin = np.sum((Y - Y_hat_lin) ** 2)
    SS_tot = np.sum((Y - Y.mean()) ** 2)
    if SS_tot < 1e-12:
        return np.nan, np.nan, np.nan, np.nan, None
    r2_lin = 1 - SS_res_lin / SS_tot

    # 三次多项式拟合
    X_cub = np.column_stack([np.ones(n), D, D**2, D**3])
    beta_cub = np.linalg.lstsq(X_cub, Y, rcond=None)[0]
    Y_hat_cub = X_cub @ beta_cub
    SS_res_cub = np.sum((Y - Y_hat_cub) ** 2)
    r2_cub = 1 - SS_res_cub / SS_tot

    # F 检验：线性 vs 三次
    df1 = 2  # 新增参数数（D² 和 D³）
    df2 = n - 4  # 三次模型参数数为 4
    if df2 <= 0 or SS_res_cub < 1e-12:
        return float(r2_lin), float(r2_cub), np.nan, np.nan, None

    f_stat = ((SS_res_lin - SS_res_cub) / df1) / (SS_res_cub / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    reject_linear = bool(p_value < alpha)

    return float(r2_lin), float(r2_cub), float(f_stat), float(p_value), reject_linear


# ═══════════════════════════════════════════════════════════════════
#  检验 3：分段线性检验（局部斜率一致性）
# ═══════════════════════════════════════════════════════════════════
def piecewise_slope_test(D: np.ndarray, Y: np.ndarray, n_segments: int = N_SEGMENTS):
    """
    将 D 按分位数分为 n_segments 段，每段拟合 Y = a + b*D，
    检验各段斜率的一致性（CV）。

    返回：(slopes_array, slope_cv, reject_linear)
    """
    n = len(D)
    if n < MIN_SAMPLES * n_segments:
        return None, np.nan, None

    # 按 D 的分位数分段
    quantiles = np.linspace(0, 100, n_segments + 1)
    boundaries = np.percentile(D, quantiles)

    slopes = []
    for i in range(n_segments):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == n_segments - 1:
            mask = (D >= lo) & (D <= hi)
        else:
            mask = (D >= lo) & (D < hi)
        D_seg, Y_seg = D[mask], Y[mask]
        if len(D_seg) < MIN_SAMPLES:
            continue
        # 线性拟合
        X_seg = np.column_stack([np.ones(len(D_seg)), D_seg])
        beta = np.linalg.lstsq(X_seg, Y_seg, rcond=None)[0]
        slopes.append(beta[1])

    if len(slopes) < 2:
        return None, np.nan, None

    slopes_arr = np.array(slopes)
    mean_slope = np.mean(np.abs(slopes_arr))
    if mean_slope < 1e-8:
        return slopes_arr, np.nan, None

    slope_cv = float(np.std(slopes_arr) / mean_slope)
    reject_linear = bool(slope_cv > SLOPE_CV_THRESH)

    return slopes_arr, slope_cv, reject_linear


# ═══════════════════════════════════════════════════════════════════
#  检验 4：Durbin-Watson 残差自相关检验
# ═══════════════════════════════════════════════════════════════════
def durbin_watson_test(D: np.ndarray, Y: np.ndarray):
    """
    在 Y = a + b*D 的线性回归残差上计算 Durbin-Watson 统计量。
    DW ∈ [0, 4]：DW ≈ 2 无自相关，DW < 2 正自相关，DW > 2 负自相关。

    若 DW 偏离 2 太多，说明线性模型残差有系统性结构（可能遗漏非线性项）。

    返回：(dw_stat, has_autocorrelation)
    """
    n = len(D)
    if n < MIN_SAMPLES:
        return np.nan, None

    X_lin = np.column_stack([np.ones(n), D])
    beta = np.linalg.lstsq(X_lin, Y, rcond=None)[0]
    residuals = Y - X_lin @ beta

    # Durbin-Watson
    diff_resid = np.diff(residuals)
    dw = float(np.sum(diff_resid ** 2) / (np.sum(residuals ** 2) + 1e-12))

    has_autocorrelation = bool(dw < DW_LOWER or dw > DW_UPPER)
    return dw, has_autocorrelation


# ═══════════════════════════════════════════════════════════════════
#  检验 5：最优滞后下的非线性检验
# ═══════════════════════════════════════════════════════════════════
def lagged_linearity_test(D: np.ndarray, Y: np.ndarray, max_lag: int = MAX_LAG):
    """
    工业数据的 D→Y 关系通常有滞后。找到最优滞后后，
    在该滞后下重复 RESET 检验。

    返回：(best_lag, best_corr, reset_f, reset_p, reject_linear)
    """
    n = len(D)
    if n < MIN_SAMPLES + max_lag:
        return 0, np.nan, np.nan, np.nan, None

    # 寻找最优滞后
    best_r, best_lag = 0.0, 0
    for lag in range(1, min(max_lag + 1, n // 2)):
        r = abs(np.corrcoef(D[:-lag], Y[lag:])[0, 1])
        if r > best_r:
            best_r, best_lag = r, lag

    if best_lag == 0:
        best_lag = 1

    # 在最优滞后下做 RESET 检验
    D_lagged = D[:-best_lag]
    Y_lagged = Y[best_lag:]

    f_stat, p_val, reject = ramsey_reset_test(D_lagged, Y_lagged)
    return best_lag, float(best_r), f_stat, p_val, reject


# ═══════════════════════════════════════════════════════════════════
#  综合评估：对单个操作变量做全部检验
# ═══════════════════════════════════════════════════════════════════
def assess_single_op(op: str, df: pd.DataFrame):
    """
    对单个操作变量 op 与 Y_grade 的关系进行全套线性工作点检验。

    返回：结果字典
    """
    D = df[op].values.astype(np.float64)
    Y = df["Y_grade"].values.astype(np.float64)

    # 去除 NaN
    mask = np.isfinite(D) & np.isfinite(Y)
    D, Y = D[mask], Y[mask]

    if len(D) < MIN_SAMPLES:
        return {"操作变量": op, "有效样本数": len(D), "可评估": False}

    # 检验 1：RESET
    reset_f, reset_p, reset_reject = ramsey_reset_test(D, Y)

    # 检验 2：多项式对比
    r2_lin, r2_cub, poly_f, poly_p, poly_reject = polynomial_comparison(D, Y)

    # 检验 3：分段斜率
    _, slope_cv, slope_reject = piecewise_slope_test(D, Y)

    # 检验 4：DW
    dw_stat, dw_reject = durbin_watson_test(D, Y)

    # 检验 5：滞后非线性
    best_lag, best_corr, lag_reset_f, lag_reset_p, lag_reject = lagged_linearity_test(D, Y)

    # 综合判定：多数检验拒绝线性 → 非线性
    tests = [reset_reject, poly_reject, slope_reject, dw_reject, lag_reject]
    valid_tests = [t for t in tests if t is not None]
    n_reject = sum(valid_tests) if valid_tests else 0
    n_valid = len(valid_tests)
    linearity_score = 1.0 - (n_reject / max(1, n_valid))  # 1.0 = 完全线性, 0.0 = 完全非线性

    return {
        "操作变量": op,
        "有效样本数": len(D),
        "可评估": True,
        # RESET 检验
        "RESET_F": round(reset_f, 4) if not np.isnan(reset_f) else None,
        "RESET_p": round(reset_p, 6) if not np.isnan(reset_p) else None,
        "RESET_拒绝线性": reset_reject,
        # 多项式对比
        "R2_线性": round(r2_lin, 4) if not np.isnan(r2_lin) else None,
        "R2_三次": round(r2_cub, 4) if not np.isnan(r2_cub) else None,
        "多项式_F": round(poly_f, 4) if not np.isnan(poly_f) else None,
        "多项式_p": round(poly_p, 6) if not np.isnan(poly_p) else None,
        "多项式_拒绝线性": poly_reject,
        # 分段斜率
        "斜率_CV": round(slope_cv, 4) if not np.isnan(slope_cv) else None,
        "分段_拒绝线性": slope_reject,
        # Durbin-Watson
        "DW统计量": round(dw_stat, 4) if not np.isnan(dw_stat) else None,
        "DW_自相关显著": dw_reject,
        # 滞后非线性
        "最优滞后": best_lag,
        "滞后相关系数": round(best_corr, 4) if not np.isnan(best_corr) else None,
        "滞后RESET_F": round(lag_reset_f, 4) if not np.isnan(lag_reset_f) else None,
        "滞后RESET_p": round(lag_reset_p, 6) if not np.isnan(lag_reset_p) else None,
        "滞后_拒绝线性": lag_reject,
        # 综合
        "拒绝线性检验数": n_reject,
        "有效检验数": n_valid,
        "线性度得分": round(linearity_score, 3),
    }


# ═══════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════
def run_assessment(line: str = "xin2", alpha: float = ALPHA_RESET):
    """对指定产线的所有操作变量运行线性工作点检验。"""
    global ALPHA_RESET, ALPHA_POLY
    ALPHA_RESET = alpha
    ALPHA_POLY = alpha

    print("=" * 70)
    print(f" 线性工作点假设检验  |  产线: {line}  |  显著性水平 α = {alpha}")
    print("=" * 70)

    df, operable_in_df, observable_in_df = load_modeling_data(line)
    ops = sorted(operable_in_df & set(df.columns))
    print(f"\n待检验操作变量：{len(ops)} 个\n")

    # 逐操作变量检验
    results = []
    for op in ops:
        if df[op].std() < 1e-4:
            continue
        result = assess_single_op(op, df)
        results.append(result)
        if result["可评估"]:
            score = result["线性度得分"]
            flag = "✓ 线性" if score >= 0.6 else "⚠ 非线性"
            print(f"  {op:<30s}  线性度={score:.3f}  "
                  f"拒绝={result['拒绝线性检验数']}/{result['有效检验数']}  [{flag}]")

    # ── 保存结果 ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / f"linearity_assessment_{line}.csv"
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存：{csv_path}")

    # ── 生成汇总报告 ─────────────────────────────────────────────────
    evaluable = df_results[df_results["可评估"] == True]
    if evaluable.empty:
        print("[警告] 没有可评估的操作变量，请检查数据。")
        return df_results

    n_total = len(evaluable)
    n_linear = int((evaluable["线性度得分"] >= 0.6).sum())
    n_nonlinear = n_total - n_linear
    avg_score = evaluable["线性度得分"].mean()

    # 各检验的拒绝率
    reset_reject_rate = evaluable["RESET_拒绝线性"].mean() if "RESET_拒绝线性" in evaluable else 0
    poly_reject_rate = evaluable["多项式_拒绝线性"].mean() if "多项式_拒绝线性" in evaluable else 0
    slope_reject_rate = evaluable["分段_拒绝线性"].mean() if "分段_拒绝线性" in evaluable else 0
    dw_reject_rate = evaluable["DW_自相关显著"].mean() if "DW_自相关显著" in evaluable else 0
    lag_reject_rate = evaluable["滞后_拒绝线性"].mean() if "滞后_拒绝线性" in evaluable else 0

    report_lines = [
        f"# 线性工作点假设检验报告 — 产线 {line}",
        f"",
        f"## 总体结论",
        f"",
    ]

    if avg_score >= 0.7:
        report_lines.append(
            f"**✓ 总体支持线性工作点假设。**"
        )
        report_lines.append(
            f"在 α={alpha} 的显著性水平下，多数操作变量与 Y 的关系可以认为近似线性。"
        )
        report_lines.append(
            f"当前数据支持使用线性 PLM 框架进行双重机器学习因果效应估计。"
        )
    elif avg_score >= 0.4:
        report_lines.append(
            f"**⚠ 线性工作点假设部分成立，部分操作变量表现出显著非线性。**"
        )
        report_lines.append(
            f"建议对非线性显著的操作变量考虑使用非线性 DML 方法（如 R-Learner 或非参数 nuisance 估计器）。"
        )
    else:
        report_lines.append(
            f"**✗ 线性工作点假设不成立。**"
        )
        report_lines.append(
            f"多数操作变量与 Y 的关系表现出显著非线性，不宜假设处在线性工作点。"
        )
        report_lines.append(
            f"强烈建议使用非线性双重机器学习方法进行因果效应估计。"
        )

    report_lines += [
        f"",
        f"## 统计摘要",
        f"",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 可评估操作变量数 | {n_total} |",
        f"| 判定为线性（得分 ≥ 0.6） | {n_linear} ({n_linear/n_total:.0%}) |",
        f"| 判定为非线性（得分 < 0.6） | {n_nonlinear} ({n_nonlinear/n_total:.0%}) |",
        f"| 平均线性度得分 | {avg_score:.3f} |",
        f"| 显著性水平 α | {alpha} |",
        f"",
        f"## 各检验拒绝率",
        f"",
        f"| 检验 | 拒绝率 | 说明 |",
        f"|------|--------|------|",
        f"| Ramsey RESET | {reset_reject_rate:.1%} | 非线性项联合显著性（Ŷ² + Ŷ³） |",
        f"| 多项式 F 检验 | {poly_reject_rate:.1%} | 三次多项式相比线性显著改善 |",
        f"| 分段斜率 CV | {slope_reject_rate:.1%} | 各段斜率不一致（CV > {SLOPE_CV_THRESH}） |",
        f"| Durbin-Watson | {dw_reject_rate:.1%} | 残差自相关显著（系统性模式） |",
        f"| 滞后 RESET | {lag_reject_rate:.1%} | 考虑最优滞后后仍非线性 |",
        f"",
        f"## 解读指引",
        f"",
        f"- **线性度得分 ≥ 0.8**：强线性，PLM 框架可靠",
        f"- **线性度得分 0.6~0.8**：弱线性，PLM 可用但精度有损",
        f"- **线性度得分 0.4~0.6**：边界情况，建议辅以非线性方法对照",
        f"- **线性度得分 < 0.4**：显著非线性，应使用非线性 DML",
        f"",
        f"## 方法说明",
        f"",
        f"本脚本对每个操作变量 D 与结果变量 Y (精矿品位) 的关系进行五项独立的线性假设检验：",
        f"",
        f"1. **Ramsey RESET 检验**：在 Y = a + bD 上追加 Ŷ² 和 Ŷ³，F 检验非线性项联合显著性",
        f"2. **多项式 F 检验**：比较一次 vs 三次多项式拟合（嵌套模型 F 检验）",
        f"3. **分段线性检验**：将 D 按分位数分 {N_SEGMENTS} 段，检验各段回归斜率的一致性",
        f"4. **Durbin-Watson 检验**：检验线性回归残差的一阶自相关（系统性结构 → 模型遗漏）",
        f"5. **滞后 RESET 检验**：在最优滞后 D(t-k) → Y(t) 下重做非线性检验（考虑工业延迟）",
        f"",
        f"线性度得分 = 1 - (拒绝线性的检验数 / 有效检验总数)。",
    ]

    md_path = OUTPUT_DIR / f"linearity_summary_{line}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"汇总报告已保存：{md_path}")

    # ── 终端打印核心结论 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" 核心结论")
    print("=" * 70)
    print(f"  可评估操作变量：{n_total} 个")
    print(f"  线性（得分 ≥ 0.6）：{n_linear} 个 ({n_linear/n_total:.0%})")
    print(f"  非线性（得分 < 0.6）：{n_nonlinear} 个 ({n_nonlinear/n_total:.0%})")
    print(f"  平均线性度得分：{avg_score:.3f}")
    if avg_score >= 0.7:
        print("  → 支持线性工作点假设，可继续使用线性 PLM 框架")
    elif avg_score >= 0.4:
        print("  → 线性假设部分成立，建议辅以非线性方法对照")
    else:
        print("  → 线性假设不成立，建议切换到非线性 DML 方法")
    print("=" * 70)

    return df_results


# ═══════════════════════════════════════════════════════════════════
#  命令行接口
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="线性工作点假设检验：判定操作变量与结果变量的关系是否近似线性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例：
  # 检验新2#产线（默认）
  python data_processing/assess_linearity.py

  # 检验新1#产线
  python data_processing/assess_linearity.py --line xin1

  # 使用更严格的显著性水平
  python data_processing/assess_linearity.py --alpha 0.01

输出：
  data_processing/线性工作点检验结果/linearity_assessment_{line}.csv
  data_processing/线性工作点检验结果/linearity_summary_{line}.md
        """,
    )
    p.add_argument("--line", type=str, default="xin2", choices=["xin1", "xin2"],
                   help="产线选择（默认 xin2）")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="显著性水平（默认 0.05）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_assessment(line=args.line, alpha=args.alpha)
