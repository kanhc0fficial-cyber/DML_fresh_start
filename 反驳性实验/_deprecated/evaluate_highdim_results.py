"""
evaluate_highdim_results.py
============================
高维 DML 理论验证 —— 统一评估与横向对比脚本

读取 baseline_v1/v2 + v3/v4/v5 的蒙特卡洛结果 CSV，
生成跨方法对比报告。

解决的核心评估缺失：
  1. 跨方法横向对比（同一 DAG/种子下 v1~v5 的结果并排比较）
  2. 统计检验（paired Wilcoxon 检验 v3 vs v4 RMSE 差异是否显著）
  3. SE 校准诊断（mean(SE) / std(theta)，理想值 ≈ 1.0）
  4. 正态性检验（Shapiro-Wilk on standardized bias）
  5. √n 一致性对比（各方法收敛斜率横向对比）

用法:
  python evaluate_highdim_results.py
  python evaluate_highdim_results.py --results_dir ./DML理论验证 --output_dir ./评估报告
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ─── 路径 ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "DML理论验证")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "评估报告")

# ─── 常量 ─────────────────────────────────────────────────────────
EPSILON = 1e-12  # 数值稳定性常数，防止除零
SLOPE_TOLERANCE = 0.3  # √n 一致性斜率容差：|slope + 0.5| < 此值视为接近理论值
SHAPIRO_MAX_N = 5000  # Shapiro-Wilk 样本上限（过大样本对微小偏差过于敏感）


# ══════════════════════════════════════════════════════════════════
#  1. 数据加载
# ══════════════════════════════════════════════════════════════════

def _normalize_columns(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """统一列名，兼容不同版本的输出格式"""
    # method 列兼容：ml_method → method
    if "method" not in df.columns and "ml_method" in df.columns:
        df["method"] = df["ml_method"]
    # 如果仍无 method 列，尝试从文件名推断
    if "method" not in df.columns:
        # 从文件名提取方法标识，如 monte_carlo_layered_forest_n1000.csv → forest
        basename = os.path.basename(source_file).replace(".csv", "")
        parts = basename.split("_")
        # 尝试找到 graph_type 之后的 ml_method
        method_guess = basename
        for kw in ("baseline", "v1", "v2", "v3", "v4", "v5", "forest", "gbm"):
            if kw in parts:
                method_guess = kw
                break
        df["method"] = method_guess
    return df


def load_all_results(results_dir: str) -> pd.DataFrame:
    """扫描目录下所有蒙特卡洛 CSV，合并为统一 DataFrame"""
    csv_files = sorted(glob.glob(os.path.join(results_dir, "highdim_*.csv")))
    # 也包含非 highdim 前缀的蒙特卡洛结果
    csv_files += sorted(glob.glob(os.path.join(results_dir, "monte_carlo_*.csv")))
    # 去重
    csv_files = sorted(set(csv_files))

    if not csv_files:
        raise FileNotFoundError(f"在 {results_dir} 中未找到蒙特卡洛结果 CSV 文件")

    dfs = []
    for f in csv_files:
        # 跳过 consistency 文件（单独加载）
        basename = os.path.basename(f).lower()
        if "consistency" in basename:
            continue
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            df = _normalize_columns(df, f)
            # 检查必要列是否存在
            required_cols = {"method", "theta_hat", "bias", "covers_true"}
            if not required_cols.issubset(set(df.columns)):
                print(f"  [跳过] {os.path.basename(f)}: 缺少必要列 "
                      f"{required_cols - set(df.columns)}")
                continue
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
            print(f"  已加载: {os.path.basename(f)}  ({len(df)} 行)")
        except Exception as e:
            print(f"  [跳过] {os.path.basename(f)}: {e}")

    if not dfs:
        raise ValueError("没有成功加载任何蒙特卡洛结果文件")
    return pd.concat(dfs, ignore_index=True)


def load_consistency_results(results_dir: str) -> pd.DataFrame:
    """加载一致性验证结果"""
    csv_files = sorted(glob.glob(os.path.join(results_dir, "*consistency*.csv")))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "n_samples" not in df.columns:
                continue
            # 兼容无 method 列的情况：从文件名推断
            if "method" not in df.columns:
                basename = os.path.basename(f).replace(".csv", "")
                # e.g. consistency_layered_forest → forest
                parts = basename.split("_")
                method_name = parts[-1] if len(parts) > 1 else basename
                df["method"] = method_name
            dfs.append(df)
            print(f"  已加载一致性: {os.path.basename(f)}  ({len(df)} 行)")
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  2. 核心评估指标
# ══════════════════════════════════════════════════════════════════

def compute_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按方法汇总核心指标"""
    methods = df["method"].unique()
    rows = []
    for m in sorted(methods):
        sub = df[df["method"] == m]
        thetas = sub["theta_hat"].values
        biases = sub["bias"].values
        covers = sub["covers_true"].values

        # 安全获取 ate_true 和 se
        ate_true = sub["ate_true"].iloc[0] if "ate_true" in sub.columns else np.nan
        ses = sub["se"].values if "se" in sub.columns else np.full(len(sub), np.nan)

        n_exp = len(sub)
        mean_bias = float(np.mean(biases))
        median_bias = float(np.median(biases))
        rmse = float(np.sqrt(np.mean(biases ** 2)))
        mae = float(np.mean(np.abs(biases)))
        std_theta = float(np.std(thetas))
        mean_se = float(np.nanmean(ses))
        coverage = float(np.mean(covers))

        # SE 校准比：mean(SE) / std(theta)，理想值 ≈ 1.0
        se_calibration = mean_se / (std_theta + EPSILON) if not np.isnan(mean_se) else np.nan

        # 正态性检验（Shapiro-Wilk on standardized bias）
        p_normal = np.nan
        if n_exp >= 20 and std_theta > EPSILON:
            z_scores = biases / std_theta
            try:
                _, p_normal = stats.shapiro(z_scores[:min(n_exp, SHAPIRO_MAX_N)])
            except Exception:
                pass

        rows.append({
            "method": m,
            "n_experiments": n_exp,
            "ate_true": round(float(ate_true), 6) if not np.isnan(ate_true) else None,
            "mean_theta": round(float(np.mean(thetas)), 6),
            "mean_bias": round(mean_bias, 6),
            "median_bias": round(median_bias, 6),
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "std_theta": round(std_theta, 6),
            "mean_se": round(mean_se, 6) if not np.isnan(mean_se) else None,
            "se_calibration": round(se_calibration, 3) if not np.isnan(se_calibration) else None,
            "coverage_95": round(coverage, 4),
            "p_normality": round(float(p_normal), 4) if not np.isnan(p_normal) else None,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
#  3. 成对统计检验
# ══════════════════════════════════════════════════════════════════

def pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    对共享相同 seed 的实验做成对比较。
    检验两方法的 |bias| 差异是否显著（paired Wilcoxon signed-rank test）。
    """
    if "seed" not in df.columns:
        print("  （CSV 中无 seed 列，无法执行成对检验）")
        return pd.DataFrame()

    methods = sorted(df["method"].unique())
    if len(methods) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m_a, m_b = methods[i], methods[j]
            df_a = df[df["method"] == m_a].set_index("seed")
            df_b = df[df["method"] == m_b].set_index("seed")
            common_seeds = df_a.index.intersection(df_b.index)

            if len(common_seeds) < 10:
                continue

            abs_bias_a = np.abs(df_a.loc[common_seeds, "bias"].values)
            abs_bias_b = np.abs(df_b.loc[common_seeds, "bias"].values)

            # diff > 0 means abs_bias_a > abs_bias_b, so B is better (lower absolute bias)
            diff = abs_bias_a - abs_bias_b
            try:
                stat, p_val = stats.wilcoxon(diff, alternative="two-sided")
            except Exception:
                stat, p_val = np.nan, np.nan

            mean_abs_bias_a = float(np.mean(abs_bias_a))
            mean_abs_bias_b = float(np.mean(abs_bias_b))
            winner = m_b if mean_abs_bias_b < mean_abs_bias_a else m_a

            rows.append({
                "method_A": m_a,
                "method_B": m_b,
                "n_paired": len(common_seeds),
                "mean_|bias|_A": round(mean_abs_bias_a, 6),
                "mean_|bias|_B": round(mean_abs_bias_b, 6),
                "winner": winner,
                "wilcoxon_p": round(float(p_val), 6) if not np.isnan(p_val) else None,
                "significant_0.05": bool(p_val < 0.05) if not np.isnan(p_val) else None,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
#  4. 一致性对比
# ══════════════════════════════════════════════════════════════════

def compare_consistency(df_cons: pd.DataFrame) -> pd.DataFrame:
    """对比各方法的 √n 收敛斜率"""
    if df_cons.empty:
        return pd.DataFrame()

    if "rmse" not in df_cons.columns or "n_samples" not in df_cons.columns:
        return pd.DataFrame()

    methods = df_cons["method"].unique()
    rows = []
    for m in sorted(methods):
        sub = df_cons[df_cons["method"] == m].sort_values("n_samples")
        if len(sub) < 2:
            continue
        # 过滤掉 rmse <= 0 的行
        sub = sub[sub["rmse"] > 0]
        if len(sub) < 2:
            continue

        log_n = np.log(sub["n_samples"].values)
        log_rmse = np.log(sub["rmse"].values)
        slope, intercept, r_val, _, _ = stats.linregress(log_n, log_rmse)
        rows.append({
            "method": m,
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "R2": round(r_val ** 2, 4),
            "close_to_minus0.5": abs(slope + 0.5) < SLOPE_TOLERANCE,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
#  5. 报告生成
# ══════════════════════════════════════════════════════════════════

def generate_report(results_dir: str, output_dir: str):
    """生成完整评估报告"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  高维 DML 理论验证 —— 统一评估报告")
    print("=" * 70)

    # ── 加载数据 ──
    print("\n[1/5] 加载蒙特卡洛结果...")
    df_mc = load_all_results(results_dir)

    # ── 方法汇总 ──
    print("\n[2/5] 计算各方法汇总指标...")
    df_summary = compute_method_summary(df_mc)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(
        os.path.join(output_dir, "method_summary.csv"),
        index=False, encoding="utf-8-sig",
    )

    # ── 成对检验 ──
    print("\n[3/5] 成对统计检验...")
    df_pairs = pairwise_comparisons(df_mc)
    if not df_pairs.empty:
        print(df_pairs.to_string(index=False))
        df_pairs.to_csv(
            os.path.join(output_dir, "pairwise_tests.csv"),
            index=False, encoding="utf-8-sig",
        )
    else:
        print("  （无法进行成对检验：共享种子不足或方法不足 2 个）")

    # ── 一致性对比 ──
    print("\n[4/5] 一致性对比...")
    df_cons = load_consistency_results(results_dir)
    df_cons_cmp = compare_consistency(df_cons)
    if not df_cons_cmp.empty:
        print(df_cons_cmp.to_string(index=False))
        df_cons_cmp.to_csv(
            os.path.join(output_dir, "consistency_comparison.csv"),
            index=False, encoding="utf-8-sig",
        )
    else:
        print("  （未找到一致性验证结果）")

    # ── 文本报告 ──
    print("\n[5/5] 生成文本报告...")
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("高维 DML 理论验证 —— 评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"结果目录: {results_dir}\n")
        f.write(f"总实验数: {len(df_mc)}\n")
        f.write(f"方法数量: {df_mc['method'].nunique()}\n\n")

        f.write("一、各方法汇总指标\n")
        f.write("-" * 60 + "\n")
        f.write(df_summary.to_string(index=False) + "\n\n")

        f.write("二、关键发现\n")
        f.write("-" * 60 + "\n")
        if not df_summary.empty:
            best_rmse = df_summary.loc[df_summary["rmse"].idxmin()]
            best_cov = df_summary.loc[
                (df_summary["coverage_95"] - 0.95).abs().idxmin()
            ]
            f.write(f"  - 最低 RMSE: {best_rmse['method']} "
                    f"(RMSE={best_rmse['rmse']})\n")
            f.write(f"  - 最佳 Coverage: {best_cov['method']} "
                    f"(Coverage={best_cov['coverage_95']})\n")

            # SE 校准诊断
            for _, row in df_summary.iterrows():
                cal = row.get("se_calibration")
                if cal is not None and (cal < 0.7 or cal > 1.3):
                    f.write(f"  - ⚠ {row['method']} SE 校准偏离 "
                            f"(ratio={cal}, 理想=1.0)\n")

            # 正态性诊断
            for _, row in df_summary.iterrows():
                p_val = row.get("p_normality")
                if p_val is not None and p_val < 0.05:
                    f.write(f"  - ⚠ {row['method']} 估计分布偏离正态 "
                            f"(Shapiro p={p_val})\n")

        if not df_pairs.empty:
            f.write("\n三、成对检验结果\n")
            f.write("-" * 60 + "\n")
            f.write(df_pairs.to_string(index=False) + "\n")
            # 标注显著结果
            sig_pairs = df_pairs[df_pairs["significant_0.05"].fillna(False)]
            if not sig_pairs.empty:
                f.write("\n  显著差异对:\n")
                for _, row in sig_pairs.iterrows():
                    f.write(f"    {row['method_A']} vs {row['method_B']}: "
                            f"优胜={row['winner']} (p={row['wilcoxon_p']})\n")

        if not df_cons_cmp.empty:
            f.write("\n四、√n 一致性收敛分析\n")
            f.write("-" * 60 + "\n")
            f.write(df_cons_cmp.to_string(index=False) + "\n")
            f.write("\n  理论期望斜率: -0.5 (DML √n-一致性)\n")
            for _, row in df_cons_cmp.iterrows():
                status = "✓" if row["close_to_minus0.5"] else "⚠"
                f.write(f"  {status} {row['method']}: 斜率={row['slope']}, "
                        f"R²={row['R2']}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("报告生成完毕\n")

    # ── 输出汇总 ──
    print(f"\n{'─' * 60}")
    print(f"  报告已保存至: {output_dir}/")
    print(f"    - method_summary.csv       (各方法汇总指标)")
    if not df_pairs.empty:
        print(f"    - pairwise_tests.csv       (成对统计检验)")
    if not df_cons_cmp.empty:
        print(f"    - consistency_comparison.csv (一致性对比)")
    print(f"    - evaluation_report.txt    (文本报告)")
    print(f"{'─' * 60}")


# ══════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="高维 DML 理论验证 —— 统一评估脚本",
    )
    parser.add_argument(
        "--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="蒙特卡洛结果 CSV 所在目录 (默认: ./DML理论验证)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="评估报告输出目录 (默认: ./评估报告)",
    )
    args = parser.parse_args()
    generate_report(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
