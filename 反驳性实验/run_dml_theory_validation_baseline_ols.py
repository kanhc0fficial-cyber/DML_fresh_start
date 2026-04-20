"""
run_dml_theory_validation_baseline_ols.py
=========================================
DML 理论验证 —— 朴素回归基线方法（无 DML 对照组）

═══════════════════════════════════════════════════════════════════
  本脚本作为「不使用 DML」的对照基线，展示朴素回归方法在因果推断中的
  缺陷（遗漏变量偏差、正则化偏差），从而突出 DML 交叉拟合的必要性。

  3 种朴素基线方法：

    1. Naive Regression (朴素回归)
       —— Y ~ D，完全不控制混杂变量
       —— 暴露遗漏变量偏差 (Omitted Variable Bias)

    2. Partialling-Out without Cross-Fitting (全样本残差化)
       —— 与 DML 相同的残差化思路，但不做样本分割
       —— 在全样本上训练 g(X) 和 m(X)，再在同一样本上预测残差
       —— 暴露过拟合偏差 / 正则化偏差 (Overfitting / Regularization Bias)

    3. OLS with Linear Controls (线性控制变量法)
       —— Y ~ D + X，将混杂变量作为线性控制直接加入回归
       —— 参数化假设正确时无偏，但无法捕捉非线性混杂

  本脚本的目的是与 DML 基线 (run_dml_theory_validation_baseline_dml.py)
  形成对照，展示 DML 交叉拟合在去偏方面的优势。
═══════════════════════════════════════════════════════════════════

用法：
  # 快速验证（少量实验）
  python run_dml_theory_validation_baseline_ols.py --mode quick

  # 完整蒙特卡洛验证（默认 200 次实验，3 种方法）
  python run_dml_theory_validation_baseline_ols.py --mode full

  # 方法对比（含 DML 对照）
  python run_dml_theory_validation_baseline_ols.py --mode compare

  # 全部实验
  python run_dml_theory_validation_baseline_ols.py --mode all
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ─── 路径配置 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import dml_validation_common as dvc

# 输出目录
OUT_DIR = os.path.join(BASE_DIR, "DML理论验证")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  朴素基线估计器定义
# ═══════════════════════════════════════════════════════════════════

def naive_regression_estimator(Y, D, X_ctrl, seed):
    """
    朴素回归估计器：Y ~ D（不控制任何混杂变量）。

    缺陷：存在遗漏变量偏差 (OVB)，当混杂变量同时影响 D 和 Y 时，
    θ̂_naive ≠ θ_true。

    标准误计算：
      Y = α + θ·D + ε
      SE(θ̂) = sqrt(σ̂² / Σ(D_i - D̄)²)
      其中 σ̂² = Σε̂² / (n - 2)
    """
    from sklearn.linear_model import LinearRegression

    n = len(Y)
    D_design = D.reshape(-1, 1)

    model = LinearRegression()
    model.fit(D_design, Y)

    theta_hat = float(model.coef_[0])

    # 计算 OLS 标准误
    Y_pred = model.predict(D_design)
    residuals = Y - Y_pred
    p = 2  # 截距 + D 系数
    sigma2 = float(np.sum(residuals ** 2)) / (n - p)

    # SE = sqrt(sigma^2 * (X'X)^{-1}[1,1])
    # 对于含截距的设计矩阵 [1, D]
    X_design = np.column_stack([np.ones(n), D])
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))

    ci_lower = theta_hat - 1.96 * se
    ci_upper = theta_hat + 1.96 * se

    return theta_hat, se, ci_lower, ci_upper


def partialling_out_no_crossfit_estimator(Y, D, X_ctrl, seed):
    """
    全样本残差化估计器（Partialling-out without cross-fitting）。

    思路与 DML 相同（Frisch-Waugh-Lovell 定理）：
      1. 用 ML 模型估计 E[Y|X] = g(X)，计算 Ỹ = Y - ĝ(X)
      2. 用 ML 模型估计 E[D|X] = m(X)，计算 D̃ = D - m̂(X)
      3. θ̂ = Σ(D̃·Ỹ) / Σ(D̃²)

    关键区别：在全样本上训练和预测（不做交叉拟合/样本分割）。
    这会导致过拟合偏差 —— ĝ(X) 和 m̂(X) 在训练样本上过度拟合，
    残差被人为压缩，θ̂ 产生正则化偏差。

    使用 Random Forest 作为 nuisance 估计器（与 DML 基线一致）。
    """
    # 全样本训练（无交叉拟合）
    model_Y = dvc.fit_ml_model(X_ctrl, Y, method="forest", seed=seed)
    model_D = dvc.fit_ml_model(X_ctrl, D, method="forest", seed=seed)

    # 在相同样本上预测（过拟合源头）
    Y_hat = model_Y.predict(X_ctrl)
    D_hat = model_D.predict(X_ctrl)

    res_Y = Y - Y_hat
    res_D = D - D_hat

    # Wald 估计量
    denom = np.sum(res_D ** 2) + 1e-12
    theta_hat = float(np.sum(res_D * res_Y) / denom)

    # Neyman 式标准误（与 DML 一致的公式）
    n = len(Y)
    psi = res_D * (res_Y - theta_hat * res_D)
    J = np.mean(res_D ** 2)
    var_neyman = float(np.mean(psi ** 2) / (n * J ** 2 + 1e-12))
    se = float(np.sqrt(max(var_neyman, 1e-12)))

    ci_lower = theta_hat - 1.96 * se
    ci_upper = theta_hat + 1.96 * se

    return theta_hat, se, ci_lower, ci_upper


def ols_with_controls_estimator(Y, D, X_ctrl, seed):
    """
    OLS 线性控制变量法：Y ~ D + X_ctrl。

    将混杂变量作为线性协变量直接纳入回归方程。
    当 DGP 中混杂关系为线性时，此方法无偏；
    当混杂关系为非线性时，产生模型错配偏差。

    标准误计算：
      Y = X_full @ β + ε，X_full = [D, X_ctrl]（含截距）
      SE(β̂_D) = sqrt(σ̂² * (X'X)^{-1}[1,1])
      其中 σ̂² = Σε̂² / (n - p)
    """
    from sklearn.linear_model import LinearRegression

    n = len(Y)

    # 设计矩阵：[截距, D, X_ctrl]
    X_design = np.column_stack([np.ones(n), D, X_ctrl])
    p = X_design.shape[1]  # 参数数量

    model = LinearRegression(fit_intercept=False)
    model.fit(X_design, Y)

    # D 的系数是第 2 个（索引 1，因为第 0 个是截距）
    theta_hat = float(model.coef_[1])

    # 计算 OLS 标准误
    Y_pred = model.predict(X_design)
    residuals = Y - Y_pred
    sigma2 = float(np.sum(residuals ** 2)) / (n - p)

    # (X'X)^{-1} 对角线元素
    XtX_inv = np.linalg.inv(X_design.T @ X_design + 1e-10 * np.eye(p))
    se = float(np.sqrt(max(sigma2 * XtX_inv[1, 1], 1e-12)))

    ci_lower = theta_hat - 1.96 * se
    ci_upper = theta_hat + 1.96 * se

    return theta_hat, se, ci_lower, ci_upper


# 方法配置字典
BASELINE_OLS_METHODS = {
    "naive_regression": {
        "name": "Naive Regression",
        "tag": "NaiveOLS",
        "fn": naive_regression_estimator,
        "description": "朴素回归 Y~D —— 不控制混杂，暴露遗漏变量偏差",
    },
    "partialling_out_no_cf": {
        "name": "Partialling-Out (No CF)",
        "tag": "PartialNoCF",
        "fn": partialling_out_no_crossfit_estimator,
        "description": "全样本残差化 —— 无交叉拟合，暴露过拟合/正则化偏差",
    },
    "ols_with_controls": {
        "name": "OLS with Controls",
        "tag": "OLS_Ctrl",
        "fn": ols_with_controls_estimator,
        "description": "线性控制变量法 Y~D+X —— 参数化线性假设",
    },
}


# ═══════════════════════════════════════════════════════════════════
#  实验模式实现
# ═══════════════════════════════════════════════════════════════════

def run_quick(args):
    """快速验证模式：使用 Naive OLS 进行小规模蒙特卡洛"""
    print("\n" + "█" * 70)
    print("  [快速模式] 朴素回归基线验证 (Naive Regression)")
    print("█" * 70)

    dag_info = dvc.setup_fixed_dag(
        n_nodes=20,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
        enforce_linear_ty=True,
    )
    ate_true = dvc.compute_ate_for_dag(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    n_exp = min(args.n_experiments, 30)

    df, summary = dvc.run_monte_carlo(
        estimator_fn=naive_regression_estimator,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=n_exp,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="baseline_ols_quick",
        method_name="Naive Regression (Quick)",
    )

    if not df.empty:
        dvc.save_results(df, summary, "baseline_ols_quick_NaiveOLS")
    return df, summary


def run_full(args):
    """完整蒙特卡洛验证：对每种朴素方法分别运行"""
    print("\n" + "█" * 70)
    print("  [完整模式] 朴素回归基线验证 (3种非DML方法)")
    print("█" * 70)

    dag_info = dvc.setup_fixed_dag(
        n_nodes=20,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
        enforce_linear_ty=True,
    )
    ate_true = dvc.compute_ate_for_dag(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    all_dfs = []
    all_summaries = []

    for method_key, config in BASELINE_OLS_METHODS.items():
        print(f"\n{'─' * 60}")
        print(f"  方法: {config['name']} ({config['tag']})")
        print(f"  说明: {config['description']}")
        print(f"{'─' * 60}")

        df, summary = dvc.run_monte_carlo(
            estimator_fn=config["fn"],
            dag_info=dag_info,
            ate_true=ate_true,
            n_experiments=args.n_experiments,
            n_samples=args.n_samples,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            tag=f"baseline_ols_{config['tag']}",
            method_name=config["name"],
        )

        if not df.empty:
            dvc.save_results(df, summary, f"baseline_ols_full_{config['tag']}")
            all_dfs.append(df)
            all_summaries.append(summary)

    # 合并所有方法的结果
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv = os.path.join(OUT_DIR, "baseline_ols_full_all_methods.csv")
        combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
        print(f"\n  合并结果已保存：{combined_csv}")

    return all_dfs, all_summaries


def run_compare(args):
    """
    方法对比模式：朴素方法 vs DML，展示 DML 交叉拟合的优势。

    对比维度：
      - |Bias|: 绝对偏差（越小越好）
      - RMSE: 均方根误差（越小越好）
      - 95% CI 覆盖率（越接近 95% 越好）
    """
    print("\n" + "█" * 70)
    print("  [对比模式] 朴素方法 vs DML —— 展示交叉拟合优势")
    print("█" * 70)

    dag_info = dvc.setup_fixed_dag(
        n_nodes=20,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
        enforce_linear_ty=True,
    )
    ate_true = dvc.compute_ate_for_dag(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    comparison_results = []

    # ── 运行 3 种朴素方法 ──
    for method_key, config in BASELINE_OLS_METHODS.items():
        df, summary = dvc.run_monte_carlo(
            estimator_fn=config["fn"],
            dag_info=dag_info,
            ate_true=ate_true,
            n_experiments=args.n_experiments,
            n_samples=args.n_samples,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            tag=f"compare_ols_{config['tag']}",
            method_name=config["name"],
        )

        if not df.empty:
            comparison_results.append({
                "method": config["name"],
                "tag": config["tag"],
                "category": "Naive (No DML)",
                "mean_bias": summary["mean_bias"],
                "abs_bias": abs(summary["mean_bias"]),
                "rmse": summary["rmse"],
                "coverage_95": summary["coverage_95"],
                "mean_se": summary["mean_se"],
                "bias_p_value": summary["bias_p_value"],
                "n_experiments": summary["n_experiments"],
            })

    # ── 运行 DML 作为对照 ──
    print(f"\n{'─' * 60}")
    print("  对照方法: DML with Cross-Fitting (Random Forest)")
    print(f"{'─' * 60}")

    def dml_rf_estimator(Y, D, X_ctrl, seed):
        return dvc.dml_estimate_cross_fitting(
            Y, D, X_ctrl,
            n_folds=5,
            ml_method="forest",
            seed=seed,
            n_repeats=5,
        )

    df_dml, summary_dml = dvc.run_monte_carlo(
        estimator_fn=dml_rf_estimator,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=args.n_experiments,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="compare_ols_DML_RF",
        method_name="DML-RF (Cross-Fitting)",
    )

    if not df_dml.empty:
        comparison_results.append({
            "method": "DML-RF (Cross-Fitting)",
            "tag": "DML_RF",
            "category": "DML",
            "mean_bias": summary_dml["mean_bias"],
            "abs_bias": abs(summary_dml["mean_bias"]),
            "rmse": summary_dml["rmse"],
            "coverage_95": summary_dml["coverage_95"],
            "mean_se": summary_dml["mean_se"],
            "bias_p_value": summary_dml["bias_p_value"],
            "n_experiments": summary_dml["n_experiments"],
        })

    if not comparison_results:
        print("\n  [错误] 所有方法均失败，无法进行对比")
        return None

    # ── 生成对比报告 ──
    compare_df = pd.DataFrame(comparison_results)

    print("\n" + "═" * 70)
    print("  朴素方法 vs DML 对比结果")
    print("═" * 70)
    print(f"\n{'方法':<28s} {'类别':<16s} {'|Bias|':<12s} "
          f"{'RMSE':<12s} {'覆盖率':<10s} {'p值':<10s}")
    print("─" * 88)

    for _, row in compare_df.iterrows():
        print(f"  {row['method']:<26s} {row['category']:<14s} "
              f"{row['abs_bias']:<12.6f} {row['rmse']:<12.6f} "
              f"{row['coverage_95']:<10.1%} {row['bias_p_value']:<10.4f}")

    # 找最优方法
    best_bias_idx = compare_df["abs_bias"].idxmin()
    best_rmse_idx = compare_df["rmse"].idxmin()
    best_coverage_idx = (compare_df["coverage_95"] - 0.95).abs().idxmin()

    print(f"\n  最小 |Bias|:   {compare_df.loc[best_bias_idx, 'method']}"
          f" ({compare_df.loc[best_bias_idx, 'abs_bias']:.6f})")
    print(f"  最小 RMSE:    {compare_df.loc[best_rmse_idx, 'method']}"
          f" ({compare_df.loc[best_rmse_idx, 'rmse']:.6f})")
    print(f"  最优覆盖率:   {compare_df.loc[best_coverage_idx, 'method']}"
          f" ({compare_df.loc[best_coverage_idx, 'coverage_95']:.1%})")

    # DML 优势分析
    dml_rows = compare_df[compare_df["category"] == "DML"]
    naive_rows = compare_df[compare_df["category"] == "Naive (No DML)"]
    if not dml_rows.empty and not naive_rows.empty:
        dml_rmse = dml_rows["rmse"].min()
        naive_best_rmse = naive_rows["rmse"].min()
        improvement = (naive_best_rmse - dml_rmse) / naive_best_rmse * 100
        print(f"\n  ▶ DML 相对最优朴素方法的 RMSE 改善: {improvement:.1f}%")

        dml_coverage = dml_rows["coverage_95"].values[0]
        naive_best_coverage = naive_rows.loc[
            (naive_rows["coverage_95"] - 0.95).abs().idxmin()
        ]["coverage_95"] if len(naive_rows) > 0 else 0.0
        print(f"  ▶ DML 覆盖率: {dml_coverage:.1%} vs 朴素最优: "
              f"{naive_best_coverage:.1%} (目标: 95%)")

    print("═" * 70)

    # 保存对比结果
    compare_csv = os.path.join(OUT_DIR, "baseline_ols_comparison.csv")
    compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")
    print(f"\n  对比结果已保存：{compare_csv}")

    compare_json = os.path.join(OUT_DIR, "baseline_ols_comparison_summary.json")
    compare_summary = {
        "description": "朴素回归方法 vs DML 对比（展示交叉拟合去偏优势）",
        "conclusion": "DML 通过交叉拟合消除正则化偏差，"
                      "相比朴素方法具有更低偏差和更准确的覆盖率",
        "n_experiments": args.n_experiments,
        "n_samples": args.n_samples,
        "noise_type": args.noise_type,
        "noise_scale": args.noise_scale,
        "graph_type": args.graph_type,
        "ate_true": round(ate_true, 6),
        "best_bias": compare_df.loc[best_bias_idx, "method"],
        "best_rmse": compare_df.loc[best_rmse_idx, "method"],
        "best_coverage": compare_df.loc[best_coverage_idx, "method"],
        "results": comparison_results,
    }
    with open(compare_json, "w", encoding="utf-8") as f:
        json.dump(compare_summary, f, ensure_ascii=False, indent=2)
    print(f"  对比汇总已保存：{compare_json}")

    return compare_df


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DML 理论验证 —— 朴素回归基线方法（无 DML 对照组）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_dml_theory_validation_baseline_ols.py --mode quick
  python run_dml_theory_validation_baseline_ols.py --mode full --n_experiments 200
  python run_dml_theory_validation_baseline_ols.py --mode compare --n_samples 3000
  python run_dml_theory_validation_baseline_ols.py --mode all --graph_type layered
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "full", "compare", "all"],
        help="运行模式: quick=快速验证, full=完整蒙特卡洛, "
             "compare=与DML对比, all=全部",
    )
    parser.add_argument(
        "--n_experiments", type=int, default=200,
        help="蒙特卡洛实验次数 (默认: 200)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=2000,
        help="每次实验的样本量 (默认: 2000)",
    )
    parser.add_argument(
        "--graph_type", type=str, default="layered",
        choices=["layered", "er", "scale_free"],
        help="DAG 图类型 (默认: layered)",
    )
    parser.add_argument(
        "--noise_type", type=str, default="gaussian",
        choices=["gaussian", "student_t", "heteroscedastic"],
        help="噪声类型 (默认: gaussian)",
    )
    parser.add_argument(
        "--noise_scale", type=float, default=0.3,
        help="噪声标准差 (默认: 0.3)",
    )
    parser.add_argument(
        "--use_industrial", action="store_true",
        help="使用工业过程特征的边函数",
    )

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   DML 理论验证 —— 朴素回归基线（无 DML 对照组）               ║")
    print("║   目的：展示不使用 DML 交叉拟合时的估计缺陷                   ║")
    print("║   方法：Naive OLS / Partialling-Out (No CF) / OLS+Controls    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  模式:       {args.mode}")
    print(f"  实验次数:   {args.n_experiments}")
    print(f"  样本量:     {args.n_samples}")
    print(f"  图类型:     {args.graph_type}")
    print(f"  噪声类型:   {args.noise_type}")
    print(f"  噪声标准差: {args.noise_scale}")
    print(f"  工业函数:   {args.use_industrial}")

    t_start = time.perf_counter()

    if args.mode == "quick":
        run_quick(args)
    elif args.mode == "full":
        run_full(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "all":
        print("\n\n" + "▓" * 70)
        print("  第 1 步：完整蒙特卡洛验证（3 种朴素方法）")
        print("▓" * 70)
        run_full(args)

        print("\n\n" + "▓" * 70)
        print("  第 2 步：朴素方法 vs DML 横向对比")
        print("▓" * 70)
        run_compare(args)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'═' * 70}")
    print(f"  全部任务完成  总耗时: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")
    print(f"  结果输出目录: {OUT_DIR}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
