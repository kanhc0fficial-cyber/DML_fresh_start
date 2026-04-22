"""
run_dml_theory_validation_baseline_dml.py
=========================================
DML 理论验证 —— 传统机器学习基线方法

═══════════════════════════════════════════════════════════════════
  本脚本作为标准 DML-PLM 估计器的基线验证，使用 3 种传统机器学习方法
  作为 nuisance function 估计器：

    1. Random Forest (RF) —— 最常见的 DML 基线方法
    2. Lasso          —— 经典稀疏线性方法
    3. Gradient Boosting Machine (GBM) —— 梯度提升树

  本脚本的目的是为创新方案（v3/v4/v5）提供可对比的传统基线参考。
  所有方法在相同的合成 DAG 数据生成过程下评估，确保公平对比。
═══════════════════════════════════════════════════════════════════

验证内容：
  1. 无偏性（Unbiasedness）：E[θ̂] ≈ θ_true
  2. √n-一致性（√n-Consistency）：RMSE ∝ 1/√n
  3. 置信区间覆盖率（Coverage）：95% CI 覆盖 θ_true 的频率 ≈ 95%
  4. 方法对比：哪种 nuisance 估计器在偏差、RMSE、覆盖率方面最优

用法：
  # 快速验证（少量实验，单方法）
  python run_dml_theory_validation_baseline_dml.py --mode quick

  # 完整蒙特卡洛验证（默认 200 次实验，3 种方法）
  python run_dml_theory_validation_baseline_dml.py --mode full

  # √n-一致性验证
  python run_dml_theory_validation_baseline_dml.py --mode consistency

  # 方法对比
  python run_dml_theory_validation_baseline_dml.py --mode compare

  # 全部实验
  python run_dml_theory_validation_baseline_dml.py --mode all

  # 指定图类型和噪声
  python run_dml_theory_validation_baseline_dml.py --mode full --graph_type layered --noise_type gaussian
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
#  基线估计器定义
# ═══════════════════════════════════════════════════════════════════

# 3 种传统 ML 方法配置
BASELINE_METHODS = {
    "forest": {
        "name": "Random Forest",
        "tag": "RF",
        "description": "随机森林 —— DML 最常用的 nuisance 估计器",
    },
    "lasso": {
        "name": "Lasso",
        "tag": "Lasso",
        "description": "Lasso 正则化线性回归 —— 经典稀疏方法",
    },
    "gbm": {
        "name": "Gradient Boosting",
        "tag": "GBM",
        "description": "梯度提升树 —— 集成学习代表方法",
    },
}


def make_estimator_fn(ml_method: str, n_folds: int = 5, n_repeats: int = 5):
    """
    创建给定 ML 方法的估计器函数。

    参数:
        ml_method: 机器学习方法 ('forest', 'lasso', 'gbm')
        n_folds:   交叉拟合折数 (默认 5)
        n_repeats: 重复交叉拟合次数，用于中位数聚合 (默认 5)

    返回签名为 (Y, D, X_ctrl, seed) -> (theta, se, ci_lo, ci_hi) 的函数，
    适配 run_monte_carlo 的 estimator_fn 接口。
    """
    def estimator_fn(Y, D, X_ctrl, seed):
        return dvc.dml_estimate_cross_fitting(
            Y, D, X_ctrl,
            n_folds=n_folds,
            ml_method=ml_method,
            seed=seed,
            n_repeats=n_repeats,
        )
    return estimator_fn


# ═══════════════════════════════════════════════════════════════════
#  实验模式实现
# ═══════════════════════════════════════════════════════════════════

def run_quick(args):
    """快速验证模式：使用 Random Forest 进行小规模蒙特卡洛"""
    print("\n" + "█" * 70)
    print("  [快速模式] 标准 DML 基线验证 (Random Forest)")
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

    estimator_fn = make_estimator_fn("forest")
    n_exp = min(args.n_experiments, 30)

    df, summary = dvc.run_monte_carlo(
        estimator_fn=estimator_fn,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=n_exp,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="baseline_quick",
        method_name="DML-RF (Quick)",
    )

    if not df.empty:
        dvc.save_results(df, summary, "baseline_dml_quick_RF")
    return df, summary


def run_full(args):
    """完整蒙特卡洛验证：对每种 ML 方法分别运行"""
    print("\n" + "█" * 70)
    print("  [完整模式] 标准 DML 基线验证 (3种传统方法)")
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

    for ml_method, config in BASELINE_METHODS.items():
        print(f"\n{'─' * 60}")
        print(f"  方法: {config['name']} ({config['tag']})")
        print(f"  说明: {config['description']}")
        print(f"{'─' * 60}")

        estimator_fn = make_estimator_fn(ml_method)

        df, summary = dvc.run_monte_carlo(
            estimator_fn=estimator_fn,
            dag_info=dag_info,
            ate_true=ate_true,
            n_experiments=args.n_experiments,
            n_samples=args.n_samples,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            tag=f"baseline_{config['tag']}",
            method_name=f"DML-{config['tag']}",
        )

        if not df.empty:
            dvc.save_results(df, summary, f"baseline_dml_full_{config['tag']}")
            all_dfs.append(df)
            all_summaries.append(summary)

    # 合并所有方法的结果
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv = os.path.join(OUT_DIR, "baseline_dml_full_all_methods.csv")
        combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
        print(f"\n  合并结果已保存：{combined_csv}")

    return all_dfs, all_summaries


def run_consistency(args):
    """√n-一致性验证：对每种方法验证 RMSE ∝ 1/√n"""
    print("\n" + "█" * 70)
    print("  [一致性模式] √n-一致性验证")
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

    all_consistency_dfs = []

    for ml_method, config in BASELINE_METHODS.items():
        print(f"\n  === {config['name']} ({config['tag']}) ===")
        estimator_fn = make_estimator_fn(ml_method)

        df_cons = dvc.run_consistency_validation(
            estimator_fn=estimator_fn,
            dag_info=dag_info,
            ate_true=ate_true,
            sample_sizes=[500, 1000, 2000, 4000, 8000],
            n_experiments_per_size=50,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            method_name=f"DML-{config['tag']}",
        )

        if not df_cons.empty:
            csv_path = os.path.join(
                OUT_DIR, f"baseline_dml_consistency_{config['tag']}.csv"
            )
            df_cons.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"  已保存：{csv_path}")
            all_consistency_dfs.append(df_cons)

    # 合并一致性结果
    if all_consistency_dfs:
        combined_cons = pd.concat(all_consistency_dfs, ignore_index=True)
        combined_path = os.path.join(OUT_DIR, "baseline_dml_consistency_all.csv")
        combined_cons.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"\n  合并一致性结果已保存：{combined_path}")

    return all_consistency_dfs


def run_compare(args):
    """方法对比模式：横向比较 3 种方法的偏差、RMSE、覆盖率"""
    print("\n" + "█" * 70)
    print("  [对比模式] 传统 DML 方法横向对比")
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

    for ml_method, config in BASELINE_METHODS.items():
        estimator_fn = make_estimator_fn(ml_method)

        # 使用较多实验次数以获得稳定对比
        n_exp = args.n_experiments

        df, summary = dvc.run_monte_carlo(
            estimator_fn=estimator_fn,
            dag_info=dag_info,
            ate_true=ate_true,
            n_experiments=n_exp,
            n_samples=args.n_samples,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            tag=f"compare_{config['tag']}",
            method_name=f"DML-{config['tag']}",
        )

        if not df.empty:
            comparison_results.append({
                "method": config["name"],
                "tag": config["tag"],
                "ml_method": ml_method,
                "mean_bias": summary["mean_bias"],
                "abs_bias": abs(summary["mean_bias"]),
                "rmse": summary["rmse"],
                "coverage_95": summary["coverage_95"],
                "mean_se": summary["mean_se"],
                "bias_p_value": summary["bias_p_value"],
                "n_experiments": summary["n_experiments"],
            })

    if not comparison_results:
        print("\n  [错误] 所有方法均失败，无法进行对比")
        return None

    # 生成对比报告
    compare_df = pd.DataFrame(comparison_results)

    print("\n" + "═" * 70)
    print("  传统 DML 方法对比结果")
    print("═" * 70)
    print(f"\n{'方法':<20s} {'|Bias|':<12s} {'RMSE':<12s} {'覆盖率':<10s} {'p值':<10s}")
    print("─" * 64)

    for _, row in compare_df.iterrows():
        print(f"  {row['method']:<18s} {row['abs_bias']:<12.6f} "
              f"{row['rmse']:<12.6f} {row['coverage_95']:<10.1%} "
              f"{row['bias_p_value']:<10.4f}")

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
    print("═" * 70)

    # 保存对比结果
    compare_csv = os.path.join(OUT_DIR, "baseline_dml_comparison.csv")
    compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")
    print(f"\n  对比结果已保存：{compare_csv}")

    compare_json = os.path.join(OUT_DIR, "baseline_dml_comparison_summary.json")
    compare_summary = {
        "description": "传统 DML 基线方法对比 (RF / Lasso / GBM)",
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
        description="DML 理论验证 —— 传统基线方法 (RF / Lasso / GBM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_dml_theory_validation_baseline_dml.py --mode quick
  python run_dml_theory_validation_baseline_dml.py --mode full --n_experiments 200
  python run_dml_theory_validation_baseline_dml.py --mode compare --n_samples 3000
  python run_dml_theory_validation_baseline_dml.py --mode all --graph_type layered
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "full", "consistency", "compare", "all"],
        help="运行模式: quick=快速验证, full=完整蒙特卡洛, "
             "consistency=√n一致性, compare=方法对比, all=全部",
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
    print("║   DML 理论验证 —— 传统基线方法 (RF / Lasso / GBM)              ║")
    print("║   用途：为创新方案 (v3/v4/v5) 提供可对比的标准基线             ║")
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
    elif args.mode == "consistency":
        run_consistency(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "all":
        print("\n\n" + "▓" * 70)
        print("  第 1 步：完整蒙特卡洛验证")
        print("▓" * 70)
        run_full(args)

        print("\n\n" + "▓" * 70)
        print("  第 2 步：√n-一致性验证")
        print("▓" * 70)
        run_consistency(args)

        print("\n\n" + "▓" * 70)
        print("  第 3 步：方法横向对比")
        print("▓" * 70)
        run_compare(args)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'═' * 70}")
    print(f"  全部任务完成  总耗时: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")
    print(f"  结果输出目录: {OUT_DIR}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
