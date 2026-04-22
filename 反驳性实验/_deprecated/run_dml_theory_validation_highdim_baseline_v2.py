"""
run_dml_theory_validation_highdim_baseline_v2.py
=================================================
DML 理论验证（高维合成数据）—— 基线方法 V2：梯度提升 DML（GBM-DML）

═══════════════════════════════════════════════════════════════════
  高维场景基线：GBM DML

  本脚本是高维合成数据管线中的基线 V2，对应真实管线中的
  run_refutation_xin2_baseline_v2.py（GBM-DML 基线）。

  核心方法：
    - Nuisance function 估计器：GradientBoostingRegressor
    - 交叉拟合：标准 K-Fold（Chernozhukov 2018）
    - 中位数聚合 SE（多次重复交叉拟合）

  GBM vs RF 在高维场景的预期差异：
    - GBM 偏差更低（序贯 Boosting），残差可能更干净
    - GBM 在高维冗余特征下仍可能过拟合，但通常优于 RF
    - 两者均无法像 VAE 那样利用流形结构

  高维场景设计（同 baseline_v1）：
    - DAG 节点 60 个，7 层分层工业图
    - 混杂变量经观测扩展后约 80~120 维
    - 异方差噪声 + 时间滞后

用法：
  python run_dml_theory_validation_highdim_baseline_v2.py --mode quick
  python run_dml_theory_validation_highdim_baseline_v2.py --mode full
  python run_dml_theory_validation_highdim_baseline_v2.py --mode consistency
  python run_dml_theory_validation_highdim_baseline_v2.py --mode all
═══════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# ─── 路径配置 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import dml_validation_common_highdim as dvch

# 输出目录
OUT_DIR = os.path.join(BASE_DIR, "DML理论验证")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  GBM-DML 估计器
# ═══════════════════════════════════════════════════════════════════

def gbm_dml_estimator(Y, D, X_ctrl, seed, n_folds=5, n_repeats=5):
    """
    GBM DML 估计器（标准交叉拟合 + 中位数聚合）。

    使用 GradientBoostingRegressor 作为 nuisance function 估计器。
    """
    return dvch.dml_estimate_cross_fitting(
        Y, D, X_ctrl,
        n_folds=n_folds,
        ml_method="gbm",
        seed=seed,
        n_repeats=n_repeats,
    )


# ═══════════════════════════════════════════════════════════════════
#  实验模式
# ═══════════════════════════════════════════════════════════════════

def run_quick(args):
    """快速验证：少量实验验证 GBM-DML 在高维数据上的基本正确性"""
    print("\n" + "█" * 70)
    print("  [快速模式] 高维基线 V2: GBM-DML 验证")
    print("█" * 70)

    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
    )
    ate_true = dvch.compute_ate_for_dag_highdim(
        dag_info, args.noise_scale, args.noise_type
    )
    print(f"  真实 ATE = {ate_true:.6f}")
    print(f"  混杂变量数: {len(dag_info['confounder_indices'])}")

    exp_dim = dvch.compute_expanded_dim(len(dag_info['confounder_indices']))
    print(f"  扩展后特征维度: ~{exp_dim}")

    n_exp = min(args.n_experiments, 30)
    df, summary = dvch.run_monte_carlo_highdim(
        estimator_fn=gbm_dml_estimator,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=n_exp,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="highdim_baseline_v2_quick",
        method_name="GBM-DML (高维基线V2, Quick)",
    )

    if isinstance(df, pd.DataFrame) and not df.empty:
        dvch.save_results(df, summary, "highdim_baseline_v2_gbm_quick")
    return df, summary


def run_full(args):
    """完整蒙特卡洛验证"""
    print("\n" + "█" * 70)
    print("  [完整模式] 高维基线 V2: GBM-DML 完整蒙特卡洛验证")
    print("█" * 70)

    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
    )
    ate_true = dvch.compute_ate_for_dag_highdim(
        dag_info, args.noise_scale, args.noise_type
    )
    print(f"  真实 ATE = {ate_true:.6f}")

    df, summary = dvch.run_monte_carlo_highdim(
        estimator_fn=gbm_dml_estimator,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=args.n_experiments,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="highdim_baseline_v2_full",
        method_name="GBM-DML (高维基线V2)",
    )

    if isinstance(df, pd.DataFrame) and not df.empty:
        dvch.save_results(df, summary, "highdim_baseline_v2_gbm_full")
    return df, summary


def run_consistency(args):
    """√n-一致性验证"""
    print("\n" + "█" * 70)
    print("  [一致性模式] 高维基线 V2: GBM-DML √n-一致性验证")
    print("█" * 70)

    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
    )
    ate_true = dvch.compute_ate_for_dag_highdim(
        dag_info, args.noise_scale, args.noise_type
    )
    print(f"  真实 ATE = {ate_true:.6f}")

    df_cons = dvch.run_consistency_validation_highdim(
        estimator_fn=gbm_dml_estimator,
        dag_info=dag_info,
        ate_true=ate_true,
        sample_sizes=[500, 1000, 2000, 4000, 8000],
        n_experiments_per_size=50,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        method_name="GBM-DML (高维基线V2)",
    )

    if not df_cons.empty:
        csv_path = os.path.join(OUT_DIR, "highdim_baseline_v2_gbm_consistency.csv")
        df_cons.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  一致性结果已保存：{csv_path}")

    return df_cons


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DML 理论验证（高维）—— 基线 V2: GBM-DML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full", "consistency", "all"])
    parser.add_argument("--n_experiments", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_nodes", type=int, default=60)
    parser.add_argument("--graph_type", type=str, default="layered",
                        choices=["layered", "er", "scale_free"])
    parser.add_argument("--noise_type", type=str, default="heteroscedastic",
                        choices=["gaussian", "heteroscedastic", "heavy_tail"])
    parser.add_argument("--noise_scale", type=float, default=0.3)
    parser.add_argument("--use_industrial", action="store_true", default=True)

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   DML 理论验证（高维）—— 基线 V2: GBM-DML                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  模式:       {args.mode}")
    print(f"  DAG 节点:   {args.n_nodes}")
    print(f"  样本量:     {args.n_samples}")
    print(f"  噪声类型:   {args.noise_type}")
    print(f"  噪声标准差: {args.noise_scale}")

    t_start = time.perf_counter()

    if args.mode == "quick":
        run_quick(args)
    elif args.mode == "full":
        run_full(args)
    elif args.mode == "consistency":
        run_consistency(args)
    elif args.mode == "all":
        run_full(args)
        run_consistency(args)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═' * 70}")
    print(f"  完成  耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
