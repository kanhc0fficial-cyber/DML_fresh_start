"""
run_dml_theory_validation.py
============================
DML（双重机器学习）理论验证 —— 基于合成 DAG 数据生成器（已修正版）

═══════════════════════════════════════════════════════════════════
  核心目标：在已知真实因果效应的合成数据上验证标准 DML 的理论性质
═══════════════════════════════════════════════════════════════════

  本脚本作为基线参考脚本，验证标准 DML-PLM 的理论性质。
  创新方案的验证请使用 run_dml_theory_validation_v3/v4/v5.py。

  已修正的 4 个关键漏洞（参考 AI Review 反馈）：
    1. DGP 错配：ATE 仿真使用 generate_data 的 do_interventions 参数，
       复用原生噪声分布（异方差/重尾），不再外部硬编码高斯噪声
    2. PLM 假设：强制 T→Y 路径线性化（enforce_linear_treatment_paths），
       混杂路径可以是任意非线性
    3. SE 聚合：使用中位数聚合公式 V = Median(V_b + (θ_b - θ)²)，
       正确合并抽样方差和分割方差
    4. ATE 兜底：移除 "仿真=0 则用解析值替换" 的逻辑，
       仿真 do-calculus 始终是基准真相

  验证项目：
    1. 无偏性（Unbiasedness）：E[θ̂] ≈ θ_true
    2. √n-一致性（√n-Consistency）：RMSE ∝ 1/√n
    3. 置信区间覆盖率（Coverage）：95% CI 覆盖 θ_true 的频率 ≈ 95%
    4. 混杂控制有效性：正确控制混杂 vs 遗漏混杂 的效果对比

用法：
  python run_dml_theory_validation.py --mode quick
  python run_dml_theory_validation.py --mode full
  python run_dml_theory_validation.py --mode consistency
  python run_dml_theory_validation.py --mode confounding
  python run_dml_theory_validation.py --mode all
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ─── 路径配置 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
CAUSAL_DISCOVERY_DIR = os.path.join(REPO_ROOT, "因果的发现算法理论验证")

# 导入合成 DAG 生成器
sys.path.insert(0, CAUSAL_DISCOVERY_DIR)
from synthetic_dag_generator import SyntheticDAGGenerator

# 导入共享模块（包含修正后的 ATE 计算和 DML 估计器）
from dml_validation_common import (
    compute_true_ate,
    compute_true_ate_linear,
    enforce_linear_treatment_paths,
    select_treatment_outcome,
    convert_roles_to_dag_parser_format,
    build_adjustment_variables,
    dml_estimate_cross_fitting,
    fit_ml_model,
    setup_fixed_dag,
    compute_ate_for_dag,
    run_monte_carlo,
    run_consistency_validation,
    save_results,
    OUT_DIR,
)


# ═══════════════════════════════════════════════════════════════════
#  混杂控制对比实验（保留此独立实验，使用修正后的共享模块函数）
# ═══════════════════════════════════════════════════════════════════

def run_confounding_comparison(
    n_experiments: int = 100,
    n_samples: int = 2000,
    graph_type: str = "layered",
    noise_scale: float = 0.3,
    noise_type: str = "gaussian",
    use_industrial: bool = False,
    ml_method: str = "forest",
):
    """
    混杂控制有效性对比：
      A. 正确控制混杂（使用 DAG 识别的后门调整集）
      B. 遗漏混杂（不控制任何变量，朴素回归）
      C. 过度控制（控制所有变量，包括碰撞变量和中介变量）

    验证：方案 A 的偏差应最小，覆盖率最高。
    """
    import networkx as nx

    print("\n" + "=" * 70)
    print(" 混杂控制对比实验（修正版）")
    print(f" 实验次数: {n_experiments} | 样本量: {n_samples}")
    print(" 方案 A：正确控制混杂（后门调整集）")
    print(" 方案 B：遗漏混杂（无控制变量）")
    print(" 方案 C：过度控制（所有非 T/Y 变量）")
    print("=" * 70)

    results = {
        "A_correct": [],
        "B_omitted": [],
        "C_overcntrl": [],
    }

    n_success = 0
    n_nodes = 20
    for exp_i in range(n_experiments):
        try:
            seed = exp_i * 97 + 42
            gen = SyntheticDAGGenerator(n_nodes=n_nodes, seed=seed)
            X_data, adj_true, metadata = gen.generate_complete_synthetic_dataset(
                graph_type=graph_type,
                n_samples=n_samples,
                noise_scale=noise_scale,
                noise_type=noise_type,
                add_time_lag=False,
                use_industrial_functions=use_industrial,
            )
            edge_funcs = metadata["edge_funcs"]
            layer_indices = metadata.get("layer_indices")

            t_idx, y_idx, roles = select_treatment_outcome(
                gen, adj_true, edge_funcs, layer_indices, seed=seed,
            )

            # 强制 T→Y 路径线性化（修正漏洞二）
            edge_funcs = enforce_linear_treatment_paths(
                adj_true, edge_funcs, t_idx, y_idx,
            )

            # 使用修正后的 ATE 计算（修正漏洞一和四）
            ate_true = compute_true_ate(
                gen, adj_true, edge_funcs, t_idx, y_idx,
                noise_scale=noise_scale,
                noise_type=noise_type,
            )

            D = X_data[:, t_idx]
            Y = X_data[:, y_idx]

            # 方案 A：正确控制混杂
            X_A, _ = build_adjustment_variables(
                gen, adj_true, X_data, t_idx, y_idx, n_nodes,
            )
            theta_A, se_A, ci_lo_A, ci_hi_A = dml_estimate_cross_fitting(
                Y, D, X_A, ml_method=ml_method, seed=seed,
            )
            results["A_correct"].append({
                "bias": theta_A - ate_true,
                "covers": bool(ci_lo_A <= ate_true <= ci_hi_A),
                "se": se_A,
            })

            # 方案 B：遗漏混杂（使用随机无关变量代替）
            rng = np.random.RandomState(seed + 1000)
            X_B = rng.randn(n_samples, 3)
            theta_B, se_B, ci_lo_B, ci_hi_B = dml_estimate_cross_fitting(
                Y, D, X_B, ml_method=ml_method, seed=seed,
            )
            results["B_omitted"].append({
                "bias": theta_B - ate_true,
                "covers": bool(ci_lo_B <= ate_true <= ci_hi_B),
                "se": se_B,
            })

            # 方案 C：过度控制（包含碰撞/中介）
            all_others = [i for i in range(n_nodes) if i != t_idx and i != y_idx]
            X_C = X_data[:, all_others]
            theta_C, se_C, ci_lo_C, ci_hi_C = dml_estimate_cross_fitting(
                Y, D, X_C, ml_method=ml_method, seed=seed,
            )
            results["C_overcntrl"].append({
                "bias": theta_C - ate_true,
                "covers": bool(ci_lo_C <= ate_true <= ci_hi_C),
                "se": se_C,
            })

            n_success += 1

            if (exp_i + 1) % max(1, n_experiments // 5) == 0:
                print(f"  [{exp_i + 1}/{n_experiments}]  成功={n_success}")

        except Exception as e:
            if n_success == 0 and exp_i < 5:
                print(f"  [实验 {exp_i}] 失败: {e}")

    print(f"\n完成：{n_success}/{n_experiments} 成功")

    if n_success == 0:
        print("[错误] 所有实验失败")
        return pd.DataFrame()

    # 汇总
    print("\n" + "─" * 60)
    print("混杂控制对比结果")
    print(f"{'方案':<20s}  {'偏差均值':>10s}  {'RMSE':>10s}  {'覆盖率':>8s}")
    print("─" * 60)

    rows = []
    for label, res_list in results.items():
        if not res_list:
            continue
        biases = np.array([r["bias"] for r in res_list])
        covers = np.array([r["covers"] for r in res_list])
        mean_bias = float(np.mean(biases))
        rmse = float(np.sqrt(np.mean(biases ** 2)))
        coverage = float(np.mean(covers))

        name_map = {
            "A_correct": "A. 正确控制",
            "B_omitted": "B. 遗漏混杂",
            "C_overcntrl": "C. 过度控制",
        }
        name = name_map.get(label, label)
        print(f"{name:<20s}  {mean_bias:>+10.6f}  {rmse:>10.6f}  {coverage:>8.1%}")
        rows.append({
            "方案": name,
            "偏差均值": round(mean_bias, 6),
            "RMSE": round(rmse, 6),
            "覆盖率": round(coverage, 4),
            "实验次数": len(res_list),
        })
    print("─" * 60)

    if len(rows) >= 2:
        rmse_A = rows[0]["RMSE"]
        rmse_B = rows[1]["RMSE"]
        if rmse_A < rmse_B:
            print("✓ 正确控制混杂的 RMSE 低于遗漏混杂，符合理论预期")
        else:
            print("⚠ 正确控制混杂的 RMSE 未明显优于遗漏混杂")

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f"confounding_comparison_{graph_type}_{ml_method}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="DML 理论验证（修正版）：基于合成 DAG 数据生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
验证项目：
  1. quick:         快速验证（30 次实验，确认脚本正常运行）
  2. full:          完整蒙特卡洛验证（默认 200 次实验）
  3. consistency:   √n-一致性验证（不同样本量对比）
  4. confounding:   混杂控制对比实验（正确 vs 遗漏 vs 过度控制）
  5. all:           全部实验

修正内容：
  - 漏洞一：ATE 仿真使用 generate_data 的 do_interventions（复用原生 DGP）
  - 漏洞二：T→Y 路径强制线性化（满足 PLM 假设）
  - 漏洞三：SE 使用中位数聚合公式（Chernozhukov 2018）
  - 漏洞四：移除 ATE 兜底逻辑（仿真值始终为基准真相）

结果保存在 反驳性实验/DML理论验证/ 目录下。
        """,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["quick", "full", "consistency", "confounding", "all"],
    )
    p.add_argument("--n_experiments", type=int, default=200)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_nodes", type=int, default=20)
    p.add_argument("--graph_type", type=str, default="layered",
                   choices=["er", "scale_free", "layered"])
    p.add_argument("--noise_scale", type=float, default=0.3)
    p.add_argument("--noise_type", type=str, default="gaussian",
                   choices=["gaussian", "heteroscedastic", "heavy_tail"])
    p.add_argument("--ml_method", type=str, default="forest",
                   choices=["forest", "linear", "lasso"])
    p.add_argument("--use_industrial", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f" DML 理论验证（修正版）  |  模式: {args.mode.upper()}")
    print(f" 数据来源: SyntheticDAGGenerator (因果的发现算法理论验证)")
    print(f" 已修正: DGP 错配 / PLM 假设 / SE 聚合 / ATE 兜底")
    print("=" * 70)

    mode = args.mode

    # 创建标准 DML 估计器函数
    def standard_dml_estimator(Y, D, X_ctrl, seed):
        return dml_estimate_cross_fitting(
            Y, D, X_ctrl,
            n_folds=5,
            ml_method=args.ml_method,
            seed=seed,
        )

    # 设置固定 DAG（所有蒙特卡洛实验共享）
    dag_info = setup_fixed_dag(
        n_nodes=args.n_nodes,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        enforce_linear_ty=True,  # 修正漏洞二
    )
    ate_true = compute_ate_for_dag(
        dag_info,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
    )

    if mode in ("quick",):
        df, summary = run_monte_carlo(
            estimator_fn=standard_dml_estimator,
            dag_info=dag_info,
            ate_true=ate_true,
            n_experiments=30,
            n_samples=1000,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            tag="quick",
            method_name=f"Standard DML ({args.ml_method})",
        )
        save_results(df, summary, f"monte_carlo_{args.graph_type}_{args.ml_method}_n1000_quick")

    if mode in ("full", "all"):
        df, summary = run_monte_carlo(
            estimator_fn=standard_dml_estimator,
            dag_info=dag_info,
            ate_true=ate_true,
            n_experiments=args.n_experiments,
            n_samples=args.n_samples,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            method_name=f"Standard DML ({args.ml_method})",
        )
        save_results(df, summary,
                     f"monte_carlo_{args.graph_type}_{args.ml_method}_n{args.n_samples}")

    if mode in ("consistency", "all"):
        run_consistency_validation(
            estimator_fn=standard_dml_estimator,
            dag_info=dag_info,
            ate_true=ate_true,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            method_name=f"Standard DML ({args.ml_method})",
        )

    if mode in ("confounding", "all"):
        run_confounding_comparison(
            n_experiments=min(args.n_experiments, 100),
            n_samples=args.n_samples,
            graph_type=args.graph_type,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            use_industrial=args.use_industrial,
            ml_method=args.ml_method,
        )

    print("\n" + "=" * 70)
    print(" 全部实验完成")
    print(f" 结果保存目录: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
