"""
dml_validation_common_highdim.py
=================================
高维合成数据 DML 理论验证共享模块

在 dml_validation_common.py 基础上，增加高维观测扩展支持。
核心差异：
  - 使用 HighDimSyntheticDAGGenerator（60 节点 vs 20 节点）
  - setup_fixed_dag_highdim 默认 60 节点 + 异方差噪声 + 时间滞后
  - build_expanded_ctrl 将 ~10 维混杂变量扩展为 ~100 维高维观测
  - ATE / 蒙特卡洛 / 一致性框架完全复用原版

T→Y 线性化由 enforce_linear_treatment_paths() 保证，不受维度影响。
"""

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

sys.path.insert(0, CAUSAL_DISCOVERY_DIR)
sys.path.insert(0, BASE_DIR)

# 导入原版公共模块（复用 ATE 计算、蒙特卡洛框架等）
import dml_validation_common as dvc

# 导入高维生成器
from synthetic_dag_generator_highdim import (
    HighDimSyntheticDAGGenerator,
    expand_to_highdim,
    compute_expanded_dim,
)
# 同时暴露原版生成器（某些函数需要基类）
from synthetic_dag_generator import SyntheticDAGGenerator

# 输出目录
OUT_DIR = os.path.join(BASE_DIR, "DML理论验证")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  高维 DAG 配置
# ═══════════════════════════════════════════════════════════════════

# 默认高维参数
HIGHDIM_N_NODES = 60        # DAG 节点数
HIGHDIM_N_LAYERS = 7        # 分层图层数
HIGHDIM_NOISE_TYPE = "heteroscedastic"  # 默认异方差噪声
HIGHDIM_NOISE_SCALE = 0.3   # 噪声标准差
HIGHDIM_ADD_TIME_LAG = True  # 默认启用时间滞后

# 观测扩展参数
EXPAND_NOISY_COPIES = 3
EXPAND_NONLINEAR_TRANSFORMS = 3
EXPAND_INTERACTION_PAIRS = 8
EXPAND_BLOCK_GROUPS = 3
EXPAND_BLOCK_DIM = 5
EXPAND_COPY_NOISE_SCALE = 0.3
EXPAND_BLOCK_NOISE_SCALE = 0.5


# ═══════════════════════════════════════════════════════════════════
#  高维 DAG 固定配置
# ═══════════════════════════════════════════════════════════════════

def setup_fixed_dag_highdim(
    n_nodes: int = HIGHDIM_N_NODES,
    n_layers: int = HIGHDIM_N_LAYERS,
    graph_type: str = "layered",
    use_industrial: bool = True,
    dag_seed: int = 42,
    enforce_linear_ty: bool = True,
) -> dict:
    """
    创建并配置高维固定 DAG 结构。

    返回:
        dict 包含:
          gen_base, adj_true, edge_funcs, layer_indices,
          t_idx, y_idx, roles, confounder_indices,
          n_nodes（用于数据生成时重建生成器）
    """
    gen_base = HighDimSyntheticDAGGenerator(n_nodes=n_nodes, seed=dag_seed)

    if graph_type == "er":
        adj_true = gen_base.generate_er_dag()
        layer_indices = None
    elif graph_type == "scale_free":
        adj_true = gen_base.generate_scale_free_dag()
        layer_indices = None
    elif graph_type == "layered":
        adj_true, layer_indices = gen_base.generate_layered_industrial_dag(
            n_layers=n_layers
        )
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    edge_funcs = gen_base.assign_edge_functions(
        adj_true, layer_indices, use_industrial
    )

    t_idx, y_idx, roles = dvc.select_treatment_outcome(
        gen_base, adj_true, edge_funcs, layer_indices, seed=dag_seed,
    )

    # 强制 T→Y 路径线性化（PLM 假设）
    if enforce_linear_ty:
        edge_funcs = dvc.enforce_linear_treatment_paths(
            adj_true, edge_funcs, t_idx, y_idx,
        )

    # 构建控制变量索引（高维版：使用更宽泛的安全控制变量集）
    # 在高维验证场景中，我们需要足够多的控制变量来测试 VAE 的降维能力
    # 策略：使用所有 T 的安全祖先（不是 T 或 Y 的后代），而非仅限于严格调整集
    import networkx as nx
    G_dag = nx.DiGraph(adj_true)
    t_ancestors = nx.ancestors(G_dag, t_idx)
    y_ancestors = nx.ancestors(G_dag, y_idx)
    t_descendants = nx.descendants(G_dag, t_idx)
    y_descendants = nx.descendants(G_dag, y_idx)

    # 安全控制变量 = T 的祖先中，不是 T/Y 后代的变量
    # 这是一个有效的过度控制策略（在 PLM 中是安全的）
    safe_ctrl_vars = [
        v for v in range(n_nodes)
        if v != t_idx and v != y_idx
        and v not in t_descendants and v not in y_descendants
        and (v in t_ancestors or v in y_ancestors)
    ]
    confounder_indices = sorted(safe_ctrl_vars)

    # 如果安全控制变量仍然太少（<5），回退到包含更多变量
    if len(confounder_indices) < 5:
        # 纳入所有非后代变量
        broader_vars = [
            v for v in range(n_nodes)
            if v != t_idx and v != y_idx
            and v not in t_descendants and v not in y_descendants
        ]
        confounder_indices = sorted(broader_vars)

    return {
        "gen_base": gen_base,
        "adj_true": adj_true,
        "edge_funcs": edge_funcs,
        "layer_indices": layer_indices,
        "t_idx": t_idx,
        "y_idx": y_idx,
        "roles": roles,
        "confounder_indices": confounder_indices,
    }


def compute_ate_for_dag_highdim(
    dag_info: dict,
    noise_scale: float = HIGHDIM_NOISE_SCALE,
    noise_type: str = HIGHDIM_NOISE_TYPE,
) -> float:
    """基于固定高维 DAG 计算真实 ATE"""
    return dvc.compute_true_ate(
        dag_info["gen_base"],
        dag_info["adj_true"],
        dag_info["edge_funcs"],
        dag_info["t_idx"],
        dag_info["y_idx"],
        noise_scale=noise_scale,
        noise_type=noise_type,
    )


# ═══════════════════════════════════════════════════════════════════
#  高维数据生成（含观测扩展）
# ═══════════════════════════════════════════════════════════════════

def generate_highdim_data(
    dag_info: dict,
    n_samples: int = 3000,
    noise_scale: float = HIGHDIM_NOISE_SCALE,
    noise_type: str = HIGHDIM_NOISE_TYPE,
    add_time_lag: bool = HIGHDIM_ADD_TIME_LAG,
    data_seed: int = 1000,
    expand_seed: int = None,
) -> tuple:
    """
    生成高维合成数据（含观测扩展）。

    步骤:
      1. 用 DAG 生成器生成原始 60 维时序数据
      2. 提取 D, Y, X_ctrl_raw（混杂变量的原始维度）
      3. 将 X_ctrl_raw 扩展为高维观测 X_ctrl_hd

    参数:
        dag_info:      setup_fixed_dag_highdim 的返回值
        n_samples:     样本数
        noise_scale:   噪声标准差
        noise_type:    噪声类型
        add_time_lag:  是否启用时间滞后
        data_seed:     数据生成种子
        expand_seed:   观测扩展种子（默认 = data_seed + 7777）

    返回:
        (D, Y, X_ctrl_hd, X_ctrl_raw):
          D:          处理变量 (n_samples,)
          Y:          结果变量 (n_samples,)
          X_ctrl_hd:  高维观测控制变量 (n_samples, expanded_dim)
          X_ctrl_raw: 原始低维混杂变量 (n_samples, n_confounders)
    """
    n_nodes = dag_info["gen_base"].n_nodes
    adj_true = dag_info["adj_true"]
    edge_funcs = dag_info["edge_funcs"]
    t_idx = dag_info["t_idx"]
    y_idx = dag_info["y_idx"]
    confounder_indices = dag_info["confounder_indices"]

    if expand_seed is None:
        expand_seed = data_seed + 7777

    # 生成 DAG 数据
    gen_data = HighDimSyntheticDAGGenerator(n_nodes=n_nodes, seed=data_seed)
    X_data = gen_data.generate_data(
        adj_true, edge_funcs,
        n_samples=n_samples,
        noise_scale=noise_scale,
        noise_type=noise_type,
        add_time_lag=add_time_lag,
    )

    D = X_data[:, t_idx]
    Y = X_data[:, y_idx]

    # 构建控制变量
    if len(confounder_indices) > 0:
        X_ctrl_raw = X_data[:, confounder_indices]
    else:
        X_ctrl_raw = np.ones((n_samples, 1))

    # 观测扩展
    X_ctrl_hd = expand_to_highdim(
        X_ctrl_raw,
        seed=expand_seed,
        n_noisy_copies=EXPAND_NOISY_COPIES,
        n_nonlinear_transforms=EXPAND_NONLINEAR_TRANSFORMS,
        n_interaction_pairs=EXPAND_INTERACTION_PAIRS,
        n_block_noise_groups=EXPAND_BLOCK_GROUPS,
        block_noise_dim=EXPAND_BLOCK_DIM,
        copy_noise_scale=EXPAND_COPY_NOISE_SCALE,
        block_noise_scale=EXPAND_BLOCK_NOISE_SCALE,
    )

    return D, Y, X_ctrl_hd, X_ctrl_raw


# ═══════════════════════════════════════════════════════════════════
#  高维蒙特卡洛验证框架
# ═══════════════════════════════════════════════════════════════════

def run_monte_carlo_highdim(
    estimator_fn,
    dag_info: dict,
    ate_true: float,
    n_experiments: int = 200,
    n_samples: int = 3000,
    noise_scale: float = HIGHDIM_NOISE_SCALE,
    noise_type: str = HIGHDIM_NOISE_TYPE,
    add_time_lag: bool = HIGHDIM_ADD_TIME_LAG,
    tag: str = "",
    method_name: str = "DML",
):
    """
    高维蒙特卡洛验证框架。

    与原版 run_monte_carlo 的区别：
      - 每次实验先生成 DAG 数据，再扩展为高维观测
      - estimator_fn 的接口为 (Y, D, X_ctrl_hd, seed) -> (theta, se, ci_lo, ci_hi)

    参数:
        estimator_fn:  估计器函数
        dag_info:      setup_fixed_dag_highdim 的返回值
        ate_true:      真实 ATE
        其他参数同原版
    """
    tag_str = f"_{tag}" if tag else ""

    print(f"\n{'=' * 70}")
    print(f" {method_name} 高维蒙特卡洛理论验证{tag_str}")
    print(f" 实验次数: {n_experiments} | 样本量: {n_samples}")
    print(f" 噪声: {noise_type}({noise_scale}) | 时间滞后: {add_time_lag}")
    print(f" 真实 ATE: {ate_true:.6f}")
    print(f" Treatment: X_{dag_info['t_idx']}, Outcome: X_{dag_info['y_idx']}")
    print(f" 混杂变量: {len(dag_info['roles']['confounders'])} 个  "
          f"调整集: {len(dag_info['confounder_indices'])} 个")
    print(f"{'=' * 70}")

    results = []
    n_success = 0
    n_fail = 0
    t0 = time.perf_counter()

    for exp_i in range(n_experiments):
        try:
            data_seed = exp_i * 13 + 1000

            D, Y, X_ctrl_hd, _ = generate_highdim_data(
                dag_info,
                n_samples=n_samples,
                noise_scale=noise_scale,
                noise_type=noise_type,
                add_time_lag=add_time_lag,
                data_seed=data_seed,
            )

            theta_hat, se, ci_lower, ci_upper = estimator_fn(
                Y, D, X_ctrl_hd, data_seed,
            )

            bias = theta_hat - ate_true
            covers = bool(ci_lower <= ate_true <= ci_upper)

            results.append({
                "experiment": exp_i,
                "seed": data_seed,
                "method": method_name,
                "n_samples": n_samples,
                "ate_true": round(ate_true, 6),
                "theta_hat": round(theta_hat, 6),
                "se": round(se, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
                "bias": round(bias, 6),
                "covers_true": covers,
            })
            n_success += 1

            if (exp_i + 1) % max(1, n_experiments // 10) == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{exp_i + 1}/{n_experiments}]  成功={n_success}  "
                      f"失败={n_fail}  耗时={elapsed:.1f}s")
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  [实验 {exp_i}] 失败: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\n完成：{n_success}/{n_experiments} 成功  "
          f"耗时 {elapsed:.1f}s  ({elapsed / max(1, n_success):.2f}s/实验)")

    if not results:
        print("[错误] 所有实验失败，无法汇总")
        return pd.DataFrame(), {}

    df = pd.DataFrame(results)
    summary = dvc.summarize_monte_carlo(df, method_name)
    return df, summary


def run_consistency_validation_highdim(
    estimator_fn,
    dag_info: dict,
    ate_true: float,
    sample_sizes: list = None,
    n_experiments_per_size: int = 50,
    noise_scale: float = HIGHDIM_NOISE_SCALE,
    noise_type: str = HIGHDIM_NOISE_TYPE,
    add_time_lag: bool = HIGHDIM_ADD_TIME_LAG,
    method_name: str = "DML",
):
    """高维版 √n-一致性验证"""
    if sample_sizes is None:
        sample_sizes = [500, 1000, 2000, 4000, 8000]

    print(f"\n{'=' * 70}")
    print(f" {method_name} 高维 √n-一致性验证")
    print(f" 样本量: {sample_sizes}")
    print(f" 每个样本量实验 {n_experiments_per_size} 次")
    print(f"{'=' * 70}")

    consistency_results = []

    for n_s in sample_sizes:
        print(f"\n--- 样本量 n = {n_s} ---")
        biases = []

        for exp_i in range(n_experiments_per_size):
            try:
                data_seed = exp_i * 13 + n_s
                D, Y, X_ctrl_hd, _ = generate_highdim_data(
                    dag_info,
                    n_samples=n_s,
                    noise_scale=noise_scale,
                    noise_type=noise_type,
                    add_time_lag=add_time_lag,
                    data_seed=data_seed,
                )

                theta_hat, se, _, _ = estimator_fn(Y, D, X_ctrl_hd, data_seed)
                biases.append(theta_hat - ate_true)
            except Exception:
                pass

        if biases:
            rmse = float(np.sqrt(np.mean(np.array(biases) ** 2)))
            mean_bias = float(np.mean(biases))
            print(f"  n={n_s:>5d}  RMSE={rmse:.6f}  Bias={mean_bias:+.6f}  "
                  f"(成功 {len(biases)}/{n_experiments_per_size})")
            consistency_results.append({
                "method": method_name,
                "n_samples": n_s,
                "rmse": round(rmse, 6),
                "mean_bias": round(mean_bias, 6),
                "n_success": len(biases),
            })

    if len(consistency_results) >= 2:
        log_n = np.log([r["n_samples"] for r in consistency_results])
        log_rmse = np.log([r["rmse"] + 1e-12 for r in consistency_results])
        slope, intercept, r_value, _, _ = stats.linregress(log_n, log_rmse)

        print(f"\n{'─' * 60}")
        print(f"{method_name} 高维 √n-一致性检验")
        print(f"{'─' * 60}")
        print(f"  log(RMSE) = {slope:.3f} × log(n) + {intercept:.3f}")
        print(f"  R² = {r_value ** 2:.3f}")
        print(f"  斜率 = {slope:.3f}  (理论值: -0.5)")
        if -0.8 < slope < -0.2:
            print(f"  ✓ 斜率 {slope:.3f} 接近理论值 -0.5，支持 √n-一致性")
        else:
            print(f"  ⚠ 斜率 {slope:.3f} 偏离理论值 -0.5")
        print(f"{'─' * 60}")

    return pd.DataFrame(consistency_results)


# ═══════════════════════════════════════════════════════════════════
#  透传原版公共函数（供脚本直接使用）
# ═══════════════════════════════════════════════════════════════════

# DML 交叉拟合估计（传统 ML 方法）
dml_estimate_cross_fitting = dvc.dml_estimate_cross_fitting
fit_ml_model = dvc.fit_ml_model
save_results = dvc.save_results
summarize_monte_carlo = dvc.summarize_monte_carlo
