"""
dml_validation_common.py
========================
DML 理论验证共享模块 —— 供所有验证脚本（基线 + 创新方案）复用

包含：
  1. 真实 ATE 计算（已修正 DGP 错配漏洞，使用 do_interventions 干预仿真）
  2. DAG 约束工具（强制 T→Y 路径线性化，满足 PLM 假设）
  3. 正确的 DML 标准误聚合（中位数聚合 + 方差校正，Chernozhukov 2018）
  4. 蒙特卡洛验证框架
  5. 实验结果输出
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx

warnings.filterwarnings("ignore")

# ─── 路径配置 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
CAUSAL_DISCOVERY_DIR = os.path.join(REPO_ROOT, "因果的发现算法理论验证")

sys.path.insert(0, CAUSAL_DISCOVERY_DIR)
from synthetic_dag_generator import SyntheticDAGGenerator

OUT_DIR = os.path.join(BASE_DIR, "DML理论验证")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  边函数计算（与 SyntheticDAGGenerator._compute_causal_contribution 一致）
# ═══════════════════════════════════════════════════════════════════

def apply_edge_func(func_info: dict, parent_val: float) -> float:
    """应用单条边的因果函数"""
    func_type = func_info["type"]
    params = func_info["params"]

    if func_type == "linear":
        return params["a"] * parent_val
    elif func_type == "sin":
        return np.sin(params["b"] * parent_val)
    elif func_type == "exp":
        return np.exp(-params["c"] * np.abs(parent_val))
    elif func_type == "saturation":
        return params["x_max"] * (1 - np.exp(-params["k"] * np.abs(parent_val)))
    elif func_type == "threshold":
        return 1 / (1 + np.exp(-params["slope"] * (parent_val - params["threshold"])))
    elif func_type == "inverted_u":
        return np.exp(-((parent_val - params["optimal"]) ** 2)
                      / (2 * params["width"] ** 2))
    elif func_type == "poly":
        return params["a"] * parent_val ** 2 + params["b"] * parent_val
    return 0.0


# ═══════════════════════════════════════════════════════════════════
#  DAG 约束：强制 T→Y 路径线性化（修正漏洞二）
# ═══════════════════════════════════════════════════════════════════

def enforce_linear_treatment_paths(
    adj: np.ndarray,
    edge_funcs: dict,
    treatment_idx: int,
    outcome_idx: int,
) -> dict:
    """
    强制 Treatment → Outcome 的所有有向路径上的边为线性函数。

    PLM（偏线性模型）假设因果效应 θ 是常数（线性的），
    即 Y = θ·D + g(X) + ε。如果 T→Y 路径上存在非线性边，
    PLM 估计器 θ̂ 会与仿真 ATE 产生系统偏差（非 DML 的问题，
    而是模型假设不满足）。

    此函数将 T→Y 所有有向路径上的非线性边替换为线性边，
    保留原始参数中的线性成分（如 poly 的 b 参数）。
    混杂路径 (X→T, X→Y) 上的边不受影响，可以是任意非线性。

    参数:
        adj:           邻接矩阵
        edge_funcs:    边函数字典（会被修改）
        treatment_idx: 处理变量索引
        outcome_idx:   结果变量索引

    返回:
        edge_funcs:    修改后的边函数字典（原地修改并返回）
    """
    G = nx.DiGraph()
    n = adj.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(i, j)

    if not nx.has_path(G, treatment_idx, outcome_idx):
        return edge_funcs

    # 找到 T→Y 所有有向路径上的所有边
    edges_on_causal_paths = set()
    for path in nx.all_simple_paths(G, treatment_idx, outcome_idx):
        for k in range(len(path) - 1):
            edges_on_causal_paths.add((path[k], path[k + 1]))

    # 将这些边强制替换为线性
    rng = np.random.RandomState(42)
    for edge in edges_on_causal_paths:
        if edge not in edge_funcs:
            continue
        func_info = edge_funcs[edge]
        if func_info["type"] == "linear":
            continue  # 已经是线性，不需要修改

        # 提取线性成分
        if func_info["type"] == "poly":
            # poly: a*x² + b*x → 保留 b（线性项系数）
            linear_coeff = func_info["params"]["b"]
        else:
            # 其他非线性函数：用随机线性系数替代
            linear_coeff = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])

        edge_funcs[edge] = {
            "type": "linear",
            "params": {"a": linear_coeff},
        }

    return edge_funcs


# ═══════════════════════════════════════════════════════════════════
#  真实 ATE 计算（修正漏洞一：使用 do_interventions 复用原生 DGP）
# ═══════════════════════════════════════════════════════════════════

def compute_true_ate_linear(
    adj: np.ndarray,
    edge_funcs: dict,
    treatment_idx: int,
    outcome_idx: int,
) -> float:
    """
    解析计算线性 SCM 下的真实 ATE。
    总因果效应 = 沿所有 T→Y 有向路径的系数乘积之和。
    仅考虑全线性路径。
    """
    G = nx.DiGraph()
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(i, j)

    if not nx.has_path(G, treatment_idx, outcome_idx):
        return 0.0

    total_effect = 0.0
    for path in nx.all_simple_paths(G, treatment_idx, outcome_idx):
        path_coeff = 1.0
        all_linear = True
        for k in range(len(path) - 1):
            edge_key = (path[k], path[k + 1])
            if edge_key not in edge_funcs:
                all_linear = False
                break
            func_info = edge_funcs[edge_key]
            if func_info["type"] == "linear":
                path_coeff *= func_info["params"]["a"]
            else:
                all_linear = False
                break
        if all_linear:
            total_effect += path_coeff

    return total_effect


def compute_true_ate_simulation(
    gen: SyntheticDAGGenerator,
    adj: np.ndarray,
    edge_funcs: dict,
    treatment_idx: int,
    outcome_idx: int,
    n_samples: int = 10000,
    delta: float = 1.0,
    noise_scale: float = 0.1,
    noise_type: str = "gaussian",
) -> float:
    """
    仿真计算真实 ATE（修正版：复用 SyntheticDAGGenerator.generate_data 的原生 DGP）。

    原理（do-calculus 干预仿真）：
      ATE = E[Y | do(T = t + δ)] - E[Y | do(T = t)] / δ

    修正要点（对应漏洞一）：
      - 不再在外部重写 DGP（旧版硬编码高斯噪声，与异方差/重尾噪声不一致）
      - 使用 generate_data 的 do_interventions 参数实现干预
      - 两次生成使用相同种子，保证 RNG 序列同步（噪声一致）
      - 干预仅通过因果路径传播，非因果路径被固定

    参数:
        gen:           SyntheticDAGGenerator 实例（用于获取 seed 和 n_nodes）
        adj:           邻接矩阵
        edge_funcs:    边函数字典
        treatment_idx: 处理变量索引
        outcome_idx:   结果变量索引
        n_samples:     仿真样本数（越大越精确）
        delta:         干预增量
        noise_scale:   噪声标准差
        noise_type:    噪声类型（与数据生成一致）

    返回:
        true_ate: 真实平均处理效应（每单位 T 变化对 Y 的因果效应）
    """
    ate_seed = gen.seed + 9999

    # Step 1: 生成基准数据（使用 generate_data 原生 DGP）
    gen_base = SyntheticDAGGenerator(n_nodes=gen.n_nodes, seed=ate_seed)
    X_base = gen_base.generate_data(
        adj, edge_funcs,
        n_samples=n_samples,
        noise_scale=noise_scale,
        noise_type=noise_type,
        add_time_lag=False,
    )

    # Step 2: 生成干预数据 do(T = T_base + δ)
    # 使用相同种子，保证 RNG 序列完全同步（外生噪声一致）
    gen_int = SyntheticDAGGenerator(n_nodes=gen.n_nodes, seed=ate_seed)
    intervention_values = X_base[:, treatment_idx] + delta
    X_int = gen_int.generate_data(
        adj, edge_funcs,
        n_samples=n_samples,
        noise_scale=noise_scale,
        noise_type=noise_type,
        add_time_lag=False,
        do_interventions={treatment_idx: intervention_values},
    )

    # Step 3: ATE = E[Y(do(T+δ)) - Y(do(T))] / δ
    Y_base = X_base[:, outcome_idx]
    Y_int = X_int[:, outcome_idx]
    ate = float(np.mean(Y_int - Y_base)) / delta

    return ate


def compute_true_ate(
    gen: SyntheticDAGGenerator,
    adj: np.ndarray,
    edge_funcs: dict,
    treatment_idx: int,
    outcome_idx: int,
    noise_scale: float = 0.1,
    noise_type: str = "gaussian",
) -> float:
    """
    计算真实 ATE。

    修正版（对应漏洞四）：
      - 仿真 do-calculus 的结果始终是基准真相
      - 解析值仅用于纯线性图的断言检查（Assertion），不作为兜底
      - 移除了旧版中 "仿真=0 则用解析值替换" 的危险逻辑

    返回:
        ate_true: 仿真 ATE（基准真相）
    """
    ate_analytic = compute_true_ate_linear(adj, edge_funcs, treatment_idx, outcome_idx)
    ate_sim = compute_true_ate_simulation(
        gen, adj, edge_funcs, treatment_idx, outcome_idx,
        noise_scale=noise_scale,
        noise_type=noise_type,
    )

    # 纯线性路径时进行交叉验证（仅用于日志，不影响返回值）
    if abs(ate_analytic) > 1e-6 and abs(ate_sim) > 1e-6:
        relative_diff = abs(ate_analytic - ate_sim) / max(abs(ate_analytic), abs(ate_sim))
        if relative_diff > 0.2:
            warnings.warn(
                f"解析 ATE ({ate_analytic:.4f}) 与仿真 ATE ({ate_sim:.4f}) "
                f"差异较大 ({relative_diff:.1%})，可能存在非线性路径"
            )

    return ate_sim


# ═══════════════════════════════════════════════════════════════════
#  DAG 角色转换（兼容 DAG 解析器 load_dag_roles 输出格式）
# ═══════════════════════════════════════════════════════════════════

def convert_roles_to_dag_parser_format(
    roles: dict,
    treatment_name: str,
    node_names: list,
) -> dict:
    """
    将 SyntheticDAGGenerator.identify_causal_roles 的输出转换为
    与 DAG 解析器 (load_dag_roles) 兼容的格式。
    """
    return {
        treatment_name: {
            "confounders": {node_names[i] for i in roles["confounders"]},
            "mediators":   {node_names[i] for i in roles["mediators"]},
            "colliders":   {node_names[i] for i in roles["colliders"]},
            "instruments": {node_names[i] for i in roles["instruments"]},
        }
    }


# ═══════════════════════════════════════════════════════════════════
#  选择 Treatment-Outcome 对
# ═══════════════════════════════════════════════════════════════════

def select_treatment_outcome(
    gen: SyntheticDAGGenerator,
    adj: np.ndarray,
    edge_funcs: dict,
    layer_indices: list = None,
    seed: int = 42,
) -> tuple:
    """
    从 DAG 中选择合适的 treatment-outcome 对。

    选择标准：
    1. treatment → outcome 之间有因果路径
    2. 存在混杂变量（确保 DML 的必要性）
    3. 优先选择不同层的变量

    返回:
        (treatment_idx, outcome_idx, roles)
    """
    rng = np.random.RandomState(seed)
    G = nx.DiGraph()
    n = adj.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(i, j)

    candidates = []

    if layer_indices is not None and len(layer_indices) >= 3:
        mid_layers = layer_indices[1:-1]
        last_layer = layer_indices[-1]
        treatment_candidates = [node for layer in mid_layers for node in layer]
        outcome_candidates = last_layer
    else:
        treatment_candidates = list(range(n))
        outcome_candidates = list(range(n))

    for t_idx in treatment_candidates:
        for y_idx in outcome_candidates:
            if t_idx == y_idx:
                continue
            if not nx.has_path(G, t_idx, y_idx):
                continue
            roles = gen.identify_causal_roles(adj, t_idx, y_idx)
            n_confounders = len(roles["confounders"])
            if n_confounders > 0:
                candidates.append((t_idx, y_idx, n_confounders, roles))

    if not candidates:
        for t_idx in treatment_candidates:
            for y_idx in outcome_candidates:
                if t_idx == y_idx:
                    continue
                if nx.has_path(G, t_idx, y_idx):
                    roles = gen.identify_causal_roles(adj, t_idx, y_idx)
                    candidates.append((t_idx, y_idx, 0, roles))
                    break
            if candidates:
                break

    if not candidates:
        raise ValueError("无法找到合适的 treatment-outcome 对")

    candidates.sort(key=lambda x: -x[2])
    top_k = min(5, len(candidates))
    chosen = candidates[rng.choice(top_k)]
    return chosen[0], chosen[1], chosen[3]


# ═══════════════════════════════════════════════════════════════════
#  构建控制变量（后门调整集）
# ═══════════════════════════════════════════════════════════════════

def build_adjustment_variables(
    gen: SyntheticDAGGenerator,
    adj: np.ndarray,
    X_data: np.ndarray,
    treatment_idx: int,
    outcome_idx: int,
    n_nodes: int,
) -> tuple:
    """
    构建 DML 的控制变量矩阵。

    返回:
        (X_ctrl, confounder_indices): 控制变量矩阵和索引列表
    """
    adjustment_set = gen.find_adjustment_set(adj, treatment_idx, outcome_idx)
    confounder_indices = sorted(adjustment_set)

    if len(confounder_indices) == 0:
        G_dag = nx.DiGraph(adj)
        t_ancestors = nx.ancestors(G_dag, treatment_idx)
        t_descendants = nx.descendants(G_dag, treatment_idx)
        y_descendants = nx.descendants(G_dag, outcome_idx)
        safe_vars = [
            v for v in range(n_nodes)
            if v != treatment_idx and v != outcome_idx
            and v not in t_descendants and v not in y_descendants
            and v in t_ancestors
        ]
        if safe_vars:
            confounder_indices = sorted(safe_vars)

    if len(confounder_indices) > 0:
        X_ctrl = X_data[:, confounder_indices]
    else:
        warnings.warn(
            f"Treatment={treatment_idx}, Outcome={outcome_idx}: "
            "无混杂变量且无安全祖先，DML 退化为朴素回归"
        )
        X_ctrl = np.ones((X_data.shape[0], 1))

    return X_ctrl, confounder_indices


# ═══════════════════════════════════════════════════════════════════
#  ML 模型训练
# ═══════════════════════════════════════════════════════════════════

def fit_ml_model(X_train, y_train, method, seed):
    """训练一个机器学习模型用于 nuisance function 估计"""
    if method == "forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    elif method == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed,
        )
    elif method == "lasso":
        from sklearn.linear_model import LassoCV
        model = LassoCV(cv=3, random_state=seed)
    elif method == "linear":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    else:
        raise ValueError(f"未知的 ML 方法: {method}")

    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════════
#  DML 估计器（修正漏洞三：正确的重复交叉拟合 SE 聚合）
# ═══════════════════════════════════════════════════════════════════

def dml_estimate_cross_fitting(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    ml_method: str = "forest",
    seed: int = 42,
    n_repeats: int = 5,
) -> tuple:
    """
    DML 交叉拟合估计器（Chernozhukov et al., 2018）。

    偏线性模型：Y = θ₀D + g₀(X) + ε,  D = m₀(X) + V
    交叉拟合步骤：
      1. 将数据分为 K 折
      2. 对第 k 折：用其余 K-1 折训练 ĝ_k 和 m̂_k
      3. 在第 k 折上计算残差：Ỹ = Y - ĝ_k(X), D̃ = D - m̂_k(X)
      4. 汇总所有折的残差，计算 θ̂ = Σ(D̃Ỹ) / Σ(D̃²)

    标准误修正（对应漏洞三）：
      使用 Chernozhukov et al. (2018) 推荐的中位数聚合公式：
        θ_final = Median(θ_b)
        V_final = Median(V_b + (θ_b - θ_final)²)
        SE = √V_final
      其中 V_b 是每次重复交叉拟合的 Neyman 方差估计，
      (θ_b - θ_final)² 是数据分割带来的额外方差。

    参数:
        Y:          结果变量 (n,)
        D:          处理变量 (n,)
        X:          控制变量（混杂变量）(n, p)
        n_folds:    交叉拟合折数
        ml_method:  机器学习方法 ('forest', 'linear', 'lasso', 'gbm')
        seed:       随机种子
        n_repeats:  重复交叉拟合次数

    返回:
        (theta, se, ci_lower, ci_upper): 估计值、标准误、95% CI
    """
    from sklearn.model_selection import KFold

    def _single_dml_with_se(seed_k):
        """单次交叉拟合 DML 估计（含 Neyman SE）"""
        n = len(Y)
        res_Y = np.zeros(n)
        res_D = np.zeros(n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_k)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            D_train, D_test = D[train_idx], D[test_idx]
            model_Y = fit_ml_model(X_train, Y_train, ml_method, seed_k)
            model_D = fit_ml_model(X_train, D_train, ml_method, seed_k)
            res_Y[test_idx] = Y_test - model_Y.predict(X_test)
            res_D[test_idx] = D_test - model_D.predict(X_test)

        denom = np.sum(res_D ** 2) + 1e-12
        theta_k = np.sum(res_D * res_Y) / denom

        # Neyman 式解析 SE
        psi = res_D * (res_Y - theta_k * res_D)
        J = np.mean(res_D ** 2)
        var_neyman = float(np.mean(psi ** 2) / (n * J ** 2 + 1e-12))
        return theta_k, var_neyman

    # 重复交叉拟合（不同随机分折）
    theta_boots = []
    var_boots = []
    for b in range(n_repeats):
        boot_seed = seed * 1000 + b * 7
        theta_b, var_b = _single_dml_with_se(boot_seed)
        theta_boots.append(theta_b)
        var_boots.append(var_b)

    theta_boots = np.array(theta_boots)
    var_boots = np.array(var_boots)

    # 中位数聚合（修正漏洞三）
    theta_final = float(np.median(theta_boots))
    # 结合抽样方差和分割方差
    var_final = float(np.median(var_boots + (theta_boots - theta_final) ** 2))
    se_final = np.sqrt(max(var_final, 1e-12))

    ci_lower = theta_final - 1.96 * se_final
    ci_upper = theta_final + 1.96 * se_final

    return float(theta_final), se_final, float(ci_lower), float(ci_upper)


# ═══════════════════════════════════════════════════════════════════
#  固定 DAG 工具
# ═══════════════════════════════════════════════════════════════════

def setup_fixed_dag(
    n_nodes: int = 20,
    graph_type: str = "layered",
    use_industrial: bool = False,
    dag_seed: int = 42,
    enforce_linear_ty: bool = True,
) -> dict:
    """
    创建并配置固定的 DAG 结构，供蒙特卡洛实验使用。

    返回:
        dict 包含 gen_base, adj_true, edge_funcs, layer_indices,
             t_idx, y_idx, roles, confounder_indices, ate_true
    """
    gen_base = SyntheticDAGGenerator(n_nodes=n_nodes, seed=dag_seed)

    if graph_type == "er":
        adj_true = gen_base.generate_er_dag()
        layer_indices = None
    elif graph_type == "scale_free":
        adj_true = gen_base.generate_scale_free_dag()
        layer_indices = None
    elif graph_type == "layered":
        adj_true, layer_indices = gen_base.generate_layered_industrial_dag()
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    edge_funcs = gen_base.assign_edge_functions(adj_true, layer_indices, use_industrial)

    t_idx, y_idx, roles = select_treatment_outcome(
        gen_base, adj_true, edge_funcs, layer_indices, seed=dag_seed,
    )

    # 强制 T→Y 路径线性化（满足 PLM 假设）
    if enforce_linear_ty:
        edge_funcs = enforce_linear_treatment_paths(
            adj_true, edge_funcs, t_idx, y_idx,
        )

    # 构建控制变量索引
    adjustment_set = gen_base.find_adjustment_set(adj_true, t_idx, y_idx)
    confounder_indices = sorted(adjustment_set)
    if len(confounder_indices) == 0:
        G_dag = nx.DiGraph(adj_true)
        t_ancestors = nx.ancestors(G_dag, t_idx)
        t_descendants = nx.descendants(G_dag, t_idx)
        y_descendants = nx.descendants(G_dag, y_idx)
        safe_vars = [
            v for v in range(n_nodes)
            if v != t_idx and v != y_idx
            and v not in t_descendants and v not in y_descendants
            and v in t_ancestors
        ]
        if safe_vars:
            confounder_indices = sorted(safe_vars)

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


def compute_ate_for_dag(dag_info: dict, noise_scale: float, noise_type: str) -> float:
    """基于固定 DAG 计算真实 ATE"""
    return compute_true_ate(
        dag_info["gen_base"],
        dag_info["adj_true"],
        dag_info["edge_funcs"],
        dag_info["t_idx"],
        dag_info["y_idx"],
        noise_scale=noise_scale,
        noise_type=noise_type,
    )


# ═══════════════════════════════════════════════════════════════════
#  蒙特卡洛验证框架
# ═══════════════════════════════════════════════════════════════════

def run_monte_carlo(
    estimator_fn,
    dag_info: dict,
    ate_true: float,
    n_experiments: int = 200,
    n_samples: int = 2000,
    noise_scale: float = 0.3,
    noise_type: str = "gaussian",
    tag: str = "",
    method_name: str = "DML",
):
    """
    蒙特卡洛验证框架：固定 DAG 结构，多次独立生成数据并运行估计器。

    参数:
        estimator_fn:  估计器函数，签名为
                       (Y, D, X_ctrl, seed) -> (theta_hat, se, ci_lower, ci_upper)
        dag_info:      setup_fixed_dag 的返回值
        ate_true:      真实 ATE
        n_experiments: 实验次数
        n_samples:     每次实验的样本量
        noise_scale:   噪声标准差
        noise_type:    噪声类型
        tag:           输出文件标签
        method_name:   方法名称

    返回:
        df: 实验结果 DataFrame
    """
    tag_str = f"_{tag}" if tag else ""
    n_nodes = dag_info["gen_base"].n_nodes
    adj_true = dag_info["adj_true"]
    edge_funcs = dag_info["edge_funcs"]
    t_idx = dag_info["t_idx"]
    y_idx = dag_info["y_idx"]
    roles = dag_info["roles"]
    confounder_indices = dag_info["confounder_indices"]

    print(f"\n{'=' * 70}")
    print(f" {method_name} 蒙特卡洛理论验证{tag_str}")
    print(f" 实验次数: {n_experiments} | 样本量: {n_samples}")
    print(f" 噪声: {noise_type} | 真实 ATE: {ate_true:.6f}")
    print(f" Treatment: X_{t_idx}, Outcome: X_{y_idx}")
    print(f" 混杂变量: {len(roles['confounders'])} 个  调整集: {len(confounder_indices)} 个")
    print(f"{'=' * 70}")

    results = []
    n_success = 0
    n_fail = 0
    t0 = time.perf_counter()

    for exp_i in range(n_experiments):
        try:
            data_seed = exp_i * 13 + 1000
            gen_data = SyntheticDAGGenerator(n_nodes=n_nodes, seed=data_seed)
            X_data = gen_data.generate_data(
                adj_true, edge_funcs,
                n_samples=n_samples,
                noise_scale=noise_scale,
                noise_type=noise_type,
                add_time_lag=False,
            )

            D = X_data[:, t_idx]
            Y = X_data[:, y_idx]

            if len(confounder_indices) > 0:
                X_ctrl = X_data[:, confounder_indices]
            else:
                X_ctrl = np.ones((n_samples, 1))

            theta_hat, se, ci_lower, ci_upper = estimator_fn(
                Y, D, X_ctrl, data_seed,
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
        return pd.DataFrame()

    df = pd.DataFrame(results)
    summary = summarize_monte_carlo(df, method_name)
    return df, summary


def summarize_monte_carlo(df: pd.DataFrame, method_name: str) -> dict:
    """汇总蒙特卡洛实验结果"""
    biases = df["bias"].values
    covers = df["covers_true"].values
    theta_hats = df["theta_hat"].values

    mean_bias = float(np.mean(biases))
    median_bias = float(np.median(biases))
    rmse = float(np.sqrt(np.mean(biases ** 2)))
    coverage = float(np.mean(covers))
    mean_se = float(df["se"].mean())

    if len(biases) > 1 and np.std(biases) > 1e-12:
        bias_t_stat = mean_bias / (np.std(biases) / np.sqrt(len(biases)))
        bias_p_value = float(2 * (1 - stats.norm.cdf(abs(bias_t_stat))))
    else:
        bias_t_stat = 0.0
        bias_p_value = 1.0

    print(f"\n{'─' * 60}")
    print(f"{method_name} 理论验证汇总")
    print(f"{'─' * 60}")
    print(f"  实验次数:              {len(df)}")
    print(f"  平均偏差 (Mean Bias):  {mean_bias:+.6f}")
    print(f"  中位偏差 (Median Bias):{median_bias:+.6f}")
    print(f"  RMSE:                  {rmse:.6f}")
    print(f"  95% CI 覆盖率:         {coverage:.1%}  (理论值: 95%)")
    print(f"  平均标准误:            {mean_se:.6f}")
    print(f"  偏差 t 统计量:         {bias_t_stat:.3f}")
    print(f"  偏差 p 值:             {bias_p_value:.4f}")
    print(f"{'─' * 60}")

    summary = {
        "method": method_name,
        "n_experiments": len(df),
        "mean_bias": round(mean_bias, 6),
        "median_bias": round(median_bias, 6),
        "rmse": round(rmse, 6),
        "coverage_95": round(coverage, 4),
        "mean_se": round(mean_se, 6),
        "bias_t_stat": round(bias_t_stat, 3),
        "bias_p_value": round(bias_p_value, 4),
    }
    return summary


def save_results(df: pd.DataFrame, summary: dict, filename_prefix: str):
    """保存实验结果和汇总"""
    csv_path = os.path.join(OUT_DIR, f"{filename_prefix}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = os.path.join(OUT_DIR, f"{filename_prefix}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  结果已保存：{csv_path}")
    print(f"  汇总已保存：{json_path}")


# ═══════════════════════════════════════════════════════════════════
#  √n-一致性验证框架
# ═══════════════════════════════════════════════════════════════════

def run_consistency_validation(
    estimator_fn,
    dag_info: dict,
    ate_true: float,
    sample_sizes: list = None,
    n_experiments_per_size: int = 50,
    noise_scale: float = 0.3,
    noise_type: str = "gaussian",
    method_name: str = "DML",
):
    """√n-一致性验证：随着样本量增大，RMSE 应以 1/√n 速率下降。"""
    if sample_sizes is None:
        sample_sizes = [500, 1000, 2000, 4000, 8000]

    n_nodes = dag_info["gen_base"].n_nodes
    adj_true = dag_info["adj_true"]
    edge_funcs = dag_info["edge_funcs"]
    t_idx = dag_info["t_idx"]
    y_idx = dag_info["y_idx"]
    confounder_indices = dag_info["confounder_indices"]

    print(f"\n{'=' * 70}")
    print(f" {method_name} √n-一致性验证")
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
                gen_data = SyntheticDAGGenerator(n_nodes=n_nodes, seed=data_seed)
                X_data = gen_data.generate_data(
                    adj_true, edge_funcs,
                    n_samples=n_s,
                    noise_scale=noise_scale,
                    noise_type=noise_type,
                    add_time_lag=False,
                )
                D = X_data[:, t_idx]
                Y = X_data[:, y_idx]
                if len(confounder_indices) > 0:
                    X_ctrl = X_data[:, confounder_indices]
                else:
                    X_ctrl = np.ones((n_s, 1))

                theta_hat, se, _, _ = estimator_fn(Y, D, X_ctrl, data_seed)
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
        print(f"{method_name} √n-一致性检验")
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
