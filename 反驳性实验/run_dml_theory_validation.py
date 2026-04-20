"""
run_dml_theory_validation.py
============================
DML（双重机器学习）理论验证 —— 基于合成 DAG 数据生成器

═══════════════════════════════════════════════════════════════════
  核心目标：在已知真实因果效应的合成数据上验证 DML 的理论性质
═══════════════════════════════════════════════════════════════════

  本脚本复用「因果的发现算法理论验证」中的 SyntheticDAGGenerator 生成
  已知 DAG 结构和因果关系的合成数据，并使用与反驳性实验相同的 DAG 解析
  接口（load_dag_roles 兼容格式），在此基础上运行 DML 估计，验证：

  1. 无偏性（Unbiasedness）：E[θ̂] ≈ θ_true
  2. √n-一致性（√n-Consistency）：RMSE ∝ 1/√n
  3. 置信区间覆盖率（Coverage）：95% CI 覆盖 θ_true 的频率 ≈ 95%
  4. 渐近正态性（Asymptotic Normality）：(θ̂ - θ_true)/SE ∼ N(0,1)
  5. 混杂控制有效性：正确控制混杂 vs 遗漏混杂 的效果对比

  数据生成流程：
    SyntheticDAGGenerator
      → generate_complete_synthetic_dataset()     # 生成 DAG + 数据
      → identify_causal_roles(adj, T_idx, Y_idx)  # 识别因果角色
      → find_adjustment_set(adj, T_idx, Y_idx)     # 找后门调整集
    这些接口与 DAG 解析器 (analyze_dag_causal_roles_v4_1.py) 的角色
    定义完全一致：混杂因子、中介变量、碰撞节点、工具变量。

  DML 估计方法（Chernozhukov et al., 2018）：
    偏线性模型：Y = θ₀D + g₀(X) + ε,  D = m₀(X) + V
    交叉拟合：K 折前向分割，每折只用训练集估计 ĝ、m̂
    残差回归：θ̂ = Σ(D̃ᵢ Ỹᵢ) / Σ(D̃ᵢ²)
    其中 Ỹ = Y - ĝ(X),  D̃ = D - m̂(X)

  真实因果效应计算：
    线性 SCM：解析计算（沿所有 T→Y 有向路径的系数乘积之和）
    非线性 SCM：仿真计算（do-calculus 干预仿真）

用法：
  # 快速验证（小规模）
  python run_dml_theory_validation.py --mode quick

  # 完整蒙特卡洛验证（默认 200 次实验）
  python run_dml_theory_validation.py --mode full

  # √n-一致性验证（不同样本量）
  python run_dml_theory_validation.py --mode consistency

  # 混杂控制对比实验
  python run_dml_theory_validation.py --mode confounding

  # 全部实验
  python run_dml_theory_validation.py --mode all

  # 指定图类型和函数类型
  python run_dml_theory_validation.py --mode full --graph_type layered --use_industrial
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

# 输出目录
OUT_DIR = os.path.join(BASE_DIR, "DML理论验证")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  真实因果效应计算
# ═══════════════════════════════════════════════════════════════════

def compute_true_ate_linear(
    adj: np.ndarray,
    edge_funcs: dict,
    treatment_idx: int,
    outcome_idx: int,
) -> float:
    """
    解析计算线性 SCM 下的真实 ATE。

    对于线性 SCM：X_j = Σ_{i∈pa(j)} a_{i,j} X_i + noise_j
    总因果效应 = 沿所有 T→Y 有向路径的系数乘积之和。

    如果某条路径上存在非线性边，该路径的贡献设为 0（仅保留线性路径）。

    参数:
        adj:           邻接矩阵
        edge_funcs:    边函数字典 {(i,j): {'type':..., 'params':...}}
        treatment_idx: 处理变量索引
        outcome_idx:   结果变量索引

    返回:
        true_ate: 真实平均处理效应
    """
    import networkx as nx

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
            elif func_info["type"] == "poly":
                # poly: a*x^2 + b*x → 在 x=0 附近的局部线性效应 ≈ b
                path_coeff *= func_info["params"]["b"]
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
    n_samples: int = 5000,
    delta: float = 1.0,
    noise_scale: float = 0.1,
) -> float:
    """
    仿真计算真实 ATE（适用于非线性 SCM）。

    原理（do-calculus 干预仿真）：
      ATE = E[Y | do(T = t + δ)] - E[Y | do(T = t)]

    实现（正确的干预仿真）：
      1. 生成基准数据 X_base（正常 DAG 数据生成，固定所有噪声）
      2. 生成干预数据 X_int：
         - 非 T 后代的节点：值与基准完全相同
         - T 节点：X_int[T] = X_base[T] + δ
         - T 的后代：按拓扑顺序，用 X_int 中的父节点值重新计算
      3. ATE = mean(Y_int - Y_base) / δ

    此方法确保干预仅通过因果路径传播，非因果路径被固定。

    参数:
        gen:           SyntheticDAGGenerator 实例
        adj:           邻接矩阵
        edge_funcs:    边函数字典
        treatment_idx: 处理变量索引
        outcome_idx:   结果变量索引
        n_samples:     仿真样本数
        delta:         干预增量
        noise_scale:   噪声标准差

    返回:
        true_ate: 真实平均处理效应（每单位 T 变化对 Y 的因果效应）
    """
    import networkx as nx

    G = nx.DiGraph()
    n = adj.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(i, j)

    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXError:
        topo_order = list(range(n))

    # T 的所有后代（不含 T）
    t_descendants = nx.descendants(G, treatment_idx)

    # 预生成噪声矩阵（基准和干预共享相同噪声）
    rng = np.random.RandomState(gen.seed + 9999)
    noise_matrix = rng.randn(n_samples, n) * noise_scale

    # 1. 生成基准数据
    X_base = np.zeros((n_samples, n))
    for t_step in range(n_samples):
        for node in topo_order:
            parents = [i for i in range(n) if adj[i, node] > 0]
            if len(parents) == 0:
                X_base[t_step, node] = noise_matrix[t_step, node]
            else:
                contribution = 0.0
                for parent in parents:
                    func_info = edge_funcs[(parent, node)]
                    p_val = X_base[t_step, parent]
                    contribution += _apply_edge_func(func_info, p_val)
                if len(parents) > 1:
                    contribution /= len(parents)
                X_base[t_step, node] = contribution + noise_matrix[t_step, node]

    # 2. 生成干预数据：do(T = T_base + δ)
    X_int = X_base.copy()
    X_int[:, treatment_idx] = X_base[:, treatment_idx] + delta

    # 重新计算 T 的所有后代（按拓扑顺序）
    descendants_topo = [node for node in topo_order if node in t_descendants]
    for t_step in range(n_samples):
        for node in descendants_topo:
            parents = [i for i in range(n) if adj[i, node] > 0]
            contribution = 0.0
            for parent in parents:
                func_info = edge_funcs[(parent, node)]
                p_val = X_int[t_step, parent]
                contribution += _apply_edge_func(func_info, p_val)
            if len(parents) > 1:
                contribution /= len(parents)
            X_int[t_step, node] = contribution + noise_matrix[t_step, node]

    # 3. ATE = E[Y(do(T+δ)) - Y(do(T))] / δ
    Y_base = X_base[:, outcome_idx]
    Y_int = X_int[:, outcome_idx]
    ate = float(np.mean(Y_int - Y_base)) / delta

    return ate


def _apply_edge_func(func_info: dict, parent_val: float) -> float:
    """应用单条边的因果函数（与 SyntheticDAGGenerator._compute_causal_contribution 一致）"""
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


def compute_true_ate(
    gen: SyntheticDAGGenerator,
    adj: np.ndarray,
    edge_funcs: dict,
    treatment_idx: int,
    outcome_idx: int,
    noise_scale: float = 0.1,
) -> tuple:
    """
    计算真实 ATE，同时返回线性解析值和仿真值。

    返回:
        (ate_analytic, ate_sim): 线性解析 ATE 和仿真 ATE
    """
    ate_analytic = compute_true_ate_linear(adj, edge_funcs, treatment_idx, outcome_idx)
    ate_sim = compute_true_ate_simulation(
        gen, adj, edge_funcs, treatment_idx, outcome_idx,
        noise_scale=noise_scale,
    )
    return ate_analytic, ate_sim


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
    与 DAG 解析器 (analyze_dag_causal_roles_v4_1.py) 的 load_dag_roles
    兼容的格式。

    生成器输出: {'confounders': [idx...], 'mediators': [idx...], ...}
    DAG 解析器输出: {treatment_name: {'confounders': {name...}, ...}}

    参数:
        roles:          identify_causal_roles 的输出
        treatment_name: 处理变量名称
        node_names:     节点名称列表（索引→名称的映射）

    返回:
        dag_roles: 与 load_dag_roles 兼容的字典
    """
    dag_roles = {
        treatment_name: {
            "confounders": {node_names[i] for i in roles["confounders"]},
            "mediators":   {node_names[i] for i in roles["mediators"]},
            "colliders":   {node_names[i] for i in roles["colliders"]},
            "instruments": {node_names[i] for i in roles["instruments"]},
        }
    }
    return dag_roles


# ═══════════════════════════════════════════════════════════════════
#  DML 估计器（交叉拟合）
# ═══════════════════════════════════════════════════════════════════

def dml_estimate_cross_fitting(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    ml_method: str = "forest",
    seed: int = 42,
    n_bootstrap: int = 5,
) -> tuple:
    """
    DML 交叉拟合估计器（Chernozhukov et al., 2018）。

    偏线性模型：Y = θ₀D + g₀(X) + ε,  D = m₀(X) + V
    交叉拟合步骤：
      1. 将数据分为 K 折
      2. 对第 k 折：用其余 K-1 折训练 ĝ_k 和 m̂_k
      3. 在第 k 折上计算残差：Ỹ = Y - ĝ_k(X), D̃ = D - m̂_k(X)
      4. 汇总所有折的残差，计算 θ̂ = Σ(D̃Ỹ) / Σ(D̃²)

    标准误采用双重方案取最大值：
      - Neyman 式解析 SE（DML 理论推导）
      - 多次重复交叉拟合的 Bootstrap SE（更稳健）

    参数:
        Y:          结果变量 (n,)
        D:          处理变量 (n,)
        X:          控制变量（混杂变量）(n, p)
        n_folds:    交叉拟合折数
        ml_method:  机器学习方法 ('forest', 'linear', 'lasso')
        seed:       随机种子
        n_bootstrap: 重复交叉拟合次数（用于 Bootstrap SE）

    返回:
        (theta, se, ci_lower, ci_upper): 估计值、标准误、95% CI
    """
    from sklearn.model_selection import KFold

    def _single_dml(seed_k):
        """单次交叉拟合 DML 估计"""
        n = len(Y)
        res_Y = np.zeros(n)
        res_D = np.zeros(n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_k)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            D_train, D_test = D[train_idx], D[test_idx]
            model_Y = _fit_ml_model(X_train, Y_train, ml_method, seed_k)
            model_D = _fit_ml_model(X_train, D_train, ml_method, seed_k)
            res_Y[test_idx] = Y_test - model_Y.predict(X_test)
            res_D[test_idx] = D_test - model_D.predict(X_test)
        theta_k = np.sum(res_D * res_Y) / (np.sum(res_D ** 2) + 1e-12)
        return theta_k, res_Y, res_D

    # 主估计（使用基础种子）
    theta_main, res_Y_main, res_D_main = _single_dml(seed)

    # Neyman 式解析 SE
    n = len(Y)
    psi = res_D_main * (res_Y_main - theta_main * res_D_main)
    J = np.mean(res_D_main ** 2)
    se_neyman = float(np.sqrt(np.mean(psi ** 2) / (n * J ** 2 + 1e-12)))

    # 重复交叉拟合 Bootstrap SE（Chernozhukov et al. 推荐）
    theta_boots = [theta_main]
    for b in range(1, n_bootstrap):
        boot_seed = seed * 1000 + b * 7
        theta_b, _, _ = _single_dml(boot_seed)
        theta_boots.append(theta_b)
    theta_boots = np.array(theta_boots)
    se_boot = float(np.std(theta_boots))

    # 取两者最大值（更保守、更稳健的 SE 估计）
    se = max(se_neyman, se_boot)

    # 使用 Bootstrap 中位数作为点估计（更稳健）
    theta = float(np.median(theta_boots))

    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se

    return float(theta), se, float(ci_lower), float(ci_upper)


def _fit_ml_model(X_train, y_train, method, seed):
    """训练一个机器学习模型用于 nuisance function 估计"""
    if method == "forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
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
    3. 优先选择不同层的变量（模拟真实工业场景）

    返回:
        (treatment_idx, outcome_idx, roles)
    """
    import networkx as nx
    rng = np.random.RandomState(seed)

    G = nx.DiGraph()
    n = adj.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(i, j)

    # 候选 treatment-outcome 对
    candidates = []

    if layer_indices is not None and len(layer_indices) >= 3:
        # 分层图：treatment 从中间层选，outcome 从最后一层选
        mid_layers = layer_indices[1:-1]
        last_layer = layer_indices[-1]
        treatment_candidates = [n for layer in mid_layers for n in layer]
        outcome_candidates = last_layer
    else:
        # 非分层图：任选
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
        # 无混杂的备用方案：选择任何有因果路径的对
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

    # 按混杂变量数量排序，优先选择混杂较多的（更能体现 DML 价值）
    candidates.sort(key=lambda x: -x[2])
    # 从 top-5 中随机选一个
    top_k = min(5, len(candidates))
    chosen = candidates[rng.choice(top_k)]

    return chosen[0], chosen[1], chosen[3]


# ═══════════════════════════════════════════════════════════════════
#  单次 DML 实验
# ═══════════════════════════════════════════════════════════════════

def run_single_experiment(
    n_nodes: int = 20,
    n_samples: int = 2000,
    graph_type: str = "layered",
    noise_scale: float = 0.3,
    noise_type: str = "gaussian",
    use_industrial: bool = False,
    ml_method: str = "forest",
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    运行单次 DML 理论验证实验。

    返回:
        result: 包含真实 ATE、估计 ATE、标准误、覆盖率等指标的字典
    """
    # 1. 生成合成数据
    gen = SyntheticDAGGenerator(n_nodes=n_nodes, seed=seed)
    X_data, adj_true, metadata = gen.generate_complete_synthetic_dataset(
        graph_type=graph_type,
        n_samples=n_samples,
        noise_scale=noise_scale,
        noise_type=noise_type,
        add_time_lag=False,  # 截面数据，便于标准 DML
        use_industrial_functions=use_industrial,
    )

    edge_funcs = metadata["edge_funcs"]
    layer_indices = metadata.get("layer_indices")

    # 2. 选择 treatment-outcome 对
    t_idx, y_idx, roles = select_treatment_outcome(
        gen, adj_true, edge_funcs, layer_indices, seed=seed,
    )

    # 3. 计算真实 ATE
    ate_analytic, ate_sim = compute_true_ate(
        gen, adj_true, edge_funcs, t_idx, y_idx,
        noise_scale=noise_scale,
    )

    # 选择参考真实值
    # 仿真 ATE 是最可靠的参考值（正确处理非线性 SCM）
    # 解析 ATE 仅在全线性路径下有效，作为辅助参考
    ate_true = ate_sim
    ate_source = "simulation"
    # 如果仿真值接近 0 但解析值不为 0，可能仿真精度不够，使用解析值
    if abs(ate_sim) < 1e-6 and abs(ate_analytic) > 1e-6:
        ate_true = ate_analytic
        ate_source = "analytic"

    # 4. 识别因果角色（与 DAG 解析器兼容）
    node_names = [f"X_{i}" for i in range(n_nodes)]
    dag_roles = convert_roles_to_dag_parser_format(roles, node_names[t_idx], node_names)

    # 5. 构建 DML 输入
    D = X_data[:, t_idx]       # 处理变量
    Y = X_data[:, y_idx]       # 结果变量

    # 控制变量：混杂变量（后门调整集）
    # 注意：仅使用混杂变量，不包含中介变量（会阻断间接效应）或碰撞变量（会引入偏差）
    adjustment_set = gen.find_adjustment_set(adj_true, t_idx, y_idx)
    confounder_indices = sorted(adjustment_set)

    if len(confounder_indices) == 0:
        # 无混杂变量时，使用 T 和 Y 的共同祖先中非 T 后代的安全变量
        # 这些变量不会阻断因果路径也不会引入偏差
        import networkx as nx
        G_dag = nx.DiGraph(adj_true)
        t_ancestors = nx.ancestors(G_dag, t_idx)
        t_descendants = nx.descendants(G_dag, t_idx)
        y_descendants = nx.descendants(G_dag, y_idx)
        # 安全变量：T 的祖先中不是 T 后代也不是 Y 后代的节点
        safe_vars = []
        for v in range(n_nodes):
            if v == t_idx or v == y_idx:
                continue
            if v in t_descendants or v in y_descendants:
                continue  # 排除后代（含中介和碰撞）
            if v in t_ancestors:
                safe_vars.append(v)
        if not safe_vars:
            # 兜底：使用 T 的祖先（即使不存在共同祖先）
            safe_vars = [v for v in t_ancestors if v != t_idx and v != y_idx]
        if not safe_vars:
            # 最终兜底：生成一个随机无关变量作为占位符
            rng_placeholder = np.random.RandomState(seed + 5555)
            X_confounders = rng_placeholder.randn(n_samples, 1)
            confounder_indices = [-1]  # 标记占位
        else:
            confounder_indices = sorted(safe_vars)

    if confounder_indices and confounder_indices[0] != -1:
        X_confounders = X_data[:, confounder_indices]

    # 6. DML 估计
    theta_hat, se, ci_lower, ci_upper = dml_estimate_cross_fitting(
        Y, D, X_confounders,
        n_folds=n_folds,
        ml_method=ml_method,
        seed=seed,
    )

    # 7. 计算指标
    bias = theta_hat - ate_true
    covers = bool(ci_lower <= ate_true <= ci_upper)

    result = {
        "seed": seed,
        "n_samples": n_samples,
        "n_nodes": n_nodes,
        "graph_type": graph_type,
        "noise_type": noise_type,
        "ml_method": ml_method,
        "treatment_idx": t_idx,
        "outcome_idx": y_idx,
        "n_confounders": len(roles["confounders"]),
        "n_mediators": len(roles["mediators"]),
        "n_instruments": len(roles["instruments"]),
        "n_adjustment": len(confounder_indices),
        "ate_true": round(ate_true, 6),
        "ate_source": ate_source,
        "ate_analytic": round(ate_analytic, 6),
        "ate_simulation": round(ate_sim, 6),
        "theta_hat": round(theta_hat, 6),
        "se": round(se, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "bias": round(bias, 6),
        "covers_true": covers,
    }

    return result


# ═══════════════════════════════════════════════════════════════════
#  蒙特卡洛验证
# ═══════════════════════════════════════════════════════════════════

def run_monte_carlo_validation(
    n_experiments: int = 200,
    n_nodes: int = 20,
    n_samples: int = 2000,
    graph_type: str = "layered",
    noise_scale: float = 0.3,
    noise_type: str = "gaussian",
    use_industrial: bool = False,
    ml_method: str = "forest",
    n_folds: int = 5,
    tag: str = "",
):
    """
    蒙特卡洛实验：固定一个 DAG 结构，多次独立生成数据并运行 DML。

    关键设计：
      - 固定 DAG 结构和因果函数（不变的 DGP）
      - 每次实验只改变噪声实现（不同的随机样本）
      - 这是验证 DML 渐近理论的正确范式

    指标汇总：
      - 平均偏差（Mean Bias）
      - RMSE（均方根误差）
      - 95% CI 覆盖率（Coverage）
      - 偏差 t 检验（Bias ≈ 0 ?）
    """
    tag_str = f"_{tag}" if tag else ""
    print("\n" + "=" * 70)
    print(f" DML 蒙特卡洛理论验证{tag_str}")
    print(f" 实验次数: {n_experiments} | 样本量: {n_samples}")
    print(f" 图类型: {graph_type} | 噪声: {noise_type} | ML: {ml_method}")
    print(f" 工业函数: {use_industrial}")
    print("=" * 70)

    # ── 固定 DAG 结构和因果函数（所有实验共享）─────────────────
    dag_seed = 42
    gen_base = SyntheticDAGGenerator(n_nodes=n_nodes, seed=dag_seed)

    if graph_type == "er":
        adj_true = gen_base.generate_er_dag()
    elif graph_type == "scale_free":
        adj_true = gen_base.generate_scale_free_dag()
    elif graph_type == "layered":
        adj_true, layer_indices_base = gen_base.generate_layered_industrial_dag()
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    layer_indices = layer_indices_base if graph_type == "layered" else None
    edge_funcs = gen_base.assign_edge_functions(
        adj_true, layer_indices, use_industrial,
    )

    # ── 选择 treatment-outcome 对（固定）─────────────────────────
    t_idx, y_idx, roles = select_treatment_outcome(
        gen_base, adj_true, edge_funcs, layer_indices, seed=dag_seed,
    )

    # ── 计算真实 ATE（固定 DGP 下只需算一次）─────────────────────
    ate_analytic, ate_sim = compute_true_ate(
        gen_base, adj_true, edge_funcs, t_idx, y_idx,
        noise_scale=noise_scale,
    )
    ate_true = ate_sim
    ate_source = "simulation"
    if abs(ate_sim) < 1e-6 and abs(ate_analytic) > 1e-6:
        ate_true = ate_analytic
        ate_source = "analytic"

    # ── 构建控制变量索引（固定）─────────────────────────────────
    import networkx as nx
    adjustment_set = gen_base.find_adjustment_set(adj_true, t_idx, y_idx)
    confounder_indices = sorted(adjustment_set)

    if len(confounder_indices) == 0:
        G_dag = nx.DiGraph(adj_true)
        t_ancestors = nx.ancestors(G_dag, t_idx)
        t_descendants = nx.descendants(G_dag, t_idx)
        y_descendants = nx.descendants(G_dag, y_idx)
        safe_vars = [v for v in range(n_nodes)
                     if v != t_idx and v != y_idx
                     and v not in t_descendants
                     and v not in y_descendants
                     and v in t_ancestors]
        if safe_vars:
            confounder_indices = sorted(safe_vars)

    node_names = [f"X_{i}" for i in range(n_nodes)]
    dag_roles = convert_roles_to_dag_parser_format(roles, node_names[t_idx], node_names)

    print(f"\n固定 DAG 信息:")
    print(f"  节点数: {n_nodes}, 边数: {int(adj_true.sum())}")
    print(f"  Treatment: X_{t_idx}, Outcome: X_{y_idx}")
    print(f"  混杂变量: {len(roles['confounders'])} 个  "
          f"中介变量: {len(roles['mediators'])} 个")
    print(f"  调整集大小: {len(confounder_indices)}")
    print(f"  真实 ATE: {ate_true:.6f} ({ate_source})")
    print(f"  解析 ATE: {ate_analytic:.6f}  仿真 ATE: {ate_sim:.6f}")
    print(f"  DAG 角色（兼容 load_dag_roles 格式）: {dag_roles}")

    # ── 蒙特卡洛实验（仅改变数据采样种子）────────────────────────
    results = []
    n_success = 0
    n_fail = 0
    t0 = time.perf_counter()

    for exp_i in range(n_experiments):
        try:
            data_seed = exp_i * 13 + 1000  # 不同的数据种子
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
                rng_ph = np.random.RandomState(data_seed + 5555)
                X_ctrl = rng_ph.randn(n_samples, 1)

            theta_hat, se, ci_lower, ci_upper = dml_estimate_cross_fitting(
                Y, D, X_ctrl,
                n_folds=n_folds,
                ml_method=ml_method,
                seed=data_seed,
            )

            bias = theta_hat - ate_true
            covers = bool(ci_lower <= ate_true <= ci_upper)

            results.append({
                "experiment": exp_i,
                "seed": data_seed,
                "n_samples": n_samples,
                "graph_type": graph_type,
                "ml_method": ml_method,
                "treatment_idx": t_idx,
                "outcome_idx": y_idx,
                "n_confounders": len(roles["confounders"]),
                "n_adjustment": len(confounder_indices),
                "ate_true": round(ate_true, 6),
                "ate_source": ate_source,
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

    # ── 汇总统计 ────────────────────────────────────────────────
    biases = df["bias"].values
    covers = df["covers_true"].values
    theta_hats = df["theta_hat"].values
    ate_trues = df["ate_true"].values

    mean_bias = float(np.mean(biases))
    median_bias = float(np.median(biases))
    rmse = float(np.sqrt(np.mean(biases ** 2)))
    coverage = float(np.mean(covers))
    mean_se = float(df["se"].mean())

    # 偏差是否显著不为零的 t 检验
    if len(biases) > 1 and np.std(biases) > 1e-12:
        bias_t_stat = mean_bias / (np.std(biases) / np.sqrt(len(biases)))
        bias_p_value = float(2 * (1 - stats.norm.cdf(abs(bias_t_stat))))
    else:
        bias_t_stat = 0.0
        bias_p_value = 1.0

    print("\n" + "─" * 60)
    print("DML 理论验证汇总")
    print("─" * 60)
    print(f"  实验次数:              {n_success}")
    print(f"  样本量:                {n_samples}")
    print(f"  平均偏差 (Mean Bias):  {mean_bias:+.6f}")
    print(f"  中位偏差 (Median Bias):{median_bias:+.6f}")
    print(f"  RMSE:                  {rmse:.6f}")
    print(f"  95% CI 覆盖率:         {coverage:.1%}  (理论值: 95%)")
    print(f"  平均标准误:            {mean_se:.6f}")
    print(f"  偏差 t 统计量:         {bias_t_stat:.3f}")
    print(f"  偏差 p 值:             {bias_p_value:.4f}")
    print("─" * 60)

    # 判定结果
    verdicts = []
    if abs(mean_bias) < 2 * mean_se:
        verdicts.append("✓ 近似无偏（mean bias < 2×SE）")
    else:
        verdicts.append(f"⚠ 偏差可能显著（mean bias / SE = {mean_bias / (mean_se + 1e-12):.2f}）")

    if 0.85 <= coverage <= 0.99:
        verdicts.append(f"✓ 覆盖率合理 ({coverage:.1%})")
    elif coverage < 0.85:
        verdicts.append(f"⚠ 覆盖率偏低 ({coverage:.1%})，CI 可能过窄")
    else:
        verdicts.append(f"⚠ 覆盖率偏高 ({coverage:.1%})，CI 可能过宽")

    if bias_p_value > 0.05:
        verdicts.append("✓ 偏差检验不显著（p > 0.05），支持无偏性")
    else:
        verdicts.append(f"⚠ 偏差检验显著（p = {bias_p_value:.4f}），可能存在系统偏差")

    print("\n验证结论：")
    for v in verdicts:
        print(f"  {v}")

    # 保存结果
    out_path = os.path.join(
        OUT_DIR, f"monte_carlo_{graph_type}_{ml_method}_n{n_samples}{tag_str}.csv"
    )
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 保存汇总
    summary = {
        "n_experiments": n_success,
        "n_samples": n_samples,
        "graph_type": graph_type,
        "noise_type": noise_type,
        "ml_method": ml_method,
        "use_industrial": use_industrial,
        "mean_bias": round(mean_bias, 6),
        "median_bias": round(median_bias, 6),
        "rmse": round(rmse, 6),
        "coverage_95": round(coverage, 4),
        "mean_se": round(mean_se, 6),
        "bias_t_stat": round(bias_t_stat, 3),
        "bias_p_value": round(bias_p_value, 4),
    }
    summary_path = os.path.join(
        OUT_DIR, f"summary_{graph_type}_{ml_method}_n{n_samples}{tag_str}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存：{out_path}")
    print(f"汇总已保存：{summary_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
#  √n-一致性验证
# ═══════════════════════════════════════════════════════════════════

def run_consistency_validation(
    sample_sizes: list = None,
    n_experiments_per_size: int = 50,
    graph_type: str = "layered",
    noise_scale: float = 0.3,
    use_industrial: bool = False,
    ml_method: str = "forest",
):
    """
    √n-一致性验证：随着样本量增大，RMSE 应以 1/√n 速率下降。

    使用固定 DAG 结构，仅改变样本量和数据种子。
    对不同的 n 分别做 Monte Carlo 实验，记录 RMSE，
    然后检验 log(RMSE) vs log(n) 的斜率是否 ≈ -0.5。
    """
    if sample_sizes is None:
        sample_sizes = [500, 1000, 2000, 4000, 8000]

    print("\n" + "=" * 70)
    print(" √n-一致性验证")
    print(f" 样本量: {sample_sizes}")
    print(f" 每个样本量实验 {n_experiments_per_size} 次")
    print("=" * 70)

    # 固定 DAG
    dag_seed = 42
    n_nodes = 20
    gen_base = SyntheticDAGGenerator(n_nodes=n_nodes, seed=dag_seed)

    if graph_type == "layered":
        adj_true, layer_indices = gen_base.generate_layered_industrial_dag()
    elif graph_type == "er":
        adj_true = gen_base.generate_er_dag()
        layer_indices = None
    else:
        adj_true = gen_base.generate_scale_free_dag()
        layer_indices = None

    edge_funcs = gen_base.assign_edge_functions(adj_true, layer_indices, use_industrial)
    t_idx, y_idx, roles = select_treatment_outcome(
        gen_base, adj_true, edge_funcs, layer_indices, seed=dag_seed,
    )
    ate_analytic, ate_sim = compute_true_ate(
        gen_base, adj_true, edge_funcs, t_idx, y_idx,
        noise_scale=noise_scale,
    )
    ate_true = ate_sim if abs(ate_sim) > 1e-6 else ate_analytic

    import networkx as nx
    adjustment_set = gen_base.find_adjustment_set(adj_true, t_idx, y_idx)
    confounder_indices = sorted(adjustment_set)
    if len(confounder_indices) == 0:
        G_dag = nx.DiGraph(adj_true)
        t_ancestors = nx.ancestors(G_dag, t_idx)
        t_descendants = nx.descendants(G_dag, t_idx)
        y_descendants = nx.descendants(G_dag, y_idx)
        safe_vars = [v for v in range(n_nodes)
                     if v != t_idx and v != y_idx
                     and v not in t_descendants
                     and v not in y_descendants
                     and v in t_ancestors]
        if safe_vars:
            confounder_indices = sorted(safe_vars)

    print(f"  固定 DAG: T=X_{t_idx}, Y=X_{y_idx}, "
          f"混杂={len(roles['confounders'])}, 真实 ATE={ate_true:.6f}")

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
                    add_time_lag=False,
                )
                D = X_data[:, t_idx]
                Y = X_data[:, y_idx]
                if len(confounder_indices) > 0:
                    X_ctrl = X_data[:, confounder_indices]
                else:
                    rng_ph = np.random.RandomState(data_seed + 5555)
                    X_ctrl = rng_ph.randn(n_s, 1)

                theta_hat, se, _, _ = dml_estimate_cross_fitting(
                    Y, D, X_ctrl, ml_method=ml_method, seed=data_seed,
                )
                biases.append(theta_hat - ate_true)
            except Exception:
                pass

        if biases:
            rmse = float(np.sqrt(np.mean(np.array(biases) ** 2)))
            mean_bias = float(np.mean(biases))
            print(f"  n={n_s:>5d}  RMSE={rmse:.6f}  Bias={mean_bias:+.6f}  "
                  f"(成功 {len(biases)}/{n_experiments_per_size})")
            consistency_results.append({
                "n_samples": n_s,
                "rmse": round(rmse, 6),
                "mean_bias": round(mean_bias, 6),
                "n_success": len(biases),
            })

    if len(consistency_results) >= 2:
        # 检验 log(RMSE) vs log(n) 斜率
        log_n = np.log([r["n_samples"] for r in consistency_results])
        log_rmse = np.log([r["rmse"] + 1e-12 for r in consistency_results])
        slope, intercept, r_value, _, _ = stats.linregress(log_n, log_rmse)

        print("\n" + "─" * 60)
        print("√n-一致性检验")
        print("─" * 60)
        print(f"  log(RMSE) = {slope:.3f} × log(n) + {intercept:.3f}")
        print(f"  R² = {r_value ** 2:.3f}")
        print(f"  斜率 = {slope:.3f}  (理论值: -0.5)")

        if -0.8 < slope < -0.2:
            print(f"  ✓ 斜率 {slope:.3f} 接近理论值 -0.5，支持 √n-一致性")
        else:
            print(f"  ⚠ 斜率 {slope:.3f} 偏离理论值 -0.5")
        print("─" * 60)

    # 保存
    df = pd.DataFrame(consistency_results)
    out_path = os.path.join(OUT_DIR, f"consistency_{graph_type}_{ml_method}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
#  混杂控制对比实验
# ═══════════════════════════════════════════════════════════════════

def run_confounding_comparison(
    n_experiments: int = 100,
    n_samples: int = 2000,
    graph_type: str = "layered",
    noise_scale: float = 0.3,
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
    print("\n" + "=" * 70)
    print(" 混杂控制对比实验")
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
    for exp_i in range(n_experiments):
        try:
            seed = exp_i * 97 + 42
            gen = SyntheticDAGGenerator(n_nodes=20, seed=seed)
            X_data, adj_true, metadata = gen.generate_complete_synthetic_dataset(
                graph_type=graph_type,
                n_samples=n_samples,
                noise_scale=noise_scale,
                add_time_lag=False,
                use_industrial_functions=use_industrial,
            )
            edge_funcs = metadata["edge_funcs"]
            layer_indices = metadata.get("layer_indices")

            t_idx, y_idx, roles = select_treatment_outcome(
                gen, adj_true, edge_funcs, layer_indices, seed=seed,
            )

            ate_analytic, ate_sim = compute_true_ate(
                gen, adj_true, edge_funcs, t_idx, y_idx,
                noise_scale=noise_scale,
            )
            ate_true = ate_analytic if abs(ate_analytic) > 1e-6 else ate_sim

            D = X_data[:, t_idx]
            Y = X_data[:, y_idx]

            # 方案 A：正确控制混杂
            adjustment_set = gen.find_adjustment_set(adj_true, t_idx, y_idx)
            adj_indices = sorted(adjustment_set)
            if len(adj_indices) > 0:
                X_A = X_data[:, adj_indices]
            else:
                # 无混杂变量：使用 T 的安全祖先
                import networkx as nx
                G_dag = nx.DiGraph(adj_true)
                t_ancestors = nx.ancestors(G_dag, t_idx)
                t_descendants = nx.descendants(G_dag, t_idx)
                y_descendants = nx.descendants(G_dag, y_idx)
                safe_vars = [v for v in range(20)
                             if v != t_idx and v != y_idx
                             and v not in t_descendants
                             and v not in y_descendants
                             and v in t_ancestors]
                if safe_vars:
                    X_A = X_data[:, sorted(safe_vars)]
                else:
                    rng_ph = np.random.RandomState(seed + 5555)
                    X_A = rng_ph.randn(n_samples, 1)

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
            X_B = rng.randn(n_samples, 3)  # 3 个随机噪声变量
            theta_B, se_B, ci_lo_B, ci_hi_B = dml_estimate_cross_fitting(
                Y, D, X_B, ml_method=ml_method, seed=seed,
            )
            results["B_omitted"].append({
                "bias": theta_B - ate_true,
                "covers": bool(ci_lo_B <= ate_true <= ci_hi_B),
                "se": se_B,
            })

            # 方案 C：过度控制（包含碰撞/中介）
            all_others = [i for i in range(20) if i != t_idx and i != y_idx]
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
        description="DML 理论验证：基于合成 DAG 数据生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
验证项目：
  1. quick:         快速验证（30 次实验，确认脚本正常运行）
  2. full:          完整蒙特卡洛验证（默认 200 次实验）
  3. consistency:   √n-一致性验证（不同样本量对比）
  4. confounding:   混杂控制对比实验（正确 vs 遗漏 vs 过度控制）
  5. all:           全部实验

建议执行顺序：
  python run_dml_theory_validation.py --mode quick
  python run_dml_theory_validation.py --mode full
  python run_dml_theory_validation.py --mode consistency
  python run_dml_theory_validation.py --mode confounding

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
    print(f" DML 理论验证  |  模式: {args.mode.upper()}")
    print(f" 数据来源: SyntheticDAGGenerator (因果的发现算法理论验证)")
    print(f" DAG 角色接口: 兼容 analyze_dag_causal_roles_v4_1.py")
    print("=" * 70)

    mode = args.mode

    if mode in ("quick",):
        run_monte_carlo_validation(
            n_experiments=30,
            n_nodes=args.n_nodes,
            n_samples=1000,
            graph_type=args.graph_type,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            use_industrial=args.use_industrial,
            ml_method=args.ml_method,
            tag="quick",
        )

    if mode in ("full", "all"):
        run_monte_carlo_validation(
            n_experiments=args.n_experiments,
            n_nodes=args.n_nodes,
            n_samples=args.n_samples,
            graph_type=args.graph_type,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type,
            use_industrial=args.use_industrial,
            ml_method=args.ml_method,
        )

    if mode in ("consistency", "all"):
        run_consistency_validation(
            graph_type=args.graph_type,
            noise_scale=args.noise_scale,
            use_industrial=args.use_industrial,
            ml_method=args.ml_method,
        )

    if mode in ("confounding", "all"):
        run_confounding_comparison(
            n_experiments=min(args.n_experiments, 100),
            n_samples=args.n_samples,
            graph_type=args.graph_type,
            noise_scale=args.noise_scale,
            use_industrial=args.use_industrial,
            ml_method=args.ml_method,
        )

    print("\n" + "=" * 70)
    print(" 全部实验完成")
    print(f" 结果保存目录: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
