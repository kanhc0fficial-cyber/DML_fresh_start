"""
synthetic_dag_generator_highdim.py
===================================
高维合成 DAG 数据生成器 —— 为 VAE / LSTM 优势场景设计

设计理念：
  原版 SyntheticDAGGenerator 默认 20 节点、低维混杂变量（约 3~8 个），
  传统 RF/GBM 在这种小维度下足以有效拟合 nuisance function，VAE 的潜在
  表征学习没有额外增益。

  本生成器通过以下方式创造 VAE/LSTM 优势场景：

  1. 更多 DAG 节点（默认 60，7 层分层工业 DAG）
     → 更复杂的因果结构，更多路径

  2. 观测扩展（Observable Expansion）
     将每个混杂变量扩展为 K 个高维相关观测特征：
       - 噪声副本（noisy copies）
       - 非线性变换（x², log|x|, tanh(x)）
       - 交互项（x_i × x_j）
       - 块相关噪声组（block-correlated noise groups）
     例如 10 个混杂变量 → ~80-120 个观测特征
     → VAE 能发现低维流形结构，RF 在高维冗余特征上过拟合

  3. 异方差噪声（默认 heteroscedastic）
     信号越强噪声越大，模拟真实工业数据的特征
     → v5 的不确定性加权 DML（微创新 D）有优势

  4. 自回归时序依赖（AR lag）
     数据带滞后效应，适合 LSTM（虽然当前验证脚本用 MLP，但数据结构保留）

  核心约束（继承）：
    - T→Y 路径保持线性（由 dml_validation_common 的 enforce_linear_treatment_paths 保证）
    - PLM 假设：Y = θ·D + g(X) + ε，其中 g(X) 可以是任意非线性
    - 与原版 SyntheticDAGGenerator 完全兼容（继承全部 DAG 生成方法）

用法示例：
  from synthetic_dag_generator_highdim import HighDimSyntheticDAGGenerator, expand_to_highdim

  gen = HighDimSyntheticDAGGenerator(n_nodes=60, seed=42)
  adj, layer_indices = gen.generate_layered_industrial_dag(n_layers=7)
  edge_funcs = gen.assign_edge_functions(adj, layer_indices, use_industrial_functions=True)
  X_data = gen.generate_data(adj, edge_funcs, n_samples=5000,
                              noise_type='heteroscedastic', add_time_lag=True)

  # 选择 T-Y 对，获取混杂变量索引
  # ...

  # 将混杂变量扩展为高维观测
  X_ctrl_raw = X_data[:, confounder_indices]  # (n, ~10)
  X_ctrl_hd = expand_to_highdim(X_ctrl_raw, seed=42)  # (n, ~100)
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple

# 导入原版生成器
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from synthetic_dag_generator import SyntheticDAGGenerator


class HighDimSyntheticDAGGenerator(SyntheticDAGGenerator):
    """
    高维合成 DAG 生成器，继承原版并增加默认参数调整。

    与原版的区别仅在默认参数：
      - n_nodes: 20 → 60
      - 推荐 graph_type: 'layered'（7 层）
      - 推荐 noise_type: 'heteroscedastic'
      - 推荐 add_time_lag: True

    DAG 生成和数据生成逻辑完全继承自 SyntheticDAGGenerator，
    不修改因果机制，保证 T→Y 线性化等约束仍然有效。
    """

    def __init__(self, n_nodes: int = 60, seed: int = 42):
        super().__init__(n_nodes=n_nodes, seed=seed)


# ═══════════════════════════════════════════════════════════════════
#  观测扩展函数（核心新增功能）
# ═══════════════════════════════════════════════════════════════════

def expand_to_highdim(
    X_raw: np.ndarray,
    seed: int = 42,
    n_noisy_copies: int = 3,
    n_nonlinear_transforms: int = 3,
    n_interaction_pairs: int = 5,
    n_block_noise_groups: int = 2,
    block_noise_dim: int = 5,
    copy_noise_scale: float = 0.3,
    block_noise_scale: float = 0.5,
) -> np.ndarray:
    """
    将低维混杂变量矩阵扩展为高维观测特征矩阵。

    扩展策略（每个原始变量 x_j 生成 ~10 个观测特征）：

    1. 噪声副本（Noisy Copies）：x_j + ε，ε ~ N(0, σ²)
       → 模拟传感器冗余（同一物理量有多个传感器测量）
       → VAE 能从多个噪声副本中恢复真实信号

    2. 非线性变换（Nonlinear Transforms）：
       x_j²/scale, tanh(x_j), sign(x_j)·log(1+|x_j|)
       → 模拟传感器非线性响应
       → VAE 的非线性编码器能提取；RF 的线性分裂效率低

    3. 交互项（Interaction Terms）：x_i × x_j
       → 模拟变量间的交互效应
       → 加入特征空间的非线性维度

    4. 块相关噪声组（Block-Correlated Noise Groups）：
       每组 k 个特征 = W @ X_raw + noise
       → 模拟传感器组内相关（如同一选矿段的多个测点）
       → VAE 的线性/非线性投影可解耦；RF 无法高效降维

    参数:
        X_raw:                  原始混杂变量矩阵 (n_samples, n_confounders)
        seed:                   随机种子
        n_noisy_copies:         每个变量的噪声副本数
        n_nonlinear_transforms: 非线性变换数（最多 3 种）
        n_interaction_pairs:    交互项的对数（从混杂变量对中随机选取）
        n_block_noise_groups:   块相关噪声组数
        block_noise_dim:        每个噪声组的维度
        copy_noise_scale:       噪声副本的标准差（相对于信号标准差）
        block_noise_scale:      块相关噪声组的噪声标准差

    返回:
        X_expanded: 高维观测特征矩阵 (n_samples, expanded_dim)
        （expanded_dim ≈ n_confounders × (1 + n_noisy_copies + n_nonlinear_transforms)
                         + n_interaction_pairs + n_block_noise_groups × block_noise_dim）
    """
    rng = np.random.RandomState(seed)
    n_samples, n_vars = X_raw.shape
    features = [X_raw.copy()]  # 保留原始特征

    # 每列的标准差（用于缩放噪声）
    col_std = np.std(X_raw, axis=0) + 1e-8

    # ─── 1. 噪声副本 ──────────────────────────────────────────────
    for _ in range(n_noisy_copies):
        noise = rng.randn(n_samples, n_vars) * (col_std * copy_noise_scale)
        features.append(X_raw + noise)

    # ─── 2. 非线性变换 ────────────────────────────────────────────
    transforms = [
        lambda x: np.tanh(x),
        lambda x: np.sign(x) * np.log1p(np.abs(x)),
        lambda x: x ** 2 / (np.std(x, axis=0, keepdims=True) + 1e-8),
    ]
    for i in range(min(n_nonlinear_transforms, len(transforms))):
        features.append(transforms[i](X_raw))

    # ─── 3. 交互项 ────────────────────────────────────────────────
    if n_vars >= 2 and n_interaction_pairs > 0:
        # 随机选取变量对
        max_pairs = n_vars * (n_vars - 1) // 2
        n_pairs = min(n_interaction_pairs, max_pairs)
        all_pairs = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
        selected_pairs = [all_pairs[k] for k in
                          rng.choice(len(all_pairs), size=n_pairs, replace=False)]
        for i, j in selected_pairs:
            # 标准化交互项，防止量纲爆炸
            xi = X_raw[:, i] / (col_std[i] + 1e-8)
            xj = X_raw[:, j] / (col_std[j] + 1e-8)
            features.append((xi * xj).reshape(-1, 1))

    # ─── 4. 块相关噪声组 ─────────────────────────────────────────
    for _ in range(n_block_noise_groups):
        # 随机投影矩阵 W: (block_noise_dim, n_vars)
        W = rng.randn(block_noise_dim, n_vars) * 0.3
        projected = X_raw @ W.T  # (n_samples, block_noise_dim)
        noise = rng.randn(n_samples, block_noise_dim) * block_noise_scale
        features.append(projected + noise)

    # ─── 合并 ─────────────────────────────────────────────────────
    X_expanded = np.hstack(features)

    # 安全检查
    if np.any(~np.isfinite(X_expanded)):
        warnings.warn("expand_to_highdim: 存在 NaN/Inf，已替换为 0")
        X_expanded = np.nan_to_num(X_expanded, nan=0.0, posinf=1e4, neginf=-1e4)

    return X_expanded


def compute_expanded_dim(
    n_confounders: int,
    n_noisy_copies: int = 3,
    n_nonlinear_transforms: int = 3,
    n_interaction_pairs: int = 5,
    n_block_noise_groups: int = 2,
    block_noise_dim: int = 5,
) -> int:
    """计算扩展后的特征维度（用于模型初始化）"""
    base = n_confounders
    copies = n_confounders * n_noisy_copies
    transforms = n_confounders * min(n_nonlinear_transforms, 3)
    max_pairs = n_confounders * (n_confounders - 1) // 2
    interactions = min(n_interaction_pairs, max_pairs)
    blocks = n_block_noise_groups * block_noise_dim
    return base + copies + transforms + interactions + blocks
