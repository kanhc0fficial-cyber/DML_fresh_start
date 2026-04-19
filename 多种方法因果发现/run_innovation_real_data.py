"""
run_innovation_real_data.py
===========================
在真实工业数据上运行三种创新因果发现算法，输出 GraphML 供 DML 下游使用。

算法：BiAttn-CUTS / MultiScale-NTS / MB-CUTS
产线：xin1 (Group A+C) / xin2 (Group B+C)
输出：GraphML (兼容 analyze_dag_causal_roles_v4_1.py)

关键适配点（相对于模拟数据的 run_monte_carlo_benchmark_fixed.py）：
  1. 全局 W 矩阵训练时施加物理拓扑掩码（topo_mask），作为硬约束惩罚
  2. y_grade 作为第 N 个变量并入数据矩阵，W 中 Y→其他 方向全部 mask
  3. MB-CUTS 的 Spearman 预筛选掩码与物理拓扑掩码取交集
  4. NOTEARS 权重降低至 0.1*h²（物理掩码已确保近似前馈，降低冗余梯度）
  5. 邻接矩阵 W[i,j] 均表示 i→j 方向（已注释验证）
  6. GraphML 节点名与 analyze_dag 的 Y_CANDIDATES 约定匹配
  7. 训练前打印 W 权重分位数供阈值调优参考
  8. 输出图强制 DAG 后处理（移除最弱环边）
  9. Conv1d groups=d 时用 hidden=d*8 保证整除
  10. 标准化在窗口化之前全局进行
"""

import gc
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config import prepare_data, can_cause

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 超参数（真实数据适配） ────────────────────────────────────────────────
# WINDOW_SIZE 与 run_tcdf_space_time_dag 保持一致（10min 采样 × 15 = 2.5h 窗口）
WINDOW_SIZE = 15
BATCH_SIZE = 32       # 真实数据节点数多，减小 batch 防显存溢出
EPOCHS = 50           # 真实数据噪声大，多训练
LR = 0.003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 邻接矩阵阈值
DEFAULT_THRESHOLD = 0.05

# 拓扑掩码惩罚权重（物理不可行边的惩罚系数）
TOPO_PENALTY_WEIGHT = 10.0

# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def build_topology_mask(valid_vars, var_to_stage, var_to_group, line):
    """
    构建 (N+1, N+1) 物理因果可行性掩码。

    前 N 个维度对应 valid_vars，第 N 个维度对应 y_grade。
    mask[i, j] = 1 表示 i→j 物理上可行。

    关键设计：
      - y_grade（索引 N）只能被其他节点指向，自身不指向任何节点
      - 对角线全为 0（无自环）
    """
    N = len(valid_vars)
    mask = np.zeros((N + 1, N + 1), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i != j and can_cause(
                var_to_stage[valid_vars[i]], var_to_stage[valid_vars[j]],
                var_to_group.get(valid_vars[i]), var_to_group.get(valid_vars[j]),
                line
            ):
                mask[i, j] = 1.0
        # i → y_grade
        if can_cause(
            var_to_stage[valid_vars[i]], "Y",
            var_to_group.get(valid_vars[i]), None,
            line
        ):
            mask[i, N] = 1.0

    # y_grade 不影响任何其他节点（反向因果禁止）
    mask[N, :] = 0.0
    # 消除对角线（自环）
    np.fill_diagonal(mask, 0.0)
    return mask


def build_windows(X):
    """
    构建时间窗口。

    注意：标准化必须在调用此函数之前完成（全局 z-score），
    以保留跨窗口的数值尺度信息。

    参数:
        X: (T, d) 已标准化数据
    返回:
        xs: (n_win, WINDOW_SIZE, d) 输入窗口
        ys: (n_win, d) 预测目标（下一时刻）
    """
    T, d = X.shape
    xs, ys = [], []
    for start in range(0, T - WINDOW_SIZE):
        xs.append(X[start:start + WINDOW_SIZE, :])
        ys.append(X[start + WINDOW_SIZE, :])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─── 模型定义 ────────────────────────────────────────────────────────────────

class BiAttnCUTSNet(nn.Module):
    """
    创新方案一：Transformer 注意力编码器 + 软阈值邻接矩阵 + NOTEARS 约束。

    W[i,j] 表示 i→j 的影响强度（einsum 'ij,bif->bjf' 验证方向正确）。

    真实数据适配：
      - d_model 固定 16，n_heads 固定 4（避免大 d 时 B*d 过大导致 OOM）
      - topo_mask 通过外部惩罚施加，模型本身不存储掩码（保持模块化）
    """
    def __init__(self, d, n_heads=4, d_model=16):
        super().__init__()
        self.d = d
        self.d_model = d_model
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.05, 0.05))
        # tau 经 softplus 后恒正，保证 sigmoid 方向不反转
        self.tau = nn.Parameter(torch.zeros(1))

        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,        # d_model=16, nhead=4 → head_dim=4 ✓
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Linear(d_model, 1)

    def _tau_pos(self):
        """保证温度系数恒正"""
        return F.softplus(self.tau)

    def forward(self, x):
        """
        参数:
            x: (B, WINDOW_SIZE, d)
        返回:
            pred: (B, d)
        """
        B, T, d = x.shape

        # 先 transpose → (B, d, T)，再 contiguous().reshape → (B·d, T, 1)
        # 保证第 k 个"伪样本"对应单一节点的完整时序，不跨节点交叉
        x_flat = x.transpose(1, 2).contiguous().reshape(B * d, T, 1)  # (B·d, T, 1)
        x_emb = self.input_proj(x_flat)                                # (B·d, T, d_model)
        h = self.transformer(x_emb)                                    # (B·d, T, d_model)
        h_last = h[:, -1, :]                                           # (B·d, d_model)
        node_feats = h_last.reshape(B, d, self.d_model)                # (B, d, d_model)

        tau_pos = self._tau_pos()
        W_soft = torch.sigmoid(self.W * tau_pos)
        W_soft = W_soft * (1 - torch.eye(d, device=W_soft.device))    # 去自环

        # W[i,j] 作用于第 i 个节点特征 → 聚合得到第 j 个节点的预测输入
        # einsum 'ij,bif->bjf'：W[i,j] = i→j ✓
        agg = torch.einsum("ij,bif->bjf", W_soft, node_feats)         # (B, d, d_model)
        pred = self.out_proj(agg).squeeze(-1)                          # (B, d)
        return pred

    def notears_penalty(self):
        """NOTEARS 无环约束（施加在 W² 上）"""
        M = self.W * self.W
        E = torch.matrix_exp(M)
        return torch.trace(E) - self.d


class MultiScaleNTSNet(nn.Module):
    """
    创新方案二：三路并行多尺度卷积 + 可学习融合权重 + 共享 NOTEARS 约束。

    W[i,j] 含义：x_agg = x @ W，W[i,j] 表示第 i 列特征对第 j 列预测的贡献，
    即 i→j ✓。

    真实数据适配：
      - hidden = d * 8 保证 Conv1d groups=d 时 in_channels % d == 0
      - WINDOW_SIZE=15 满足第三路 kernel_size=WINDOW_SIZE 的最小输入长度要求
    """
    KERNEL_SIZES = [3, 5]  # 第三路动态使用 WINDOW_SIZE

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.01, 0.01))

        kernel_sizes = self.KERNEL_SIZES + [WINDOW_SIZE]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d, d * 8, kernel_size=ks, groups=d),  # groups=d, hidden=d*8 整除 ✓
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(d * 8, d, kernel_size=1, groups=d),
            )
            for ks in kernel_sizes
        ])
        # zeros 初始化：softmax(zeros) = 均匀分布，梯度从对称点出发更稳定
        self.alpha = nn.Parameter(torch.zeros(len(kernel_sizes)))

    def forward(self, x):
        # W[i,j] = i→j（见类注释）
        x_agg = torch.matmul(x, self.W)   # (B, T, d)
        x_t = x_agg.transpose(1, 2)       # (B, d, T)

        alpha_norm = torch.softmax(self.alpha, dim=0)
        pred = sum(
            alpha_norm[i] * conv(x_t).squeeze(2)
            for i, conv in enumerate(self.convs)
        )
        return pred

    def notears_penalty(self):
        M = self.W * self.W
        E = torch.matrix_exp(M)
        return torch.trace(E) - self.d


class NTS_NOTEARSNet(nn.Module):
    """
    基线 NTS-NOTEARS（用于 MB-CUTS Stage 3 精化）。

    W[i,j] = i→j（同 MultiScaleNTSNet）✓。
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.01, 0.01))
        self.conv1 = nn.Conv1d(d, d * 16, kernel_size=WINDOW_SIZE, groups=d)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d * 16, d, kernel_size=1, groups=d)

    def forward(self, x):
        x_agg = torch.matmul(x, self.W)
        x_agg = x_agg.transpose(1, 2)
        h = self.relu(self.conv1(x_agg))
        pred = self.conv2(h).squeeze(2)
        return pred

    def notears_penalty(self):
        M = self.W * self.W
        E = torch.matrix_exp(M)
        return torch.trace(E) - self.d


class CUTSPlusNet(nn.Module):
    """
    基线 CUTS+（用于 MB-CUTS Stage 2 预热）。

    W[i,j] = i→j（einsum 'bsf,st->btf'：W[s,t] = s 对 t 的影响）✓。
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.05, 0.05))
        self.encoder = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.decoder = nn.Conv1d(16 * d, d, kernel_size=1, groups=d)

    def forward(self, x):
        B = x.shape[0]
        x_in = x.transpose(1, 2).contiguous().reshape(B * self.d, WINDOW_SIZE, 1)
        _, (h_n, _) = self.encoder(x_in)
        hiddens = h_n.squeeze(0).reshape(B, self.d, 16)

        W_attn = torch.abs(self.W)
        H_agg = torch.einsum("bsf,st->btf", hiddens, W_attn)  # W[s,t] = s→t ✓
        H_agg_flat = H_agg.reshape(B, self.d * 16, 1)
        pred = self.decoder(H_agg_flat).squeeze(2)
        return pred


# ─── 通用训练函数（施加物理拓扑约束） ──────────────────────────────────────

def train_with_topo_mask(model, X_all, topo_mask, epochs=EPOCHS, verbose=True, algo_name="model"):
    """
    通用训练循环，同时施加物理拓扑掩码约束。

    损失 = MSE
          + 0.001 * 稀疏项（物理可行区域）
          + 0.1   * h²（NOTEARS，降低权重因物理掩码已保证近前馈）
          + 10.0  * 拓扑惩罚（物理不可行区域的权重绝对值之和）

    参数:
        model:      已 .to(DEVICE) 的神经网络模型（需有 model.W 参数）
        X_all:      (T, d) 包含 y_grade 的完整数据（已包含在最后一列）
        topo_mask:  (d, d) numpy float32，物理可行性掩码
        epochs:     训练轮数
        verbose:    是否打印每 10 轮的 loss
        algo_name:  算法名称（用于打印）
    返回:
        训练后的 model
    """
    d = X_all.shape[1]
    # 全局标准化（必须在窗口化之前）
    X_norm = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)

    topo_t = torch.tensor(topo_mask, dtype=torch.float32, device=DEVICE)
    diag_mask = torch.eye(d, device=DEVICE)
    # 物理不可行区域掩码（排除对角线，对角线不参与惩罚）
    anti_topo_t = (1.0 - topo_t) * (1.0 - diag_mask)

    xb = torch.tensor(wx, device=DEVICE)
    yb = torch.tensor(wy, device=DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        total_loss = 0.0
        for b_x, b_y in loader:
            opt.zero_grad()
            pred = model(b_x)
            mse = F.mse_loss(pred, b_y)

            # 物理拓扑硬约束：图外边重罚（鼓励学到的权重在物理不可行区域归零）
            topo_penalty = torch.sum(torch.abs(model.W * anti_topo_t))

            # NOTEARS 无环约束（权重 0.1，低于模拟数据的 0.5，因为物理掩码已近似保证前馈）
            h_val = model.notears_penalty() if hasattr(model, "notears_penalty") else 0.0
            h_sq = h_val * h_val if isinstance(h_val, torch.Tensor) else 0.0

            # 稀疏性约束（仅在物理可行区域施加，避免压制有效信号）
            sparse = torch.sum(torch.abs(model.W * topo_t))

            loss = mse + 0.001 * sparse + 0.1 * h_sq + TOPO_PENALTY_WEIGHT * topo_penalty
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    [{algo_name}] Epoch {epoch+1}/{epochs}  Loss={total_loss/len(loader):.4f}")

    return model


# ─── MB-CUTS 真实数据专用训练 ─────────────────────────────────────────────────

def _compute_mb_mask(X_norm, keep_ratio=0.5):
    """
    基于 Spearman 秩相关计算近似马尔可夫毯掩码。

    Spearman 能捕捉单调非线性因果关系（Pearson 仅限线性）。
    """
    d = X_norm.shape[1]
    corr_matrix, _ = spearmanr(X_norm)
    corr = np.array(corr_matrix)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    if corr.shape != (d, d):
        raise ValueError(
            f"spearmanr 返回了意外的相关矩阵形状 {corr.shape}，期望 ({d}, {d})。"
            f"请检查 X_norm 是否含有全常数列（std=0 会导致 NaN）。"
        )
    np.fill_diagonal(corr, 0.0)

    mb_mask = np.zeros((d, d), dtype=np.float32)
    k = max(1, int(np.ceil(d * keep_ratio)))
    for j in range(d):
        col = np.abs(corr[:, j])
        top_idx = np.argsort(col)[-k:]
        mb_mask[top_idx, j] = 1.0
    return mb_mask


def train_mb_cuts_real(X_all, topo_mask, verbose=True, epochs=None):
    """
    MB-CUTS 三阶段训练（真实数据版）。

    关键适配：
      - Stage 1：Spearman 掩码 ∩ 物理拓扑掩码（取交集），同时满足统计支持和物理可行性
      - Stage 2：CUTS+ 预热时叠加交集掩码
      - Stage 3：NTS-NOTEARS 精化时同时施加交集掩码和物理拓扑惩罚

    参数:
        X_all:     (T, d) 包含 y_grade 的完整数据
        topo_mask: (d, d) numpy 物理拓扑掩码
        verbose:   是否打印进度
        epochs:    Stage 3 训练轮数（None 则使用全局 EPOCHS）
    返回:
        训练后的 NTS_NOTEARSNet 模型
    """
    eff_epochs = epochs if epochs is not None else EPOCHS
    d = X_all.shape[1]
    X_norm = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)

    xb = torch.tensor(wx, device=DEVICE)
    yb = torch.tensor(wy, device=DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ── Stage 1：Spearman 筛选 ∩ 物理拓扑掩码 ────────────────────────────
    if verbose:
        print("  [MB-CUTS] Stage 1: 马尔可夫毯候选边筛选（Spearman）∩ 物理拓扑...")
    mb_mask = _compute_mb_mask(X_norm, keep_ratio=0.5)
    # 取交集：既要统计支持，又要物理可行
    final_mask = mb_mask * topo_mask
    if verbose:
        n_mb = int(mb_mask.sum())
        n_final = int(final_mask.sum())
        n_topo = int(topo_mask.sum())
        print(f"    Spearman候选边: {n_mb}  物理可行边: {n_topo}  交集: {n_final}")

    final_mask_t = torch.tensor(final_mask, dtype=torch.float32, device=DEVICE)
    non_final_t = torch.tensor(1.0 - final_mask, dtype=torch.float32, device=DEVICE)
    diag_mask = torch.eye(d, device=DEVICE)
    non_final_t = non_final_t * (1.0 - diag_mask)

    # ── Stage 2：CUTS+ 预热（在交集掩码内） ──────────────────────────────
    if verbose:
        print("  [MB-CUTS] Stage 2: CUTS+ 预热...")
    model_rough = CUTSPlusNet(d).to(DEVICE)
    opt_rough = optim.Adam(model_rough.parameters(), lr=LR)
    for _ in range(10):
        for b_x, b_y in loader:
            opt_rough.zero_grad()
            pred = model_rough(b_x)
            W_in = model_rough.W * final_mask_t
            loss = F.mse_loss(pred, b_y) + 0.01 * torch.sum(torch.abs(W_in))
            loss.backward()
            opt_rough.step()
    adj_rough = torch.abs(model_rough.W).detach().cpu().numpy() * final_mask
    if verbose:
        nonzero = adj_rough[adj_rough > 0]
        print(f"    预热完成，粗糙邻接矩阵均值: {nonzero.mean():.4f}" if len(nonzero) else "    预热完成（无非零边）")

    # ── Stage 3：NTS-NOTEARS 精化 + 交集掩码约束 + 物理拓扑惩罚 ──────────
    if verbose:
        print("  [MB-CUTS] Stage 3: NTS-NOTEARS 精化...")
    model_fine = NTS_NOTEARSNet(d).to(DEVICE)
    with torch.no_grad():
        model_fine.W.data = torch.tensor(adj_rough * 0.1, dtype=torch.float32, device=DEVICE)
    opt_fine = optim.Adam(model_fine.parameters(), lr=LR)

    for epoch in range(eff_epochs):
        total_loss = 0.0
        for b_x, b_y in loader:
            opt_fine.zero_grad()
            pred = model_fine(b_x)
            mse = F.mse_loss(pred, b_y)
            h_val = model_fine.notears_penalty()
            sparse = torch.sum(torch.abs(model_fine.W))
            # 惩罚交集掩码外的边（同时违反统计支持和物理可行性）
            mb_penalty = torch.sum(torch.abs(model_fine.W * non_final_t))
            loss = mse + 0.001 * sparse + 0.1 * h_val * h_val + TOPO_PENALTY_WEIGHT * mb_penalty
            loss.backward()
            opt_fine.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [MB-CUTS] Epoch {epoch+1}/{eff_epochs}  Loss={total_loss/len(loader):.4f}")

    if verbose:
        print("  [MB-CUTS] 三阶段完成")
    return model_fine


# ─── 邻接矩阵提取 + GraphML 输出 ─────────────────────────────────────────────

def extract_adj_and_save(model, valid_vars, topo_mask, line, algo_name, threshold=DEFAULT_THRESHOLD):
    """
    从训练好的模型提取邻接矩阵，施加物理掩码，输出 GraphML。

    GraphML 格式与 analyze_dag_causal_roles_v4_1.py 兼容：
      - 节点名为变量名字符串（含 "y_grade"）
      - 边有 weight 属性
      - 输出前强制 DAG（移除最弱环边）

    参数:
        model:      训练后的模型（需有 model.W 参数）
        valid_vars: 特征变量名列表（不含 y_grade）
        topo_mask:  (N+1, N+1) 物理掩码
        line:       产线名
        algo_name:  算法名（用于文件名）
        threshold:  邻接矩阵阈值

    返回:
        G:   nx.DiGraph
        W:   后处理后的邻接矩阵 (N+1, N+1)
    """
    N = len(valid_vars)

    with torch.no_grad():
        if hasattr(model, "_tau_pos"):
            # BiAttn-CUTS：使用 softplus(tau) 版本的软邻接矩阵
            tau_pos = model._tau_pos()
            W = torch.sigmoid(model.W * tau_pos).cpu().numpy()
        else:
            W = np.abs(model.W.detach().cpu().numpy())

    # 二次保险：物理掩码过滤（训练时已惩罚，这里硬置零）
    W = W * topo_mask
    np.fill_diagonal(W, 0.0)

    # 打印权重分布供阈值调优
    nonzero = W[W > 0]
    if len(nonzero) > 0:
        pcts = np.percentile(nonzero, [25, 50, 75, 90, 95])
        print(f"  [{algo_name}] W 非零权重分位数 [25,50,75,90,95]: {pcts.round(4)}")
    else:
        print(f"  [{algo_name}] 警告：训练后邻接矩阵全零，请检查训练过程或调整超参数")

    # 构建 DiGraph
    G = nx.DiGraph()
    for var in valid_vars:
        G.add_node(var)
    G.add_node("y_grade")  # 必须命名为 "y_grade"，与 analyze_dag Y_CANDIDATES 匹配

    edge_count = 0
    for i in range(N):
        for j in range(N):
            if W[i, j] > threshold:
                G.add_edge(valid_vars[i], valid_vars[j], weight=float(W[i, j]))
                edge_count += 1
        # X → y_grade
        if W[i, N] > threshold:
            G.add_edge(valid_vars[i], "y_grade", weight=float(W[i, N]))
            edge_count += 1

    # DAG 后处理：移除最弱环边（物理掩码 + NOTEARS 后同 Stage 内仍可能有环）
    removed = 0
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda e: G[e[0]][e[1]].get("weight", 0.0))
        G.remove_edge(weakest[0], weakest[1])
        edge_count -= 1
        removed += 1
    if removed:
        print(f"  [{algo_name}] DAG 后处理：移除 {removed} 条环边")

    out_path = os.path.join(OUT_DIR, f"{algo_name}_real_dag_{line}.graphml")
    nx.write_graphml(G, out_path)
    print(f"  ✓ [{algo_name}] [{line}]: {edge_count} 条边 → {out_path}")
    return G, W


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def run_all(line="xin1", epochs=None, threshold=None):
    """
    对单条产线运行全部三种创新算法并输出 GraphML。

    参数:
        line:      'xin1' 或 'xin2'
        epochs:    训练轮数（None 则使用全局默认 EPOCHS）
        threshold: 邻接矩阵阈值（None 则使用全局默认 DEFAULT_THRESHOLD）
    """
    eff_epochs = epochs if epochs is not None else EPOCHS
    eff_threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
    print(f"\n{'='*70}")
    print(f"创新算法真实数据因果发现  [产线={line}]  设备={DEVICE}")
    print(f"{'='*70}")

    t0 = time.time()
    df, valid_vars, var_to_stage, var_to_group = prepare_data(line)
    N = len(valid_vars)
    print(f"变量数: {N}  样本数: {len(df)}")

    # 构建含 y_grade 的完整数据矩阵（y_grade 放在最后一列，索引 N）
    all_vars = valid_vars + ["y_grade"]
    X_all = df[all_vars].values.astype(np.float32)  # (T, N+1)

    topo_mask = build_topology_mask(valid_vars, var_to_stage, var_to_group, line)
    n_feasible = int(topo_mask.sum())
    total_possible = (N + 1) ** 2 - (N + 1)
    print(f"物理可行边数: {n_feasible} / {total_possible}  "
          f"({n_feasible / total_possible * 100:.1f}%)")

    d = N + 1  # 总维度（含 y_grade）

    # ── BiAttn-CUTS ──────────────────────────────────────────────────────────
    print(f"\n--- 训练 BiAttn-CUTS ---")
    model_biattn = BiAttnCUTSNet(d).to(DEVICE)
    model_biattn = train_with_topo_mask(
        model_biattn, X_all, topo_mask, epochs=eff_epochs, algo_name="BiAttn-CUTS"
    )
    extract_adj_and_save(model_biattn, valid_vars, topo_mask, line, "biattn_cuts",
                         threshold=eff_threshold)
    del model_biattn
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ── MultiScale-NTS ────────────────────────────────────────────────────────
    print(f"\n--- 训练 MultiScale-NTS ---")
    model_ms = MultiScaleNTSNet(d).to(DEVICE)
    model_ms = train_with_topo_mask(
        model_ms, X_all, topo_mask, epochs=eff_epochs, algo_name="MultiScale-NTS"
    )
    extract_adj_and_save(model_ms, valid_vars, topo_mask, line, "multiscale_nts",
                         threshold=eff_threshold)
    del model_ms
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ── MB-CUTS ───────────────────────────────────────────────────────────────
    print(f"\n--- 训练 MB-CUTS ---")
    model_mb = train_mb_cuts_real(X_all, topo_mask, verbose=True, epochs=eff_epochs)
    extract_adj_and_save(model_mb, valid_vars, topo_mask, line, "mb_cuts",
                         threshold=eff_threshold)
    del model_mb
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - t0
    print(f"\n[{line}] 全部完成，耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="在真实工业数据上运行创新因果发现算法（BiAttn-CUTS / MultiScale-NTS / MB-CUTS）"
    )
    parser.add_argument(
        "--line",
        choices=["xin1", "xin2", "both"],
        default="both",
        help="产线选择（默认: both）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"邻接矩阵阈值（默认: {DEFAULT_THRESHOLD}）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"训练轮数（默认: {EPOCHS}）",
    )
    args = parser.parse_args()

    lines = ["xin1", "xin2"] if args.line == "both" else [args.line]
    for ln in lines:
        run_all(ln, epochs=args.epochs, threshold=args.threshold)
