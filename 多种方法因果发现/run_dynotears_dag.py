"""
run_dynotears_dag.py
====================
基于 DYNOTEARS（动态 NOTEARS）算法在真实工业数据上进行时序因果发现，
输出 GraphML 供 DAG 解析器下游使用。

算法原理：
  DYNOTEARS 将 NOTEARS 扩展至时间序列，联合学习：
    - 同期邻接矩阵 W（即刻因果，满足 NOTEARS 无环约束）
    - 滞后邻接矩阵 A_k（k 步滞后因果，k=1,...,MAX_LAGS，允许成环）

  线性 SEM：X_t = X_t @ W^T + Σ_{k=1}^{K} X_{t-k} @ A_k^T + ε_t

  损失函数：
    MSE 预测误差
    + λ_W     * ||W||_1        （同期稀疏）
    + λ_A     * ||A_k||_1      （滞后稀疏）
    + λ_notears * h(W)^2       （NOTEARS 无环约束，仅施加在同期图）

  物理拓扑掩码（topo_mask）通过重罚方式施加，使不可行方向权重趋零。

适配点（与 run_innovation_real_data.py 保持一致）：
  1. 物理拓扑掩码同时施加于同期矩阵 W 和各滞后矩阵 A_k
  2. y_grade 并入数据末列，W/A 中 y→其他方向全部 mask
  3. 邻接矩阵 W[i,j] 均表示 i→j 方向
  4. 输出图强制 DAG 后处理（仅对同期图 W；滞后边本质无环）
  5. 标准化在滑窗化之前全局进行

用法：
  python run_dynotears_dag.py [--line xin1|xin2|both] [--epochs N]
                              [--lags K] [--threshold θ] [--lr LR]
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
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config import prepare_data, can_cause

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 超参数 ──────────────────────────────────────────────────────────────────
# 最大滞后阶数：10min 采样 × 3 = 30min 滞后窗口，覆盖浮选工序主要滞后时长
MAX_LAGS = 3

# 训练设置
EPOCHS = 300            # DYNOTEARS 梯度优化需要较多步数收敛
LR = 5e-3               # Adam 学习率；lr 过大容易绕过鞍点，过小收敛慢
BATCH_SIZE = 256        # 线性模型可用较大 batch 加速

# 正则化系数
LAMBDA_W = 0.01         # 同期稀疏惩罚（L1）
LAMBDA_A = 0.005        # 滞后稀疏惩罚（L1，略小于同期，允许更多滞后信号）
LAMBDA_NOTEARS = 0.5    # NOTEARS 无环约束权重（施加于同期图 W）
TOPO_PENALTY = 10.0     # 物理不可行边的重罚系数

# 邻接矩阵阈值（权重绝对值超过此值才保留为边）
DEFAULT_THRESHOLD = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALGO_NAME = "dynotears"


# ─── 物理拓扑掩码 ─────────────────────────────────────────────────────────────

def build_topology_mask(valid_vars, var_to_stage, var_to_group, line):
    """
    构建 (N+1, N+1) 物理因果可行性掩码（含 y_grade 在末列/末行）。
    mask[i, j] = 1 表示 i→j 物理上可行。
    y_grade（索引 N）只能被其他节点指向，自身不指向任何节点。
    """
    N = len(valid_vars)
    mask = np.zeros((N + 1, N + 1), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i != j and can_cause(
                var_to_stage[valid_vars[i]], var_to_stage[valid_vars[j]],
                var_to_group.get(valid_vars[i]), var_to_group.get(valid_vars[j]),
                line,
            ):
                mask[i, j] = 1.0
        # X_i → y_grade
        if can_cause(
            var_to_stage[valid_vars[i]], "Y",
            var_to_group.get(valid_vars[i]), None,
            line,
        ):
            mask[i, N] = 1.0

    # y_grade 不指向任何节点
    mask[N, :] = 0.0
    np.fill_diagonal(mask, 0.0)
    return mask


# ─── DYNOTEARS 模型 ───────────────────────────────────────────────────────────

class DYNOTEARSModel(nn.Module):
    """
    线性 DYNOTEARS 模型。

    参数：
      d:        变量数（含 y_grade）
      max_lags: 最大滞后阶数

    W[i,j] = 同期 i→j 的影响强度。
    A[k][i,j] = 第 k 步滞后 i→j 的影响强度。
    """

    def __init__(self, d: int, max_lags: int = MAX_LAGS):
        super().__init__()
        self.d = d
        self.max_lags = max_lags

        # 同期邻接矩阵（无自环，NOTEARS 约束）
        self.W = nn.Parameter(torch.zeros(d, d))

        # 滞后邻接矩阵列表 A_1, A_2, ..., A_K
        self.A_list = nn.ParameterList([
            nn.Parameter(torch.zeros(d, d)) for _ in range(max_lags)
        ])

    def forward(self, X_t, X_lags):
        """
        X_t:    (B, d) 当前时刻
        X_lags: [(B, d)] × max_lags，从 t-1 到 t-K
        预测值：X_hat_t = X_t @ W + Σ_k X_{t-k} @ A_k
        """
        # 去自环（对角线归零）
        W_eff = self.W * (1 - torch.eye(self.d, device=self.W.device))
        pred = X_t @ W_eff
        for k, (A_k, X_lag) in enumerate(zip(self.A_list, X_lags)):
            pred = pred + X_lag @ A_k
        return pred

    def notears_h(self):
        """NOTEARS 无环约束 h(W) = trace(exp(W∘W)) - d，施加于同期图 W。"""
        M = self.W * self.W
        E = torch.matrix_exp(M)
        return torch.trace(E) - self.d


# ─── 数据准备（滑窗） ─────────────────────────────────────────────────────────

def make_lagged_dataset(X_norm, max_lags):
    """
    将标准化时序矩阵切分为 (X_t, X_{t-1}, ..., X_{t-K}) 对。

    返回：
      X_curr:  (T_eff, d) 当前时刻
      X_lags:  [(T_eff, d)] × max_lags
    """
    T = X_norm.shape[0]
    X_curr = X_norm[max_lags:, :]              # t = max_lags, ..., T-1
    X_lags = [X_norm[max_lags - k - 1: T - k - 1, :] for k in range(max_lags)]
    return X_curr, X_lags


# ─── 训练函数 ─────────────────────────────────────────────────────────────────

def train_dynotears(X_all, topo_mask, epochs=EPOCHS, verbose=True,
                    max_lags=MAX_LAGS, lr=LR):
    """
    训练 DYNOTEARS 模型。

    损失 = MSE
         + λ_W * ||W * topo_mask||_1      （同期可行边稀疏）
         + λ_A * Σ_k||A_k * topo_mask||_1 （滞后可行边稀疏）
         + λ_notears * h(W)^2              （NOTEARS 无环约束）
         + topo_penalty * Σ(|W * anti_topo| + |A_k * anti_topo|)  （不可行边惩罚）

    参数：
      X_all:     (T, d) 含 y_grade 的全局标准化前数据
      topo_mask: (d, d) numpy float32 物理可行性掩码
      epochs:    训练轮数
      verbose:   是否打印进度
      max_lags:  最大滞后阶数
      lr:        Adam 学习率

    返回：
      训练后的 DYNOTEARSModel
    """
    d = X_all.shape[1]
    X_norm = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-8)

    X_curr, X_lags_np = make_lagged_dataset(X_norm, max_lags)

    # 转 Tensor
    X_curr_t = torch.tensor(X_curr, dtype=torch.float32, device=DEVICE)
    X_lags_t = [torch.tensor(xl, dtype=torch.float32, device=DEVICE) for xl in X_lags_np]

    topo_t = torch.tensor(topo_mask, dtype=torch.float32, device=DEVICE)
    diag_m = torch.eye(d, device=DEVICE)
    anti_topo_t = (1.0 - topo_t) * (1.0 - diag_m)

    model = DYNOTEARSModel(d, max_lags).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    # DataLoader for mini-batch
    dataset = torch.utils.data.TensorDataset(X_curr_t, *X_lags_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            b_curr = batch[0]
            b_lags = list(batch[1:])

            opt.zero_grad()
            pred = model(b_curr, b_lags)
            mse = F.mse_loss(pred, b_curr)

            # 稀疏约束（物理可行区域）
            sparse_W = LAMBDA_W * torch.sum(torch.abs(model.W * topo_t))
            sparse_A = sum(
                LAMBDA_A * torch.sum(torch.abs(A_k * topo_t))
                for A_k in model.A_list
            )

            # NOTEARS 无环约束
            h = model.notears_h()
            notears_loss = LAMBDA_NOTEARS * h * h

            # 物理拓扑惩罚（不可行区域归零）
            topo_pen_W = TOPO_PENALTY * torch.sum(torch.abs(model.W * anti_topo_t))
            topo_pen_A = TOPO_PENALTY * sum(
                torch.sum(torch.abs(A_k * anti_topo_t))
                for A_k in model.A_list
            )

            loss = mse + sparse_W + sparse_A + notears_loss + topo_pen_W + topo_pen_A
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        scheduler.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  [DYNOTEARS] Epoch {epoch+1}/{epochs}  "
                  f"Loss={epoch_loss/len(loader):.4f}  h={model.notears_h().item():.6f}")

    return model


# ─── 邻接矩阵提取与 GraphML 输出 ──────────────────────────────────────────────

def extract_and_save(model, valid_vars, topo_mask, line, threshold=DEFAULT_THRESHOLD):
    """
    从训练好的 DYNOTEARS 模型提取邻接矩阵，施加物理掩码，输出 GraphML。

    同期图 W 施加 NOTEARS 无环约束后处理（移除最弱环边）；
    滞后图 A_k 本质无环（过去→现在方向），直接阈值截断。

    GraphML 节点名与 analyze_dag_causal_roles_v4_1.py 中 Y_CANDIDATES 匹配：
      y_grade（同期发现）或 Y_grade（TCDF 命名）→ 此处统一为 y_grade。
    """
    N = len(valid_vars)
    d = N + 1
    all_vars = valid_vars + ["y_grade"]

    with torch.no_grad():
        W_np = model.W.cpu().numpy()
        A_list_np = [A_k.cpu().numpy() for A_k in model.A_list]

    # 应用物理掩码（绝对值后再比较阈值）
    W_np = np.abs(W_np) * topo_mask
    A_list_np = [np.abs(A_k) * topo_mask for A_k in A_list_np]
    np.fill_diagonal(W_np, 0.0)
    for A_k in A_list_np:
        np.fill_diagonal(A_k, 0.0)

    # 打印权重分布
    all_weights = np.concatenate([W_np.flatten()] + [A_k.flatten() for A_k in A_list_np])
    nonzero = all_weights[all_weights > 0]
    if len(nonzero):
        pcts = np.percentile(nonzero, [25, 50, 75, 90, 95])
        print(f"  [DYNOTEARS] 全局非零权重分位数 [25,50,75,90,95]: {pcts.round(4)}")
    else:
        print(f"  [DYNOTEARS] 警告：训练后邻接矩阵全零，请检查超参数或数据")

    # 构建 DiGraph（同期 + 滞后边合并，同期边权重更高时优先使用同期）
    G = nx.DiGraph()
    for var in all_vars:
        G.add_node(var)

    edge_count = 0
    # 同期边
    for i in range(d):
        for j in range(d):
            if i != j and W_np[i, j] > threshold:
                G.add_edge(all_vars[i], all_vars[j], weight=float(W_np[i, j]),
                           lag=0)
                edge_count += 1

    # 滞后边（若同期已有该边，则取最大权重更新；否则新建）
    for k, A_k in enumerate(A_list_np):
        lag = k + 1
        for i in range(d):
            for j in range(d):
                if i != j and A_k[i, j] > threshold:
                    if G.has_edge(all_vars[i], all_vars[j]):
                        # 已有边：更新 weight 为二者最大值
                        existing = G[all_vars[i]][all_vars[j]]["weight"]
                        G[all_vars[i]][all_vars[j]]["weight"] = max(existing, float(A_k[i, j]))
                    else:
                        G.add_edge(all_vars[i], all_vars[j],
                                   weight=float(A_k[i, j]), lag=lag)
                        edge_count += 1

    # DAG 后处理：移除最弱环边（NOTEARS 训练后仍可能有残余同期环）
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
        print(f"  [DYNOTEARS] DAG 后处理：移除 {removed} 条环边")

    out_path = os.path.join(OUT_DIR, f"{ALGO_NAME}_real_dag_{line}.graphml")
    nx.write_graphml(G, out_path)
    print(f"  ✓ [DYNOTEARS] [{line}]: {edge_count} 条边 → {out_path}")
    return G


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run_dynotears(line="xin1", epochs=EPOCHS, threshold=DEFAULT_THRESHOLD,
                  max_lags=MAX_LAGS, lr=LR):
    """
    对单条产线运行 DYNOTEARS 并输出 GraphML。

    参数：
      line:      'xin1' 或 'xin2'
      epochs:    训练轮数
      threshold: 邻接矩阵阈值
      max_lags:  最大滞后阶数
      lr:        Adam 学习率
    """
    print(f"\n{'='*70}")
    print(f"DYNOTEARS 因果发现  [产线={line}]  设备={DEVICE}  滞后阶={max_lags}")
    print(f"{'='*70}")

    t0 = time.time()
    df, valid_vars, var_to_stage, var_to_group = prepare_data(line)
    N = len(valid_vars)
    print(f"变量数: {N}  样本数: {len(df)}")

    all_vars = valid_vars + ["y_grade"]
    X_all = df[all_vars].values.astype(np.float32)  # (T, N+1)

    topo_mask = build_topology_mask(valid_vars, var_to_stage, var_to_group, line)
    n_feasible = int(topo_mask.sum())
    total_possible = (N + 1) ** 2 - (N + 1)
    print(f"物理可行边数: {n_feasible} / {total_possible}  "
          f"({n_feasible / total_possible * 100:.1f}%)")

    print(f"\n--- 训练 DYNOTEARS ---")
    model = train_dynotears(X_all, topo_mask, epochs=epochs, verbose=True,
                            max_lags=max_lags, lr=lr)

    extract_and_save(model, valid_vars, topo_mask, line, threshold=threshold)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - t0
    print(f"\n[{line}] DYNOTEARS 完成，耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DYNOTEARS 时序因果发现（双产线支持，输出 GraphML）"
    )
    parser.add_argument(
        "--line",
        choices=["xin1", "xin2", "both"],
        default="both",
        help="产线选择（默认: both）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"训练轮数（默认: {EPOCHS}）",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=MAX_LAGS,
        help=f"最大滞后阶数（默认: {MAX_LAGS}）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"邻接矩阵阈值（默认: {DEFAULT_THRESHOLD}）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Adam 学习率（默认: {LR}）",
    )
    args = parser.parse_args()

    lines = ["xin1", "xin2"] if args.line == "both" else [args.line]
    for ln in lines:
        run_dynotears(ln, epochs=args.epochs, threshold=args.threshold,
                      max_lags=args.lags, lr=args.lr)
