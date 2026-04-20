"""
run_neural_gc_dag.py
====================
基于神经 Granger 因果（cMLP + 组 LASSO）的时序因果图发现。

适用场景：高维（d ≥ 40）× 非线性 × 大滞后（K_LAGS = 18 → 3 小时）。

核心方法：Component-wise MLP (cMLP) + 组 LASSO
─────────────────────────────────────────────────
对于每个目标变量 j，独立训练一个 cMLP_j：
  - 输入：[X_{t-K}, ..., X_{t-1}] ∈ R^{d_j × K}
    （物理拓扑允许的源变量 × K 步滞后，K_LAGS=18 → 180min ≈ 3h）
  - 第一层权重：W1 ∈ R^{(d_j·K) × H}，按源变量 i 分组
    W1_group[i] = W1[i·K : (i+1)·K, :] ∈ R^{K × H}
  - 组 LASSO 惩罚：λ_group · Σ_i ‖W1_group[i]‖_F
    → 每个源变量的所有滞后权重作为一组，稀疏化源选择
  - 后续层：SiLU + 全连接层 → 非线性建模高阶交互

重要性分数：A[i, j] = ‖W1_group_j[i]‖_F（训练结束后提取）

后处理：
  - 物理拓扑掩码硬置零（`can_cause` 规则）
  - 阈值 → 二值邻接
  - 移除最弱环边（强制 DAG）
  - 输出 GraphML（与 analyze_dag_causal_roles_v4_1.py 兼容）

相比现有算法的优势：
  - vs TCDF：
      显式建模 K=18 步大滞后（工业过程 3h）；
      不将全部历史压缩为单一注意力值，保留各滞后的独立贡献。
  - vs NOTEARS-based（BiAttn/MultiScale-NTS/MB-CUTS）：
      无全局 d×d 邻接矩阵，高维下内存更友好（每目标独立小模型）；
      组 LASSO 直接控制源稀疏性，不依赖 NOTEARS 约束（NOTEARS 对大
      d 的收敛稳定性和计算开销均有挑战）。
  - vs Granger VAR：
      全非线性 MLP（捕捉工业过程的非线性因果机制）；
      保留 Granger 因果的"源变量的全部滞后"语义，比 TCDF 的注意力更直接。
  - vs DYNOTEARS：
      不依赖线性 VAR 假设；组 LASSO 稀疏化比 NOTEARS h(W)=0 约束更
      数值稳定。

超参数（设计说明）：
  K_LAGS   = 18  : 3h@10min；比 TCDF(15) 更大，专为大滞后工业场景
  HIDDEN   = 32  : 适中容量，防过拟合（真实数据 d ~ 40-60）
  EPOCHS   = 100 : 足够的训练轮数
  LR       = 0.005
  LAMBDA_GROUP = 0.02  : 组 LASSO 强度（适当比 L1 更大，组范数单位不同）
  LAMBDA_TOPO  = 5.0   : 物理不可行边的软惩罚（硬后处理会二次保险）
  THRESHOLD    = 0.05  : 重要性分数阈值（‖W1_group[i]‖_F 的尺度与 L2 norm 相当）

用法：
  python run_neural_gc_dag.py --line xin1
  python run_neural_gc_dag.py --line xin2
  python run_neural_gc_dag.py --line both --epochs 150 --k_lags 24 --threshold 0.04
"""

import os
import sys
import time
import warnings

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config import prepare_data, can_cause

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 超参 ──────────────────────────────────────────────────────────────────────
K_LAGS        = 18    # 滞后步数（18 × 10min = 3h，专为大滞后工业场景设计）
HIDDEN        = 32    # 隐层维度
EPOCHS        = 100   # 训练轮数
LR            = 0.005
BATCH_SIZE    = 128
LAMBDA_GROUP  = 0.02  # 组 LASSO 强度
LAMBDA_TOPO   = 5.0   # 物理不可行边惩罚（软约束）
THRESHOLD     = 0.05  # 重要性分数阈值（‖W1_group[i]‖_F）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── cMLP（Component-wise MLP）─────────────────────────────────────────────────

class cMLP(nn.Module):
    """
    针对单个目标变量 j 的 cMLP 模型。

    输入：X_flat ∈ R^{n_src · K_LAGS}
      - 源变量按 (src_0_lag0, src_0_lag1, ..., src_0_lagK,
                   src_1_lag0, ..., src_{n_src-1}_lagK) 顺序展平

    第一层：W1 ∈ R^{H × (n_src·K)} → 重整为 R^{n_src, K, H} 用于组范数计算
    后续层：SiLU + FC × 2 → 输出标量（目标变量预测值）

    组 LASSO 惩罚：Σ_i ‖W1_group[i]‖_F（i = 源变量索引）
    """
    def __init__(self, n_src: int, k_lags: int, hidden: int = HIDDEN):
        super().__init__()
        self.n_src  = n_src
        self.k_lags = k_lags
        self.hidden = hidden

        # 第一层（含组结构，按源变量分组）
        self.W1 = nn.Linear(n_src * k_lags, hidden, bias=True)
        # 后续非线性层
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_src * k_lags)
        returns: (B,)
        """
        return self.mlp(self.W1(x)).squeeze(-1)

    def group_lasso_penalty(self) -> torch.Tensor:
        """
        组 LASSO：Σ_i ‖W1.weight[:, i*K : (i+1)*K]‖_F

        W1.weight shape = (H, n_src * k_lags)
        reshape → (H, n_src, k_lags) → transpose → (n_src, k_lags, H)
        group_norm[i] = ‖W1_group[i]‖_F，和 → 标量
        """
        W = self.W1.weight  # (H, n_src * k_lags)
        W_grouped = W.view(self.hidden, self.n_src, self.k_lags)  # (H, n_src, k_lags)
        W_grouped = W_grouped.permute(1, 2, 0)                    # (n_src, k_lags, H)
        group_norms = W_grouped.norm(dim=[1, 2])                  # (n_src,)
        return group_norms.sum()

    @torch.no_grad()
    def importance(self) -> np.ndarray:
        """返回 (n_src,) 重要性向量 = ‖W1_group[i]‖_F"""
        W = self.W1.weight  # (H, n_src * k_lags)
        W_grouped = W.view(self.hidden, self.n_src, self.k_lags)
        W_grouped = W_grouped.permute(1, 2, 0)                    # (n_src, k_lags, H)
        return W_grouped.norm(dim=[1, 2]).cpu().numpy()


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────

def build_lag_matrix(X: np.ndarray, k: int) -> np.ndarray:
    """
    构建滞后特征矩阵（向量化，无 Python 循环）。

    参数:
        X: (T, d) 已标准化的时序矩阵
        k: 滞后步数
    返回:
        X_lag: (T-k, d*k)  第 t 行 = flatten(X[t:t+k])（过去 k 步）
        X_target: (T-k, d)  第 t 行 = X[t+k]（对应预测目标）
    """
    T, d = X.shape
    n = T - k
    # row_idx[i, s] = i + s，shape (n, k)
    row_idx = np.arange(n)[:, None] + np.arange(k)[None, :]
    windows = X[row_idx]                           # (n, k, d)
    # reshape to (n, d*k)：source 0 all lags first, then source 1, ...
    # 转置 windows → (n, d, k) 再 reshape → (n, d*k)
    X_lag = windows.transpose(0, 2, 1).reshape(n, d * k).astype(np.float32)
    X_target = X[k:].astype(np.float32)
    return X_lag, X_target


def build_topo_mask(valid_vars: list, var_to_stage: dict,
                    var_to_group: dict, line: str) -> np.ndarray:
    """
    构建 (N+1, N+1) 物理拓扑掩码（含 y_grade 为最后维度）。
    mask[i, j] = 1 表示 valid_vars[i] → valid_vars[j] 物理可行。
    mask[i, N] = 1 表示 valid_vars[i] → y_grade 可行。
    mask[N, :] = 0（y_grade 不影响任何节点）。
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
        if can_cause(
            var_to_stage[valid_vars[i]], "Y",
            var_to_group.get(valid_vars[i]), None, line,
        ):
            mask[i, N] = 1.0
    np.fill_diagonal(mask, 0.0)
    return mask


# ─── 训练单个目标 ──────────────────────────────────────────────────────────────

def train_one_target(
    j: int,
    target_name: str,
    X_lag_all: np.ndarray,     # (n_samples, (N+1) * K) 包含 y_grade
    X_target_all: np.ndarray,  # (n_samples, N+1)
    allowed_src: list,         # 物理允许的源变量索引（相对于全变量 N+1 维）
    n_total: int,              # N+1（总变量数含 y_grade）
    k_lags: int,
    lambda_group: float,
    lambda_topo: float,
    epochs: int,
    verbose: bool = False,
) -> np.ndarray:
    """
    训练目标 j 的 cMLP，返回 (n_total,) 重要性向量（非允许源为 0）。

    物理拓扑约束：
      - 在 `allowed_src` 范围内提取输入特征（减少输入维度，直接硬约束）
      - `lambda_topo` 软惩罚对输入维度无效，但用于兼容性
    """
    n_src = len(allowed_src)
    if n_src == 0:
        return np.zeros(n_total, dtype=np.float32)

    # 从完整 lag 矩阵中提取允许源的特征列
    # X_lag_all 排列：src_0 的 k 列，src_1 的 k 列，...（见 build_lag_matrix 转置逻辑）
    col_idx = []
    for src_i in allowed_src:
        col_idx.extend(range(src_i * k_lags, (src_i + 1) * k_lags))
    X_in = X_lag_all[:, col_idx]      # (n_samples, n_src * k_lags)
    y_in = X_target_all[:, j]         # (n_samples,)

    xb = torch.tensor(X_in, dtype=torch.float32, device=DEVICE)
    yb = torch.tensor(y_in, dtype=torch.float32, device=DEVICE)

    dataset = torch.utils.data.TensorDataset(xb, yb)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = cMLP(n_src, k_lags).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            pred  = model(bx)
            loss  = nn.functional.mse_loss(pred, by)
            loss += lambda_group * model.group_lasso_penalty()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    [目标 '{target_name}'] "
                  f"Epoch {epoch+1}/{epochs}  Loss={total_loss/len(loader):.4f}")

    # 提取重要性并映射回全局索引
    imp_local  = model.importance()       # (n_src,)
    imp_global = np.zeros(n_total, dtype=np.float32)
    for local_i, global_i in enumerate(allowed_src):
        imp_global[global_i] = imp_local[local_i]

    return imp_global


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def run_neural_gc(line: str = "xin1", epochs: int = EPOCHS,
                  k_lags: int = K_LAGS, threshold: float = THRESHOLD):
    """
    对单条产线运行 NeuralGC 因果发现并输出 GraphML。

    参数:
        line:      'xin1' 或 'xin2'
        epochs:    每目标的训练轮数
        k_lags:    滞后步数（默认 K_LAGS=18，对应 3h@10min）
        threshold: 重要性分数阈值（‖W1_group‖_F 尺度）
    """
    print("=" * 70)
    print(f"NeuralGC 因果发现（cMLP + 组 LASSO）[产线={line}]  设备={DEVICE}")
    print(f"K_LAGS={k_lags}  HIDDEN={HIDDEN}  EPOCHS={epochs}  "
          f"λ_group={LAMBDA_GROUP}  阈值={threshold}")
    print("=" * 70)

    t0 = time.time()
    df, valid_vars, var_to_stage, var_to_group = prepare_data(line)
    N = len(valid_vars)
    all_vars = valid_vars + ["y_grade"]   # y_grade 位于索引 N
    D = N + 1                             # 总维度（含 y_grade）
    print(f"变量数（含 y_grade）: {D}  样本数: {len(df)}")

    # 全局标准化（在窗口化之前）
    X_all = df[all_vars].values.astype(np.float32)
    X_norm = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-8)

    # 构建物理拓扑掩码 (D, D)
    topo_mask = build_topo_mask(valid_vars, var_to_stage, var_to_group, line)
    n_feasible = int(topo_mask.sum())
    print(f"物理可行边数: {n_feasible} / {D * (D - 1)}")

    # 构建滞后特征矩阵
    print(f"\n构建滞后特征矩阵（K_LAGS={k_lags}，{k_lags}×10min = {k_lags*10}min）...")
    X_lag, X_target = build_lag_matrix(X_norm, k_lags)
    print(f"  X_lag 形状: {X_lag.shape}  X_target 形状: {X_target.shape}")

    # 重要性矩阵：A[i, j] = 源 i → 目标 j 的重要性
    A = np.zeros((D, D), dtype=np.float32)

    print(f"\n训练各目标 cMLP（共 {D} 个目标）...")
    for j, target_name in enumerate(all_vars):
        # 允许的源变量：topo_mask[:, j] == 1
        allowed_src = [i for i in range(D) if topo_mask[i, j] > 0]
        if not allowed_src:
            continue

        print(f"  [{j+1}/{D}] 目标='{target_name}'  允许源数={len(allowed_src)}", end="", flush=True)
        imp = train_one_target(
            j=j,
            target_name=target_name,
            X_lag_all=X_lag,
            X_target_all=X_target,
            allowed_src=allowed_src,
            n_total=D,
            k_lags=k_lags,
            lambda_group=LAMBDA_GROUP,
            lambda_topo=LAMBDA_TOPO,
            epochs=epochs,
            verbose=False,
        )
        A[:, j] = imp

        # 打印最强源
        nonzero_imp = [(all_vars[i], imp[i]) for i in range(D) if imp[i] > 0]
        nonzero_imp.sort(key=lambda x: x[1], reverse=True)
        top3 = ", ".join(f"{v}={s:.3f}" for v, s in nonzero_imp[:3])
        print(f"  → 前3源: {top3}" if top3 else "  → 无非零重要性")

    # 二次保险：物理掩码硬置零（训练时已通过输入选择硬约束，此处确认无泄露）
    A = A * topo_mask
    np.fill_diagonal(A, 0.0)

    # 打印非零权重分位数供阈值调优参考
    nonzero = A[A > 0]
    if len(nonzero) > 0:
        pcts = np.percentile(nonzero, [25, 50, 75, 90, 95])
        print(f"\n[NeuralGC] A 非零重要性分位数 [25,50,75,90,95]: {pcts.round(4)}")
    else:
        print("\n[NeuralGC] 警告：重要性矩阵全零，请检查数据或调整超参数")

    # 构建有向图
    G = nx.DiGraph()
    for var in valid_vars:
        G.add_node(var)
    G.add_node("y_grade")

    edge_count = 0
    for i in range(D):
        for j in range(D):
            if i != j and A[i, j] > threshold:
                src_name = all_vars[i]
                dst_name = all_vars[j]
                G.add_edge(src_name, dst_name, weight=float(A[i, j]))
                edge_count += 1

    # 强制 DAG：移除最弱环边
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
        print(f"[NeuralGC] DAG 后处理：移除 {removed} 条最弱环边，图现为 DAG ✓")
    else:
        print("[NeuralGC] 图已是 DAG，无需后处理 ✓")

    # 写 GraphML
    out_graphml = os.path.join(OUT_DIR, f"neural_gc_real_dag_{line}.graphml")
    nx.write_graphml(G, out_graphml)

    # 写重要性 CSV（对 Y 影响排序）
    import pandas as pd
    df_imp = pd.DataFrame({
        "Source_Var":     all_vars,
        "Importance_To_Y": A[:, N],   # 对 y_grade 的重要性（索引 N）
    })
    df_imp["Stage"] = df_imp["Source_Var"].map(
        lambda v: var_to_stage.get(v, "Y") if v != "y_grade" else "Y"
    )
    df_imp = df_imp[df_imp["Importance_To_Y"] > 0].sort_values(
        "Importance_To_Y", ascending=False
    )
    out_csv = os.path.join(OUT_DIR, f"neural_gc_effects_on_y_{line}.csv")
    df_imp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    print(f"\n✓ [NeuralGC] [{line}]: {edge_count} 条边  耗时 {elapsed:.1f}s")
    print(f"  GraphML → {out_graphml}")
    print(f"  效应 CSV → {out_csv}")
    if len(df_imp) > 0:
        print(f"\n====== 对 Y 影响最强的变量（NeuralGC）[{line}] ======")
        print(df_imp.head(20).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuralGC 因果发现（cMLP + 组 LASSO，高维/非线性/大滞后场景）"
    )
    parser.add_argument("--line", choices=["xin1", "xin2", "both"], default="both",
                        help="产线选择（默认: both）")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"每目标训练轮数（默认: {EPOCHS}）")
    parser.add_argument("--k_lags", type=int, default=K_LAGS,
                        help=f"滞后步数 K（默认: {K_LAGS}，即 {K_LAGS*10}min）")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"重要性分数阈值（默认: {THRESHOLD}）")
    args = parser.parse_args()

    lines = ["xin1", "xin2"] if args.line == "both" else [args.line]
    for ln in lines:
        run_neural_gc(ln, epochs=args.epochs, k_lags=args.k_lags,
                      threshold=args.threshold)
