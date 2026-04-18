"""
run_tcdf_space_time_dag.py
==========================
基于时间和空间拓扑限制，运行 TCDF (注意力卷积神经网络) 构建因果 DAG。
【双产线版】：支持 line='xin1' / line='xin2'，Y 分别为新1/新2精矿品位。

主轴物料流：2 -> 3 -> 4 -> 5 -> 6
终端分流：6 -> 7, 6 -> 8
平行辅助：0, 1 -> 主流程
目标变量：Y (精矿品位)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import warnings
import builtins

def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

warnings.filterwarnings("ignore")

# ─── 导入公共配置 ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config import prepare_data, can_cause

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")
os.makedirs(OUT_DIR, exist_ok=True)

# TCDF 深度学习超参
EPOCHS = 30
BATCH_SIZE = 64
WINDOW_SIZE = 15
KERNEL_SIZE = 4
DILATION_C = 2
HIDDEN_LAYERS = 3
LR = 0.005
LAMBDA_SPARSE = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── TCDF 模型定义 ──────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCDFNet(nn.Module):
    def __init__(self, n_vars):
        super().__init__()
        # [修正] 放弃初始全联通的 1.0 先验，改用较小值随机探索，防止未充分训练时出现大量假阳性因果边
        self.attention = nn.Parameter(torch.empty(n_vars).uniform_(0.0, 0.1))
        layers = [CausalConv1d(n_vars, 32, KERNEL_SIZE, 1), nn.ReLU()]
        for l in range(1, HIDDEN_LAYERS + 1):
            layers += [CausalConv1d(32, 32, KERNEL_SIZE, DILATION_C ** l), nn.ReLU()]
        layers.append(nn.Conv1d(32, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        attn = torch.abs(self.attention)
        return self.net(x * attn.view(1, -1, 1))


def train_tcdf_simple(X_win, y_win, N):
    model = TCDFNet(N).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    xb = torch.tensor(X_win, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(y_win, dtype=torch.float32).to(DEVICE)

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(xb).squeeze(1)
        loss = nn.functional.mse_loss(pred, yb) + LAMBDA_SPARSE * model.attention.abs().sum()
        loss.backward()
        opt.step()

    return torch.abs(model.attention).detach().cpu().numpy()


def build_windows(X, y):
    T = X.shape[0]
    step = WINDOW_SIZE
    xs, ys = [], []
    for start in range(0, T - WINDOW_SIZE + 1, step):
        xs.append(X[start:start + WINDOW_SIZE].T)
        ys.append(y[start:start + WINDOW_SIZE])
    return np.stack(xs), np.stack(ys)


def run_tcdf_space_time_dag(line: str = "xin1"):
    print("=" * 60)
    print(f"TCDF 因果发现 [产线 = {line}]")
    print("=" * 60)

    df, valid_vars, var_to_stage, var_to_group = prepare_data(line)

    N = len(valid_vars)
    adj_matrix = np.zeros((N, N))

    print(f"\n构建变量内因果矩阵 (共 {N} 个节点，独立训练)...")
    t0 = time.time()

    X_np = df[valid_vars].values
    X_norm = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-8)

    for i in range(N):
        target_name = valid_vars[i]
        target_stage = var_to_stage[target_name]
        target_group = var_to_group.get(target_name)

        allowed_src_indices = [
            s for s in range(N)
            if s != i and can_cause(
                var_to_stage[valid_vars[s]], target_stage,
                var_to_group.get(valid_vars[s]), target_group,
                line
            )
        ]

        if not allowed_src_indices:
            continue

        y_target = X_norm[:, i]
        X_inputs = X_norm[:, allowed_src_indices]
        wx, wy = build_windows(X_inputs, y_target)
        attn = train_tcdf_simple(wx, wy, len(allowed_src_indices))

        for local_idx, global_src_idx in enumerate(allowed_src_indices):
            adj_matrix[global_src_idx, i] = attn[local_idx]

        print(f"  已完成 {i+1}/{N} 节点... ('{target_name}')")

    print(f"变量间拓扑矩阵构建完成，耗时 {time.time()-t0:.1f}s")

    # ── 单独计算对最终 Y 的因果力 ──
    print("\n计算所有受测环节对最终目标（Y_grade）的动态影响...")
    y_final = df["y_grade"].values
    y_final_norm = (y_final - y_final.mean()) / (y_final.std() + 1e-8)

    allowed_y_src_indices = [
        s for s in range(N)
        if can_cause(var_to_stage[valid_vars[s]], 'Y',
                     var_to_group.get(valid_vars[s]), None, line)
    ]

    attn_y_global = np.zeros(N)
    if allowed_y_src_indices:
        X_inputs_y = X_norm[:, allowed_y_src_indices]
        wx, wy = build_windows(X_inputs_y, y_final_norm)
        attn_local_y = train_tcdf_simple(wx, wy, len(allowed_y_src_indices))
        for local_idx, global_src_idx in enumerate(allowed_y_src_indices):
            attn_y_global[global_src_idx] = attn_local_y[local_idx]

    # ── 组装图并输出 ──
    G = nx.DiGraph()
    G.add_nodes_from(valid_vars)
    G.add_node("Y_grade")

    threshold = 0.05
    for i in range(N):
        for j in range(N):
            if adj_matrix[i, j] > threshold:
                G.add_edge(valid_vars[i], valid_vars[j], weight=float(adj_matrix[i, j]))

    for i, w in enumerate(attn_y_global):
        if w > threshold:
            G.add_edge(valid_vars[i], "Y_grade", weight=float(w))

    out_graphml = os.path.join(OUT_DIR, f"tcdf_space_time_dag_{line}.graphml")
    nx.write_graphml(G, out_graphml)
    print(f"\n[TCDF 因果图] 写入完成: {out_graphml}")

    res = pd.DataFrame({"Source_Var": valid_vars, "Attention_To_Y": attn_y_global})
    res["Stage"] = res["Source_Var"].map(var_to_stage)
    res["Group"] = res["Source_Var"].map(var_to_group)
    res = res[res["Attention_To_Y"] > 0.05].sort_values("Attention_To_Y", ascending=False)

    out_csv = os.path.join(OUT_DIR, f"tcdf_space_time_effects_on_y_{line}.csv")
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n====== 发现对 Y 具有最强 TCDF 注意力的变量 [{line}] ======")
    print(res.head(20).to_string(index=False))
    print(f"\n量化结果保存至: {out_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TCDF 双产线因果发现")
    parser.add_argument("--line", choices=["xin1", "xin2", "both"], default="both",
                        help="运行哪条产线: xin1/xin2/both (默认 both)")
    args = parser.parse_args()

    lines = ["xin1", "xin2"] if args.line == "both" else [args.line]
    for ln in lines:
        run_tcdf_space_time_dag(ln)
