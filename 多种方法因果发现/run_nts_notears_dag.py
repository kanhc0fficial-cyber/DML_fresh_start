"""
run_nts_notears_dag.py
======================
采用 NTS-NOTEARS（非线性时序 NOTEARS）架构构建工业因果图。
【双产线版】：支持 line='xin1' / line='xin2' 分别运行。
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
from tqdm import tqdm

warnings.filterwarnings("ignore")
import builtins

def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

# ─── 导入公共配置 ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config import prepare_data, can_cause

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 15
BATCH_SIZE = 128
EPOCHS = 40
LR = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── NTS-NOTEARS (Non-linear Time Series NOTEARS) ────────────────────────
class NTS_NOTEARSNet(nn.Module):
    def __init__(self, d, mask):
        super().__init__()
        self.d = d
        self.mask = torch.tensor(mask, dtype=torch.float32).to(DEVICE)
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.01, 0.01))
        # 向量化：通过分组一维卷积 (groups=d) 同步计算所有节点的独立 MLP
        self.conv1 = nn.Conv1d(d, d * 16, kernel_size=WINDOW_SIZE, groups=d)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d * 16, d, kernel_size=1, groups=d)

    def forward(self, x):
        W_masked = self.W * self.mask
        x_agg = torch.matmul(x, W_masked)
        x_agg = x_agg.transpose(1, 2)
        h = self.relu(self.conv1(x_agg))
        pred = self.conv2(h).squeeze(2)
        return pred

    def notears_penalty(self):
        W_masked = self.W * self.mask
        M = W_masked * W_masked
        E = torch.matrix_exp(M)
        return torch.trace(E) - self.d


def build_windows(X):
    T, d = X.shape
    xs, ys = [], []
    for start in range(0, T - WINDOW_SIZE):
        xs.append(X[start:start + WINDOW_SIZE, :])
        ys.append(X[start + WINDOW_SIZE, :])
    return np.array(xs), np.array(ys)


def train_causal_model(wx, wy, d, constraint_mask):
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始训练 NTS-NOTEARS 因果发现引擎")
    model = NTS_NOTEARSNet(d, constraint_mask).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)

    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)

    ds_tensor = torch.utils.data.TensorDataset(xb, yb)
    loader = torch.utils.data.DataLoader(ds_tensor, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in tqdm(range(EPOCHS), desc="Training NTS-NOTEARS"):
        total_loss = 0
        for b_x, b_y in loader:
            opt.zero_grad()
            pred = model(b_x)
            mse = nn.functional.mse_loss(pred, b_y)
            h_val = model.notears_penalty()
            # [修正] 进一步降低稀疏惩罚至 0.001，允许更多潜在因果边通过
            loss = mse + 0.001 * torch.sum(torch.abs(model.W)) + 0.5 * h_val * h_val

            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  [NTS_NOTEARS] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    final_adj = (model.W * model.mask).detach().cpu().numpy()
    return final_adj


def run_nts_notears(line: str = "xin1"):
    print("=" * 70)
    print(f"NTS-NOTEARS 高维时序因果网络构建 [产线 = {line}]")
    print("=" * 70)

    df, valid_vars, var_to_stage, var_to_group = prepare_data(line)

    vars_with_y = valid_vars + ["y_grade"]
    d = len(vars_with_y)
    print(f"构建总体图维度 (包括 Y_grade): {d} * {d}")

    # ── 构建时空拓扑约束掩码 ──
    constraint_mask = np.zeros((d, d))
    edges_allowed = 0
    for i, src in enumerate(vars_with_y):
        s_stage = 'Y' if src == "y_grade" else var_to_stage.get(src, -1)
        s_group = None if src == "y_grade" else var_to_group.get(src)
        for j, dst in enumerate(vars_with_y):
            d_stage = 'Y' if dst == "y_grade" else var_to_stage.get(dst, -1)
            d_group = None if dst == "y_grade" else var_to_group.get(dst)
            if can_cause(s_stage, d_stage, s_group, d_group, line):
                constraint_mask[i, j] = 1.0
                edges_allowed += 1

    print(f"掩码允许的潜在路径占比: {edges_allowed/(d*d)*100:.2f}%")

    X_np = df[vars_with_y].values
    X_norm = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)

    adj_notears = train_causal_model(wx, wy, d, constraint_mask)

    # ── 结果导出 ──
    threshold = 0.05
    G = nx.DiGraph()
    G.add_nodes_from(vars_with_y)

    for i in range(d):
        for j in range(d):
            val = adj_notears[i, j]
            if abs(val) > threshold and i != j:
                G.add_edge(vars_with_y[i], vars_with_y[j], weight=float(val))

    out_graphml = os.path.join(OUT_DIR, f"nts_notears_dag_{line}.graphml")
    nx.write_graphml(G, out_graphml)
    print(f"\n[NTS-NOTEARS 时空受限因果图] 导出成功: {out_graphml}")

    y_idx = d - 1
    y_weights = np.abs(adj_notears[:, y_idx])

    res = pd.DataFrame({"Source_Var": vars_with_y, "Causal_Score": y_weights})
    res["Stage"] = res["Source_Var"].map(
        lambda x: 'Y' if x == 'y_grade' else var_to_stage.get(x, "Unknown"))
    res["Group"] = res["Source_Var"].map(
        lambda x: 'Y' if x == 'y_grade' else var_to_group.get(x, "?"))
    res = res[res["Causal_Score"] > threshold].sort_values("Causal_Score", ascending=False)

    out_csv = os.path.join(OUT_DIR, f"nts_notears_effects_on_y_{line}.csv")
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(res.head(20).to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NTS-NOTEARS 双产线因果发现")
    parser.add_argument("--line", choices=["xin1", "xin2", "both"], default="both",
                        help="运行哪条产线: xin1/xin2/both (默认 both)")
    args = parser.parse_args()

    lines = ["xin1", "xin2"] if args.line == "both" else [args.line]
    for ln in lines:
        run_nts_notears(ln)
