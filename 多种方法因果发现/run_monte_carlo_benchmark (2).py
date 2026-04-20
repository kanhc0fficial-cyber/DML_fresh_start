"""
run_monte_carlo_benchmark.py  [DML-CQS 增强版]
===============================================
蒙特卡洛模拟 - 因果发现算法鲁棒性压力测试

新增内容（相对修正版）：
  [DML1] 引入 dml_causal_metrics 模块，支持 DML 控制质量得分
  [DML2] run_single_experiment 新增 DML-CQS 三子指标及综合得分
  [DML3] 统计报告、Markdown 报告、对比表均包含 DML-CQS 列
  [DML4] generate_comparison_report 新增"DML 控制质量"专栏

修正清单（沿用修正版）：
  [B1-B6] 同修正版，不再重复注释
"""

import gc
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# [DML1] 引入新指标模块（与本文件同目录）
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
# dml_causal_metrics.py 在本文件所在目录
sys.path.insert(0, _this_dir)
from dml_causal_metrics import compute_dml_cqs_multi

# synthetic_dag_generator.py 在兄弟目录"因果的发现算法理论验证"下
_dag_gen_dir = os.path.join(_parent_dir, "因果的发现算法理论验证")
if _dag_gen_dir not in sys.path:
    sys.path.insert(0, _dag_gen_dir)

from synthetic_dag_generator import (
    SyntheticDAGGenerator,
    compute_dag_metrics,
    evaluate_causal_structure_recovery,
    evaluate_multiple_treatments
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果", "monte_carlo_benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 超参数配置 ──────────────────────────────────────────────────────────
N_EXPERIMENTS            = 50
N_NODES                  = 20
N_SAMPLES                = 1000
NOISE_SCALE              = 0.1
GRAPH_TYPE               = 'layered'
NOISE_TYPE               = 'heteroscedastic'
USE_INDUSTRIAL_FUNCTIONS = True
ADD_TIME_LAG             = True

EVALUATE_CAUSAL_ROLES    = True
N_TREATMENT_SAMPLES      = 4

# [DML2] DML 评估配置
EVALUATE_DML_CQS         = True   # 是否计算 DML 控制质量得分
DML_OUTCOME_IDX          = N_NODES - 1  # 默认最后一个节点为结果变量
DML_THRESHOLD            = 0.05         # 二值化阈值

WINDOW_SIZE  = 10
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 0.001
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")
print(f"DAG 类型: {GRAPH_TYPE}")
print(f"噪声类型: {NOISE_TYPE}")
print(f"工业函数: {USE_INDUSTRIAL_FUNCTIONS}")
print(f"DML-CQS 评估: {EVALUATE_DML_CQS}")


# ─── CUTS+ 模型 ──────────────────────────────────────────────────────────
class CUTSPlusNet(nn.Module):
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
        H_agg = torch.einsum('bsf, st -> btf', hiddens, W_attn)
        H_agg_flat = H_agg.reshape(B, self.d * 16, 1)
        pred = self.decoder(H_agg_flat).squeeze(2)
        return pred


# ─── NTS-NOTEARS 模型 ────────────────────────────────────────────────────
class NTS_NOTEARSNet(nn.Module):
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


# ═══════════════════════════════════════════════════════════════════════════
# 创新方案一：BiAttn-CUTS
# ═══════════════════════════════════════════════════════════════════════════
class BiAttnCUTSNet(nn.Module):
    def __init__(self, d, n_heads: int = 4, d_model: int = 16):
        super().__init__()
        self.d = d
        self.d_model = d_model
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.05, 0.05))
        self.tau = nn.Parameter(torch.zeros(1))
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=32, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Linear(d_model, 1)

    def _tau_pos(self):
        return F.softplus(self.tau)

    def forward(self, x):
        B, T, d = x.shape
        x_flat = x.transpose(1, 2).contiguous().reshape(B * d, T, 1)
        x_emb  = self.input_proj(x_flat)
        h      = self.transformer(x_emb)
        h_last = h[:, -1, :]
        node_feats = h_last.reshape(B, d, self.d_model)
        tau_pos = self._tau_pos()
        W_soft  = torch.sigmoid(self.W * tau_pos)
        W_soft  = W_soft * (1 - torch.eye(d, device=W_soft.device))
        agg  = torch.einsum('ij,bif->bjf', W_soft, node_feats)
        pred = self.out_proj(agg).squeeze(-1)
        return pred

    def notears_penalty(self):
        M = self.W * self.W
        E = torch.matrix_exp(M)
        return torch.trace(E) - self.d


def train_biattn_cuts(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)
    model = BiAttnCUTSNet(d).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb), batch_size=BATCH_SIZE, shuffle=True
    )
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for b_x, b_y in loader:
            opt.zero_grad()
            pred    = model(b_x)
            mse     = F.mse_loss(pred, b_y)
            tau_pos = model._tau_pos()
            W_soft  = torch.sigmoid(model.W * tau_pos)
            sparse  = torch.sum(W_soft * (1 - torch.eye(d, device=W_soft.device)))
            h_val   = model.notears_penalty()
            loss    = mse + 0.5 * sparse + 2.0 * h_val * h_val
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [BiAttn-CUTS] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")
    with torch.no_grad():
        tau_pos  = model._tau_pos()
        adj_pred = torch.sigmoid(model.W * tau_pos).cpu().numpy()
    np.fill_diagonal(adj_pred, 0)
    return adj_pred


# ═══════════════════════════════════════════════════════════════════════════
# 创新方案二：MultiScale-NTS（多尺度空洞卷积版）
# ═══════════════════════════════════════════════════════════════════════════
class MultiScaleNTSNet(nn.Module):
    """
    多尺度空洞卷积（Multi-Scale Dilated Convolution, MSDC）
    + 可学习融合权重 + 共享 NOTEARS 约束

    空洞率设计：DILATION_RATES = [1, 2, 4]
      - dilation=1: 感受野 = 3（短程邻近依赖）
      - dilation=2: 感受野 = 5（中程时延依赖）
      - dilation=4: 感受野 = 9（长程跨窗依赖，几乎覆盖 WINDOW_SIZE=10）

    理论依据：
      1. 感受野多样性：不同空洞率等价于多分辨率时间分析，在同一 kernel_size=3
         下以指数级扩大感受野，参数量与标准多尺度卷积完全相同。
      2. Granger 因果完整性：工业时序数据存在多阶因果时延，空洞卷积可在
         单层内同时捕获短/中/长程 Granger 依赖，比大核卷积更参数高效。
      3. 与 NOTEARS 兼容性：空洞卷积仅改变特征提取分支，共享的可微 DAG 约束
         矩阵 W 完全不受影响，理论框架保持自洽。

    alpha 初始化为 zeros：softmax(zeros) = 均匀分布，梯度从对称点出发，
    收敛更稳定。
    """
    DILATION_RATES = [1, 2, 4]   # 对应感受野: 3, 5, 9

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.01, 0.01))
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d, d * 8, kernel_size=3, dilation=r, groups=d),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(d * 8, d, kernel_size=1, groups=d)
            )
            for r in self.DILATION_RATES
        ])
        self.alpha = nn.Parameter(torch.zeros(len(self.DILATION_RATES)))

    def forward(self, x):
        x_agg = torch.matmul(x, self.W)
        x_t   = x_agg.transpose(1, 2)
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


def train_multiscale_nts(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)
    model = MultiScaleNTSNet(d).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb), batch_size=BATCH_SIZE, shuffle=True
    )
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for b_x, b_y in loader:
            opt.zero_grad()
            pred  = model(b_x)
            mse   = F.mse_loss(pred, b_y)
            h_val = model.notears_penalty()
            loss  = mse + 0.01 * torch.sum(torch.abs(model.W)) + 2.0 * h_val * h_val
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            alpha_disp = torch.softmax(model.alpha, dim=0).detach().cpu().numpy().round(2)
            print(f"  [MultiScale-NTS] Epoch {epoch+1}/{EPOCHS} - "
                  f"Loss: {total_loss/len(loader):.4f} | α={alpha_disp}")
    adj_pred = model.W.detach().cpu().numpy()
    return adj_pred


# ═══════════════════════════════════════════════════════════════════════════
# 创新方案三：MB-CUTS
# ═══════════════════════════════════════════════════════════════════════════
def _compute_mb_mask(X_norm: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    d = X_norm.shape[1]
    corr_matrix, _ = spearmanr(X_norm)
    corr = np.array(corr_matrix)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    np.fill_diagonal(corr, 0.0)
    mb_mask = np.zeros((d, d), dtype=np.float32)
    k = max(1, int(np.ceil(d * keep_ratio)))
    for j in range(d):
        col     = np.abs(corr[:, j])
        top_idx = np.argsort(col)[-k:]
        mb_mask[top_idx, j] = 1.0
    return mb_mask


def train_mb_cuts(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)
    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb), batch_size=BATCH_SIZE, shuffle=True
    )
    mb_mask   = _compute_mb_mask(X_norm, keep_ratio=0.5)
    mb_mask_t = torch.tensor(mb_mask, dtype=torch.float32).to(DEVICE)
    non_mb_t  = torch.tensor(1.0 - mb_mask, dtype=torch.float32).to(DEVICE)

    model_rough = CUTSPlusNet(d).to(DEVICE)
    opt_rough   = optim.Adam(model_rough.parameters(), lr=LR)
    for epoch in range(10):
        for b_x, b_y in loader:
            opt_rough.zero_grad()
            pred    = model_rough(b_x)
            W_in_mb = model_rough.W * mb_mask_t
            loss    = F.mse_loss(pred, b_y) + 0.01 * torch.sum(torch.abs(W_in_mb))
            loss.backward()
            opt_rough.step()
    adj_rough = torch.abs(model_rough.W).detach().cpu().numpy() * mb_mask

    model_fine = NTS_NOTEARSNet(d).to(DEVICE)
    with torch.no_grad():
        model_fine.W.data = torch.tensor(adj_rough, dtype=torch.float32).to(DEVICE) * 0.1
    opt_fine = optim.Adam(model_fine.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for b_x, b_y in loader:
            opt_fine.zero_grad()
            pred       = model_fine(b_x)
            mse        = F.mse_loss(pred, b_y)
            h_val      = model_fine.notears_penalty()
            sparse     = torch.sum(torch.abs(model_fine.W))
            mb_penalty = torch.sum(torch.abs(model_fine.W * non_mb_t))
            loss       = mse + 0.01 * sparse + 2.0 * h_val * h_val + 5.0 * mb_penalty
            loss.backward()
            opt_fine.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [MB-CUTS] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")
    return model_fine.W.detach().cpu().numpy()


# ─── 基线算法 ────────────────────────────────────────────────────────────
def build_windows(X):
    T, d = X.shape
    xs, ys = [], []
    for start in range(0, T - WINDOW_SIZE):
        xs.append(X[start:start + WINDOW_SIZE, :])
        ys.append(X[start + WINDOW_SIZE, :])
    return np.array(xs), np.array(ys)


def train_cuts_plus(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)
    model = CUTSPlusNet(d).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb), batch_size=BATCH_SIZE, shuffle=True
    )
    for epoch in range(EPOCHS):
        for b_x, b_y in loader:
            opt.zero_grad()
            pred = model(b_x)
            loss = F.mse_loss(pred, b_y) + 0.01 * torch.sum(torch.abs(model.W))
            loss.backward()
            opt.step()
    return torch.abs(model.W).detach().cpu().numpy()


def train_nts_notears(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)
    model = NTS_NOTEARSNet(d).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb), batch_size=BATCH_SIZE, shuffle=True
    )
    for epoch in range(EPOCHS):
        for b_x, b_y in loader:
            opt.zero_grad()
            pred  = model(b_x)
            mse   = F.mse_loss(pred, b_y)
            h_val = model.notears_penalty()
            loss  = mse + 0.001 * torch.sum(torch.abs(model.W)) + 0.5 * h_val * h_val
            loss.backward()
            opt.step()
    return model.W.detach().cpu().numpy()


def train_coupled(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)
    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb), batch_size=BATCH_SIZE, shuffle=True
    )
    model_tcdf = CUTSPlusNet(d).to(DEVICE)
    opt_tcdf = optim.Adam(model_tcdf.parameters(), lr=LR)
    for epoch in range(10):
        for b_x, b_y in loader:
            opt_tcdf.zero_grad()
            pred = model_tcdf(b_x)
            loss = F.mse_loss(pred, b_y) + 0.02 * torch.sum(torch.abs(model_tcdf.W))
            loss.backward()
            opt_tcdf.step()
    adj_tcdf = torch.abs(model_tcdf.W).detach().cpu().numpy()
    skeleton_mask = np.zeros_like(adj_tcdf)
    for j in range(d):
        col = adj_tcdf[:, j].copy(); col[j] = 0
        k = max(1, int(np.ceil(d * 0.3)))
        for i in np.argsort(col)[-k:]:
            if col[i] > 0:
                skeleton_mask[i, j] = 1.0
    model_cuts = CUTSPlusNet(d).to(DEVICE)
    with torch.no_grad():
        model_cuts.W.data = torch.tensor(adj_tcdf * skeleton_mask, dtype=torch.float32).to(DEVICE) * 0.1
    opt_cuts = optim.Adam(model_cuts.parameters(), lr=LR)
    sk_tensor = torch.tensor(skeleton_mask, dtype=torch.float32).to(DEVICE)
    for epoch in range(EPOCHS):
        for b_x, b_y in loader:
            opt_cuts.zero_grad()
            pred = model_cuts(b_x)
            loss = F.mse_loss(pred, b_y) + 0.01 * torch.sum(torch.abs(model_cuts.W * sk_tensor))
            loss.backward()
            opt_cuts.step()
    adj_cuts = torch.abs(model_cuts.W).detach().cpu().numpy() * skeleton_mask
    def nm(M):
        v1, v2 = M.min(), M.max()
        return np.zeros_like(M) if v2 - v1 < 1e-9 else (M - v1) / (v2 - v1)
    M_ref = 0.5 * nm(adj_tcdf) + 0.5 * nm(adj_cuts)
    model_nts = NTS_NOTEARSNet(d).to(DEVICE)
    with torch.no_grad():
        model_nts.W.data = torch.tensor(M_ref, dtype=torch.float32).to(DEVICE) * 0.05
    opt_nts = optim.Adam(model_nts.parameters(), lr=LR)
    M_ref_t = torch.tensor(M_ref, dtype=torch.float32).to(DEVICE)
    for epoch in range(EPOCHS):
        for b_x, b_y in loader:
            opt_nts.zero_grad()
            pred    = model_nts(b_x)
            mse     = F.mse_loss(pred, b_y)
            h_val   = model_nts.notears_penalty()
            sparse  = torch.sum(torch.abs(model_nts.W))
            distill = torch.sum((model_nts.W - M_ref_t) ** 2)
            loss    = mse + 0.001 * sparse + 0.5 * h_val * h_val + 0.1 * distill
            loss.backward()
            opt_nts.step()
    return model_nts.W.detach().cpu().numpy()


# ─── 算法注册表 ──────────────────────────────────────────────────────────
ALGORITHM_REGISTRY = {
    "cuts_plus":      train_cuts_plus,
    "nts_notears":    train_nts_notears,
    "coupled":        train_coupled,
    "biattn_cuts":    train_biattn_cuts,
    "multiscale_nts": train_multiscale_nts,
    "mb_cuts":        train_mb_cuts,
}

ALGORITHM_CN_NAME = {
    "cuts_plus":      "CUTS+（基线）",
    "nts_notears":    "NTS-NOTEARS（基线）",
    "coupled":        "Coupled-三阶段（基线）",
    "biattn_cuts":    "BiAttn-CUTS（方案一：Transformer换头）",
    "multiscale_nts": "MultiScale-NTS（方案二：多尺度空洞卷积）",
    "mb_cuts":        "MB-CUTS（方案三：马尔可夫毯混合）",
}

# [DML3] 统计指标列定义
STANDARD_METRICS = ['SHD', 'TPR', 'FDR', 'Precision', 'Recall', 'F1']
DML_METRICS = ['DML_CQS', 'DML_CIS', 'DML_BCES', 'DML_IVP']


# ═══════════════════════════════════════════════════════════════════════════
# 核心实验函数（含 DML-CQS 计算）
# ═══════════════════════════════════════════════════════════════════════════
def run_single_experiment(exp_id: int, algorithm: str = 'cuts_plus') -> dict:
    """
    运行单次实验。

    新增流程 [DML2]:
      1. 在 adj_true / adj_pred 上调用 compute_dml_cqs_multi
      2. 从结果中提取 DML_CQS / DML_CIS / DML_BCES / DML_IVP 并写入 result 字典
    """
    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(f"未知算法: {algorithm}")

    gen = SyntheticDAGGenerator(n_nodes=N_NODES, seed=exp_id)
    X, adj_true, metadata = gen.generate_complete_synthetic_dataset(
        graph_type=GRAPH_TYPE,
        n_samples=N_SAMPLES,
        noise_scale=NOISE_SCALE,
        noise_type=NOISE_TYPE,
        add_time_lag=ADD_TIME_LAG,
        use_industrial_functions=USE_INDUSTRIAL_FUNCTIONS,
        n_layers=5
    )

    train_fn = ALGORITHM_REGISTRY[algorithm]
    adj_pred = train_fn(X, verbose=False)
    metrics  = compute_dag_metrics(adj_true, adj_pred, threshold=DML_THRESHOLD)

    result = {
        'exp_id':       exp_id,
        'algorithm':    algorithm,
        'n_nodes':      N_NODES,
        'n_edges_true': metadata['n_edges'],
        'n_samples':    N_SAMPLES,
        'graph_type':   GRAPH_TYPE,
        **metrics
    }

    # ── [DML2] DML 控制质量得分 ────────────────────────────────────────
    if EVALUATE_DML_CQS:
        outcome_idx = DML_OUTCOME_IDX if DML_OUTCOME_IDX < N_NODES else N_NODES - 1
        # 选取候选处理变量（排除结果变量和最近两个节点，偏向"上游"变量）
        candidate_treatments = list(range(max(0, N_NODES - 6)))
        if len(candidate_treatments) > N_TREATMENT_SAMPLES:
            rng = np.random.default_rng(exp_id)
            treatment_indices = rng.choice(
                candidate_treatments, size=N_TREATMENT_SAMPLES, replace=False
            ).tolist()
        else:
            treatment_indices = candidate_treatments

        dml_eval = compute_dml_cqs_multi(
            adj_true, adj_pred,
            treatment_indices=treatment_indices,
            outcome_idx=outcome_idx,
            threshold=DML_THRESHOLD,
        )

        result.update({
            'DML_CQS':  dml_eval['mean_dml_cqs'],   # 综合得分
            'DML_CIS':  dml_eval['mean_cis'],        # 混杂纳入 F1
            'DML_BCES': dml_eval['mean_bces'],       # 坏控制排除 F1
            'DML_IVP':  dml_eval['mean_ivp'],        # 工具精度
        })
    else:
        result.update({'DML_CQS': 0.0, 'DML_CIS': 0.0, 'DML_BCES': 0.0, 'DML_IVP': 0.0})

    # ── 传统因果角色评估（保留兼容性） ─────────────────────────────────
    if EVALUATE_CAUSAL_ROLES:
        outcome_idx  = N_NODES - 1
        potential_t  = list(range(N_NODES - 5))
        rng2         = np.random.default_rng(exp_id + 10000)
        treatment_idx2 = rng2.choice(
            potential_t, size=min(N_TREATMENT_SAMPLES, len(potential_t)), replace=False
        ).tolist()
        multi_eval = evaluate_multiple_treatments(
            adj_true, adj_pred, treatment_idx2, outcome_idx, threshold=DML_THRESHOLD
        )
        if 'summary' in multi_eval and multi_eval['summary']:
            result.update({
                'avg_confounder_f1':          multi_eval['summary']['avg_confounder_f1'],
                'avg_mediator_f1':            multi_eval['summary']['avg_mediator_f1'],
                'avg_instrument_f1':          multi_eval['summary']['avg_instrument_f1'],
                'avg_causal_structure_score': multi_eval['summary']['avg_causal_structure_score'],
                'direct_effect_accuracy':     multi_eval['summary']['direct_effect_accuracy'],
            })
        else:
            result.update({
                'avg_confounder_f1': 0.0, 'avg_mediator_f1': 0.0,
                'avg_instrument_f1': 0.0, 'avg_causal_structure_score': 0.0,
                'direct_effect_accuracy': 0.0,
            })

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 主蒙特卡洛循环
# ═══════════════════════════════════════════════════════════════════════════
def run_monte_carlo_benchmark(algorithm: str = 'cuts_plus'):
    cn_name = ALGORITHM_CN_NAME.get(algorithm, algorithm)
    print("=" * 70)
    print(f"蒙特卡洛基准测试 - {cn_name}")
    print("=" * 70)

    results = []
    for exp_id in tqdm(range(N_EXPERIMENTS), desc=f"Running {algorithm}"):
        try:
            result = run_single_experiment(exp_id, algorithm)
            results.append(result)
        except Exception as e:
            print(f"\n[警告] 实验 {exp_id} 失败: {e}")
        finally:
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    df_results = pd.DataFrame(results)

    out_csv = os.path.join(OUT_DIR, f"{algorithm}_detailed_results.csv")
    df_results.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {out_csv}")

    # ── 计算统计量 ────────────────────────────────────────────────────────
    all_metrics = STANDARD_METRICS + DML_METRICS
    if EVALUATE_CAUSAL_ROLES:
        all_metrics += [
            'avg_confounder_f1', 'avg_mediator_f1', 'avg_instrument_f1',
            'avg_causal_structure_score', 'direct_effect_accuracy'
        ]

    stats = {}
    for metric in all_metrics:
        if metric in df_results.columns:
            stats[metric] = {
                'mean': df_results[metric].mean(),
                'std':  df_results[metric].std()
            }

    # ── 控制台输出 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"统计报告 - {cn_name}")
    print("=" * 70)

    print("\n【传统 DAG 恢复指标】")
    print(f"{'指标':<15} {'均值':<12} {'标准差':<12} {'学术格式'}")
    print("-" * 55)
    for metric in STANDARD_METRICS:
        if metric in stats:
            m, s = stats[metric]['mean'], stats[metric]['std']
            print(f"{metric:<15} {m:<12.4f} {s:<12.4f} {m:.3f} ± {s:.3f}")

    print("\n【DML 控制质量指标（新增）】")
    print(f"{'指标':<15} {'均值':<12} {'标准差':<12} {'学术格式':<20} {'说明'}")
    print("-" * 78)
    dml_desc = {
        'DML_CQS':  '综合得分（0.4×CIS+0.4×BCES+0.2×IVP）',
        'DML_CIS':  '混杂纳入 F1（漏掉混杂 → 混杂偏差）',
        'DML_BCES': '坏控制排除 F1（纳入中介/对撞 → 过控制/碰撞器偏差）',
        'DML_IVP':  '工具变量精度（Precision only）',
    }
    for metric in DML_METRICS:
        if metric in stats:
            m, s = stats[metric]['mean'], stats[metric]['std']
            desc = dml_desc.get(metric, '')
            print(f"{metric:<15} {m:<12.4f} {s:<12.4f} {m:.3f} ± {s:.3f}    {desc}")

    print("=" * 70)

    # ── 保存统计 CSV ──────────────────────────────────────────────────────
    stats_df = pd.DataFrame(stats).T
    stats_df['academic_format'] = stats_df.apply(
        lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1
    )
    out_stats = os.path.join(OUT_DIR, f"{algorithm}_statistics.csv")
    stats_df.to_csv(out_stats, encoding='utf-8-sig')
    print(f"统计报告已保存: {out_stats}")

    generate_markdown_report(algorithm, df_results, stats)
    return df_results, stats


# ═══════════════════════════════════════════════════════════════════════════
# Markdown 报告生成
# ═══════════════════════════════════════════════════════════════════════════
def generate_markdown_report(algorithm: str, df_results: pd.DataFrame, stats: dict):
    cn_name = ALGORITHM_CN_NAME.get(algorithm, algorithm)

    report = f"""# 蒙特卡洛基准测试报告 - {cn_name}

## 实验配置

| 参数 | 值 |
|------|-----|
| 实验次数 | {N_EXPERIMENTS} |
| 节点数 | {N_NODES} |
| 样本数 | {N_SAMPLES} |
| 图类型 | {GRAPH_TYPE} |
| 噪声类型 | {NOISE_TYPE} |
| 噪声水平 | {NOISE_SCALE} |
| 工业函数 | {USE_INDUSTRIAL_FUNCTIONS} |
| 时序依赖 | {ADD_TIME_LAG} |
| 算法 | {algorithm} |

## 传统 DAG 恢复指标（均值 ± 标准差）

| 指标 | 均值 | 标准差 | 学术格式 |
|------|------|--------|----------|
"""
    for metric in STANDARD_METRICS:
        if metric in stats:
            m, s = stats[metric]['mean'], stats[metric]['std']
            report += f"| {metric} | {m:.4f} | {s:.4f} | {m:.3f} ± {s:.3f} |\n"

    report += """
## DML 控制质量指标（新增，均值 ± 标准差）

> 本节指标专为双重机器学习（DML）场景设计，衡量因果发现结果对 DML 估计有效性的支撑程度。
> 传统 F1 对所有边一视同仁；本指标关注"识别错误的代价"——
> 纳入中介/对撞变量会导致 ATE 偏误，其危害远大于遗漏无关边。

| 指标 | 均值 | 标准差 | 学术格式 | 说明 |
|------|------|--------|----------|------|
"""
    dml_desc = {
        'DML_CQS':  '综合控制质量得分（越高越好）',
        'DML_CIS':  '混杂纳入 F1（漏掉混杂→遗漏混杂偏差）',
        'DML_BCES': '坏控制排除 F1（纳入中介/对撞→过控制/碰撞器偏差）',
        'DML_IVP':  '工具变量发现精度（Precision，宁缺毋滥）',
    }
    for metric in DML_METRICS:
        if metric in stats:
            m, s = stats[metric]['mean'], stats[metric]['std']
            desc = dml_desc.get(metric, '')
            report += f"| {metric} | {m:.4f} | {s:.4f} | {m:.3f} ± {s:.3f} | {desc} |\n"

    report += f"""
## 指标设计说明

### DML-CQS 计算公式

```
DML-CQS = 0.4 × CIS + 0.4 × BCES + 0.2 × IVP
```

权重设计依据：
- CIS 与 BCES 权重相等（0.4），体现"纳入该纳入的"与"排除不该纳入的"同等重要。
- IVP 权重较低（0.2），因为工具变量发现是锦上添花，而非 DML 必要条件。
- 仅用 Precision（非 F1）衡量工具变量，因为错误的工具变量危害大于漏掉真实工具变量。

### 与传统指标的区别

| 维度 | 传统 F1 | DML-CQS |
|------|---------|---------|
| 边的权重 | 一视同仁 | 按 DML 代价加权 |
| 关注目标 | 图结构精度 | 估计有效性 |
| 中介误判代价 | 等同于其他错误 | 最高（破坏因果路径） |
| 适用场景 | 通用因果发现 | DML/因果推断下游任务 |

生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    out_md = os.path.join(OUT_DIR, f"{algorithm}_report.md")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Markdown 报告已保存: {out_md}")


# ═══════════════════════════════════════════════════════════════════════════
# 多算法横向对比报告（含 DML-CQS 专栏）[DML4]
# ═══════════════════════════════════════════════════════════════════════════
def generate_comparison_report(all_results: dict):
    algorithms = list(all_results.keys())

    print("\n" + "=" * 90)
    print("算法横向对比（均值 ± 标准差）")
    print("=" * 90)

    # ── 传统指标对比表 ─────────────────────────────────────────────────────
    comparison_std = []
    for metric in STANDARD_METRICS:
        row = {'Metric': metric}
        for algo in algorithms:
            if metric in all_results[algo]['stats']:
                m = all_results[algo]['stats'][metric]['mean']
                s = all_results[algo]['stats'][metric]['std']
                row[algo] = f"{m:.3f} ± {s:.3f}"
            else:
                row[algo] = "N/A"
        comparison_std.append(row)

    df_std = pd.DataFrame(comparison_std)
    print("\n【传统 DAG 恢复指标】")
    print(df_std.to_string(index=False))

    # ── DML 指标对比表 ─────────────────────────────────────────────────────
    comparison_dml = []
    for metric in DML_METRICS:
        row = {'Metric': metric}
        for algo in algorithms:
            if metric in all_results[algo]['stats']:
                m = all_results[algo]['stats'][metric]['mean']
                s = all_results[algo]['stats'][metric]['std']
                row[algo] = f"{m:.3f} ± {s:.3f}"
            else:
                row[algo] = "N/A"
        comparison_dml.append(row)

    df_dml = pd.DataFrame(comparison_dml)
    print("\n【DML 控制质量指标（新增）】")
    print(df_dml.to_string(index=False))

    # ── 保存 CSV ───────────────────────────────────────────────────────────
    df_all = pd.concat([df_std, df_dml], ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "algorithm_comparison.csv")
    df_all.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存: {out_csv}")

    # ── 生成 Markdown 对比报告 ─────────────────────────────────────────────
    algo_names = [ALGORITHM_CN_NAME.get(a, a) for a in algorithms]
    header_sep = "|" + "------|" * (len(algorithms) + 1)

    md  = f"# 因果发现算法横向对比报告\n\n"
    md += f"实验配置：{N_NODES} 节点 × {N_SAMPLES} 样本 × {N_EXPERIMENTS} 次蒙特卡洛 × {GRAPH_TYPE} 图\n\n"
    md += "## 传统 DAG 恢复指标\n\n"
    md += "| 指标 | " + " | ".join(algo_names) + " |\n"
    md += header_sep + "\n"
    for row in comparison_std:
        md += f"| {row['Metric']} | " + " | ".join(row[a] for a in algorithms) + " |\n"

    md += "\n## DML 控制质量指标（新增）\n\n"
    md += "> DML-CQS 综合得分越高，表明该算法产出的 DAG 对 DML 因果推断越有利。\n\n"
    md += "| 指标 | " + " | ".join(algo_names) + " |\n"
    md += header_sep + "\n"
    for row in comparison_dml:
        md += f"| {row['Metric']} | " + " | ".join(row[a] for a in algorithms) + " |\n"

    md += "\n## DML 指标说明\n\n"
    md += "- **DML-CQS**：综合控制质量得分，越大越好\n"
    md += "- **DML-CIS**：混杂纳入 F1，越大越好（漏掉混杂会导致 ATE 不一致）\n"
    md += "- **DML-BCES**：坏控制排除 F1，越大越好（纳入中介/对撞会扭曲 ATE 估计）\n"
    md += "- **DML-IVP**：工具变量发现精度（Precision），越大越好\n\n"
    md += f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    out_md = os.path.join(OUT_DIR, "algorithm_comparison_report.md")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"对比 Markdown 报告已保存: {out_md}")


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    ALL_BASELINES   = ["cuts_plus", "nts_notears", "coupled"]
    ALL_INNOVATIONS = ["biattn_cuts", "multiscale_nts", "mb_cuts"]
    ALL_ALGORITHMS  = ALL_BASELINES + ALL_INNOVATIONS

    parser = argparse.ArgumentParser(
        description="蒙特卡洛基准测试（DML-CQS 增强版）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--algorithm",
        choices=ALL_ALGORITHMS + ["baselines", "innovations", "all"],
        default="all",
    )
    parser.add_argument("--n_experiments",           type=int,  default=50)
    parser.add_argument("--n_nodes",                 type=int,  default=20)
    parser.add_argument("--n_samples",               type=int,  default=1000)
    parser.add_argument("--graph_type",              choices=["er", "scale_free", "layered"], default="layered")
    parser.add_argument("--noise_type",
        choices=["gaussian", "heteroscedastic", "heavy_tail", "periodic"],
        default="heteroscedastic"
    )
    parser.add_argument("--dml_threshold",           type=float, default=0.05)
    parser.add_argument("--use_industrial_functions", action="store_true", default=True)
    parser.add_argument("--add_time_lag",             action="store_true", default=True)
    parser.add_argument("--evaluate_causal_roles",   action="store_true", default=True)
    parser.add_argument("--evaluate_dml_cqs",        action="store_true", default=True)
    parser.add_argument("--n_treatment_samples",     type=int, default=4)

    args = parser.parse_args()

    N_EXPERIMENTS            = args.n_experiments
    N_NODES                  = args.n_nodes
    N_SAMPLES                = args.n_samples
    GRAPH_TYPE               = args.graph_type
    NOISE_TYPE               = args.noise_type
    DML_THRESHOLD            = args.dml_threshold
    USE_INDUSTRIAL_FUNCTIONS = args.use_industrial_functions
    ADD_TIME_LAG             = args.add_time_lag
    EVALUATE_CAUSAL_ROLES    = args.evaluate_causal_roles
    EVALUATE_DML_CQS         = args.evaluate_dml_cqs
    N_TREATMENT_SAMPLES      = args.n_treatment_samples
    DML_OUTCOME_IDX          = N_NODES - 1

    if args.algorithm == "all":
        algorithms = ALL_ALGORITHMS
    elif args.algorithm == "baselines":
        algorithms = ALL_BASELINES
    elif args.algorithm == "innovations":
        algorithms = ALL_INNOVATIONS
    else:
        algorithms = [args.algorithm]

    print(f"\n本次将运行以下算法: {algorithms}")

    all_results = {}
    for algo in algorithms:
        df_results, stats = run_monte_carlo_benchmark(algo)
        all_results[algo] = {'results': df_results, 'stats': stats}

    if len(algorithms) >= 2:
        generate_comparison_report(all_results)

    print("\n" + "=" * 70)
    print("所有实验完成！")
    print("=" * 70)
