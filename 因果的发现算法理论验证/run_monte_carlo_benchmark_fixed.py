"""
run_monte_carlo_benchmark.py  [修正版]
=============================
蒙特卡洛模拟 - 因果发现算法鲁棒性压力测试

修正清单（相对于原始版本）：
  [B1] BiAttnCUTSNet.forward: x.reshape(B*d,T,1) → 先 transpose 再 contiguous().reshape，
       消除跨节点/时间步的内存交叉污染
  [B2] BiAttnCUTSNet: tau 无正值约束 → 所有使用处改用 F.softplus(self.tau) 保证 τ > 0，
       防止 sigmoid 方向反转
  [B3] train_biattn_cuts: 缺少 NOTEARS 无环约束 → 在模型中加入 notears_penalty()
       并在训练损失中加入 0.5 * h² 惩罚
  [B4] _compute_mb_mask: Pearson 相关对非线性无效 → 改用 Spearman 秩相关，
       能正确捕捉单调非线性因果关系
  [B5] run_monte_carlo_benchmark: 50次循环无显存清理 → 每次实验后调用
       torch.cuda.empty_cache() 和 gc.collect() 防止 OOM
  [B6] MultiScaleNTSNet.alpha 初始化: ones/n → zeros，梯度空间更干净

实验设计:
  - 50 次独立实验，每次生成不同的随机 DAG（20 节点）
  - 支持 ER 和 Scale-Free 两种图模型
  - 混合线性/正弦/指数/多项式非线性关系
  - 计算 SHD、TPR、FDR、Precision、Recall、F1
  - 报告均值 ± 标准差

算法支持（原始基线）:
  - CUTS+
  - NTS-NOTEARS
  - Coupled (TCDF → CUTS+ → NTS-NOTEARS 三阶段耦合)

新增创新方案:
  - BiAttn-CUTS   [方案一] Transformer注意力编码器 + 可学习软阈值邻接矩阵 + NOTEARS约束
  - MultiScale-NTS [方案二] 多尺度并行卷积特征融合 + 共享NOTEARS约束
  - MB-CUTS        [方案三] 马尔可夫毯相关性预筛选(Spearman) + NTS-NOTEARS精化

学术意义:
  这是 NeurIPS/ICLR 等顶会标准的因果发现算法评估范式
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
from scipy.stats import spearmanr          # [B4] Spearman 秩相关
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 导入合成数据生成器（包含新的因果角色评估功能）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_dag_generator import (
    SyntheticDAGGenerator,
    compute_dag_metrics,
    evaluate_causal_structure_recovery,
    evaluate_multiple_treatments
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果", "monte_carlo_benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 超参数配置 ──────────────────────────────────────────────────────────
N_EXPERIMENTS = 50  # 蒙特卡洛实验次数
N_NODES = 20        # 节点数
N_SAMPLES = 1000    # 每个实验的样本数
NOISE_SCALE = 0.1   # 噪声水平
GRAPH_TYPE = 'layered'        # 'er', 'scale_free', 或 'layered'
NOISE_TYPE = 'heteroscedastic'  # 噪声类型
USE_INDUSTRIAL_FUNCTIONS = True  # 使用工业函数
ADD_TIME_LAG = True              # 添加时序依赖

# 因果角色评估配置
EVALUATE_CAUSAL_ROLES = True   # 是否评估因果角色识别
N_TREATMENT_SAMPLES = 4        # 随机选择多少个操作变量进行评估

# 算法训练参数
WINDOW_SIZE = 10
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")
print(f"DAG 类型: {GRAPH_TYPE}")
print(f"噪声类型: {NOISE_TYPE}")
print(f"工业函数: {USE_INDUSTRIAL_FUNCTIONS}")
print(f"时序依赖: {ADD_TIME_LAG}")
print(f"因果角色评估: {EVALUATE_CAUSAL_ROLES}")


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
        # 正确写法：先 transpose → (B, d, T)，再展平前两维
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
# 创新方案一：BiAttn-CUTS  [修正版]
# ═══════════════════════════════════════════════════════════════════════════
class BiAttnCUTSNet(nn.Module):
    """
    Transformer注意力编码器 + 可学习温度软阈值邻接矩阵 + NOTEARS无环约束

    修正点：
      [B1] forward 中的张量重塑：先 transpose(1,2).contiguous() 再 reshape，
           保证每个 (B*d, T, 1) 样本对应单一节点的完整时间序列。
      [B2] tau 正值约束：所有使用 tau 的地方改用 F.softplus(self.tau)，
           防止 tau 优化为负数时 sigmoid 逻辑反转。
      [B3] 新增 notears_penalty()，补回 DAG 无环约束。
    """
    def __init__(self, d, n_heads: int = 4, d_model: int = 16):
        super().__init__()
        self.d = d
        self.d_model = d_model
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.05, 0.05))
        # [B2] tau 初始化为 0，经 softplus 后自然 > 0（softplus(0) ≈ 0.693）
        self.tau = nn.Parameter(torch.zeros(1))

        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,       # d_model=16, nhead=4 → head_dim=4 ✓
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Linear(d_model, 1)

    def _tau_pos(self):
        """[B2] 保证温度系数恒正"""
        return F.softplus(self.tau)

    def forward(self, x):
        """
        参数:
            x: (B, WINDOW_SIZE, d)
        返回:
            pred: (B, d)
        """
        B, T, d = x.shape

        # [B1] 修正：先 transpose → (B, d, T)，再 contiguous().reshape → (B·d, T, 1)
        # 保证第 k 个"伪样本"对应第 k//B 个节点、第 k%B 个批次的完整时序
        x_flat = x.transpose(1, 2).contiguous().reshape(B * d, T, 1)  # (B·d, T, 1)
        x_emb  = self.input_proj(x_flat)                               # (B·d, T, d_model)
        h      = self.transformer(x_emb)                               # (B·d, T, d_model)
        h_last = h[:, -1, :]                                           # (B·d, d_model)
        node_feats = h_last.reshape(B, d, self.d_model)                # (B, d, d_model)

        # [B2] 使用 softplus 保证 tau > 0
        tau_pos = self._tau_pos()
        W_soft  = torch.sigmoid(self.W * tau_pos)
        W_soft  = W_soft * (1 - torch.eye(d, device=W_soft.device))   # 去自环

        agg  = torch.einsum('ij,bif->bjf', W_soft, node_feats)        # (B, d, d_model)
        pred = self.out_proj(agg).squeeze(-1)                          # (B, d)
        return pred

    def notears_penalty(self):
        """[B3] 补回 NOTEARS 无环约束（施加在原始 W² 上，与基线保持一致）"""
        M = self.W * self.W
        E = torch.matrix_exp(M)
        return torch.trace(E) - self.d


def train_biattn_cuts(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    训练创新方案一：BiAttn-CUTS [修正版]

    修正点：
      [B2] sparse 正则使用 softplus(tau) 版本的 W_soft
      [B3] 损失中加入 NOTEARS 惩罚 0.5 * h²
    """
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)

    model = BiAttnCUTSNet(d).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)

    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb),
        batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for b_x, b_y in loader:
            opt.zero_grad()
            pred = model(b_x)
            mse  = F.mse_loss(pred, b_y)

            # [B2] 使用模型内的 _tau_pos() 保证一致性
            tau_pos = model._tau_pos()
            W_soft  = torch.sigmoid(model.W * tau_pos)
            sparse  = torch.sum(W_soft * (1 - torch.eye(d, device=W_soft.device)))

            # [B3] 补回 NOTEARS 无环惩罚
            h_val = model.notears_penalty()

            loss = mse + 0.01 * sparse + 0.5 * h_val * h_val
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [BiAttn-CUTS] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    # [B2] 输出时同样使用 softplus(tau)
    with torch.no_grad():
        tau_pos  = model._tau_pos()
        adj_pred = torch.sigmoid(model.W * tau_pos).cpu().numpy()
    np.fill_diagonal(adj_pred, 0)
    return adj_pred


# ═══════════════════════════════════════════════════════════════════════════
# 创新方案二：MultiScale-NTS  [修正版]
# ═══════════════════════════════════════════════════════════════════════════
class MultiScaleNTSNet(nn.Module):
    """
    三路并行多尺度卷积 + 可学习融合权重 + 共享 NOTEARS 约束

    修正点：
      [B6] alpha 初始化改为 zeros（softmax(zeros) = 均匀分布，梯度空间更干净）

    注：原版初始化 ones/n 在数学上等价（softmax 对平移不变），但 zeros 是更规范的做法。
    """
    KERNEL_SIZES = [3, 5]   # 第三路动态追加 WINDOW_SIZE

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.01, 0.01))

        kernel_sizes = self.KERNEL_SIZES + [WINDOW_SIZE]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d, d * 8, kernel_size=ks, groups=d),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(d * 8, d, kernel_size=1, groups=d)
            )
            for ks in kernel_sizes
        ])

        # [B6] zeros 初始化：梯度更新从对称点出发，收敛更稳定
        self.alpha = nn.Parameter(torch.zeros(len(kernel_sizes)))

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
    """训练创新方案二：MultiScale-NTS"""
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)

    model = MultiScaleNTSNet(d).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)

    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb),
        batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for b_x, b_y in loader:
            opt.zero_grad()
            pred  = model(b_x)
            mse   = F.mse_loss(pred, b_y)
            h_val = model.notears_penalty()
            loss  = mse + 0.001 * torch.sum(torch.abs(model.W)) + 0.5 * h_val * h_val
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
# 创新方案三：MB-CUTS  [修正版]
# ═══════════════════════════════════════════════════════════════════════════
def _compute_mb_mask(X_norm: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    """
    基于 Spearman 秩相关计算近似马尔可夫毯掩码  [修正版 B4]

    修正原因：
      原版使用 Pearson 相关系数，仅能捕捉线性关系。
      当 USE_INDUSTRIAL_FUNCTIONS=True 时，数据含有饱和/阈值/倒U型等非线性函数，
      Pearson 会将强非线性因果边误判为弱相关进而剪除，导致 Stage 1 误剪真实边。
      Spearman 秩相关等价于对排名做 Pearson，能正确捕捉单调非线性关系，
      计算成本与 Pearson 相当，无需引入额外依赖。

    局限性说明：
      Spearman 对于非单调关系（如倒 U 型、正弦）仍可能给出接近 0 的相关值。
      若需完全非参数筛选，可替换为距离相关系数（`dcor` 库）或互信息近似。
      此处优先保证与 scipy 的零额外依赖，建议在论文局限性章节中明确声明。

    参数:
        X_norm : 标准化数据 (n_samples, d)
        keep_ratio : 每列保留的候选父节点比例（默认 50%）

    返回:
        mb_mask : (d, d) 二值候选边掩码
    """
    d = X_norm.shape[1]

    # [B4] 使用 Spearman 秩相关替换 Pearson
    # spearmanr 返回 (相关系数矩阵, p 值矩阵)，取 [0] 即相关矩阵
    corr_matrix, _ = spearmanr(X_norm)
    corr = np.array(corr_matrix)       # 确保是 ndarray，防止标量退化

    # 若 d=1 时 spearmanr 返回标量，做保护
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    np.fill_diagonal(corr, 0.0)        # 排除自相关

    mb_mask = np.zeros((d, d), dtype=np.float32)
    k = max(1, int(np.ceil(d * keep_ratio)))

    for j in range(d):
        col     = np.abs(corr[:, j])
        top_idx = np.argsort(col)[-k:]
        mb_mask[top_idx, j] = 1.0

    return mb_mask


def train_mb_cuts(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    训练创新方案三：MB-CUTS（三阶段混合因果发现）[修正版]

    修正点：
      [B4] Stage 1 改用 Spearman 秩相关筛选候选边，适应非线性数据
    """
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)

    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # ── Stage 1：马尔可夫毯相关性筛选（Spearman） ─────────────────────────
    if verbose:
        print("  [MB-CUTS] Stage 1: 马尔可夫毯候选边筛选（Spearman秩相关）...")

    mb_mask   = _compute_mb_mask(X_norm, keep_ratio=0.5)
    mb_mask_t = torch.tensor(mb_mask, dtype=torch.float32).to(DEVICE)
    non_mb_t  = torch.tensor(1.0 - mb_mask, dtype=torch.float32).to(DEVICE)

    if verbose:
        n_cand = int(mb_mask.sum())
        print(f"    候选边数量: {n_cand} / {d*d-d} （稀疏度 {n_cand/(d*d-d)*100:.1f}%）")

    # ── Stage 2：CUTS+ 风格预热训练 ────────────────────────────────────────
    if verbose:
        print("  [MB-CUTS] Stage 2: 候选图内 CUTS+ 预热...")

    model_rough = CUTSPlusNet(d).to(DEVICE)
    opt_rough   = optim.Adam(model_rough.parameters(), lr=LR)

    for epoch in range(10):
        for b_x, b_y in loader:
            opt_rough.zero_grad()
            pred     = model_rough(b_x)
            W_in_mb  = model_rough.W * mb_mask_t
            loss     = F.mse_loss(pred, b_y) + 0.01 * torch.sum(torch.abs(W_in_mb))
            loss.backward()
            opt_rough.step()

    adj_rough = torch.abs(model_rough.W).detach().cpu().numpy() * mb_mask

    if verbose:
        nonzero_vals = adj_rough[adj_rough > 0]
        mean_val = nonzero_vals.mean() if len(nonzero_vals) > 0 else 0.0
        print(f"    预热完成，粗糙邻接矩阵均值: {mean_val:.4f}")

    # ── Stage 3：NTS-NOTEARS 精化 + 图外惩罚项 ────────────────────────────
    if verbose:
        print("  [MB-CUTS] Stage 3: NOTEARS 精化 + 马尔可夫毯约束...")

    model_fine = NTS_NOTEARSNet(d).to(DEVICE)
    with torch.no_grad():
        model_fine.W.data = torch.tensor(
            adj_rough, dtype=torch.float32
        ).to(DEVICE) * 0.1

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
            loss       = mse + 0.001 * sparse + 0.5 * h_val * h_val + 1.0 * mb_penalty
            loss.backward()
            opt_fine.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [MB-CUTS] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    if verbose:
        print("  [MB-CUTS] 三阶段完成")

    adj_pred = model_fine.W.detach().cpu().numpy()
    return adj_pred


# ═══════════════════════════════════════════════════════════════════════════
# 以下为原始基线算法（保持不变）
# ═══════════════════════════════════════════════════════════════════════════

def build_windows(X):
    """构建时间窗口"""
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
        total_loss = 0
        for b_x, b_y in loader:
            opt.zero_grad()
            pred = model(b_x)
            loss = F.mse_loss(pred, b_y) + 0.01 * torch.sum(torch.abs(model.W))
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    adj_pred = torch.abs(model.W).detach().cpu().numpy()
    return adj_pred


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
        total_loss = 0
        for b_x, b_y in loader:
            opt.zero_grad()
            pred  = model(b_x)
            mse   = F.mse_loss(pred, b_y)
            h_val = model.notears_penalty()
            loss  = mse + 0.001 * torch.sum(torch.abs(model.W)) + 0.5 * h_val * h_val
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    adj_pred = model.W.detach().cpu().numpy()
    return adj_pred


def train_coupled(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    d = X.shape[1]
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm)

    if verbose:
        print("  [Coupled] Stage 1: TCDF 骨架提取...")

    model_tcdf = CUTSPlusNet(d).to(DEVICE)
    opt_tcdf = optim.Adam(model_tcdf.parameters(), lr=LR)

    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb), batch_size=BATCH_SIZE, shuffle=True
    )

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
        col = adj_tcdf[:, j].copy()
        col[j] = 0
        k = max(1, int(np.ceil(d * 0.3)))
        top_k_indices = np.argsort(col)[-k:]
        for i in top_k_indices:
            if col[i] > 0:
                skeleton_mask[i, j] = 1.0

    if verbose:
        print(f"    骨架保留 {int(skeleton_mask.sum())} / {d*d} 条边")
        print("  [Coupled] Stage 2: CUTS+ 深挖...")

    model_cuts = CUTSPlusNet(d).to(DEVICE)
    with torch.no_grad():
        model_cuts.W.data = torch.tensor(
            adj_tcdf * skeleton_mask, dtype=torch.float32
        ).to(DEVICE) * 0.1

    opt_cuts = optim.Adam(model_cuts.parameters(), lr=LR)
    sk_tensor = torch.tensor(skeleton_mask, dtype=torch.float32).to(DEVICE)

    for epoch in range(EPOCHS):
        for b_x, b_y in loader:
            opt_cuts.zero_grad()
            pred     = model_cuts(b_x)
            W_masked = model_cuts.W * sk_tensor
            loss     = F.mse_loss(pred, b_y) + 0.01 * torch.sum(torch.abs(W_masked))
            loss.backward()
            opt_cuts.step()

    adj_cuts = torch.abs(model_cuts.W).detach().cpu().numpy() * skeleton_mask

    def normalize_matrix(M):
        vmin, vmax = M.min(), M.max()
        if vmax - vmin < 1e-9:
            return np.zeros_like(M)
        return (M - vmin) / (vmax - vmin)

    M_ref = 0.5 * normalize_matrix(adj_tcdf) + 0.5 * normalize_matrix(adj_cuts)

    if verbose:
        print("  [Coupled] Stage 3: NTS-NOTEARS 知识蒸馏...")

    model_nts = NTS_NOTEARSNet(d).to(DEVICE)
    with torch.no_grad():
        model_nts.W.data = torch.tensor(M_ref, dtype=torch.float32).to(DEVICE) * 0.05

    opt_nts = optim.Adam(model_nts.parameters(), lr=LR)
    M_ref_tensor = torch.tensor(M_ref, dtype=torch.float32).to(DEVICE)

    for epoch in range(EPOCHS):
        for b_x, b_y in loader:
            opt_nts.zero_grad()
            pred    = model_nts(b_x)
            mse     = F.mse_loss(pred, b_y)
            h_val   = model_nts.notears_penalty()
            sparse  = torch.sum(torch.abs(model_nts.W))
            distill = torch.sum((model_nts.W - M_ref_tensor) ** 2)
            loss    = mse + 0.001 * sparse + 0.5 * h_val * h_val + 0.1 * distill
            loss.backward()
            opt_nts.step()

    if verbose:
        print("  [Coupled] 三阶段完成")

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
    "multiscale_nts": "MultiScale-NTS（方案二：多尺度卷积）",
    "mb_cuts":        "MB-CUTS（方案三：马尔可夫毯混合）",
}


def run_single_experiment(exp_id: int, algorithm: str = 'cuts_plus') -> dict:
    """运行单次实验（支持因果角色评估）"""
    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(f"未知算法: {algorithm}，可选: {list(ALGORITHM_REGISTRY.keys())}")

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
    metrics  = compute_dag_metrics(adj_true, adj_pred, threshold=0.05)

    result = {
        'exp_id':       exp_id,
        'algorithm':    algorithm,
        'n_nodes':      N_NODES,
        'n_edges_true': metadata['n_edges'],
        'n_samples':    N_SAMPLES,
        'graph_type':   GRAPH_TYPE,
        **metrics
    }

    if EVALUATE_CAUSAL_ROLES:
        outcome_idx         = N_NODES - 1
        potential_treatments = list(range(N_NODES - 5))
        if len(potential_treatments) > N_TREATMENT_SAMPLES:
            treatment_indices = gen.rng.choice(
                potential_treatments, size=N_TREATMENT_SAMPLES, replace=False
            ).tolist()
        else:
            treatment_indices = potential_treatments

        multi_eval = evaluate_multiple_treatments(
            adj_true, adj_pred, treatment_indices, outcome_idx, threshold=0.05
        )

        if 'summary' in multi_eval and multi_eval['summary']:
            result.update({
                'avg_confounder_f1':         multi_eval['summary']['avg_confounder_f1'],
                'avg_mediator_f1':           multi_eval['summary']['avg_mediator_f1'],
                'avg_instrument_f1':         multi_eval['summary']['avg_instrument_f1'],
                'avg_causal_structure_score': multi_eval['summary']['avg_causal_structure_score'],
                'direct_effect_accuracy':    multi_eval['summary']['direct_effect_accuracy'],
            })
        else:
            result.update({
                'avg_confounder_f1': 0.0, 'avg_mediator_f1': 0.0,
                'avg_instrument_f1': 0.0, 'avg_causal_structure_score': 0.0,
                'direct_effect_accuracy': 0.0,
            })

    return result


def run_monte_carlo_benchmark(algorithm: str = 'cuts_plus'):
    """运行完整的蒙特卡洛基准测试"""
    cn_name = ALGORITHM_CN_NAME.get(algorithm, algorithm)
    print("=" * 70)
    print(f"蒙特卡洛基准测试 - {cn_name}")
    print("=" * 70)
    print(f"实验配置:")
    print(f"  实验次数: {N_EXPERIMENTS}")
    print(f"  节点数:   {N_NODES}")
    print(f"  样本数:   {N_SAMPLES}")
    print(f"  图类型:   {GRAPH_TYPE}")
    print(f"  噪声水平: {NOISE_SCALE}")
    print("=" * 70)

    results = []

    for exp_id in tqdm(range(N_EXPERIMENTS), desc=f"Running {algorithm}"):
        try:
            result = run_single_experiment(exp_id, algorithm)
            results.append(result)
        except Exception as e:
            print(f"\n[警告] 实验 {exp_id} 失败: {e}")
        finally:
            # [B5] 每次实验后显式清理 GPU 显存，防止 50 次循环累积碎片化 / OOM
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    df_results = pd.DataFrame(results)

    out_csv = os.path.join(OUT_DIR, f"{algorithm}_detailed_results.csv")
    df_results.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {out_csv}")

    if EVALUATE_CAUSAL_ROLES:
        metrics_cols = [
            'SHD', 'TPR', 'FDR', 'Precision', 'Recall', 'F1',
            'avg_confounder_f1', 'avg_mediator_f1', 'avg_instrument_f1',
            'avg_causal_structure_score', 'direct_effect_accuracy'
        ]
    else:
        metrics_cols = ['SHD', 'TPR', 'FDR', 'Precision', 'Recall', 'F1']

    stats = {}
    for metric in metrics_cols:
        stats[metric] = {
            'mean': df_results[metric].mean(),
            'std':  df_results[metric].std()
        }

    print("\n" + "=" * 70)
    print(f"统计报告 - {cn_name}")
    print("=" * 70)
    print(f"{'指标':<15} {'均值':<15} {'标准差':<15} {'学术格式':<20}")
    print("-" * 70)
    for metric in metrics_cols:
        mean_val = stats[metric]['mean']
        std_val  = stats[metric]['std']
        print(f"{metric:<15} {mean_val:<15.4f} {std_val:<15.4f} "
              f"{mean_val:.3f} ± {std_val:.3f}")
    print("=" * 70)

    stats_df = pd.DataFrame(stats).T
    stats_df['academic_format'] = stats_df.apply(
        lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1
    )
    out_stats = os.path.join(OUT_DIR, f"{algorithm}_statistics.csv")
    stats_df.to_csv(out_stats, encoding='utf-8-sig')
    print(f"\n统计报告已保存: {out_stats}")

    generate_markdown_report(algorithm, df_results, stats)
    return df_results, stats


def generate_markdown_report(algorithm: str, df_results: pd.DataFrame, stats: dict):
    """生成 Markdown 格式的学术报告"""
    cn_name = ALGORITHM_CN_NAME.get(algorithm, algorithm)

    innovation_note = {
        "biattn_cuts": """
## 创新方案说明

**BiAttn-CUTS（方案一：Transformer注意力换头）**

- **核心改动**：将 CUTS+ 中的 LSTM 编码器替换为 TransformerEncoder，
  引入可学习温度参数 τ（经 Softplus 约束为正）对邻接矩阵做软阈值化，
  并补充 NOTEARS 无环约束。
- **学术依据**：Transformer 自注意力机制可同时建模窗口内任意时刻对的依赖，
  克服 LSTM 梯度传播随距离衰减的局限性。
- **适用场景**：工业传感器数据中存在长程、多时滞因果关系时。
""",
        "multiscale_nts": """
## 创新方案说明

**MultiScale-NTS（方案二：多尺度并行卷积特征融合）**

- **核心改动**：三路并行 Conv1d 分支（kernel=3/5/10），通过可学习融合权重 α
  （Softmax归一化，zeros初始化）自适应加权，共享单一 NOTEARS DAG 约束。
- **学术依据**：多尺度时序卷积在工业动态建模中已被广泛验证，
  引入因果图学习框架属于跨领域组合创新。
""",
        "mb_cuts": """
## 创新方案说明

**MB-CUTS（方案三：马尔可夫毯预筛选 + NTS-NOTEARS精化）**

- **核心改动**：三阶段混合框架。Stage 1 用 **Spearman 秩相关**（替换原 Pearson）
  生成候选边掩码，能正确处理单调非线性数据；Stage 2 CUTS+ 快速预热；
  Stage 3 NTS-NOTEARS + 图外惩罚精化。
- **局限性说明**：Spearman 对非单调关系（如倒U型）仍有局限，
  完全非参数方案可替换为距离相关系数（dcor库）或互信息近似。
""",
    }.get(algorithm, "")

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

## 评估指标

### 传统 DAG 恢复指标（均值 ± 标准差）

| 指标 | 均值 | 标准差 | 学术格式 |
|------|------|--------|----------|
"""
    for metric in ['SHD', 'TPR', 'FDR', 'Precision', 'Recall', 'F1']:
        if metric in stats:
            m, s = stats[metric]['mean'], stats[metric]['std']
            report += f"| {metric} | {m:.4f} | {s:.4f} | {m:.3f} ± {s:.3f} |\n"

    if EVALUATE_CAUSAL_ROLES:
        report += """
### 因果角色识别指标（均值 ± 标准差）

| 指标 | 均值 | 标准差 | 学术格式 | 说明 |
|------|------|--------|----------|------|
"""
        for k, name in [
            ('avg_confounder_f1',         '混杂变量识别 F1'),
            ('avg_mediator_f1',           '中介变量识别 F1'),
            ('avg_instrument_f1',         '工具变量识别 F1'),
            ('avg_causal_structure_score','综合因果结构得分'),
            ('direct_effect_accuracy',    '直接效应识别准确率'),
        ]:
            if k in stats:
                m, s = stats[k]['mean'], stats[k]['std']
                report += f"| {k} | {m:.4f} | {s:.4f} | {m:.3f} ± {s:.3f} | {name} |\n"

    report += innovation_note
    report += f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    out_md = os.path.join(OUT_DIR, f"{algorithm}_report.md")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Markdown 报告已保存: {out_md}")


def generate_comparison_report(all_results: dict):
    """生成多算法横向对比报告"""
    algorithms   = list(all_results.keys())
    metrics_cols = ['SHD', 'TPR', 'FDR', 'Precision', 'Recall', 'F1']

    print("\n" + "=" * 90)
    print("算法横向对比（均值 ± 标准差）")
    print("=" * 90)

    comparison = []
    for metric in metrics_cols:
        row = {'Metric': metric}
        for algo in algorithms:
            m = all_results[algo]['stats'][metric]['mean']
            s = all_results[algo]['stats'][metric]['std']
            row[algo] = f"{m:.3f} ± {s:.3f}"
        comparison.append(row)

    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))

    out_csv = os.path.join(OUT_DIR, "algorithm_comparison.csv")
    df_comparison.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存: {out_csv}")

    md_lines = [
        "# 因果发现算法横向对比报告\n",
        f"实验配置：{N_NODES} 节点 × {N_SAMPLES} 样本 × {N_EXPERIMENTS} 次蒙特卡洛 × {GRAPH_TYPE} 图\n",
        "## 指标对比（均值 ± 标准差）\n",
        "| 指标 | " + " | ".join(ALGORITHM_CN_NAME.get(a, a) for a in algorithms) + " |\n",
        "|" + "------|" * (len(algorithms) + 1) + "\n",
    ]
    for row in comparison:
        md_lines.append(
            f"| {row['Metric']} | " + " | ".join(row[a] for a in algorithms) + " |\n"
        )
    md_lines += [
        "\n## 指标说明\n",
        "- SHD（结构汉明距离）：越小越好\n",
        "- TPR / Recall：越大越好\n",
        "- FDR（假发现率）：越小越好\n",
        "- Precision / F1：越大越好\n",
        f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    out_md = os.path.join(OUT_DIR, "algorithm_comparison_report.md")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)
    print(f"对比 Markdown 报告已保存: {out_md}")


if __name__ == "__main__":
    import argparse

    ALL_BASELINES   = ["cuts_plus", "nts_notears", "coupled"]
    ALL_INNOVATIONS = ["biattn_cuts", "multiscale_nts", "mb_cuts"]
    ALL_ALGORITHMS  = ALL_BASELINES + ALL_INNOVATIONS

    parser = argparse.ArgumentParser(
        description="蒙特卡洛基准测试（修正版）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--algorithm",
        choices=ALL_ALGORITHMS + ["baselines", "innovations", "all"],
        default="all",
    )
    parser.add_argument("--n_experiments", type=int, default=50)
    parser.add_argument("--n_nodes",       type=int, default=20)
    parser.add_argument("--n_samples",     type=int, default=1000)
    parser.add_argument("--graph_type",    choices=["er", "scale_free", "layered"], default="layered")
    parser.add_argument("--noise_type",
        choices=["gaussian", "heteroscedastic", "heavy_tail", "periodic"],
        default="heteroscedastic"
    )
    parser.add_argument("--use_industrial_functions", action="store_true", default=True)
    parser.add_argument("--add_time_lag",             action="store_true", default=True)
    parser.add_argument("--evaluate_causal_roles",    action="store_true", default=True)
    parser.add_argument("--n_treatment_samples",      type=int, default=4)

    args = parser.parse_args()

    N_EXPERIMENTS            = args.n_experiments
    N_NODES                  = args.n_nodes
    N_SAMPLES                = args.n_samples
    GRAPH_TYPE               = args.graph_type
    NOISE_TYPE               = args.noise_type
    USE_INDUSTRIAL_FUNCTIONS = args.use_industrial_functions
    ADD_TIME_LAG             = args.add_time_lag
    EVALUATE_CAUSAL_ROLES    = args.evaluate_causal_roles
    N_TREATMENT_SAMPLES      = args.n_treatment_samples

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
