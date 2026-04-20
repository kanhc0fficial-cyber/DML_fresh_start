"""
run_dml_theory_validation_v4.py
===============================
DML 理论验证 —— v4 创新方案：改进的交叉拟合策略（Improved Cross-Fitting Strategies）

═══════════════════════════════════════════════════════════════════
  本脚本验证 v4 创新在合成数据上的因果推断性能。

  核心创新：在 v3（两阶段解耦 VAE-DML）基础上，新增四项交叉拟合改进：
═══════════════════════════════════════════════════════════════════

  【改进1】折边随机化（Fold Boundary Jitter / Repeated Cross-Fitting）
    每次重复使用不同的折边界：
      boundary_k = k × block_size + Uniform(-jitter_max, jitter_max)
      jitter_max = block_size × FOLD_JITTER_RATIO (默认 0.10)
    减少对单次数据分区的敏感性。
    理论依据：Chernozhukov et al. (2018) 重复交叉拟合建议。

  【改进2】分层交叉拟合（Stratified Cross-Fitting）
    当 D 不平衡时，检查每个训练折的"高处理量"样本数量；
    若 count < MIN_TREAT_SAMPLES 则跳过该折，
    防止倾向得分模型因极端不平衡而失准。

  【改进3】嵌套学习率搜索（Nested Learning Rate Search）
    在训练数据上做 80/20 内层验证，选择最佳 LR：
      候选集 NESTED_LR_CANDIDATES = [0.001, 0.003, 0.01]
    外层测试集完全隔离，严格防止超参调优数据泄露。

  【改进4】正确中位数聚合 SE（Correct Median Aggregation SE）
    V_final = Median(V_b + (θ_b - θ_final)²)
    已在 common 模块中实现，v4 继续沿用并强调。

═══════════════════════════════════════════════════════════════════
  架构：同 v3（TwoStageVAEDML with MLP encoder/decoder + prediction heads）
  交叉拟合封装层增强为 v4_dml_estimate
═══════════════════════════════════════════════════════════════════

  验证内容：
    1. 无偏性（Unbiasedness）：E[θ̂] ≈ θ_true
    2. √n-一致性（√n-Consistency）：RMSE ∝ 1/√n
    3. 置信区间覆盖率（Coverage）：95% CI 覆盖 θ_true ≈ 95%
    4. 交叉拟合策略对比（cf_compare）—— v4 的核心实验

  运行模式（argparse）：
    quick:       快速验证（30 次实验）
    full:        完整蒙特卡洛（200 次实验）
    consistency: √n-一致性检验
    cf_compare:  交叉拟合策略对比（v4 核心实验）
    all:         全部运行

用法：
  python run_dml_theory_validation_v4.py --mode quick
  python run_dml_theory_validation_v4.py --mode full --n_experiments 200
  python run_dml_theory_validation_v4.py --mode consistency
  python run_dml_theory_validation_v4.py --mode cf_compare
  python run_dml_theory_validation_v4.py --mode all
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── 路径配置 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import dml_validation_common as dvc

# 输出目录
OUT_DIR = os.path.join(BASE_DIR, "DML理论验证")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── PyTorch 依赖 ────────────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = None
    print("[警告] PyTorch 未安装，v4 估计器将不可用。")
    print("       请执行: pip install torch")


# ═══════════════════════════════════════════════════════════════════
#  v3 基础超参数
# ═══════════════════════════════════════════════════════════════════

LATENT_DIM = 32           # 潜变量维度
HIDDEN_DIM_ENCODER = 64   # Encoder MLP 隐层维度
HIDDEN_DIM_HEAD = 32      # 预测头 MLP 隐层维度
BETA_KL = 0.1             # KL 权重上限（退火终点）
ANNEAL_EPOCHS = 20        # KL 退火轮数
MAX_EPOCHS_VAE = 60       # Stage 1 最大训练轮数
MAX_EPOCHS_HEAD = 40      # Stage 2 最大训练轮数
LR_VAE = 0.001            # Stage 1 默认学习率
LR_HEAD = 0.003           # Stage 2 默认学习率
GRAD_CLIP = 1.0           # 梯度裁剪上限

# ═══════════════════════════════════════════════════════════════════
#  v4 新增超参数（交叉拟合策略改进）
# ═══════════════════════════════════════════════════════════════════

FOLD_JITTER_RATIO = 0.10        # 折边抖动比例：±10% block_size
MIN_TREAT_SAMPLES = 5           # 分层检查最低高处理量样本数
NESTED_LR_CANDIDATES = [0.001, 0.003, 0.01]  # 嵌套LR搜索候选集
DEFAULT_LR = LR_HEAD            # 默认学习率（不使用嵌套搜索时）
NESTED_INNER_RATIO = 0.2       # 内层验证集比例（训练数据的 20%）


# ═══════════════════════════════════════════════════════════════════
#  模型定义（适配表格数据的 MLP 版本，与 v3 一致）
#  仅在 PyTorch 可用时定义
# ═══════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class MLPEncoder(nn.Module):
        """
        MLP 编码器：X_confounders → (μ, logvar)

        使用 LayerNorm 提升训练稳定性。
        推断时只用 μ（mean encoding），不采样，确保确定性。
        """

        def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM_ENCODER,
                     latent_dim: int = LATENT_DIM):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x: torch.Tensor):
            """返回 (μ, logvar)"""
            h = self.shared(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
            """推断专用：只返回 μ，不采样（确定性）"""
            mu, _ = self.forward(x)
            return mu

    class MLPDecoder(nn.Module):
        """
        MLP 解码器：z → X_hat（重建混杂变量）
        用于 Stage 1 的 VAE 训练。
        """

        def __init__(self, latent_dim: int = LATENT_DIM,
                     hidden_dim: int = HIDDEN_DIM_ENCODER,
                     output_dim: int = 10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)

    class PredictionHead(nn.Module):
        """
        独立预测头：z → scalar
        Stage 2 中 Head_Y 和 Head_D 各有一个实例，完全独立训练。
        """

        def __init__(self, latent_dim: int = LATENT_DIM,
                     hidden_dim: int = HIDDEN_DIM_HEAD):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z).squeeze(-1)

    class TwoStageVAEDML:
        """
        两阶段解耦 VAE-DML 估计器（与 v3 一致）。

        Stage 1: 训练 Encoder + Decoder（VAE with KL annealing）
        Stage 2: 冻结 Encoder，独立训练 Head_Y 和 Head_D
        推断:    res_Y = Y - Head_Y(Encoder(X)), res_D = D - Head_D(Encoder(X))
        """

        def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM,
                     hidden_dim_enc: int = HIDDEN_DIM_ENCODER,
                     hidden_dim_head: int = HIDDEN_DIM_HEAD,
                     device=None):
            self.device = device or DEVICE
            self.latent_dim = latent_dim

            self.encoder = MLPEncoder(input_dim, hidden_dim_enc, latent_dim).to(self.device)
            self.decoder = MLPDecoder(latent_dim, hidden_dim_enc, input_dim).to(self.device)
            self.head_Y = PredictionHead(latent_dim, hidden_dim_head).to(self.device)
            self.head_D = PredictionHead(latent_dim, hidden_dim_head).to(self.device)

        def train_stage1(self, X_ctrl: np.ndarray, epochs: int = MAX_EPOCHS_VAE,
                         beta_kl: float = BETA_KL, anneal_epochs: int = ANNEAL_EPOCHS,
                         lr: float = LR_VAE, seed: int = 42):
            """
            Stage 1: 训练 Encoder + Decoder (VAE with KL Annealing)
            Loss = MSE_recon + β_anneal(t) × KL(q(z|x) || p(z))
            """
            torch.manual_seed(seed)
            self.encoder.train()
            self.decoder.train()

            X_tensor = torch.FloatTensor(X_ctrl).to(self.device)
            optimizer = optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=lr, weight_decay=1e-5,
            )

            n = X_tensor.shape[0]
            batch_size = min(256, n)

            for epoch in range(epochs):
                # KL 退火
                beta_t = beta_kl * min((epoch + 1) / anneal_epochs, 1.0)
                perm = torch.randperm(n, device=self.device)

                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    x_batch = X_tensor[idx]

                    mu, logvar = self.encoder(x_batch)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + std * eps

                    x_recon = self.decoder(z)
                    recon_loss = nn.functional.mse_loss(x_recon, x_batch)
                    kl_loss = -0.5 * torch.mean(
                        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    )
                    loss = recon_loss + beta_t * kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        GRAD_CLIP,
                    )
                    optimizer.step()

            # 冻结 encoder
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        def train_stage2(self, X_ctrl: np.ndarray, Y: np.ndarray, D: np.ndarray,
                         epochs: int = MAX_EPOCHS_HEAD, lr: float = LR_HEAD,
                         seed: int = 42):
            """
            Stage 2: 冻结 Encoder，独立训练 Head_Y 和 Head_D
            """
            torch.manual_seed(seed + 100)
            self.head_Y.train()
            self.head_D.train()

            X_tensor = torch.FloatTensor(X_ctrl).to(self.device)
            Y_tensor = torch.FloatTensor(Y).to(self.device)
            D_tensor = torch.FloatTensor(D).to(self.device)

            with torch.no_grad():
                z = self.encoder.encode_mean(X_tensor)

            opt_Y = optim.Adam(self.head_Y.parameters(), lr=lr, weight_decay=1e-5)
            opt_D = optim.Adam(self.head_D.parameters(), lr=lr, weight_decay=1e-5)

            n = z.shape[0]
            batch_size = min(256, n)

            for epoch in range(epochs):
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    z_batch = z[idx]
                    y_batch = Y_tensor[idx]
                    d_batch = D_tensor[idx]

                    # Head_Y
                    y_pred = self.head_Y(z_batch)
                    loss_Y = nn.functional.mse_loss(y_pred, y_batch)
                    opt_Y.zero_grad()
                    loss_Y.backward()
                    nn.utils.clip_grad_norm_(self.head_Y.parameters(), GRAD_CLIP)
                    opt_Y.step()

                    # Head_D（完全独立）
                    d_pred = self.head_D(z_batch)
                    loss_D = nn.functional.mse_loss(d_pred, d_batch)
                    opt_D.zero_grad()
                    loss_D.backward()
                    nn.utils.clip_grad_norm_(self.head_D.parameters(), GRAD_CLIP)
                    opt_D.step()

            self.head_Y.eval()
            self.head_D.eval()

        def predict_residuals(self, X_ctrl: np.ndarray, Y: np.ndarray,
                              D: np.ndarray) -> tuple:
            """
            计算残差：res_Y = Y - Head_Y(Encoder(X)), res_D = D - Head_D(Encoder(X))
            """
            self.encoder.eval()
            self.head_Y.eval()
            self.head_D.eval()

            X_tensor = torch.FloatTensor(X_ctrl).to(self.device)
            with torch.no_grad():
                z = self.encoder.encode_mean(X_tensor)
                y_hat = self.head_Y(z).cpu().numpy()
                d_hat = self.head_D(z).cpu().numpy()

            res_Y = Y - y_hat
            res_D = D - d_hat
            return res_Y, res_D


# ═══════════════════════════════════════════════════════════════════
#  v4 交叉拟合改进工具函数
# ═══════════════════════════════════════════════════════════════════

def _generate_jittered_folds(n: int, n_folds: int, seed: int,
                             jitter_ratio: float = FOLD_JITTER_RATIO):
    """
    生成带有折边随机抖动的交叉拟合分割。

    对每个折的边界施加 Uniform(-jitter_max, jitter_max) 的随机偏移，
    使得每次重复的数据切分不同，消除单次分组随机性。

    参数:
        n:            样本总数
        n_folds:      折数
        seed:         随机种子
        jitter_ratio: 抖动比例（相对于 block_size）

    返回:
        list of (train_idx, test_idx) 元组
    """
    rng = np.random.RandomState(seed)
    block_size = n // n_folds
    jitter_max = int(block_size * jitter_ratio)

    # 生成带抖动的折边界
    boundaries = [0]
    for k in range(1, n_folds):
        base_boundary = k * block_size
        if jitter_max > 0:
            offset = rng.randint(-jitter_max, jitter_max + 1)
        else:
            offset = 0
        boundary = max(1, min(n - 1, base_boundary + offset))
        boundaries.append(boundary)
    boundaries.append(n)

    # 确保边界单调递增（修正可能的交叉）
    boundaries = sorted(set(boundaries))
    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    if boundaries[-1] != n:
        boundaries.append(n)

    # 生成训练/测试索引对
    indices = rng.permutation(n)  # 先打乱索引
    folds = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < 2:  # 太小的折跳过
            continue
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, test_idx))

    return folds


def _generate_standard_folds(n: int, n_folds: int, seed: int):
    """
    生成标准 K-Fold 分割（与 v3 行为一致）。

    参数:
        n:       样本总数
        n_folds: 折数
        seed:    随机种子

    返回:
        list of (train_idx, test_idx) 元组
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(n)
    return list(kf.split(indices))


def _nested_lr_search(X_train: np.ndarray, Y_train: np.ndarray,
                      D_train: np.ndarray, seed: int = 42,
                      input_dim: int = None,
                      lr_candidates: list = None) -> float:
    """
    嵌套学习率搜索：在训练数据上做 80/20 内层验证。

    在训练数据末 20% 作为内层验证集，前 80% 用于训练；
    比较所有候选学习率的预测损失（Head_Y + Head_D），
    选择总损失最小的学习率。

    参数:
        X_train: 训练集特征
        Y_train: 训练集结果变量
        D_train: 训练集处理变量
        seed:    随机种子
        input_dim: 输入维度
        lr_candidates: 学习率候选列表

    返回:
        best_lr: 最佳学习率
    """
    if lr_candidates is None:
        lr_candidates = NESTED_LR_CANDIDATES
    if input_dim is None:
        input_dim = X_train.shape[1]

    n_train = len(Y_train)
    n_inner_val = max(10, int(n_train * NESTED_INNER_RATIO))
    n_inner_train = n_train - n_inner_val

    # 分割内层训练/验证集
    X_inner_train = X_train[:n_inner_train]
    Y_inner_train = Y_train[:n_inner_train]
    D_inner_train = D_train[:n_inner_train]
    X_inner_val = X_train[n_inner_train:]
    Y_inner_val = Y_train[n_inner_train:]
    D_inner_val = D_train[n_inner_train:]

    best_lr = lr_candidates[0]
    best_loss = float('inf')

    for lr_cand in lr_candidates:
        try:
            # 训练模型
            model = TwoStageVAEDML(input_dim=input_dim)
            model.train_stage1(X_inner_train, lr=LR_VAE, seed=seed,
                               epochs=max(20, MAX_EPOCHS_VAE // 2))
            model.train_stage2(X_inner_train, Y_inner_train, D_inner_train,
                               lr=lr_cand, seed=seed,
                               epochs=max(15, MAX_EPOCHS_HEAD // 2))

            # 在内层验证集上评估
            X_val_tensor = torch.FloatTensor(X_inner_val).to(DEVICE)
            with torch.no_grad():
                z_val = model.encoder.encode_mean(X_val_tensor)
                y_pred = model.head_Y(z_val).cpu().numpy()
                d_pred = model.head_D(z_val).cpu().numpy()

            loss_y = np.mean((Y_inner_val - y_pred) ** 2)
            loss_d = np.mean((D_inner_val - d_pred) ** 2)
            total_loss = loss_y + loss_d

            if total_loss < best_loss:
                best_loss = total_loss
                best_lr = lr_cand
        except Exception:
            continue

    return best_lr


# ═══════════════════════════════════════════════════════════════════
#  v4 DML 估计器（含改进的交叉拟合策略）
# ═══════════════════════════════════════════════════════════════════

def v4_dml_estimate(Y: np.ndarray, D: np.ndarray, X_ctrl: np.ndarray,
                    seed: int = 42, n_folds: int = 5, n_repeats: int = 5,
                    fold_jitter_ratio: float = FOLD_JITTER_RATIO,
                    use_stratified: bool = True,
                    nested_lr_search: bool = False) -> tuple:
    """
    v4 改进交叉拟合 DML 估计器。

    在 v3 两阶段解耦 VAE-DML 的基础上，增强交叉拟合层：
    1. 折边随机化（Fold Boundary Jitter）：每次重复使用不同折边
    2. 分层检查（Stratified）：跳过高处理量样本不足的折
    3. 嵌套 LR 搜索（Nested LR Search）：训练数据内层验证选最佳 LR
    4. 正确中位数聚合 SE（Median Aggregation）

    参数:
        Y:                  结果变量 (n,)
        D:                  处理变量 (n,)
        X_ctrl:             控制变量（混杂变量）(n, p)
        seed:               随机种子
        n_folds:            交叉拟合折数
        n_repeats:          重复交叉拟合次数
        fold_jitter_ratio:  折边抖动比例（0 = 退化为固定折边）
        use_stratified:     是否启用分层检查
        nested_lr_search:   是否启用嵌套学习率搜索

    返回:
        (theta, se, ci_lower, ci_upper)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 未安装，无法使用 v4 估计器")

    n = len(Y)
    input_dim = X_ctrl.shape[1]

    # 数据标准化（神经网络必须）
    X_mean = X_ctrl.mean(axis=0)
    X_std = X_ctrl.std(axis=0) + 1e-8
    X_normed = (X_ctrl - X_mean) / X_std

    Y_mean = Y.mean()
    Y_std = Y.std() + 1e-8
    D_mean = D.mean()
    D_std = D.std() + 1e-8
    Y_normed = (Y - Y_mean) / Y_std
    D_normed = (D - D_mean) / D_std

    def _single_crossfit_v4(seed_b: int):
        """单次 v4 改进交叉拟合 DML 估计"""
        torch.manual_seed(seed_b)
        np.random.seed(seed_b)

        res_Y_all = np.full(n, np.nan)
        res_D_all = np.full(n, np.nan)

        # 改进1：折边随机化 vs 标准分折
        if fold_jitter_ratio > 0:
            folds = _generate_jittered_folds(n, n_folds, seed=seed_b,
                                             jitter_ratio=fold_jitter_ratio)
        else:
            folds = _generate_standard_folds(n, n_folds, seed=seed_b)

        # 计算 D 中位数（用于分层检查）
        d_median = np.median(D_normed)

        for train_idx, test_idx in folds:
            # 改进2：分层检查
            if use_stratified:
                n_high = np.sum(D_normed[train_idx] > d_median)
                if n_high < MIN_TREAT_SAMPLES:
                    continue  # 跳过高处理量样本不足的折

            X_train = X_normed[train_idx]
            X_test = X_normed[test_idx]
            Y_train = Y_normed[train_idx]
            Y_test = Y_normed[test_idx]
            D_train = D_normed[train_idx]
            D_test = D_normed[test_idx]

            # 改进3：嵌套学习率搜索
            if nested_lr_search:
                best_lr = _nested_lr_search(
                    X_train, Y_train, D_train,
                    seed=seed_b, input_dim=input_dim,
                )
            else:
                best_lr = DEFAULT_LR

            # 训练 v3 模型
            model = TwoStageVAEDML(input_dim=input_dim)
            model.train_stage1(X_train, seed=seed_b)
            model.train_stage2(X_train, Y_train, D_train,
                               lr=best_lr, seed=seed_b)

            # 在测试集上计算残差
            res_Y_fold, res_D_fold = model.predict_residuals(X_test, Y_test, D_test)
            res_Y_all[test_idx] = res_Y_fold
            res_D_all[test_idx] = res_D_fold

        # 排除未被覆盖的样本（因跳过折或边界问题）
        valid_mask = ~np.isnan(res_Y_all)
        if valid_mask.sum() < n // 2:
            # 有效样本太少，回退到标准分折
            return _fallback_standard_crossfit(seed_b)

        res_Y_valid = res_Y_all[valid_mask]
        res_D_valid = res_D_all[valid_mask]
        n_valid = valid_mask.sum()

        # 反标准化残差
        res_Y_orig = res_Y_valid * Y_std
        res_D_orig = res_D_valid * D_std

        # θ = Σ(res_D × res_Y) / Σ(res_D²)
        denom = np.sum(res_D_orig ** 2) + 1e-12
        theta_k = np.sum(res_D_orig * res_Y_orig) / denom

        # Neyman 式解析 SE
        psi = res_D_orig * (res_Y_orig - theta_k * res_D_orig)
        J = np.mean(res_D_orig ** 2)
        var_neyman = float(np.mean(psi ** 2) / (n_valid * J ** 2 + 1e-12))

        return theta_k, var_neyman

    def _fallback_standard_crossfit(seed_b: int):
        """回退标准交叉拟合（不带任何 v4 改进）"""
        torch.manual_seed(seed_b)
        np.random.seed(seed_b)

        res_Y_all = np.zeros(n)
        res_D_all = np.zeros(n)

        folds = _generate_standard_folds(n, n_folds, seed=seed_b)
        for train_idx, test_idx in folds:
            model = TwoStageVAEDML(input_dim=input_dim)
            model.train_stage1(X_normed[train_idx], seed=seed_b)
            model.train_stage2(X_normed[train_idx], Y_normed[train_idx],
                               D_normed[train_idx], seed=seed_b)
            res_Y_fold, res_D_fold = model.predict_residuals(
                X_normed[test_idx], Y_normed[test_idx], D_normed[test_idx])
            res_Y_all[test_idx] = res_Y_fold
            res_D_all[test_idx] = res_D_fold

        res_Y_orig = res_Y_all * Y_std
        res_D_orig = res_D_all * D_std
        denom = np.sum(res_D_orig ** 2) + 1e-12
        theta_k = np.sum(res_D_orig * res_Y_orig) / denom
        psi = res_D_orig * (res_Y_orig - theta_k * res_D_orig)
        J = np.mean(res_D_orig ** 2)
        var_neyman = float(np.mean(psi ** 2) / (n * J ** 2 + 1e-12))
        return theta_k, var_neyman

    # 重复交叉拟合（每次使用不同的折边）
    theta_boots = []
    var_boots = []

    for b in range(n_repeats):
        boot_seed = seed * 1000 + b * 7
        theta_b, var_b = _single_crossfit_v4(boot_seed)
        theta_boots.append(theta_b)
        var_boots.append(var_b)

    theta_boots = np.array(theta_boots)
    var_boots = np.array(var_boots)

    # 改进4：正确中位数聚合（Chernozhukov 2018）
    # V_final = Median(V_b + (θ_b - θ_final)²)
    theta_final = float(np.median(theta_boots))
    var_final = float(np.median(var_boots + (theta_boots - theta_final) ** 2))
    se_final = np.sqrt(max(var_final, 1e-12))

    ci_lower = theta_final - 1.96 * se_final
    ci_upper = theta_final + 1.96 * se_final

    return float(theta_final), se_final, float(ci_lower), float(ci_upper)


# ═══════════════════════════════════════════════════════════════════
#  各策略的包装函数（供 cf_compare 使用）
# ═══════════════════════════════════════════════════════════════════

def _strategy_a_estimator(Y, D, X_ctrl, seed):
    """策略A：标准固定折边（v3 基线行为）"""
    return v4_dml_estimate(
        Y, D, X_ctrl, seed=seed,
        fold_jitter_ratio=0.0,  # 无抖动
        use_stratified=False,   # 无分层
        nested_lr_search=False, # 无嵌套LR
    )


def _strategy_b_estimator(Y, D, X_ctrl, seed):
    """策略B：标准 + 分层折检查"""
    return v4_dml_estimate(
        Y, D, X_ctrl, seed=seed,
        fold_jitter_ratio=0.0,
        use_stratified=True,    # 启用分层
        nested_lr_search=False,
    )


def _strategy_c_estimator(Y, D, X_ctrl, seed):
    """策略C：标准 + 折边随机化"""
    return v4_dml_estimate(
        Y, D, X_ctrl, seed=seed,
        fold_jitter_ratio=FOLD_JITTER_RATIO,  # 启用抖动
        use_stratified=False,
        nested_lr_search=False,
    )


def _strategy_d_estimator(Y, D, X_ctrl, seed):
    """策略D：标准 + 嵌套LR搜索"""
    return v4_dml_estimate(
        Y, D, X_ctrl, seed=seed,
        fold_jitter_ratio=0.0,
        use_stratified=False,
        nested_lr_search=True,  # 启用嵌套LR
    )


def _strategy_e_estimator(Y, D, X_ctrl, seed):
    """策略E：全部改进（完整 v4）"""
    return v4_dml_estimate(
        Y, D, X_ctrl, seed=seed,
        fold_jitter_ratio=FOLD_JITTER_RATIO,
        use_stratified=True,
        nested_lr_search=False,  # 嵌套LR开销大，默认关闭以加速
    )


# ═══════════════════════════════════════════════════════════════════
#  实验模式实现
# ═══════════════════════════════════════════════════════════════════

def run_quick(args):
    """快速验证模式：少量实验验证 v4 估计器基本正确性"""
    print("\n" + "█" * 70)
    print("  [快速模式] v4 改进交叉拟合 VAE-DML 验证")
    print("█" * 70)

    dag_info = dvc.setup_fixed_dag(
        n_nodes=20,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
        enforce_linear_ty=True,
    )
    ate_true = dvc.compute_ate_for_dag(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    n_exp = min(args.n_experiments, 30)

    # 使用完整 v4 估计器
    def estimator_fn(Y, D, X_ctrl, seed):
        return v4_dml_estimate(
            Y, D, X_ctrl, seed=seed,
            fold_jitter_ratio=args.fold_jitter_ratio,
            use_stratified=args.use_stratified,
            nested_lr_search=args.nested_lr_search,
        )

    df, summary = dvc.run_monte_carlo(
        estimator_fn=estimator_fn,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=n_exp,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="v4_quick",
        method_name="VAE-DML-v4 (Quick)",
    )

    if not df.empty:
        dvc.save_results(df, summary, "v4_vae_dml_quick")
    return df, summary


def run_full(args):
    """完整蒙特卡洛验证：200 次独立实验评估无偏性、覆盖率"""
    print("\n" + "█" * 70)
    print("  [完整模式] v4 改进交叉拟合 VAE-DML 完整蒙特卡洛验证")
    print("█" * 70)

    dag_info = dvc.setup_fixed_dag(
        n_nodes=20,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
        enforce_linear_ty=True,
    )
    ate_true = dvc.compute_ate_for_dag(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    def estimator_fn(Y, D, X_ctrl, seed):
        return v4_dml_estimate(
            Y, D, X_ctrl, seed=seed,
            fold_jitter_ratio=args.fold_jitter_ratio,
            use_stratified=args.use_stratified,
            nested_lr_search=args.nested_lr_search,
        )

    df, summary = dvc.run_monte_carlo(
        estimator_fn=estimator_fn,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=args.n_experiments,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="v4_full",
        method_name="VAE-DML-v4",
    )

    if not df.empty:
        dvc.save_results(df, summary, "v4_vae_dml_full")
    return df, summary


def run_consistency(args):
    """√n-一致性验证：验证 RMSE ∝ 1/√n"""
    print("\n" + "█" * 70)
    print("  [一致性模式] v4 改进交叉拟合 VAE-DML √n-一致性验证")
    print("█" * 70)

    dag_info = dvc.setup_fixed_dag(
        n_nodes=20,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
        enforce_linear_ty=True,
    )
    ate_true = dvc.compute_ate_for_dag(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    def estimator_fn(Y, D, X_ctrl, seed):
        return v4_dml_estimate(
            Y, D, X_ctrl, seed=seed,
            fold_jitter_ratio=args.fold_jitter_ratio,
            use_stratified=args.use_stratified,
            nested_lr_search=args.nested_lr_search,
        )

    df_cons = dvc.run_consistency_validation(
        estimator_fn=estimator_fn,
        dag_info=dag_info,
        ate_true=ate_true,
        sample_sizes=[500, 1000, 2000, 4000, 8000],
        n_experiments_per_size=50,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        method_name="VAE-DML-v4",
    )

    if not df_cons.empty:
        csv_path = os.path.join(OUT_DIR, "v4_vae_dml_consistency.csv")
        df_cons.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  一致性结果已保存：{csv_path}")

    return df_cons


def run_cf_compare(args):
    """
    交叉拟合策略对比实验（v4 核心实验）

    对比五种策略在同一数据上的 ATE 估计性能差异：
      策略A：标准固定折边（v3 基线）
      策略B：标准 + 分层折检查
      策略C：标准 + 折边随机化
      策略D：标准 + 嵌套LR搜索
      策略E：全部改进组合（完整 v4）

    输出对比表：bias, RMSE, coverage, SE, 计算耗时
    保存为 refutation_cf_compare_v4.csv
    """
    print("\n" + "█" * 70)
    print("  [策略对比模式] v4 交叉拟合策略对比实验")
    print("█" * 70)

    dag_info = dvc.setup_fixed_dag(
        n_nodes=20,
        graph_type=args.graph_type,
        use_industrial=args.use_industrial,
        dag_seed=42,
        enforce_linear_ty=True,
    )
    ate_true = dvc.compute_ate_for_dag(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    # 定义策略
    strategies = {
        "A_标准固定折边": _strategy_a_estimator,
        "B_分层折检查": _strategy_b_estimator,
        "C_折边随机化": _strategy_c_estimator,
        "D_嵌套LR搜索": _strategy_d_estimator,
        "E_全部改进": _strategy_e_estimator,
    }

    n_exp_cf = min(args.n_experiments, 50)  # 策略对比用适中实验数
    print(f"  每策略实验次数: {n_exp_cf}")
    print(f"  样本量: {args.n_samples}")

    # 数据生成参数（SyntheticDAGGenerator 从 common 模块获取）
    SyntheticDAGGenerator = dvc.SyntheticDAGGenerator

    n_nodes = dag_info["gen_base"].n_nodes
    adj_true = dag_info["adj_true"]
    edge_funcs = dag_info["edge_funcs"]
    t_idx = dag_info["t_idx"]
    y_idx = dag_info["y_idx"]
    confounder_indices = dag_info["confounder_indices"]

    all_results = []

    for strategy_name, estimator_fn in strategies.items():
        print(f"\n{'─' * 60}")
        print(f"  策略: {strategy_name}")
        print(f"{'─' * 60}")

        t_start = time.perf_counter()
        biases = []
        rmses = []
        coverages = []
        ses = []
        n_success = 0

        for exp_i in range(n_exp_cf):
            try:
                data_seed = exp_i * 13 + 1000
                gen_data = SyntheticDAGGenerator(n_nodes=n_nodes, seed=data_seed)
                X_data = gen_data.generate_data(
                    adj_true, edge_funcs,
                    n_samples=args.n_samples,
                    noise_scale=args.noise_scale,
                    noise_type=args.noise_type,
                    add_time_lag=False,
                )

                D = X_data[:, t_idx]
                Y = X_data[:, y_idx]
                if len(confounder_indices) > 0:
                    X_ctrl = X_data[:, confounder_indices]
                else:
                    X_ctrl = np.ones((args.n_samples, 1))

                theta_hat, se, ci_lower, ci_upper = estimator_fn(
                    Y, D, X_ctrl, data_seed
                )

                bias = theta_hat - ate_true
                covers = bool(ci_lower <= ate_true <= ci_upper)

                biases.append(bias)
                ses.append(se)
                coverages.append(covers)
                n_success += 1

            except Exception as e:
                if n_success == 0 and exp_i < 3:
                    print(f"    [实验 {exp_i}] 失败: {e}")
                continue

        elapsed = time.perf_counter() - t_start

        if n_success > 0:
            biases_arr = np.array(biases)
            mean_bias = float(np.mean(biases_arr))
            rmse = float(np.sqrt(np.mean(biases_arr ** 2)))
            coverage = float(np.mean(coverages))
            mean_se = float(np.mean(ses))
            time_per_exp = elapsed / n_success

            print(f"    成功: {n_success}/{n_exp_cf}")
            print(f"    Mean Bias: {mean_bias:+.6f}")
            print(f"    RMSE:      {rmse:.6f}")
            print(f"    Coverage:  {coverage:.1%}")
            print(f"    Mean SE:   {mean_se:.6f}")
            print(f"    耗时:      {elapsed:.1f}s ({time_per_exp:.2f}s/实验)")

            all_results.append({
                "strategy": strategy_name,
                "n_experiments": n_success,
                "mean_bias": round(mean_bias, 6),
                "rmse": round(rmse, 6),
                "coverage_95": round(coverage, 4),
                "mean_se": round(mean_se, 6),
                "time_total_s": round(elapsed, 1),
                "time_per_exp_s": round(time_per_exp, 2),
            })
        else:
            print(f"    ⚠ 所有实验失败")

    # 汇总并输出对比表
    if all_results:
        df_compare = pd.DataFrame(all_results)

        print("\n" + "═" * 70)
        print("  v4 交叉拟合策略对比结果")
        print("═" * 70)
        print(df_compare.to_string(index=False))
        print("═" * 70)

        # 保存结果
        csv_path = os.path.join(OUT_DIR, "refutation_cf_compare_v4.csv")
        df_compare.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  对比结果已保存：{csv_path}")

        return df_compare
    else:
        print("  [错误] 所有策略均失败")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DML 理论验证 —— v4 改进交叉拟合策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_dml_theory_validation_v4.py --mode quick
  python run_dml_theory_validation_v4.py --mode full --n_experiments 200
  python run_dml_theory_validation_v4.py --mode consistency
  python run_dml_theory_validation_v4.py --mode cf_compare
  python run_dml_theory_validation_v4.py --mode all --fold_jitter_ratio 0.1 --use_stratified
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "full", "consistency", "cf_compare", "all"],
        help="运行模式: quick=快速验证, full=完整蒙特卡洛, "
             "consistency=√n一致性, cf_compare=策略对比, all=全部",
    )
    parser.add_argument(
        "--n_experiments", type=int, default=200,
        help="蒙特卡洛实验次数 (默认: 200)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=2000,
        help="每次实验的样本量 (默认: 2000)",
    )
    parser.add_argument(
        "--graph_type", type=str, default="layered",
        choices=["layered", "er", "scale_free"],
        help="DAG 图类型 (默认: layered)",
    )
    parser.add_argument(
        "--noise_type", type=str, default="gaussian",
        choices=["gaussian", "student_t", "heteroscedastic"],
        help="噪声类型 (默认: gaussian)",
    )
    parser.add_argument(
        "--noise_scale", type=float, default=0.3,
        help="噪声标准差 (默认: 0.3)",
    )
    parser.add_argument(
        "--use_industrial", action="store_true",
        help="使用工业过程特征的边函数",
    )
    # v4 特有参数
    parser.add_argument(
        "--fold_jitter_ratio", type=float, default=FOLD_JITTER_RATIO,
        help=f"折边抖动比例 (默认: {FOLD_JITTER_RATIO}，设 0 退化为 v3)",
    )
    parser.add_argument(
        "--use_stratified", action="store_true", default=True,
        help="启用分层交叉拟合检查 (默认: 开启)",
    )
    parser.add_argument(
        "--no_stratified", action="store_true",
        help="关闭分层交叉拟合检查",
    )
    parser.add_argument(
        "--nested_lr_search", action="store_true", default=False,
        help="启用嵌套学习率搜索 (默认: 关闭，计算开销约 3×)",
    )

    args = parser.parse_args()

    # 处理 --no_stratified 标志
    if args.no_stratified:
        args.use_stratified = False

    # 检查 PyTorch 可用性
    if not TORCH_AVAILABLE:
        print("[错误] PyTorch 未安装，无法运行 v4 验证。")
        print("       请执行: pip install torch")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   DML 理论验证 —— v4 改进交叉拟合策略                         ║")
    print("║   创新点: 折边随机化 + 分层检查 + 嵌套LR + 中位数聚合         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  模式:           {args.mode}")
    print(f"  实验次数:       {args.n_experiments}")
    print(f"  样本量:         {args.n_samples}")
    print(f"  图类型:         {args.graph_type}")
    print(f"  噪声类型:       {args.noise_type}")
    print(f"  噪声标准差:     {args.noise_scale}")
    print(f"  工业函数:       {args.use_industrial}")
    print(f"  设备:           {DEVICE}")
    print(f"  ── v4 交叉拟合参数 ──")
    print(f"  FOLD_JITTER_RATIO:    {args.fold_jitter_ratio}")
    print(f"  USE_STRATIFIED:       {args.use_stratified}")
    print(f"  NESTED_LR_SEARCH:     {args.nested_lr_search}")
    print(f"  MIN_TREAT_SAMPLES:    {MIN_TREAT_SAMPLES}")
    print(f"  NESTED_LR_CANDIDATES: {NESTED_LR_CANDIDATES}")
    print(f"  ── v3 基础超参 ──")
    print(f"  LATENT_DIM:       {LATENT_DIM}")
    print(f"  HIDDEN_DIM_ENC:   {HIDDEN_DIM_ENCODER}")
    print(f"  HIDDEN_DIM_HEAD:  {HIDDEN_DIM_HEAD}")
    print(f"  BETA_KL:          {BETA_KL}")
    print(f"  MAX_EPOCHS_VAE:   {MAX_EPOCHS_VAE}")
    print(f"  MAX_EPOCHS_HEAD:  {MAX_EPOCHS_HEAD}")
    print(f"  LR_VAE:           {LR_VAE}")
    print(f"  LR_HEAD:          {LR_HEAD}")

    t_start = time.perf_counter()

    if args.mode == "quick":
        run_quick(args)
    elif args.mode == "full":
        run_full(args)
    elif args.mode == "consistency":
        run_consistency(args)
    elif args.mode == "cf_compare":
        run_cf_compare(args)
    elif args.mode == "all":
        print("\n\n" + "▓" * 70)
        print("  第 1 步：完整蒙特卡洛验证")
        print("▓" * 70)
        run_full(args)

        print("\n\n" + "▓" * 70)
        print("  第 2 步：√n-一致性验证")
        print("▓" * 70)
        run_consistency(args)

        print("\n\n" + "▓" * 70)
        print("  第 3 步：交叉拟合策略对比（v4 核心实验）")
        print("▓" * 70)
        run_cf_compare(args)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'═' * 70}")
    print(f"  全部任务完成  总耗时: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")
    print(f"  结果输出目录: {OUT_DIR}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
