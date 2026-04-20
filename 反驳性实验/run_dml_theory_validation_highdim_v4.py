"""
run_dml_theory_validation_highdim_v4.py
========================================
DML 理论验证（高维合成数据）—— V4 改进交叉拟合策略

═══════════════════════════════════════════════════════════════════
  高维场景创新方案 V4

  在 V3（两阶段解耦 VAE-DML）基础上，新增四项交叉拟合改进：

  【改进1】折边随机化（Fold Boundary Jitter / Repeated Cross-Fitting）
    每次重复使用不同的折边界，减少对单次数据分区的敏感性。

  【改进2】分层交叉拟合（Stratified Cross-Fitting）
    检查每个训练折的高处理量样本数量，跳过极端不平衡的折。

  【改进3】嵌套学习率搜索（Nested Learning Rate Search）
    在训练数据上做 80/20 内层验证，选择最佳 LR。

  【改进4】正确中位数聚合 SE
    V_final = Median(V_b + (θ_b - θ_final)²)

  高维优势（在 V3 基础上增强）：
    - 折边随机化在高维场景下更重要（数据分割对高维更敏感）
    - 嵌套 LR 搜索能适应高维输入对学习率的不同需求

用法：
  python run_dml_theory_validation_highdim_v4.py --mode quick
  python run_dml_theory_validation_highdim_v4.py --mode full
  python run_dml_theory_validation_highdim_v4.py --mode consistency
  python run_dml_theory_validation_highdim_v4.py --mode cf_compare
  python run_dml_theory_validation_highdim_v4.py --mode all
═══════════════════════════════════════════════════════════════════
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

import dml_validation_common_highdim as dvch

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


# ═══════════════════════════════════════════════════════════════════
#  超参数
# ═══════════════════════════════════════════════════════════════════

# v3 基础（高维版）
LATENT_DIM = 48
HIDDEN_DIM_ENCODER = 128
HIDDEN_DIM_HEAD = 64
BETA_KL = 0.1
ANNEAL_EPOCHS = 20
MAX_EPOCHS_VAE = 80
MAX_EPOCHS_HEAD = 50
LR_VAE = 0.001
LR_HEAD = 0.003
GRAD_CLIP = 1.0

# v4 新增
FOLD_JITTER_RATIO = 0.10
MIN_TREAT_SAMPLES = 5
NESTED_LR_CANDIDATES = [0.001, 0.003, 0.01]
DEFAULT_LR = LR_HEAD
NESTED_INNER_RATIO = 0.2


# ═══════════════════════════════════════════════════════════════════
#  模型定义（与 v3 高维版相同）
# ═══════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class MLPEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=HIDDEN_DIM_ENCODER,
                     latent_dim=LATENT_DIM):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.SiLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        def forward(self, x):
            h = self.shared(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def encode_mean(self, x):
            mu, _ = self.forward(x)
            return mu

    class MLPDecoder(nn.Module):
        def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM_ENCODER,
                     output_dim=100):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, z):
            return self.net(z)

    class PredictionHead(nn.Module):
        def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM_HEAD):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, z):
            return self.net(z).squeeze(-1)

    class TwoStageVAEDML:
        """两阶段解耦 VAE-DML（与 v3 高维版一致）"""

        def __init__(self, input_dim, latent_dim=LATENT_DIM,
                     hidden_dim_enc=HIDDEN_DIM_ENCODER,
                     hidden_dim_head=HIDDEN_DIM_HEAD, device=None):
            self.device = device or DEVICE
            self.latent_dim = latent_dim
            self.encoder = MLPEncoder(input_dim, hidden_dim_enc, latent_dim).to(self.device)
            self.decoder = MLPDecoder(latent_dim, hidden_dim_enc, input_dim).to(self.device)
            self.head_Y = PredictionHead(latent_dim, hidden_dim_head).to(self.device)
            self.head_D = PredictionHead(latent_dim, hidden_dim_head).to(self.device)

        def train_stage1(self, X_ctrl, epochs=MAX_EPOCHS_VAE,
                         beta_kl=BETA_KL, anneal_epochs=ANNEAL_EPOCHS,
                         lr=LR_VAE, seed=42):
            torch.manual_seed(seed)
            self.encoder.train()
            self.decoder.train()
            X_tensor = torch.FloatTensor(X_ctrl).to(self.device)
            optimizer = optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=lr, weight_decay=1e-5,
            )
            n = X_tensor.shape[0]
            batch_size = min(512, n)
            for epoch in range(epochs):
                beta_t = beta_kl * min((epoch + 1) / anneal_epochs, 1.0)
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    x_batch = X_tensor[idx]
                    mu, logvar = self.encoder(x_batch)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    x_recon = self.decoder(z)
                    recon_loss = nn.functional.mse_loss(x_recon, x_batch)
                    kl_loss = -0.5 * torch.mean(
                        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    loss = recon_loss + beta_t * kl_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        GRAD_CLIP)
                    optimizer.step()
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        def train_stage2(self, X_ctrl, Y, D, epochs=MAX_EPOCHS_HEAD,
                         lr=LR_HEAD, seed=42):
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
            batch_size = min(512, n)
            for epoch in range(epochs):
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    z_b, y_b, d_b = z[idx], Y_tensor[idx], D_tensor[idx]
                    loss_Y = nn.functional.mse_loss(self.head_Y(z_b), y_b)
                    opt_Y.zero_grad(); loss_Y.backward()
                    nn.utils.clip_grad_norm_(self.head_Y.parameters(), GRAD_CLIP)
                    opt_Y.step()
                    loss_D = nn.functional.mse_loss(self.head_D(z_b), d_b)
                    opt_D.zero_grad(); loss_D.backward()
                    nn.utils.clip_grad_norm_(self.head_D.parameters(), GRAD_CLIP)
                    opt_D.step()
            self.head_Y.eval()
            self.head_D.eval()

        def predict_residuals(self, X_ctrl, Y, D):
            X_tensor = torch.FloatTensor(X_ctrl).to(self.device)
            with torch.no_grad():
                z = self.encoder.encode_mean(X_tensor)
                y_hat = self.head_Y(z).cpu().numpy()
                d_hat = self.head_D(z).cpu().numpy()
            return Y - y_hat, D - d_hat


# ═══════════════════════════════════════════════════════════════════
#  v4 交叉拟合改进工具函数
# ═══════════════════════════════════════════════════════════════════

def _generate_jittered_folds(n, n_folds, seed, jitter_ratio=FOLD_JITTER_RATIO):
    """带折边随机抖动的交叉拟合分割"""
    rng = np.random.RandomState(seed)
    block_size = n // n_folds
    jitter_max = int(block_size * jitter_ratio)
    boundaries = [0]
    for k in range(1, n_folds):
        base = k * block_size
        offset = rng.randint(-jitter_max, jitter_max + 1) if jitter_max > 0 else 0
        boundaries.append(max(1, min(n - 1, base + offset)))
    boundaries.append(n)
    boundaries = sorted(set(boundaries))
    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    if boundaries[-1] != n:
        boundaries.append(n)
    indices = rng.permutation(n)
    folds = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < 2:
            continue
        test_idx = indices[s:e]
        train_idx = np.concatenate([indices[:s], indices[e:]])
        folds.append((train_idx, test_idx))
    return folds


def _generate_standard_folds(n, n_folds, seed):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


def _nested_lr_search(X_train, Y_train, D_train, seed=42, input_dim=None):
    """嵌套学习率搜索"""
    if input_dim is None:
        input_dim = X_train.shape[1]
    n_train = len(Y_train)
    n_inner_val = max(10, int(n_train * NESTED_INNER_RATIO))
    n_inner_train = n_train - n_inner_val
    X_it, Y_it, D_it = X_train[:n_inner_train], Y_train[:n_inner_train], D_train[:n_inner_train]
    X_iv, Y_iv, D_iv = X_train[n_inner_train:], Y_train[n_inner_train:], D_train[n_inner_train:]
    best_lr, best_loss = NESTED_LR_CANDIDATES[0], float('inf')
    for lr_cand in NESTED_LR_CANDIDATES:
        try:
            model = TwoStageVAEDML(input_dim=input_dim)
            model.train_stage1(X_it, lr=LR_VAE, seed=seed, epochs=max(20, MAX_EPOCHS_VAE // 2))
            model.train_stage2(X_it, Y_it, D_it, lr=lr_cand, seed=seed,
                               epochs=max(15, MAX_EPOCHS_HEAD // 2))
            X_val_t = torch.FloatTensor(X_iv).to(DEVICE)
            with torch.no_grad():
                z_val = model.encoder.encode_mean(X_val_t)
                loss = float(nn.functional.mse_loss(model.head_Y(z_val),
                             torch.FloatTensor(Y_iv).to(DEVICE)).item() +
                             nn.functional.mse_loss(model.head_D(z_val),
                             torch.FloatTensor(D_iv).to(DEVICE)).item())
            if loss < best_loss:
                best_loss, best_lr = loss, lr_cand
        except Exception:
            continue
    return best_lr


# ═══════════════════════════════════════════════════════════════════
#  v4 DML 估计器（高维版）
# ═══════════════════════════════════════════════════════════════════

def v4_highdim_dml_estimate(Y, D, X_ctrl, seed=42, n_folds=5, n_repeats=5,
                            fold_jitter_ratio=FOLD_JITTER_RATIO,
                            use_stratified=True, nested_lr_search=False):
    """v4 改进交叉拟合 DML 估计器（高维版）"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 未安装")

    n = len(Y)
    input_dim = X_ctrl.shape[1]
    X_mean = X_ctrl.mean(axis=0)
    X_std = X_ctrl.std(axis=0) + 1e-8
    X_normed = (X_ctrl - X_mean) / X_std
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8
    D_mean, D_std = D.mean(), D.std() + 1e-8
    Y_normed = (Y - Y_mean) / Y_std
    D_normed = (D - D_mean) / D_std

    def _single_crossfit_v4(seed_b):
        torch.manual_seed(seed_b)
        np.random.seed(seed_b)
        res_Y_all = np.full(n, np.nan)
        res_D_all = np.full(n, np.nan)
        if fold_jitter_ratio > 0:
            folds = _generate_jittered_folds(n, n_folds, seed=seed_b,
                                             jitter_ratio=fold_jitter_ratio)
        else:
            folds = _generate_standard_folds(n, n_folds, seed=seed_b)
        d_median = np.median(D_normed)
        for train_idx, test_idx in folds:
            if use_stratified:
                n_high = np.sum(D_normed[train_idx] > d_median)
                if n_high < MIN_TREAT_SAMPLES:
                    continue
            X_train, X_test = X_normed[train_idx], X_normed[test_idx]
            Y_train, Y_test = Y_normed[train_idx], Y_normed[test_idx]
            D_train, D_test = D_normed[train_idx], D_normed[test_idx]
            if nested_lr_search:
                best_lr = _nested_lr_search(X_train, Y_train, D_train,
                                            seed=seed_b, input_dim=input_dim)
            else:
                best_lr = DEFAULT_LR
            model = TwoStageVAEDML(input_dim=input_dim)
            model.train_stage1(X_train, seed=seed_b)
            model.train_stage2(X_train, Y_train, D_train, lr=best_lr, seed=seed_b)
            res_Y_fold, res_D_fold = model.predict_residuals(X_test, Y_test, D_test)
            res_Y_all[test_idx] = res_Y_fold
            res_D_all[test_idx] = res_D_fold

        valid_mask = ~np.isnan(res_Y_all)
        if valid_mask.sum() < n // 2:
            return _fallback(seed_b)
        res_Y_orig = res_Y_all[valid_mask] * Y_std
        res_D_orig = res_D_all[valid_mask] * D_std
        n_valid = valid_mask.sum()
        denom = np.sum(res_D_orig ** 2) + 1e-12
        theta_k = np.sum(res_D_orig * res_Y_orig) / denom
        psi = res_D_orig * (res_Y_orig - theta_k * res_D_orig)
        J = np.mean(res_D_orig ** 2)
        var_neyman = float(np.mean(psi ** 2) / (n_valid * J ** 2 + 1e-12))
        return theta_k, var_neyman

    def _fallback(seed_b):
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
            res_Y_f, res_D_f = model.predict_residuals(
                X_normed[test_idx], Y_normed[test_idx], D_normed[test_idx])
            res_Y_all[test_idx] = res_Y_f
            res_D_all[test_idx] = res_D_f
        res_Y_orig = res_Y_all * Y_std
        res_D_orig = res_D_all * D_std
        denom = np.sum(res_D_orig ** 2) + 1e-12
        theta_k = np.sum(res_D_orig * res_Y_orig) / denom
        psi = res_D_orig * (res_Y_orig - theta_k * res_D_orig)
        J = np.mean(res_D_orig ** 2)
        return theta_k, float(np.mean(psi ** 2) / (n * J ** 2 + 1e-12))

    theta_boots, var_boots = [], []
    for b in range(n_repeats):
        boot_seed = seed * 1000 + b * 7
        theta_b, var_b = _single_crossfit_v4(boot_seed)
        theta_boots.append(theta_b)
        var_boots.append(var_b)

    theta_boots = np.array(theta_boots)
    var_boots = np.array(var_boots)
    theta_final = float(np.median(theta_boots))
    var_final = float(np.median(var_boots + (theta_boots - theta_final) ** 2))
    se_final = np.sqrt(max(var_final, 1e-12))
    return float(theta_final), se_final, float(theta_final - 1.96 * se_final), float(theta_final + 1.96 * se_final)


# ═══════════════════════════════════════════════════════════════════
#  策略包装函数（用于 cf_compare）
# ═══════════════════════════════════════════════════════════════════

def _strategy_a(Y, D, X_ctrl, seed):
    return v4_highdim_dml_estimate(Y, D, X_ctrl, seed=seed,
                                   fold_jitter_ratio=0.0, use_stratified=False, nested_lr_search=False)

def _strategy_b(Y, D, X_ctrl, seed):
    return v4_highdim_dml_estimate(Y, D, X_ctrl, seed=seed,
                                   fold_jitter_ratio=0.0, use_stratified=True, nested_lr_search=False)

def _strategy_c(Y, D, X_ctrl, seed):
    return v4_highdim_dml_estimate(Y, D, X_ctrl, seed=seed,
                                   fold_jitter_ratio=FOLD_JITTER_RATIO, use_stratified=False, nested_lr_search=False)

def _strategy_e(Y, D, X_ctrl, seed):
    return v4_highdim_dml_estimate(Y, D, X_ctrl, seed=seed,
                                   fold_jitter_ratio=FOLD_JITTER_RATIO, use_stratified=True, nested_lr_search=False)


# ═══════════════════════════════════════════════════════════════════
#  实验模式
# ═══════════════════════════════════════════════════════════════════

def run_quick(args):
    print("\n" + "█" * 70)
    print("  [快速模式] 高维 V4: 改进交叉拟合 VAE-DML 验证")
    print("█" * 70)
    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42)
    ate_true = dvch.compute_ate_for_dag_highdim(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")
    n_exp = min(args.n_experiments, 30)
    def est_fn(Y, D, X, s):
        return v4_highdim_dml_estimate(Y, D, X, seed=s,
                                       fold_jitter_ratio=args.fold_jitter_ratio,
                                       use_stratified=args.use_stratified,
                                       nested_lr_search=args.nested_lr_search)
    df, summary = dvch.run_monte_carlo_highdim(
        estimator_fn=est_fn, dag_info=dag_info, ate_true=ate_true,
        n_experiments=n_exp, n_samples=args.n_samples,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        tag="highdim_v4_quick", method_name="VAE-DML-v4 (高维, Quick)")
    if isinstance(df, pd.DataFrame) and not df.empty:
        dvch.save_results(df, summary, "highdim_v4_vae_dml_quick")
    return df, summary


def run_full(args):
    print("\n" + "█" * 70)
    print("  [完整模式] 高维 V4: 改进交叉拟合 VAE-DML 完整蒙特卡洛验证")
    print("█" * 70)
    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42)
    ate_true = dvch.compute_ate_for_dag_highdim(dag_info, args.noise_scale, args.noise_type)
    def est_fn(Y, D, X, s):
        return v4_highdim_dml_estimate(Y, D, X, seed=s,
                                       fold_jitter_ratio=args.fold_jitter_ratio,
                                       use_stratified=args.use_stratified,
                                       nested_lr_search=args.nested_lr_search)
    df, summary = dvch.run_monte_carlo_highdim(
        estimator_fn=est_fn, dag_info=dag_info, ate_true=ate_true,
        n_experiments=args.n_experiments, n_samples=args.n_samples,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        tag="highdim_v4_full", method_name="VAE-DML-v4 (高维)")
    if isinstance(df, pd.DataFrame) and not df.empty:
        dvch.save_results(df, summary, "highdim_v4_vae_dml_full")
    return df, summary


def run_consistency(args):
    print("\n" + "█" * 70)
    print("  [一致性模式] 高维 V4: VAE-DML √n-一致性验证")
    print("█" * 70)
    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42)
    ate_true = dvch.compute_ate_for_dag_highdim(dag_info, args.noise_scale, args.noise_type)
    def est_fn(Y, D, X, s):
        return v4_highdim_dml_estimate(Y, D, X, seed=s,
                                       fold_jitter_ratio=args.fold_jitter_ratio,
                                       use_stratified=args.use_stratified,
                                       nested_lr_search=args.nested_lr_search)
    df_cons = dvch.run_consistency_validation_highdim(
        estimator_fn=est_fn, dag_info=dag_info, ate_true=ate_true,
        sample_sizes=[500, 1000, 2000, 4000, 8000], n_experiments_per_size=50,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        method_name="VAE-DML-v4 (高维)")
    if not df_cons.empty:
        csv_path = os.path.join(OUT_DIR, "highdim_v4_vae_dml_consistency.csv")
        df_cons.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_cons


def run_cf_compare(args):
    """交叉拟合策略对比实验（v4 核心实验）"""
    print("\n" + "█" * 70)
    print("  [策略对比模式] 高维 V4 交叉拟合策略对比")
    print("█" * 70)
    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42)
    ate_true = dvch.compute_ate_for_dag_highdim(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    from synthetic_dag_generator_highdim import HighDimSyntheticDAGGenerator
    strategies = {
        "A_标准固定折边": _strategy_a,
        "B_分层折检查": _strategy_b,
        "C_折边随机化": _strategy_c,
        "E_全部改进": _strategy_e,
    }

    n_exp_cf = min(args.n_experiments, 50)
    all_results = []
    for strategy_name, estimator_fn in strategies.items():
        print(f"\n{'─' * 60}\n  策略: {strategy_name}\n{'─' * 60}")
        t_start = time.perf_counter()
        biases, coverages, ses, n_success = [], [], [], 0
        for exp_i in range(n_exp_cf):
            try:
                data_seed = exp_i * 13 + 1000
                D, Y, X_ctrl_hd, _ = dvch.generate_highdim_data(
                    dag_info, n_samples=args.n_samples,
                    noise_scale=args.noise_scale, noise_type=args.noise_type,
                    data_seed=data_seed)
                theta_hat, se, ci_lo, ci_hi = estimator_fn(Y, D, X_ctrl_hd, data_seed)
                biases.append(theta_hat - ate_true)
                ses.append(se)
                coverages.append(bool(ci_lo <= ate_true <= ci_hi))
                n_success += 1
            except Exception as e:
                if n_success == 0 and exp_i < 3:
                    print(f"    [实验 {exp_i}] 失败: {e}")
        elapsed = time.perf_counter() - t_start
        if n_success > 0:
            ba = np.array(biases)
            all_results.append({
                "strategy": strategy_name, "n_experiments": n_success,
                "mean_bias": round(float(np.mean(ba)), 6),
                "rmse": round(float(np.sqrt(np.mean(ba ** 2))), 6),
                "coverage_95": round(float(np.mean(coverages)), 4),
                "mean_se": round(float(np.mean(ses)), 6),
                "time_per_exp_s": round(elapsed / n_success, 2),
            })
            print(f"    成功: {n_success}/{n_exp_cf}  RMSE={all_results[-1]['rmse']:.6f}")

    if all_results:
        df_compare = pd.DataFrame(all_results)
        print("\n" + "═" * 70)
        print(df_compare.to_string(index=False))
        csv_path = os.path.join(OUT_DIR, "highdim_v4_cf_compare.csv")
        df_compare.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return df_compare
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DML 理论验证（高维）—— V4 改进交叉拟合")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full", "consistency", "cf_compare", "all"])
    parser.add_argument("--n_experiments", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_nodes", type=int, default=60)
    parser.add_argument("--graph_type", type=str, default="layered",
                        choices=["layered", "er", "scale_free"])
    parser.add_argument("--noise_type", type=str, default="heteroscedastic",
                        choices=["gaussian", "heteroscedastic", "heavy_tail"])
    parser.add_argument("--noise_scale", type=float, default=0.3)
    parser.add_argument("--use_industrial", action="store_true", default=True)
    parser.add_argument("--fold_jitter_ratio", type=float, default=FOLD_JITTER_RATIO)
    parser.add_argument("--use_stratified", action="store_true", default=True)
    parser.add_argument("--no_stratified", action="store_true")
    parser.add_argument("--nested_lr_search", action="store_true", default=False)

    args = parser.parse_args()
    if args.no_stratified:
        args.use_stratified = False

    if not TORCH_AVAILABLE:
        print("[错误] PyTorch 未安装"); sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   DML 理论验证（高维）—— V4 改进交叉拟合                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  模式: {args.mode}  节点: {args.n_nodes}  设备: {DEVICE}")

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
        run_full(args)
        run_consistency(args)
        run_cf_compare(args)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═' * 70}\n  完成  耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)\n{'═' * 70}")


if __name__ == "__main__":
    main()
