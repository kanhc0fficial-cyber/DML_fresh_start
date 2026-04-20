"""
run_dml_theory_validation_highdim_v5.py
========================================
DML 理论验证（高维合成数据）—— V5 双流 VAE-DML + 四项微创新

═══════════════════════════════════════════════════════════════════
  高维场景完整创新方案 V5

  在 V3/V4 基础上集成四项微创新（对应真实管线 run_refutation_xin2_v5.py）：

  微创新 A：因果优先不对称梯度投影
    共享参数上因果损失和重建损失梯度冲突时，
    将重建梯度投影到因果梯度的正交补空间。

  微创新 B：双流潜变量结构
    z_causal（低维 16D，确定性投影）→ 因果预测头
    z_recon（高维 48D，随机投影）→ 解码器重建
    正交性约束：||W_causal @ W_recon^T||_F²

  微创新 C：课程式三阶段训练
    Phase 1（预热 20%）：只训练 VAE
    Phase 2（过渡 30%）：α 线性增加引入因果
    Phase 3（精调 50%）：因果为主，重建降权

  微创新 D：不确定性加权 DML 残差
    w_i ∝ 1/σ_i（重建流提供的不确定性）
    θ = Σ(w·res_D·res_Y) / Σ(w·res_D²)

  高维场景优势叠加：
    - B（双流）：z_causal 自动发现真实混杂子空间，z_recon 处理冗余
    - D（加权）：异方差噪声下降低高噪声样本的权重
    - A（投影）：高维输入下重建与因果更易冲突，投影保护因果方向

用法：
  python run_dml_theory_validation_highdim_v5.py --mode quick
  python run_dml_theory_validation_highdim_v5.py --mode full
  python run_dml_theory_validation_highdim_v5.py --mode consistency
  python run_dml_theory_validation_highdim_v5.py --mode ablation
  python run_dml_theory_validation_highdim_v5.py --mode all
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import dml_validation_common_highdim as dvch

OUT_DIR = os.path.join(BASE_DIR, "DML理论验证")
os.makedirs(OUT_DIR, exist_ok=True)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = None
    print("[警告] PyTorch 未安装，v5 估计器将不可用。")


# ═══════════════════════════════════════════════════════════════════
#  超参数（高维版：更大的网络容量）
# ═══════════════════════════════════════════════════════════════════

# 基础
HIDDEN_DIM_ENCODER = 128
HIDDEN_DIM_HEAD = 64
BETA_KL = 0.1
GRAD_CLIP = 1.0
LR_JOINT = 0.001

# v4 继承
FOLD_JITTER_RATIO = 0.10
MIN_TREAT_SAMPLES = 5

# v5 微创新
LATENT_DIM_CAUSAL = 24          # z_causal（高维需稍大）
LATENT_DIM_RECON = 64           # z_recon
LAMBDA_ORTH = 0.01
MAX_EPOCHS_JOINT = 100          # 高维需要更多轮次
PHASE1_RATIO = 0.20
PHASE2_RATIO = 0.30
LAMBDA_RECON_FINAL = 0.3
MC_SAMPLES = 5
UNCERTAINTY_CLIP_QUANTILE = 0.90


# ═══════════════════════════════════════════════════════════════════
#  模型定义
# ═══════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class DualStreamEncoder(nn.Module):
        """双流 MLP 编码器（微创新 B，高维版）"""

        def __init__(self, input_dim, hidden_dim=HIDDEN_DIM_ENCODER,
                     latent_dim_causal=LATENT_DIM_CAUSAL,
                     latent_dim_recon=LATENT_DIM_RECON):
            super().__init__()
            self.latent_dim_causal = latent_dim_causal
            self.latent_dim_recon = latent_dim_recon
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
            self.proj_causal = nn.Linear(hidden_dim // 2, latent_dim_causal)
            self.fc_mu_recon = nn.Linear(hidden_dim // 2, latent_dim_recon)
            self.fc_logvar_recon = nn.Linear(hidden_dim // 2, latent_dim_recon)

        def forward(self, x):
            h = self.shared(x)
            z_causal = self.proj_causal(h)
            mu_recon = self.fc_mu_recon(h)
            logvar_recon = self.fc_logvar_recon(h)
            return z_causal, mu_recon, logvar_recon

        def encode_causal(self, x):
            h = self.shared(x)
            return self.proj_causal(h)

        def get_uncertainty(self, x):
            h = self.shared(x)
            logvar = self.fc_logvar_recon(h)
            sigma = torch.exp(0.5 * logvar)
            return sigma.mean(dim=1)

        def orthogonality_loss(self):
            W_c = self.proj_causal.weight
            W_r = self.fc_mu_recon.weight
            cross = W_c @ W_r.T
            return (cross ** 2).sum()

    class MLPDecoder(nn.Module):
        def __init__(self, latent_dim=LATENT_DIM_RECON,
                     hidden_dim=HIDDEN_DIM_ENCODER, output_dim=100):
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
        def __init__(self, latent_dim=LATENT_DIM_CAUSAL,
                     hidden_dim=HIDDEN_DIM_HEAD):
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

    class DualStreamVAEDML:
        """完整 v5 模型：双流 + 课程 + 梯度投影 + 不确定性加权（高维版）"""

        def __init__(self, input_dim, latent_dim_causal=LATENT_DIM_CAUSAL,
                     latent_dim_recon=LATENT_DIM_RECON,
                     hidden_dim_enc=HIDDEN_DIM_ENCODER,
                     hidden_dim_head=HIDDEN_DIM_HEAD, device=None):
            self.device = device or DEVICE
            self.input_dim = input_dim
            self.encoder = DualStreamEncoder(
                input_dim, hidden_dim_enc, latent_dim_causal, latent_dim_recon
            ).to(self.device)
            self.decoder = MLPDecoder(
                latent_dim_recon, hidden_dim_enc, input_dim
            ).to(self.device)
            self.head_Y = PredictionHead(latent_dim_causal, hidden_dim_head).to(self.device)
            self.head_D = PredictionHead(latent_dim_causal, hidden_dim_head).to(self.device)

        def _reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def train_curriculum(self, X, Y, D,
                             total_epochs=MAX_EPOCHS_JOINT,
                             beta_kl=BETA_KL,
                             lambda_orth=LAMBDA_ORTH,
                             lambda_recon_final=LAMBDA_RECON_FINAL,
                             lr=LR_JOINT,
                             use_dual_stream=True,
                             use_curriculum=True,
                             use_grad_proj=True,
                             seed=42):
            """课程式三阶段联合训练（微创新 C），集成梯度投影（微创新 A）"""
            torch.manual_seed(seed)
            X_tensor = torch.FloatTensor(X).to(self.device)
            Y_tensor = torch.FloatTensor(Y).to(self.device)
            D_tensor = torch.FloatTensor(D).to(self.device)
            n = X_tensor.shape[0]
            batch_size = min(512, n)

            all_params = (list(self.encoder.parameters())
                          + list(self.decoder.parameters())
                          + list(self.head_Y.parameters())
                          + list(self.head_D.parameters()))
            optimizer = optim.Adam(all_params, lr=lr, weight_decay=1e-5)

            phase1_end = int(total_epochs * PHASE1_RATIO)
            phase2_end = int(total_epochs * (PHASE1_RATIO + PHASE2_RATIO))

            self.encoder.train()
            self.decoder.train()
            self.head_Y.train()
            self.head_D.train()

            shared_param_names = []
            shared_params_dict = {}
            for name, param in self.encoder.named_parameters():
                if 'shared' in name:
                    shared_param_names.append(name)
                    shared_params_dict[name] = param

            for epoch in range(total_epochs):
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    x_b, y_b, d_b = X_tensor[idx], Y_tensor[idx], D_tensor[idx]

                    z_causal, mu_recon, logvar_recon = self.encoder(x_b)
                    z_recon = self._reparameterize(mu_recon, logvar_recon)
                    x_hat = self.decoder(z_recon)

                    recon_loss = nn.functional.mse_loss(x_hat, x_b)
                    kl_loss = -0.5 * torch.mean(
                        torch.sum(1 + logvar_recon - mu_recon.pow(2) - logvar_recon.exp(), dim=1))
                    loss_Y = nn.functional.mse_loss(self.head_Y(z_causal), y_b)
                    loss_D = nn.functional.mse_loss(self.head_D(z_causal), d_b)
                    loss_causal = loss_Y + loss_D
                    orth_loss = self.encoder.orthogonality_loss() if use_dual_stream else torch.tensor(0.0, device=self.device)

                    if not use_curriculum:
                        loss = loss_causal + recon_loss + beta_kl * kl_loss + lambda_orth * orth_loss
                        optimizer.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(all_params, GRAD_CLIP); optimizer.step()
                    elif epoch < phase1_end:
                        loss_vae = recon_loss + beta_kl * kl_loss + lambda_orth * orth_loss
                        optimizer.zero_grad(); loss_vae.backward()
                        for p in list(self.head_Y.parameters()) + list(self.head_D.parameters()):
                            if p.grad is not None: p.grad.zero_()
                        nn.utils.clip_grad_norm_(all_params, GRAD_CLIP); optimizer.step()
                    elif epoch < phase2_end:
                        alpha = (epoch - phase1_end) / max(1, phase2_end - phase1_end)
                        loss_recon_full = recon_loss + beta_kl * kl_loss + lambda_orth * orth_loss
                        if use_grad_proj and shared_param_names:
                            optimizer.zero_grad()
                            loss_causal.backward(retain_graph=True)
                            g_causal = {n: shared_params_dict[n].grad.clone() if shared_params_dict[n].grad is not None
                                        else torch.zeros_like(shared_params_dict[n]) for n in shared_param_names}
                            # 保存因果头和投影层的梯度（它们不参与重建损失的计算图）
                            causal_only_params = (list(self.head_Y.parameters())
                                                  + list(self.head_D.parameters())
                                                  + list(self.encoder.proj_causal.parameters()))
                            g_causal_only = [p.grad.clone() if p.grad is not None
                                             else torch.zeros_like(p) for p in causal_only_params]
                            optimizer.zero_grad()
                            loss_recon_full.backward()
                            for name in shared_param_names:
                                p = shared_params_dict[name]
                                g_r = p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                g_c = g_causal[name]
                                dot = (g_c * g_r).sum()
                                if dot < 0:
                                    g_r = g_r - (dot / ((g_c ** 2).sum() + 1e-12)) * g_c
                                p.grad = g_c + alpha * g_r
                            # 恢复因果头和投影层的梯度
                            for p, g in zip(causal_only_params, g_causal_only):
                                if p.grad is not None:
                                    p.grad = p.grad + g
                                else:
                                    p.grad = g
                            nn.utils.clip_grad_norm_(all_params, GRAD_CLIP); optimizer.step()
                        else:
                            loss = loss_causal + alpha * loss_recon_full
                            optimizer.zero_grad(); loss.backward()
                            nn.utils.clip_grad_norm_(all_params, GRAD_CLIP); optimizer.step()
                    else:
                        loss = loss_causal + lambda_recon_final * (recon_loss + beta_kl * kl_loss) + lambda_orth * orth_loss
                        optimizer.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(all_params, GRAD_CLIP); optimizer.step()

            self.encoder.eval(); self.decoder.eval()
            self.head_Y.eval(); self.head_D.eval()

        def predict_residuals(self, X, Y, D):
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                z_c = self.encoder.encode_causal(X_tensor)
                y_hat = self.head_Y(z_c).cpu().numpy()
                d_hat = self.head_D(z_c).cpu().numpy()
            return Y - y_hat, D - d_hat

        def predict_residuals_weighted(self, X, Y, D):
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                z_c = self.encoder.encode_causal(X_tensor)
                sigma = self.encoder.get_uncertainty(X_tensor)
                y_hat = self.head_Y(z_c).cpu().numpy()
                d_hat = self.head_D(z_c).cpu().numpy()
                sigma_np = sigma.cpu().numpy()
            res_Y, res_D = Y - y_hat, D - d_hat
            weights = 1.0 / (sigma_np + 1e-6)
            clip_val = np.quantile(weights, UNCERTAINTY_CLIP_QUANTILE)
            weights = np.clip(weights, a_min=None, a_max=clip_val)
            weights = weights / (weights.mean() + 1e-12)
            return res_Y, res_D, weights


# ═══════════════════════════════════════════════════════════════════
#  交叉拟合工具函数
# ═══════════════════════════════════════════════════════════════════

def _generate_jittered_folds(n, n_folds, seed, jitter_ratio=FOLD_JITTER_RATIO):
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
    if boundaries[0] != 0: boundaries = [0] + boundaries
    if boundaries[-1] != n: boundaries.append(n)
    indices = rng.permutation(n)
    folds = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < 2: continue
        test_idx = indices[s:e]
        train_idx = np.concatenate([indices[:s], indices[e:]])
        folds.append((train_idx, test_idx))
    return folds


def _generate_standard_folds(n, n_folds, seed):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


# ═══════════════════════════════════════════════════════════════════
#  v5 DML 估计器（高维版）
# ═══════════════════════════════════════════════════════════════════

def v5_highdim_dml_estimate(Y, D, X_ctrl, seed=42, n_folds=5, n_repeats=5,
                            use_dual_stream=True, use_curriculum=True,
                            use_grad_proj=True, use_uncertainty_weight=True,
                            fold_jitter_ratio=FOLD_JITTER_RATIO):
    """完整 v5 DML 估计器（高维版），含全部四项微创新"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 未安装")

    n = len(Y)
    input_dim = X_ctrl.shape[1]
    X_mean = X_ctrl.mean(axis=0); X_std = X_ctrl.std(axis=0) + 1e-8
    X_normed = (X_ctrl - X_mean) / X_std
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8
    D_mean, D_std = D.mean(), D.std() + 1e-8
    Y_normed = (Y - Y_mean) / Y_std
    D_normed = (D - D_mean) / D_std

    def _single_crossfit_v5(seed_b):
        torch.manual_seed(seed_b); np.random.seed(seed_b)
        res_Y_all = np.full(n, np.nan)
        res_D_all = np.full(n, np.nan)
        weights_all = np.full(n, np.nan)
        if fold_jitter_ratio > 0:
            folds = _generate_jittered_folds(n, n_folds, seed=seed_b, jitter_ratio=fold_jitter_ratio)
        else:
            folds = _generate_standard_folds(n, n_folds, seed=seed_b)
        d_median = np.median(D_normed)
        for train_idx, test_idx in folds:
            n_high = np.sum(D_normed[train_idx] > d_median)
            if n_high < MIN_TREAT_SAMPLES: continue
            X_tr, X_te = X_normed[train_idx], X_normed[test_idx]
            Y_tr, Y_te = Y_normed[train_idx], Y_normed[test_idx]
            D_tr, D_te = D_normed[train_idx], D_normed[test_idx]
            model = DualStreamVAEDML(input_dim=input_dim)
            model.train_curriculum(X_tr, Y_tr, D_tr,
                                   use_dual_stream=use_dual_stream,
                                   use_curriculum=use_curriculum,
                                   use_grad_proj=use_grad_proj, seed=seed_b)
            if use_uncertainty_weight:
                res_Y_f, res_D_f, w_f = model.predict_residuals_weighted(X_te, Y_te, D_te)
                weights_all[test_idx] = w_f
            else:
                res_Y_f, res_D_f = model.predict_residuals(X_te, Y_te, D_te)
                weights_all[test_idx] = 1.0
            res_Y_all[test_idx] = res_Y_f
            res_D_all[test_idx] = res_D_f

        valid = ~np.isnan(res_Y_all)
        if valid.sum() < n // 2:
            return _fallback_v5(seed_b)
        rY = res_Y_all[valid] * Y_std; rD = res_D_all[valid] * D_std
        w = weights_all[valid]; n_v = valid.sum()
        denom = np.sum(w * rD ** 2) + 1e-12
        theta_k = np.sum(w * rD * rY) / denom
        psi = w * rD * (rY - theta_k * rD)
        J = np.mean(w * rD ** 2)
        return theta_k, float(np.mean(psi ** 2) / (n_v * J ** 2 + 1e-12))

    def _fallback_v5(seed_b):
        torch.manual_seed(seed_b); np.random.seed(seed_b)
        res_Y_all = np.zeros(n); res_D_all = np.zeros(n); w_all = np.ones(n)
        folds = _generate_standard_folds(n, n_folds, seed=seed_b)
        for train_idx, test_idx in folds:
            model = DualStreamVAEDML(input_dim=input_dim)
            model.train_curriculum(X_normed[train_idx], Y_normed[train_idx], D_normed[train_idx],
                                   use_dual_stream=use_dual_stream, use_curriculum=False,
                                   use_grad_proj=False, seed=seed_b)
            if use_uncertainty_weight:
                rYf, rDf, wf = model.predict_residuals_weighted(
                    X_normed[test_idx], Y_normed[test_idx], D_normed[test_idx])
                w_all[test_idx] = wf
            else:
                rYf, rDf = model.predict_residuals(
                    X_normed[test_idx], Y_normed[test_idx], D_normed[test_idx])
            res_Y_all[test_idx] = rYf; res_D_all[test_idx] = rDf
        rY = res_Y_all * Y_std; rD = res_D_all * D_std; w = w_all
        denom = np.sum(w * rD ** 2) + 1e-12
        theta_k = np.sum(w * rD * rY) / denom
        psi = w * rD * (rY - theta_k * rD)
        J = np.mean(w * rD ** 2)
        return theta_k, float(np.mean(psi ** 2) / (n * J ** 2 + 1e-12))

    theta_boots, var_boots = [], []
    for b in range(n_repeats):
        boot_seed = seed * 1000 + b * 7
        theta_b, var_b = _single_crossfit_v5(boot_seed)
        theta_boots.append(theta_b); var_boots.append(var_b)

    theta_boots = np.array(theta_boots); var_boots = np.array(var_boots)
    theta_final = float(np.median(theta_boots))
    var_final = float(np.median(var_boots + (theta_boots - theta_final) ** 2))
    se_final = np.sqrt(max(var_final, 1e-12))
    return float(theta_final), se_final, float(theta_final - 1.96 * se_final), float(theta_final + 1.96 * se_final)


# ═══════════════════════════════════════════════════════════════════
#  实验模式
# ═══════════════════════════════════════════════════════════════════

def run_quick(args):
    print("\n" + "█" * 70)
    print("  [快速模式] 高维 V5: 双流 VAE-DML + 四项微创新")
    print("█" * 70)
    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42)
    ate_true = dvch.compute_ate_for_dag_highdim(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")
    n_exp = min(args.n_experiments, 30)
    def est_fn(Y, D, X, s):
        return v5_highdim_dml_estimate(Y, D, X, seed=s,
            use_dual_stream=args.use_dual_stream, use_curriculum=args.use_curriculum,
            use_grad_proj=args.use_grad_proj, use_uncertainty_weight=args.use_uncertainty_weight,
            fold_jitter_ratio=args.fold_jitter_ratio)
    df, summary = dvch.run_monte_carlo_highdim(
        estimator_fn=est_fn, dag_info=dag_info, ate_true=ate_true,
        n_experiments=n_exp, n_samples=args.n_samples,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        tag="highdim_v5_quick", method_name="VAE-DML-v5 (高维, Quick)")
    if isinstance(df, pd.DataFrame) and not df.empty:
        dvch.save_results(df, summary, "highdim_v5_vae_dml_quick")
    return df, summary


def run_full(args):
    print("\n" + "█" * 70)
    print("  [完整模式] 高维 V5: 双流 VAE-DML 完整蒙特卡洛验证")
    print("█" * 70)
    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42)
    ate_true = dvch.compute_ate_for_dag_highdim(dag_info, args.noise_scale, args.noise_type)
    def est_fn(Y, D, X, s):
        return v5_highdim_dml_estimate(Y, D, X, seed=s,
            use_dual_stream=args.use_dual_stream, use_curriculum=args.use_curriculum,
            use_grad_proj=args.use_grad_proj, use_uncertainty_weight=args.use_uncertainty_weight,
            fold_jitter_ratio=args.fold_jitter_ratio)
    df, summary = dvch.run_monte_carlo_highdim(
        estimator_fn=est_fn, dag_info=dag_info, ate_true=ate_true,
        n_experiments=args.n_experiments, n_samples=args.n_samples,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        tag="highdim_v5_full", method_name="VAE-DML-v5 (高维)")
    if isinstance(df, pd.DataFrame) and not df.empty:
        dvch.save_results(df, summary, "highdim_v5_vae_dml_full")
    return df, summary


def run_consistency(args):
    print("\n" + "█" * 70)
    print("  [一致性模式] 高维 V5: VAE-DML √n-一致性验证")
    print("█" * 70)
    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42)
    ate_true = dvch.compute_ate_for_dag_highdim(dag_info, args.noise_scale, args.noise_type)
    def est_fn(Y, D, X, s):
        return v5_highdim_dml_estimate(Y, D, X, seed=s,
            use_dual_stream=args.use_dual_stream, use_curriculum=args.use_curriculum,
            use_grad_proj=args.use_grad_proj, use_uncertainty_weight=args.use_uncertainty_weight,
            fold_jitter_ratio=args.fold_jitter_ratio)
    df_cons = dvch.run_consistency_validation_highdim(
        estimator_fn=est_fn, dag_info=dag_info, ate_true=ate_true,
        sample_sizes=[500, 1000, 2000, 4000, 8000], n_experiments_per_size=50,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        method_name="VAE-DML-v5 (高维)")
    if not df_cons.empty:
        csv_path = os.path.join(OUT_DIR, "highdim_v5_vae_dml_consistency.csv")
        df_cons.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_cons


def run_ablation(args):
    """消融实验：逐项累加微创新"""
    print("\n" + "█" * 70)
    print("  [消融模式] 高维 V5 微创新消融实验")
    print("█" * 70)
    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42)
    ate_true = dvch.compute_ate_for_dag_highdim(dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    ablation_groups = [
        {"name": "G1_baseline(无微创新)", "use_dual_stream": False,
         "use_curriculum": False, "use_grad_proj": False, "use_uncertainty_weight": False},
        {"name": "G2_+B(双流潜变量)", "use_dual_stream": True,
         "use_curriculum": False, "use_grad_proj": False, "use_uncertainty_weight": False},
        {"name": "G3_+B+C(双流+课程)", "use_dual_stream": True,
         "use_curriculum": True, "use_grad_proj": False, "use_uncertainty_weight": False},
        {"name": "G4_+B+C+A(+梯度投影)", "use_dual_stream": True,
         "use_curriculum": True, "use_grad_proj": True, "use_uncertainty_weight": False},
        {"name": "G5_+B+C+A+D(完整v5)", "use_dual_stream": True,
         "use_curriculum": True, "use_grad_proj": True, "use_uncertainty_weight": True},
    ]

    n_exp_ablation = min(args.n_experiments, 50)
    all_results = []

    for group in ablation_groups:
        group_name = group["name"]
        print(f"\n{'─' * 60}\n  消融组: {group_name}\n{'─' * 60}")
        t_start = time.perf_counter()
        biases, coverages, ses, n_success = [], [], [], 0
        for exp_i in range(n_exp_ablation):
            try:
                data_seed = exp_i * 13 + 1000
                D, Y, X_ctrl_hd, _ = dvch.generate_highdim_data(
                    dag_info, n_samples=args.n_samples,
                    noise_scale=args.noise_scale, noise_type=args.noise_type,
                    data_seed=data_seed)
                theta_hat, se, ci_lo, ci_hi = v5_highdim_dml_estimate(
                    Y, D, X_ctrl_hd, seed=data_seed,
                    use_dual_stream=group["use_dual_stream"],
                    use_curriculum=group["use_curriculum"],
                    use_grad_proj=group["use_grad_proj"],
                    use_uncertainty_weight=group["use_uncertainty_weight"],
                    fold_jitter_ratio=args.fold_jitter_ratio)
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
            result = {
                "group": group_name, "n_experiments": n_success,
                "mean_bias": round(float(np.mean(ba)), 6),
                "rmse": round(float(np.sqrt(np.mean(ba ** 2))), 6),
                "coverage_95": round(float(np.mean(coverages)), 4),
                "mean_se": round(float(np.mean(ses)), 6),
                "time_per_exp_s": round(elapsed / n_success, 2),
            }
            all_results.append(result)
            print(f"    成功: {n_success}/{n_exp_ablation}  RMSE={result['rmse']:.6f}  Coverage={result['coverage_95']:.1%}")

    if all_results:
        df_ablation = pd.DataFrame(all_results)
        print("\n" + "═" * 70)
        print("  高维 V5 微创新消融实验结果")
        print("═" * 70)
        print(df_ablation.to_string(index=False))
        if len(all_results) >= 2:
            print("\n  微创新边际贡献（RMSE 变化）：")
            names = ["B(双流)", "C(课程)", "A(梯度投影)", "D(不确定性加权)"]
            for i in range(1, len(all_results)):
                dr = all_results[i]["rmse"] - all_results[i - 1]["rmse"]
                dc = all_results[i]["coverage_95"] - all_results[i - 1]["coverage_95"]
                nm = names[i - 1] if i - 1 < len(names) else "?"
                print(f"    +{nm}: ΔRMSE={dr:+.6f}  ΔCoverage={dc:+.4f}")
        csv_path = os.path.join(OUT_DIR, "highdim_v5_ablation.csv")
        df_ablation.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  消融结果已保存：{csv_path}")
        return df_ablation
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DML 理论验证（高维）—— V5 双流 VAE-DML + 四项微创新")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full", "consistency", "ablation", "all"])
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
    parser.add_argument("--no_dual_stream", action="store_true")
    parser.add_argument("--no_curriculum", action="store_true")
    parser.add_argument("--no_grad_proj", action="store_true")
    parser.add_argument("--no_uncertainty_weight", action="store_true")

    args = parser.parse_args()
    args.use_dual_stream = not args.no_dual_stream
    args.use_curriculum = not args.no_curriculum
    args.use_grad_proj = not args.no_grad_proj
    args.use_uncertainty_weight = not args.no_uncertainty_weight

    if not TORCH_AVAILABLE:
        print("[错误] PyTorch 未安装"); sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   DML 理论验证（高维）—— V5 双流 VAE-DML + 四项微创新        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  模式: {args.mode}  节点: {args.n_nodes}  设备: {DEVICE}")
    print(f"  A(梯度投影): {args.use_grad_proj}  B(双流): {args.use_dual_stream}")
    print(f"  C(课程训练): {args.use_curriculum}  D(不确定性加权): {args.use_uncertainty_weight}")
    print(f"  LATENT_CAUSAL: {LATENT_DIM_CAUSAL}  LATENT_RECON: {LATENT_DIM_RECON}")

    t_start = time.perf_counter()
    if args.mode == "quick":
        run_quick(args)
    elif args.mode == "full":
        run_full(args)
    elif args.mode == "consistency":
        run_consistency(args)
    elif args.mode == "ablation":
        run_ablation(args)
    elif args.mode == "all":
        run_quick(args)
        run_full(args)
        run_consistency(args)
        run_ablation(args)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═' * 70}\n  v5 高维验证完成！耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)\n{'═' * 70}")


if __name__ == "__main__":
    main()
