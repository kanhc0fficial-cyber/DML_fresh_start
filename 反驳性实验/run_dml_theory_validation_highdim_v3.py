"""
run_dml_theory_validation_highdim_v3.py
========================================
DML 理论验证（高维合成数据）—— V3 两阶段解耦 VAE-DML

═══════════════════════════════════════════════════════════════════
  高维场景创新方案 V3：Two-Stage Decoupled VAE-DML

  本脚本是高维合成数据管线中的 V3，对应真实管线中的
  run_refutation_xin2_v3.py（两阶段解耦 VAE-DML）。

  核心架构（继承自原版 v3）：
  ─────────────────────────────────────────────────────────────────
  【第一阶段】VAE 表征学习
    输入：X_ctrl_hd（高维观测扩展后的混杂变量，~100 维）
    Loss = MSE_recon + β_anneal(t) × KL
    输出：低维确定性均值 μ（latent_dim=32）
    训练完成后 encoder 冻结

  【第二阶段】双独立预测头
    Head_Y：μ → Ŷ = E[Y|X]
    Head_D：μ → D̂ = E[D|X]
    独立训练，无共享梯度

  【DML 估计】
    θ = Σ(res_D × res_Y) / Σ(res_D²)
    交叉拟合 + 中位数聚合 SE

  高维优势：
    - 100+ 维输入中存在大量冗余/相关特征
    - VAE 能发现低维流形结构（~10 个真实混杂变量的流形）
    - 压缩到 32 维潜空间后，预测头在干净的表征上工作
    - RF/GBM 无法有效降维，在冗余特征上过拟合

用法：
  python run_dml_theory_validation_highdim_v3.py --mode quick
  python run_dml_theory_validation_highdim_v3.py --mode full
  python run_dml_theory_validation_highdim_v3.py --mode consistency
  python run_dml_theory_validation_highdim_v3.py --mode all
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

# 输出目录
OUT_DIR = os.path.join(BASE_DIR, "DML理论验证")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── PyTorch 依赖 ────────────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import KFold
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = None
    print("[警告] PyTorch 未安装，v3 估计器将不可用。")


# ═══════════════════════════════════════════════════════════════════
#  v3 超参数（与低维版一致，但潜空间增大以匹配高维输入）
# ═══════════════════════════════════════════════════════════════════

LATENT_DIM = 48               # 潜变量维度（高维输入需要更大潜空间）
HIDDEN_DIM_ENCODER = 128      # Encoder MLP 隐层维度（高维需要更宽）
HIDDEN_DIM_HEAD = 64          # 预测头 MLP 隐层维度
BETA_KL = 0.1                 # KL 权重上限（退火终点）
ANNEAL_EPOCHS = 20            # KL 退火轮数
MAX_EPOCHS_VAE = 80           # Stage 1 最大训练轮数（高维需要更长）
MAX_EPOCHS_HEAD = 50          # Stage 2 最大训练轮数
LR_VAE = 0.001                # Stage 1 学习率
LR_HEAD = 0.003               # Stage 2 学习率
GRAD_CLIP = 1.0               # 梯度裁剪上限


# ═══════════════════════════════════════════════════════════════════
#  模型定义
# ═══════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class MLPEncoder(nn.Module):
        """MLP 编码器：X_hd → (μ, logvar)"""

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
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.SiLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        def forward(self, x: torch.Tensor):
            h = self.shared(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
            mu, _ = self.forward(x)
            return mu

    class MLPDecoder(nn.Module):
        """MLP 解码器：z → X_hat"""

        def __init__(self, latent_dim: int = LATENT_DIM,
                     hidden_dim: int = HIDDEN_DIM_ENCODER,
                     output_dim: int = 100):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)

    class PredictionHead(nn.Module):
        """独立预测头：z → scalar"""

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
        """两阶段解耦 VAE-DML 估计器（高维版）"""

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
                    z_batch = z[idx]
                    y_batch = Y_tensor[idx]
                    d_batch = D_tensor[idx]
                    y_pred = self.head_Y(z_batch)
                    loss_Y = nn.functional.mse_loss(y_pred, y_batch)
                    opt_Y.zero_grad()
                    loss_Y.backward()
                    nn.utils.clip_grad_norm_(self.head_Y.parameters(), GRAD_CLIP)
                    opt_Y.step()
                    d_pred = self.head_D(z_batch)
                    loss_D = nn.functional.mse_loss(d_pred, d_batch)
                    opt_D.zero_grad()
                    loss_D.backward()
                    nn.utils.clip_grad_norm_(self.head_D.parameters(), GRAD_CLIP)
                    opt_D.step()
            self.head_Y.eval()
            self.head_D.eval()

        def predict_residuals(self, X_ctrl, Y, D):
            self.encoder.eval()
            self.head_Y.eval()
            self.head_D.eval()
            X_tensor = torch.FloatTensor(X_ctrl).to(self.device)
            with torch.no_grad():
                z = self.encoder.encode_mean(X_tensor)
                y_hat = self.head_Y(z).cpu().numpy()
                d_hat = self.head_D(z).cpu().numpy()
            return Y - y_hat, D - d_hat


# ═══════════════════════════════════════════════════════════════════
#  v3 DML 估计器（高维版）
# ═══════════════════════════════════════════════════════════════════

def v3_highdim_dml_estimate(Y, D, X_ctrl, seed=42, n_folds=5, n_repeats=5):
    """两阶段解耦 VAE-DML 估计器（高维版 + 交叉拟合 + 中位数聚合）"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 未安装")

    n = len(Y)
    input_dim = X_ctrl.shape[1]

    # 数据标准化
    X_mean = X_ctrl.mean(axis=0)
    X_std = X_ctrl.std(axis=0) + 1e-8
    X_normed = (X_ctrl - X_mean) / X_std
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8
    D_mean, D_std = D.mean(), D.std() + 1e-8
    Y_normed = (Y - Y_mean) / Y_std
    D_normed = (D - D_mean) / D_std

    def _single_crossfit(seed_k):
        torch.manual_seed(seed_k)
        np.random.seed(seed_k)
        res_Y_all = np.zeros(n)
        res_D_all = np.zeros(n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_k)
        for train_idx, test_idx in kf.split(X_normed):
            model = TwoStageVAEDML(input_dim=input_dim)
            model.train_stage1(X_normed[train_idx], seed=seed_k)
            model.train_stage2(X_normed[train_idx], Y_normed[train_idx],
                               D_normed[train_idx], seed=seed_k)
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

    theta_boots = []
    var_boots = []
    for b in range(n_repeats):
        boot_seed = seed * 1000 + b * 7
        theta_b, var_b = _single_crossfit(boot_seed)
        theta_boots.append(theta_b)
        var_boots.append(var_b)

    theta_boots = np.array(theta_boots)
    var_boots = np.array(var_boots)
    theta_final = float(np.median(theta_boots))
    var_final = float(np.median(var_boots + (theta_boots - theta_final) ** 2))
    se_final = np.sqrt(max(var_final, 1e-12))
    ci_lower = theta_final - 1.96 * se_final
    ci_upper = theta_final + 1.96 * se_final
    return float(theta_final), se_final, float(ci_lower), float(ci_upper)


# ═══════════════════════════════════════════════════════════════════
#  实验模式
# ═══════════════════════════════════════════════════════════════════

def run_quick(args):
    print("\n" + "█" * 70)
    print("  [快速模式] 高维 V3: 两阶段解耦 VAE-DML 验证")
    print("█" * 70)

    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42,
    )
    ate_true = dvch.compute_ate_for_dag_highdim(
        dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    n_exp = min(args.n_experiments, 30)
    df, summary = dvch.run_monte_carlo_highdim(
        estimator_fn=v3_highdim_dml_estimate,
        dag_info=dag_info, ate_true=ate_true,
        n_experiments=n_exp, n_samples=args.n_samples,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        tag="highdim_v3_quick", method_name="VAE-DML-v3 (高维, Quick)",
    )
    if isinstance(df, pd.DataFrame) and not df.empty:
        dvch.save_results(df, summary, "highdim_v3_vae_dml_quick")
    return df, summary


def run_full(args):
    print("\n" + "█" * 70)
    print("  [完整模式] 高维 V3: 两阶段解耦 VAE-DML 完整蒙特卡洛验证")
    print("█" * 70)

    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42,
    )
    ate_true = dvch.compute_ate_for_dag_highdim(
        dag_info, args.noise_scale, args.noise_type)
    print(f"  真实 ATE = {ate_true:.6f}")

    df, summary = dvch.run_monte_carlo_highdim(
        estimator_fn=v3_highdim_dml_estimate,
        dag_info=dag_info, ate_true=ate_true,
        n_experiments=args.n_experiments, n_samples=args.n_samples,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        tag="highdim_v3_full", method_name="VAE-DML-v3 (高维)",
    )
    if isinstance(df, pd.DataFrame) and not df.empty:
        dvch.save_results(df, summary, "highdim_v3_vae_dml_full")
    return df, summary


def run_consistency(args):
    print("\n" + "█" * 70)
    print("  [一致性模式] 高维 V3: VAE-DML √n-一致性验证")
    print("█" * 70)

    dag_info = dvch.setup_fixed_dag_highdim(
        n_nodes=args.n_nodes, graph_type=args.graph_type,
        use_industrial=args.use_industrial, dag_seed=42,
    )
    ate_true = dvch.compute_ate_for_dag_highdim(
        dag_info, args.noise_scale, args.noise_type)

    df_cons = dvch.run_consistency_validation_highdim(
        estimator_fn=v3_highdim_dml_estimate,
        dag_info=dag_info, ate_true=ate_true,
        sample_sizes=[500, 1000, 2000, 4000, 8000],
        n_experiments_per_size=50,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        method_name="VAE-DML-v3 (高维)",
    )
    if not df_cons.empty:
        csv_path = os.path.join(OUT_DIR, "highdim_v3_vae_dml_consistency.csv")
        df_cons.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  结果已保存：{csv_path}")
    return df_cons


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DML 理论验证（高维）—— V3 两阶段解耦 VAE-DML",
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full", "consistency", "all"])
    parser.add_argument("--n_experiments", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_nodes", type=int, default=60)
    parser.add_argument("--graph_type", type=str, default="layered",
                        choices=["layered", "er", "scale_free"])
    parser.add_argument("--noise_type", type=str, default="heteroscedastic",
                        choices=["gaussian", "heteroscedastic", "heavy_tail"])
    parser.add_argument("--noise_scale", type=float, default=0.3)
    parser.add_argument("--use_industrial", action="store_true", default=True)

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("[错误] PyTorch 未安装")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   DML 理论验证（高维）—— V3 两阶段解耦 VAE-DML               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  模式:       {args.mode}")
    print(f"  DAG 节点:   {args.n_nodes}")
    print(f"  样本量:     {args.n_samples}")
    print(f"  LATENT_DIM: {LATENT_DIM}")
    print(f"  HIDDEN_ENC: {HIDDEN_DIM_ENCODER}")
    print(f"  设备:       {DEVICE}")

    t_start = time.perf_counter()

    if args.mode == "quick":
        run_quick(args)
    elif args.mode == "full":
        run_full(args)
    elif args.mode == "consistency":
        run_consistency(args)
    elif args.mode == "all":
        run_full(args)
        run_consistency(args)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═' * 70}")
    print(f"  完成  耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
