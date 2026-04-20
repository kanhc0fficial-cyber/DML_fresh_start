"""
run_dml_theory_validation_v3.py
===============================
DML 理论验证 —— v3 创新方案：两阶段解耦 Autoencoder-DML

═══════════════════════════════════════════════════════════════════
  本脚本验证 v3 创新架构在合成数据上的因果推断性能，对标传统 ML 基线。

  核心创新：两阶段解耦 VAE-DML（适配表格数据版本）
═══════════════════════════════════════════════════════════════════

  【第一阶段】Autoencoder 表征学习（学习 X 混杂变量的紧凑潜在表示 z）
    Loss_stage1 = MSE_recon + β_anneal(t) × KL
    输入：X_confounders（高维混杂变量）
    输出：低维确定性均值 μ（latent_dim=32）
    训练完成后 encoder 权重全部冻结 (requires_grad=False)

    与原版 v3 (run_refutation_xin2_v3.py) 的区别：
    - 原版使用 LSTM encoder 处理时序工业数据（329 状态变量 × SEQ_LEN）
    - 本脚本使用 MLP encoder 处理合成表格数据（无时序结构）
    - 保留 VAE 的 KL-annealing 训练策略

  【第二阶段】双独立预测头（在冻结的 μ 上训练）
    Head_Y：μ → Ŷ = E[Y|X]    （与 Head_D 完全独立，无共享梯度）
    Head_D：μ → D̂ = E[D|X]
    残差：res_Y = Y - Ŷ，res_D = D - D̂
    θ = Cov(res_Y, res_D) / Var(res_D)  [Double ML]

  【推断时只用 μ，不采样】
    彻底消除 reparameterize 的随机性（eval 阶段 ε=0）

  对比基线：
    - run_dml_theory_validation_baseline_dml.py（RF / Lasso / GBM）
    - run_dml_theory_validation_baseline_ols.py（OLS）

═══════════════════════════════════════════════════════════════════
  验证内容
═══════════════════════════════════════════════════════════════════
  1. 无偏性（Unbiasedness）：E[θ̂] ≈ θ_true
  2. √n-一致性（√n-Consistency）：RMSE ∝ 1/√n
  3. 置信区间覆盖率（Coverage）：95% CI 覆盖 θ_true 的频率 ≈ 95%
  4. 与传统基线的对比

用法：
  # 快速验证（少量实验）
  python run_dml_theory_validation_v3.py --mode quick

  # 完整蒙特卡洛验证（默认 200 次实验）
  python run_dml_theory_validation_v3.py --mode full

  # √n-一致性验证
  python run_dml_theory_validation_v3.py --mode consistency

  # 全部实验
  python run_dml_theory_validation_v3.py --mode all
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
    from sklearn.model_selection import KFold
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = None
    print("[警告] PyTorch 未安装，v3 估计器将不可用。")
    print("       请执行: pip install torch")


# ═══════════════════════════════════════════════════════════════════
#  v3 超参数（与 run_refutation_xin2_v3.py 对齐）
# ═══════════════════════════════════════════════════════════════════

LATENT_DIM = 32           # 潜变量维度
HIDDEN_DIM_ENCODER = 64   # Encoder MLP 隐层维度
HIDDEN_DIM_HEAD = 32      # 预测头 MLP 隐层维度
BETA_KL = 0.1             # KL 权重上限（退火终点）
ANNEAL_EPOCHS = 20        # KL 退火轮数：前 20 轮 β 从 0 线性升至 BETA_KL
MAX_EPOCHS_VAE = 60       # Stage 1 最大训练轮数
MAX_EPOCHS_HEAD = 40      # Stage 2 最大训练轮数
LR_VAE = 0.001            # Stage 1 学习率
LR_HEAD = 0.003           # Stage 2 学习率
GRAD_CLIP = 1.0           # 梯度裁剪上限


# ═══════════════════════════════════════════════════════════════════
#  模型定义（适配表格数据的 MLP 版本）
#  仅在 PyTorch 可用时定义
# ═══════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class MLPEncoder(nn.Module):
        """
        MLP 编码器：X_confounders → (μ, logvar)

        对应原版 v3 中的 VAEEncoder（LSTM 替换为 MLP）。
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

        用于 Stage 1 的 VAE 训练，学习有意义的潜在表征。
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

        Stage 2 中 Head_Y 和 Head_D 各有一个实例，完全独立训练，
        不共享任何参数或梯度。
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
        完整的两阶段解耦 VAE-DML 估计器。

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
            β_anneal(t) = min(t / anneal_epochs, 1.0) × beta_kl

            参数:
                X_ctrl:        混杂变量矩阵 (n, p)
                epochs:        最大训练轮数
                beta_kl:       KL 权重上限
                anneal_epochs: 退火轮数
                lr:            学习率
                seed:          随机种子
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
                # KL 退火：前 anneal_epochs 轮从 0 线性升至 beta_kl
                beta_t = beta_kl * min((epoch + 1) / anneal_epochs, 1.0)

                # Mini-batch 训练
                perm = torch.randperm(n, device=self.device)

                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    x_batch = X_tensor[idx]

                    mu, logvar = self.encoder(x_batch)

                    # Reparameterization trick
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + std * eps

                    x_recon = self.decoder(z)

                    # 重建损失 (MSE)
                    recon_loss = nn.functional.mse_loss(x_recon, x_batch)

                    # KL 散度: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
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

            # Stage 1 训练完毕，冻结 encoder
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        def train_stage2(self, X_ctrl: np.ndarray, Y: np.ndarray, D: np.ndarray,
                         epochs: int = MAX_EPOCHS_HEAD, lr: float = LR_HEAD,
                         seed: int = 42):
            """
            Stage 2: 冻结 Encoder，独立训练 Head_Y 和 Head_D

            在冻结的潜在表征 z = Encoder.encode_mean(X) 上训练两个独立预测头。
            Head_Y 和 Head_D 不共享梯度，各自独立优化 MSE 损失。

            参数:
                X_ctrl: 混杂变量矩阵 (n, p)
                Y:      结果变量 (n,)
                D:      处理变量 (n,)
                epochs: 最大训练轮数
                lr:     学习率
                seed:   随机种子
            """
            torch.manual_seed(seed + 100)
            self.head_Y.train()
            self.head_D.train()

            X_tensor = torch.FloatTensor(X_ctrl).to(self.device)
            Y_tensor = torch.FloatTensor(Y).to(self.device)
            D_tensor = torch.FloatTensor(D).to(self.device)

            # 获取冻结的潜在表征
            with torch.no_grad():
                z = self.encoder.encode_mean(X_tensor)

            # 独立优化器（确保无共享梯度）
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

                    # 训练 Head_Y
                    y_pred = self.head_Y(z_batch)
                    loss_Y = nn.functional.mse_loss(y_pred, y_batch)
                    opt_Y.zero_grad()
                    loss_Y.backward()
                    nn.utils.clip_grad_norm_(self.head_Y.parameters(), GRAD_CLIP)
                    opt_Y.step()

                    # 训练 Head_D（完全独立）
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
            使用训练好的模型计算残差。

            res_Y = Y - Head_Y(Encoder.encode_mean(X))
            res_D = D - Head_D(Encoder.encode_mean(X))

            参数:
                X_ctrl: 混杂变量 (n, p)
                Y:      结果变量 (n,)
                D:      处理变量 (n,)

            返回:
                (res_Y, res_D): 残差数组
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
#  v3 DML 估计器（含交叉拟合与中位数聚合）
# ═══════════════════════════════════════════════════════════════════

def v3_dml_estimate(Y: np.ndarray, D: np.ndarray, X_ctrl: np.ndarray,
                    seed: int = 42, n_folds: int = 5, n_repeats: int = 5) -> tuple:
    """
    两阶段解耦 Autoencoder-DML 估计器。

    使用 K-fold 交叉拟合 + 多次重复 + 中位数聚合（修正漏洞三）。

    对每次重复（不同随机分折）：
      1. 将数据分为 K 折
      2. 对第 k 折：
         - 在训练集上训练 Stage 1 (VAE)，再训练 Stage 2 (Heads)
         - 在测试集上用训练好的模型计算残差
      3. 汇总所有折的残差，计算 θ̂_b 和 Neyman 方差 V_b

    最终聚合（Chernozhukov 2018 中位数聚合）：
      θ_final = Median(θ_b)
      V_final = Median(V_b + (θ_b - θ_final)²)
      SE = √V_final

    参数:
        Y:       结果变量 (n,)
        D:       处理变量 (n,)
        X_ctrl:  控制变量（混杂变量）(n, p)
        seed:    随机种子
        n_folds: 交叉拟合折数
        n_repeats: 重复交叉拟合次数

    返回:
        (theta, se, ci_lower, ci_upper)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 未安装，无法使用 v3 估计器")

    n = len(Y)
    input_dim = X_ctrl.shape[1]

    # 数据标准化（对神经网络至关重要）
    X_mean = X_ctrl.mean(axis=0)
    X_std = X_ctrl.std(axis=0) + 1e-8
    X_normed = (X_ctrl - X_mean) / X_std

    Y_mean = Y.mean()
    Y_std = Y.std() + 1e-8
    D_mean = D.mean()
    D_std = D.std() + 1e-8
    Y_normed = (Y - Y_mean) / Y_std
    D_normed = (D - D_mean) / D_std

    def _single_crossfit_dml(seed_k: int):
        """单次交叉拟合 DML 估计（含 Neyman SE）"""
        torch.manual_seed(seed_k)
        np.random.seed(seed_k)

        res_Y_all = np.zeros(n)
        res_D_all = np.zeros(n)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_k)

        for train_idx, test_idx in kf.split(X_normed):
            X_train = X_normed[train_idx]
            X_test = X_normed[test_idx]
            Y_train = Y_normed[train_idx]
            Y_test = Y_normed[test_idx]
            D_train = D_normed[train_idx]
            D_test = D_normed[test_idx]

            # 创建新模型实例
            model = TwoStageVAEDML(input_dim=input_dim)

            # Stage 1: 训练 VAE (Encoder + Decoder)
            model.train_stage1(X_train, seed=seed_k)

            # Stage 2: 冻结 Encoder, 独立训练 Head_Y 和 Head_D
            model.train_stage2(X_train, Y_train, D_train, seed=seed_k)

            # 在测试集上计算残差
            res_Y_fold, res_D_fold = model.predict_residuals(X_test, Y_test, D_test)
            res_Y_all[test_idx] = res_Y_fold
            res_D_all[test_idx] = res_D_fold

        # 反标准化残差以恢复原始尺度
        # 标准化空间中: res_Y_normed = Y_normed - Head_Y(z)
        # 原始空间中:   res_Y_orig = res_Y_normed × Y_std
        # 注：均值项 Y_mean 不需要加回，因为残差本身已是零均值偏差量，
        #     Head 预测的是标准化后目标，残差只需还原方差尺度即可。
        res_Y_orig = res_Y_all * Y_std
        res_D_orig = res_D_all * D_std

        # θ = Σ(res_D × res_Y) / Σ(res_D²)
        denom = np.sum(res_D_orig ** 2) + 1e-12
        theta_k = np.sum(res_D_orig * res_Y_orig) / denom

        # Neyman 式解析 SE
        psi = res_D_orig * (res_Y_orig - theta_k * res_D_orig)
        J = np.mean(res_D_orig ** 2)
        var_neyman = float(np.mean(psi ** 2) / (n * J ** 2 + 1e-12))

        return theta_k, var_neyman

    # 重复交叉拟合（不同随机分折）
    theta_boots = []
    var_boots = []

    for b in range(n_repeats):
        boot_seed = seed * 1000 + b * 7
        theta_b, var_b = _single_crossfit_dml(boot_seed)
        theta_boots.append(theta_b)
        var_boots.append(var_b)

    theta_boots = np.array(theta_boots)
    var_boots = np.array(var_boots)

    # 中位数聚合（Chernozhukov 2018，修正漏洞三）
    theta_final = float(np.median(theta_boots))
    # 结合抽样方差和分割方差
    var_final = float(np.median(var_boots + (theta_boots - theta_final) ** 2))
    se_final = np.sqrt(max(var_final, 1e-12))

    ci_lower = theta_final - 1.96 * se_final
    ci_upper = theta_final + 1.96 * se_final

    return float(theta_final), se_final, float(ci_lower), float(ci_upper)


# ═══════════════════════════════════════════════════════════════════
#  实验模式实现
# ═══════════════════════════════════════════════════════════════════

def run_quick(args):
    """快速验证模式：使用少量实验验证 v3 估计器的基本正确性"""
    print("\n" + "█" * 70)
    print("  [快速模式] v3 两阶段解耦 Autoencoder-DML 验证")
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

    df, summary = dvc.run_monte_carlo(
        estimator_fn=v3_dml_estimate,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=n_exp,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="v3_quick",
        method_name="VAE-DML-v3 (Quick)",
    )

    if not df.empty:
        dvc.save_results(df, summary, "v3_vae_dml_quick")
    return df, summary


def run_full(args):
    """完整蒙特卡洛验证：200 次独立实验评估无偏性、覆盖率"""
    print("\n" + "█" * 70)
    print("  [完整模式] v3 两阶段解耦 Autoencoder-DML 完整蒙特卡洛验证")
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

    df, summary = dvc.run_monte_carlo(
        estimator_fn=v3_dml_estimate,
        dag_info=dag_info,
        ate_true=ate_true,
        n_experiments=args.n_experiments,
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        tag="v3_full",
        method_name="VAE-DML-v3",
    )

    if not df.empty:
        dvc.save_results(df, summary, "v3_vae_dml_full")
    return df, summary


def run_consistency(args):
    """√n-一致性验证：验证 RMSE ∝ 1/√n"""
    print("\n" + "█" * 70)
    print("  [一致性模式] v3 两阶段解耦 Autoencoder-DML √n-一致性验证")
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

    df_cons = dvc.run_consistency_validation(
        estimator_fn=v3_dml_estimate,
        dag_info=dag_info,
        ate_true=ate_true,
        sample_sizes=[500, 1000, 2000, 4000, 8000],
        n_experiments_per_size=50,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        method_name="VAE-DML-v3",
    )

    if not df_cons.empty:
        csv_path = os.path.join(OUT_DIR, "v3_vae_dml_consistency.csv")
        df_cons.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  一致性结果已保存：{csv_path}")

    return df_cons


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DML 理论验证 —— v3 两阶段解耦 Autoencoder-DML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_dml_theory_validation_v3.py --mode quick
  python run_dml_theory_validation_v3.py --mode full --n_experiments 200
  python run_dml_theory_validation_v3.py --mode consistency
  python run_dml_theory_validation_v3.py --mode all --graph_type layered
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "full", "consistency", "all"],
        help="运行模式: quick=快速验证, full=完整蒙特卡洛, "
             "consistency=√n一致性, all=全部",
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

    args = parser.parse_args()

    # 检查 PyTorch 可用性
    if not TORCH_AVAILABLE:
        print("[错误] PyTorch 未安装，无法运行 v3 验证。")
        print("       请执行: pip install torch")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   DML 理论验证 —— v3 两阶段解耦 Autoencoder-DML               ║")
    print("║   创新点: VAE 表征学习 + 双独立预测头 + KL-Annealing          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  模式:       {args.mode}")
    print(f"  实验次数:   {args.n_experiments}")
    print(f"  样本量:     {args.n_samples}")
    print(f"  图类型:     {args.graph_type}")
    print(f"  噪声类型:   {args.noise_type}")
    print(f"  噪声标准差: {args.noise_scale}")
    print(f"  工业函数:   {args.use_industrial}")
    print(f"  设备:       {DEVICE}")
    print(f"  ── v3 超参 ──")
    print(f"  LATENT_DIM:       {LATENT_DIM}")
    print(f"  HIDDEN_DIM_ENC:   {HIDDEN_DIM_ENCODER}")
    print(f"  HIDDEN_DIM_HEAD:  {HIDDEN_DIM_HEAD}")
    print(f"  BETA_KL:          {BETA_KL}")
    print(f"  ANNEAL_EPOCHS:    {ANNEAL_EPOCHS}")
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
    elif args.mode == "all":
        print("\n\n" + "▓" * 70)
        print("  第 1 步：完整蒙特卡洛验证")
        print("▓" * 70)
        run_full(args)

        print("\n\n" + "▓" * 70)
        print("  第 2 步：√n-一致性验证")
        print("▓" * 70)
        run_consistency(args)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'═' * 70}")
    print(f"  全部任务完成  总耗时: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")
    print(f"  结果输出目录: {OUT_DIR}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
