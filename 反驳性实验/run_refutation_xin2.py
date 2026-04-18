"""
run_refutation_xin2.py
=================================
XIN_2 因果推断管线的三种反驳性实验脚本
基于 run_joint_causal_xin2_fixed.py 的核心逻辑

支持三种反驳实验模式（通过命令行 --mode 参数选择）：
  placebo        安慰剂实验：将操作变量 D 替换为随机噪声（伪干预），期望 θ ≈ 0
  random_confounder  随机混杂变量实验：向 X 中注入随机混杂列，期望 θ 稳健不变
  data_subset    数据子集实验：对数据随机抽取多个子集，期望 θ 估计值收敛

用法示例：
  python run_refutation_xin2.py --mode placebo
  python run_refutation_xin2.py --mode random_confounder --n_confounders 5
  python run_refutation_xin2.py --mode data_subset --n_subsets 10 --subset_frac 0.8

反驳实验通过标准：
  placebo           ：绝大多数操作节点的 P_Value ≥ 0.1（不显著），|θ| 应接近 0
  random_confounder ：注入混杂后 θ 与原始 θ 的相对偏差 < 10%
  data_subset       ：子集间 θ 的变异系数 (CV) < 0.3，且符号一致

作者备注：
  本脚本与原始管线共用 JointCausalVAELSTM 模型定义和 train_one_op 训练逻辑，
  仅在数据准备阶段注入扰动，确保反驳实验的可比性。
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# ================= 配置与路径（与原始脚本保持一致）=================
DATA_DIR        = r"C:\DML_fresh_start\数据存储"
X_PARQUET       = os.path.join(DATA_DIR, "X_features_new.parquet")
Y_CSV           = os.path.join(DATA_DIR, "y_target_new.csv")
OPERABILITY_CSV = (
    r"C:\DML_fresh_start\数据预处理"
    r"\数据与处理结果-分阶段-去共线性后"
    r"\non_collinear_representative_vars_operability.csv"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLACEBO_OUT_DIR = os.path.join(BASE_DIR, "安慰剂实验")
RANDOM_CONFOUNDER_OUT_DIR = os.path.join(BASE_DIR, "随机混杂变量实验")
DATA_SUBSET_OUT_DIR = os.path.join(BASE_DIR, "数据子集实验")

os.makedirs(PLACEBO_OUT_DIR, exist_ok=True)
os.makedirs(RANDOM_CONFOUNDER_OUT_DIR, exist_ok=True)
os.makedirs(DATA_SUBSET_OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 超参数（与原始脚本保持一致）=================
SEQ_LEN              = 6
EMBARGO_GAP          = 4
K_FOLDS              = 4
LATENT_DIM           = 8
ALPHA_Y              = 1.0
ALPHA_D              = 1.0
BETA_KL              = 0.5
MAX_EPOCHS           = 60
PATIENCE             = 8
MIN_TRAIN_SIZE       = 100
MIN_VALID_RESIDUALS  = 50
F_STAT_THRESHOLD     = 10.0


# ================= 模型定义（与原始脚本完全一致）=================
class JointCausalVAELSTM(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim=8):
        super().__init__()
        self.seq_len   = seq_len
        self.input_dim = input_dim

        self.encoder_lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.encoder_fc   = nn.Sequential(nn.Linear(32, 16), nn.SiLU())
        self.fc_mu        = nn.Linear(16, latent_dim)
        self.fc_logvar    = nn.Linear(16, latent_dim)

        self.decoder_recon = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.SiLU(),
            nn.Linear(16, input_dim * seq_len)
        )
        self.head_Y = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.SiLU(), nn.Linear(16, 1)
        )
        self.head_D = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.SiLU(), nn.Linear(16, 1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h      = self.encoder_fc(h_n[-1])
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = self.reparameterize(mu, logvar)
        x_recon = self.decoder_recon(z).view(-1, self.seq_len, self.input_dim)
        y_hat   = self.head_Y(z).squeeze(-1)
        d_hat   = self.head_D(z).squeeze(-1)
        return x_recon, y_hat, d_hat, mu, logvar


# ================= 数据准备（与原始脚本保持一致）=================
def build_xin2_data():
    op_df = pd.read_csv(OPERABILITY_CSV, encoding="utf-8-sig")
    op_df["Group"] = op_df["Group"].str.strip().str.upper()
    xin2_df = op_df[op_df["Group"].isin(["B", "C"])].copy()

    operable_set   = set(xin2_df[xin2_df["Operability"].str.strip() == "operable" ]["Variable_Name"].str.strip())
    observable_set = set(xin2_df[xin2_df["Operability"].str.strip() == "observable"]["Variable_Name"].str.strip())
    all_bc_vars    = operable_set | observable_set

    print(f"[数据准备] Group B+C 共 {len(all_bc_vars)} 个变量，"
          f"其中 operable={len(operable_set)}，observable={len(observable_set)}")

    X  = pd.read_parquet(X_PARQUET)
    y  = pd.read_csv(Y_CSV, parse_dates=["time"]).dropna(subset=["y_fx_xin2"])
    X.index  = pd.to_datetime(X.index).tz_localize(None)
    y["time"] = y["time"].dt.tz_localize(None)

    X_re = X.resample("10min").mean().ffill().bfill()
    y_re = y.set_index("time")["y_fx_xin2"].resample("10min").mean().interpolate()
    comm = X_re.index.intersection(y_re.index)
    df   = pd.concat([X_re.loc[comm], y_re.loc[comm].rename("Y_grade")], axis=1).dropna()
    df   = df.loc[:, (df.std() > 1e-4)]

    valid_cols   = [c for c in df.columns if c in all_bc_vars or c == "Y_grade"]
    df_filtered  = df[valid_cols]

    cols_in_df      = set(df_filtered.columns) - {"Y_grade"}
    operable_in_df  = operable_set  & cols_in_df
    observable_in_df = observable_set & cols_in_df
    print(f"[数据准备] 实际进入 DataFrame 的列："
          f"operable={len(operable_in_df)}，observable={len(observable_in_df)}")

    return df_filtered, operable_in_df, observable_in_df


# ================= 核心训练函数（复用原始逻辑，支持数据替换）=================
def train_one_op(op, df, safe_x, override_D=None):
    """
    与原始 train_one_op 完全一致，新增 override_D 参数：
      - override_D=None：正常训练（原始逻辑）
      - override_D=np.ndarray：用外部数组替换操作变量 D（安慰剂/子集实验使用）
    """
    X_mat = df[safe_x].values
    X_mat = (X_mat - X_mat.mean(axis=0)) / (X_mat.std(axis=0) + 1e-8)
    Y_mat = df["Y_grade"].values
    D_mat = df[op].values if override_D is None else override_D

    seqs_X, targets_Y, targets_D = [], [], []
    for i in range(len(X_mat) - SEQ_LEN):
        seqs_X.append(X_mat[i: i + SEQ_LEN])
        targets_Y.append(Y_mat[i + SEQ_LEN])
        targets_D.append(D_mat[i + SEQ_LEN])

    seqs_X    = np.array(seqs_X)
    targets_Y = np.array(targets_Y)
    targets_D = np.array(targets_D)

    N          = len(seqs_X)
    block_size = N // K_FOLDS

    all_res_Y, all_res_D = [], []

    for k in range(1, K_FOLDS):
        train_end = k * block_size - EMBARGO_GAP
        if train_end < MIN_TRAIN_SIZE:
            continue

        val_start = k * block_size
        val_end   = (k + 1) * block_size if k < K_FOLDS - 1 else N

        X_train = torch.tensor(seqs_X[:train_end],    dtype=torch.float32).to(DEVICE)
        Y_train = torch.tensor(targets_Y[:train_end], dtype=torch.float32).to(DEVICE)
        D_train = torch.tensor(targets_D[:train_end], dtype=torch.float32).to(DEVICE)

        X_val = torch.tensor(seqs_X[val_start:val_end], dtype=torch.float32).to(DEVICE)
        Y_val = targets_Y[val_start:val_end]
        D_val = targets_D[val_start:val_end]

        model     = JointCausalVAELSTM(
            input_dim=len(safe_x), seq_len=SEQ_LEN, latent_dim=LATENT_DIM
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        dataset_train = TensorDataset(X_train, Y_train, D_train)
        loader_train  = DataLoader(dataset_train, batch_size=256, shuffle=False)

        val_split    = max(1, int(train_end * 0.1))
        X_inner_val  = X_train[-val_split:]
        Y_inner_val  = Y_train[-val_split:]
        D_inner_val  = D_train[-val_split:]

        best_val_loss    = float("inf")
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            for b_x, b_y, b_d in loader_train:
                optimizer.zero_grad()
                x_recon, y_hat, d_hat, mu, logvar = model(b_x)
                loss_recon = nn.functional.mse_loss(x_recon, b_x)
                loss_kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss_y     = nn.functional.mse_loss(y_hat, b_y)
                loss_d     = nn.functional.mse_loss(d_hat, b_d)
                total_loss = loss_recon + BETA_KL * loss_kl + ALPHA_Y * loss_y + ALPHA_D * loss_d
                total_loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                _, y_iv, d_iv, _, _ = model(X_inner_val)
                iv_loss = (
                    nn.functional.mse_loss(y_iv, Y_inner_val)
                    + nn.functional.mse_loss(d_iv, D_inner_val)
                ).item()

            if iv_loss < best_val_loss - 1e-5:
                best_val_loss    = iv_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        model.eval()
        with torch.no_grad():
            _, (h_n, _) = model.encoder_lstm(X_val)
            h      = model.encoder_fc(h_n[-1])
            z_mean = model.fc_mu(h)
            pred_Y = model.head_Y(z_mean).squeeze(-1).cpu().numpy()
            pred_D = model.head_D(z_mean).squeeze(-1).cpu().numpy()

        all_res_Y.extend(Y_val - pred_Y)
        all_res_D.extend(D_val - pred_D)

    res_Y = np.array(all_res_Y)
    res_D = np.array(all_res_D)

    mask  = (np.abs(res_Y) < 3 * np.std(res_Y)) & (np.abs(res_D) < 3 * np.std(res_D))
    res_Y, res_D = res_Y[mask], res_D[mask]

    if len(res_D) < MIN_VALID_RESIDUALS:
        return None

    denom_var = np.var(res_D)
    n         = len(res_D)
    f_stat    = (np.mean(res_D ** 2)) / (denom_var / n + 1e-12) if denom_var > 0 else 0
    if f_stat < F_STAT_THRESHOLD:
        return None

    sum_DD    = np.dot(res_D, res_D)
    sum_DY    = np.dot(res_D, res_Y)
    theta_hat = sum_DY / sum_DD

    J_hat      = np.mean(res_D ** 2)
    score_func = res_Y - theta_hat * res_D
    Omega_hat  = np.mean((score_func ** 2) * (res_D ** 2))
    Sigma_hat  = Omega_hat / (J_hat ** 2 + 1e-12)
    SE         = np.sqrt(Sigma_hat / n)

    t_stat = theta_hat / (SE + 1e-12)
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    return theta_hat, p_val, SE, n, f_stat


# ==================== 辅助：构建 safe_x 列表 ====================
def get_safe_x(op, df, states):
    """对单个操作变量构建混杂特征集合（保持与原始脚本一致的逻辑）"""
    y_vals = df["Y_grade"].values
    x_vals = df[op].values

    best_t_lag, best_t_r = 0, 0
    for lag in range(1, 15):
        r = abs(np.corrcoef(x_vals[:-lag], y_vals[lag:])[0, 1])
        if r > best_t_r:
            best_t_r, best_t_lag = r, lag

    safe_x = []
    for st in states:
        best_s_r = 0
        best_s_lag = 0
        s_vals = df[st].values
        for lag in range(1, 15):
            r = abs(np.corrcoef(s_vals[:-lag], y_vals[lag:])[0, 1])
            if r > best_s_r:
                best_s_r, best_s_lag = r, lag
        if best_s_r > 0.05 and best_s_lag >= best_t_lag:
            safe_x.append(st)

    return safe_x


# ================================================================
#  实验一：安慰剂反驳实验
# ================================================================
def run_placebo(df, ops, states, n_permutations=5):
    """
    安慰剂设计：将操作变量 D 替换为随机排列版本（permutation），
    切断 D 与 Y 之间的真实因果链。

    反驳通过条件：
      - θ_placebo 的均值应接近 0（|mean(θ)| < 0.5 * |θ_original|）
      - 显著性比例（P < 0.05）应明显低于原始结果

    参数 n_permutations：每个操作变量重复随机排列的次数（增强统计稳定性）
    """
    print("\n" + "=" * 70)
    print(" 实验一：安慰剂反驳实验（随机排列操作变量 D）")
    print(f" 每个操作变量重复 {n_permutations} 次排列，期望所有 θ ≈ 0")
    print("=" * 70)

    results = []

    for op in ops:
        if df[op].std() < 0.1:
            continue

        safe_x = get_safe_x(op, df, states)
        if len(safe_x) < 2:
            continue

        D_original = df[op].values.copy()
        print(f"\n[操作节点: {op}]  混杂维度: {len(safe_x)}")

        for perm_idx in range(n_permutations):
            # 安慰剂核心：随机打乱 D，破坏时间因果结构
            rng = np.random.default_rng(seed=perm_idx * 42)
            D_placebo = rng.permutation(D_original)

            result = train_one_op(op, df, safe_x, override_D=D_placebo)
            if result is None:
                print(f"  [排列 {perm_idx+1}] 被过滤（弱工具/样本不足）")
                continue

            theta_hat, p_val, SE, n_res, f_stat = result
            sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "—"))
            print(f"  [排列 {perm_idx+1}] θ={theta_hat:+.5f}, P={p_val:.4f}{sig}, F={f_stat:.1f}")

            results.append({
                "操作节点":   op,
                "排列索引":   perm_idx + 1,
                "θ_安慰剂":  round(float(theta_hat), 5),
                "P_Value":   round(float(p_val), 4),
                "SE标准误":  round(float(SE), 5),
                "有效残差数": n_res,
                "F统计量":   round(float(f_stat), 2),
                "显著":      p_val < 0.05,
            })

    df_out = pd.DataFrame(results)
    if df_out.empty:
        print("[警告] 安慰剂实验无有效结果，请检查数据")
        return df_out

    # ---- 反驳汇总统计 ----
    sig_rate = df_out["显著"].mean()
    theta_mean = df_out["θ_安慰剂"].mean()
    theta_abs_mean = df_out["θ_安慰剂"].abs().mean()

    print("\n[安慰剂汇总]")
    print(f"  θ 均值  = {theta_mean:+.5f}  (期望 ≈ 0)")
    print(f"  |θ| 均值 = {theta_abs_mean:.5f}")
    print(f"  显著率 (P<0.05) = {sig_rate:.1%}  (期望 ≈ 5% 随机水平)")

    if sig_rate > 0.2:
        print("  [⚠] 安慰剂显著率偏高，因果估计可能存在虚假相关，请排查混杂变量！")
    else:
        print("  [✓] 安慰剂反驳通过：模型未从随机干预中检出虚假效应")

    out_path = os.path.join(PLACEBO_OUT_DIR, "refutation_placebo.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")
    return df_out


# ================================================================
#  实验二：随机混杂变量反驳实验
# ================================================================
def run_random_confounder(df, ops, states, n_confounders=5, n_repeats=3):
    """
    随机混杂设计：向控制变量集 X 中注入若干独立同分布的纯噪声列，
    然后重新估计 θ。

    反驳通过条件：
      - θ_perturbed 与 θ_original 的相对偏差 < 10%
        即：|θ_p - θ_o| / (|θ_o| + 1e-8) < 0.10
      - 注入噪声不应改变因果方向（符号一致）

    参数：
      n_confounders：每次注入的噪声列数量
      n_repeats：每个操作变量重复注入次数（增强稳定性）
    """
    print("\n" + "=" * 70)
    print(f" 实验二：随机混杂变量反驳实验（注入 {n_confounders} 个随机噪声列）")
    print(f" 每个操作变量重复 {n_repeats} 次，期望 θ 估计值稳健不变")
    print("=" * 70)

    results = []

    for op in ops:
        if df[op].std() < 0.1:
            continue

        safe_x_original = get_safe_x(op, df, states)
        if len(safe_x_original) < 2:
            continue

        print(f"\n[操作节点: {op}]  原始混杂维度: {len(safe_x_original)}")

        # 先计算原始 θ（无注入噪声）
        result_orig = train_one_op(op, df, safe_x_original)
        if result_orig is None:
            print(f"  [跳过] 原始估计失败")
            continue

        theta_orig, p_orig, SE_orig, n_orig, f_orig = result_orig
        print(f"  [原始] θ={theta_orig:+.5f}, P={p_orig:.4f}, F={f_orig:.1f}")

        for rep in range(n_repeats):
            rng = np.random.default_rng(seed=rep * 1000 + hash(op) % 10000)

            # 构造带噪声的临时 DataFrame（注意不修改原始 df）
            df_noisy = df.copy()
            noise_cols = []
            for nc in range(n_confounders):
                col_name = f"__random_confounder_{nc}__"
                # 噪声为标准正态，与任何真实变量无关
                df_noisy[col_name] = rng.standard_normal(len(df_noisy))
                noise_cols.append(col_name)

            # 在注入噪声后重新构建 safe_x（噪声列会因弱相关而被自动过滤）
            safe_x_noisy = safe_x_original + noise_cols  # 强制加入，测试模型鲁棒性

            result = train_one_op(op, df_noisy, safe_x_noisy)
            if result is None:
                print(f"  [注入 {rep+1}] 被过滤")
                continue

            theta_p, p_val, SE_p, n_p, f_p = result
            rel_dev = abs(theta_p - theta_orig) / (abs(theta_orig) + 1e-8)
            sign_ok = (np.sign(theta_p) == np.sign(theta_orig))

            sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "—"))
            verdict = "✓通过" if (rel_dev < 0.10 and sign_ok) else "⚠偏差"
            print(
                f"  [注入 {rep+1}] θ={theta_p:+.5f}, P={p_val:.4f}{sig}, "
                f"相对偏差={rel_dev:.1%}, 方向{'一致' if sign_ok else '反转'}  [{verdict}]"
            )

            results.append({
                "操作节点":     op,
                "重复索引":     rep + 1,
                "θ_原始":      round(float(theta_orig), 5),
                "θ_注入噪声":  round(float(theta_p), 5),
                "相对偏差":    round(float(rel_dev), 4),
                "方向一致":    sign_ok,
                "P_Value":     round(float(p_val), 4),
                "SE标准误":    round(float(SE_p), 5),
                "有效残差数":  n_p,
                "F统计量":     round(float(f_p), 2),
                "通过反驳":    rel_dev < 0.10 and sign_ok,
            })

    df_out = pd.DataFrame(results)
    if df_out.empty:
        print("[警告] 随机混杂实验无有效结果")
        return df_out

    pass_rate = df_out["通过反驳"].mean()
    print(f"\n[随机混杂汇总]  反驳通过率 = {pass_rate:.1%}  (期望 ≥ 80%)")

    if pass_rate < 0.8:
        print("  [⚠] 通过率偏低，θ 对混杂变量敏感，可能存在遗漏混杂！")
    else:
        print("  [✓] 随机混杂反驳通过：θ 估计值对随机混杂注入保持稳健")

    out_path = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "refutation_random_confounder.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")
    return df_out


# ================================================================
#  实验三：数据子集反驳实验
# ================================================================
def run_data_subset(df, ops, states, n_subsets=8, subset_frac=0.8):
    """
    数据子集设计：对时序数据做多个连续滑动子集（而非随机抽样，保持时序性），
    在每个子集上重新估计 θ，检验估计的稳定性。

    时序子集构造方式：
      - 将时间轴均分为 (n_subsets + 1) 段
      - 每个子集取 [起点, 起点 + subset_frac * T] 的连续时段
      - 相邻子集有重叠，覆盖不同时间窗口

    反驳通过条件：
      - 子集间 θ 的变异系数 CV = std(θ) / (|mean(θ)| + 1e-8) < 0.30
      - 至少 70% 子集的 θ 符号与众数符号一致

    参数：
      n_subsets：子集数量（建议 5~10）
      subset_frac：每个子集使用的数据比例（建议 0.7~0.9）
    """
    print("\n" + "=" * 70)
    print(f" 实验三：数据子集反驳实验（{n_subsets} 个子集，每个取 {subset_frac:.0%} 数据）")
    print(" 时序滑动窗口，期望 θ 在各子集上收敛")
    print("=" * 70)

    T = len(df)
    subset_len = int(T * subset_frac)
    # 滑动步长：将剩余数据均分给 n_subsets 个起始点
    step = max(1, (T - subset_len) // max(1, n_subsets - 1))

    results = []

    for op in ops:
        if df[op].std() < 0.1:
            continue

        safe_x = get_safe_x(op, df, states)
        if len(safe_x) < 2:
            continue

        print(f"\n[操作节点: {op}]  混杂维度: {len(safe_x)}")

        theta_list = []

        for sub_idx in range(n_subsets):
            start = min(sub_idx * step, T - subset_len)
            end   = start + subset_len
            df_sub = df.iloc[start:end].copy()

            # 子集内部需要足够的样本跑 K_FOLDS 交叉验证
            if len(df_sub) < SEQ_LEN + K_FOLDS * MIN_TRAIN_SIZE:
                print(f"  [子集 {sub_idx+1}] 样本不足，跳过")
                continue

            result = train_one_op(op, df_sub, safe_x)
            if result is None:
                print(f"  [子集 {sub_idx+1}] 被过滤（弱工具/样本不足）")
                continue

            theta_hat, p_val, SE, n_res, f_stat = result
            theta_list.append(theta_hat)

            sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "—"))
            print(
                f"  [子集 {sub_idx+1}] 时段 [{start}:{end}]  "
                f"θ={theta_hat:+.5f}, P={p_val:.4f}{sig}, F={f_stat:.1f}, n={n_res}"
            )

            results.append({
                "操作节点":  op,
                "子集索引":  sub_idx + 1,
                "时段起点":  start,
                "时段终点":  end,
                "θ_子集":   round(float(theta_hat), 5),
                "P_Value":  round(float(p_val), 4),
                "SE标准误": round(float(SE), 5),
                "有效残差数": n_res,
                "F统计量":  round(float(f_stat), 2),
            })

        # 单操作变量的子集稳定性汇总
        if len(theta_list) >= 3:
            arr = np.array(theta_list)
            cv  = np.std(arr) / (abs(np.mean(arr)) + 1e-8)
            dominant_sign = np.sign(np.median(arr))
            sign_consistent = np.mean(np.sign(arr) == dominant_sign)

            verdict = "✓通过" if (cv < 0.30 and sign_consistent >= 0.70) else "⚠不稳定"
            print(
                f"  [稳定性] CV={cv:.3f}（<0.30 为佳），"
                f"符号一致率={sign_consistent:.0%}（≥70% 为佳）  [{verdict}]"
            )

    df_out = pd.DataFrame(results)
    if df_out.empty:
        print("[警告] 数据子集实验无有效结果")
        return df_out

    # 全局稳定性统计
    stable_ops = 0
    total_ops_with_results = 0
    for op, grp in df_out.groupby("操作节点"):
        if len(grp) < 3:
            continue
        total_ops_with_results += 1
        arr = grp["θ_子集"].values
        cv  = np.std(arr) / (abs(np.mean(arr)) + 1e-8)
        dominant_sign = np.sign(np.median(arr))
        sign_consistent = np.mean(np.sign(arr) == dominant_sign)
        if cv < 0.30 and sign_consistent >= 0.70:
            stable_ops += 1

    if total_ops_with_results > 0:
        global_pass = stable_ops / total_ops_with_results
        print(f"\n[数据子集汇总]  全局稳定通过率 = {global_pass:.1%}  (期望 ≥ 70%)")
        if global_pass < 0.70:
            print("  [⚠] 子集间 θ 方差较大，估计稳定性不足，请检查数据平稳性！")
        else:
            print("  [✓] 数据子集反驳通过：θ 在各时段子集上收敛稳定")

    out_path = os.path.join(OUT_DIR, "refutation_data_subset.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")
    return df_out


# ================================================================
#  主函数：命令行参数解析与调度
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="XIN_2 因果推断反驳性实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run_refutation_xin2.py --mode placebo
  python run_refutation_xin2.py --mode placebo --n_permutations 10

  python run_refutation_xin2.py --mode random_confounder
  python run_refutation_xin2.py --mode random_confounder --n_confounders 8 --n_repeats 5

  python run_refutation_xin2.py --mode data_subset
  python run_refutation_xin2.py --mode data_subset --n_subsets 10 --subset_frac 0.75

  python run_refutation_xin2.py --mode all   # 按顺序运行全部三种实验
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["placebo", "random_confounder", "data_subset", "all"],
        help=(
            "选择反驳实验模式：\n"
            "  placebo           - 安慰剂实验（随机排列 D）\n"
            "  random_confounder - 随机混杂变量实验（注入噪声列）\n"
            "  data_subset       - 数据子集实验（滑动时序窗口）\n"
            "  all               - 依次运行全部三种实验"
        ),
    )

    # 安慰剂参数
    parser.add_argument(
        "--n_permutations", type=int, default=5,
        help="[placebo] 每个操作变量的随机排列次数（默认 5）"
    )

    # 随机混杂参数
    parser.add_argument(
        "--n_confounders", type=int, default=5,
        help="[random_confounder] 每次注入的噪声列数量（默认 5）"
    )
    parser.add_argument(
        "--n_repeats", type=int, default=3,
        help="[random_confounder] 每个操作变量的重复注入次数（默认 3）"
    )

    # 数据子集参数
    parser.add_argument(
        "--n_subsets", type=int, default=8,
        help="[data_subset] 子集数量（默认 8）"
    )
    parser.add_argument(
        "--subset_frac", type=float, default=0.8,
        help="[data_subset] 每个子集使用的数据比例（默认 0.8）"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f" XIN_2 因果推断反驳性实验  |  模式: {args.mode.upper()}")
    print(f" 设备: {DEVICE}")
    print("=" * 70)

    # 加载数据（三种实验共用）
    df, operable_in_df, observable_in_df = build_xin2_data()
    ops    = sorted(operable_in_df  & set(df.columns))
    states = sorted(observable_in_df & set(df.columns))
    print(f"操作变量 {len(ops)} 个，状态变量 {len(states)} 个\n")

    mode = args.mode

    if mode in ("placebo", "all"):
        run_placebo(df, ops, states, n_permutations=args.n_permutations)

    if mode in ("random_confounder", "all"):
        run_random_confounder(
            df, ops, states,
            n_confounders=args.n_confounders,
            n_repeats=args.n_repeats,
        )

    if mode in ("data_subset", "all"):
        run_data_subset(
            df, ops, states,
            n_subsets=args.n_subsets,
            subset_frac=args.subset_frac,
        )

    print("\n" + "=" * 70)
    print(" 全部实验完成，结果保存至：")
    print(f"  安慰剂实验: {PLACEBO_OUT_DIR}")
    print(f"  随机混杂变量实验: {RANDOM_CONFOUNDER_OUT_DIR}")
    print(f"  数据子集实验: {DATA_SUBSET_OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
