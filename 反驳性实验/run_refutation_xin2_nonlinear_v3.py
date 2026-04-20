"""
run_refutation_xin2_nonlinear_v3.py
=================================
XIN_2 因果推断反驳实验 v3-nonlinear ——「两阶段解耦 VAE-DML + R-Learner 非线性因果效应估计」

═══════════════════════════════════════════════════════════════════
  核心创新：在 v3 两阶段解耦 VAE-DML 基础上增加 R-Learner 非线性估计
═══════════════════════════════════════════════════════════════════

  本脚本继承 v3 的全部架构和训练流程（两阶段解耦 VAE-DML），
  并在计算残差之后额外引入 R-Learner 作为第二种因果效应估计器：

  【Standard DML θ】（与 v3 完全一致）
    θ_DML = Cov(res_Y, res_D) / Var(res_D)
    假设线性工作点成立

  【R-Learner θ】（新增，处理非线性情况）
    伪结果 W = res_Y / (res_D + ε)
    权重    w = res_D²
    在 VAE 潜表征 μ 上拟合加权 GBM：W ~ μ，权重 = w
    θ_R-Learner = Σ(w_i · τ̂(μ_i)) / Σ(w_i)
    允许因果效应 τ(x) 关于 x 非线性

  当真实因果效应为常数时，θ_DML ≈ θ_R-Learner；
  当存在异质性（treatment effect heterogeneity）时，R-Learner 更准确。
  两个估计量同时报告，便于对比和诊断。

  ────────────────────────────────────────────────────────────────
  除 R-Learner 估计步骤外，本脚本的所有代码（数据加载、DAG过滤、
  VAE 架构、两阶段训练、残差计算、Bootstrap、并行调度、断点续传）
  均与 run_refutation_xin2_v3.py 完全一致。
  ────────────────────────────────────────────────────────────────

用法：
  python run_refutation_xin2_nonlinear_v3.py --mode stability --sample_size 2000 --n_bootstrap 3
  python run_refutation_xin2_nonlinear_v3.py --mode stability
  python run_refutation_xin2_nonlinear_v3.py --mode random_confounder --workers 4
  python run_refutation_xin2_nonlinear_v3.py --mode all --workers 6
"""

import argparse
import hashlib
import json
import os
import warnings
import concurrent.futures

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        items = list(it)
        print(f"[{kw.get('desc', '')}] 共 {len(items)} 个任务（建议 pip install tqdm）")
        return items


# ═══════════════════════════════════════════════════════════════════
#  路径配置（基于仓库根目录，自动适配 Linux / Windows / macOS）
# ═══════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))       # 反驳性实验/
REPO_ROOT = os.path.dirname(BASE_DIR)                        # 仓库根目录
DATA_DIR  = os.path.join(REPO_ROOT, "data")                  # data/

# ── 新数据管线产物（由 data_processing/ 脚本生成）─────────────────
#   优先读取 merge_final.py 输出的已对齐建模宽表；
#   若不存在，则回退到 X + Y 分别读取并在脚本内对齐。
MODELING_DATASET_XIN2 = os.path.join(DATA_DIR, "modeling_dataset_xin2_final.parquet")
X_PARQUET             = os.path.join(DATA_DIR, "X_features_final.parquet")
Y_PARQUET             = os.path.join(DATA_DIR, "y_target_final.parquet")

# ── 操作性分类表（classify_operability.py 输出）──────────────────
#   可通过 --operability-csv 命令行参数覆盖
DEFAULT_OPERABILITY_CSV = os.path.join(
    REPO_ROOT, "数据预处理",
    "数据与处理结果-分阶段-去共线性后",
    "non_collinear_representative_vars_operability.csv",
)

# ── 实验结果输出目录 ────────────────────────────────────────────
PLACEBO_OUT_DIR           = os.path.join(BASE_DIR, "安慰剂实验")
RANDOM_CONFOUNDER_OUT_DIR = os.path.join(BASE_DIR, "随机混杂变量实验")
DATA_SUBSET_OUT_DIR       = os.path.join(BASE_DIR, "数据子集实验")
STABILITY_OUT_DIR         = os.path.join(BASE_DIR, "稳定性诊断")

for _d in [PLACEBO_OUT_DIR, RANDOM_CONFOUNDER_OUT_DIR,
           DATA_SUBSET_OUT_DIR, STABILITY_OUT_DIR]:
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
#  超参
# ═══════════════════════════════════════════════════════════════════
SEQ_LEN             = 6
EMBARGO_GAP         = 4
K_FOLDS             = 4
MAX_EPOCHS_VAE      = 60     # Stage 1：VAE 表征学习最大轮数
MAX_EPOCHS_HEAD     = 40     # Stage 2：预测头最大轮数（更快收敛）
PATIENCE            = 8
MIN_TRAIN_SIZE      = 100
MIN_VALID_RESIDUALS = 50
F_STAT_THRESHOLD    = 10.0

# ─── v3 核心超参 ──────────────────────────────────────────────────
LATENT_DIM      = 32     # 扩大潜变量维度（原版 8，v3→32）
BETA_KL         = 0.1    # KL 权重上限（原版 0.5，降低防止 posterior collapse）
ANNEAL_EPOCHS   = 20     # KL 退火轮数：前 20 轮 β 从 0 线性升至 BETA_KL
HIDDEN_DIM_ENC  = 64     # LSTM encoder 隐层维度
NUM_LSTM_LAYERS = 2      # LSTM 堆叠层数
HIDDEN_DIM_HEAD = 32     # 预测头 MLP 隐层维度
GRAD_CLIP       = 1.0    # 梯度裁剪上限
N_BOOTSTRAP     = 5      # Bootstrap 重复次数
CV_WARN         = 0.30   # CV 超过此值视为不稳定
SIGN_RATE_MIN   = 0.70   # 符号一致率低于此值视为不可信

# ─── R-Learner 超参 ──────────────────────────────────────────────
RLEARNER_N_ESTIMATORS    = 100
RLEARNER_MAX_DEPTH       = 5
RLEARNER_MIN_SAMPLES_LEAF = 5
RLEARNER_LR              = 0.1
RLEARNER_SUBSAMPLE       = 0.8


# ══════════════════════════════════════════════════════════════════
#  DAG 因果角色过滤（v3.1 新增）
# ══════════════════════════════════════════════════════════════════

# 默认路径：DAG 分析脚本输出的角色明细 CSV
# 用户可通过 --dag-roles-csv 覆盖
DEFAULT_DAG_ROLES_CSV = os.path.join(
    REPO_ROOT, "DAG图分析", "DAG解析结果",
    # 替换为实际 GraphML 文件名对应的输出，例如：
    # "xin2_dag_Roles_Table.csv"
    ""  # 留空表示未指定，回退到纯相关性筛选
)


def load_dag_roles(csv_path: str) -> dict:
    """
    加载 DAG 角色明细表（analyze_dag_causal_roles_v4_1.py 的输出）。

    CSV 必须包含列：Treatment_T, Role, Node_Name
    Role 取值：0-Direct, 1-Confounder, 2-Mediator, 3-Collider, 4-Instrument

    返回：
      dag_roles[T_name] = {
          "confounders": set(...),
          "mediators":   set(...),
          "colliders":   set(...),
          "instruments": set(...),
      }
    如果文件不存在或为空，返回空 dict（回退到纯相关性筛选）。
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_cols = {"Treatment_T", "Role", "Node_Name"}
    if not required_cols.issubset(set(df.columns)):
        print(f"[警告] DAG 角色表缺少必要列 {required_cols - set(df.columns)}，跳过 DAG 过滤")
        return {}

    role_map = {
        "1-Confounder": "confounders",
        "2-Mediator":   "mediators",
        "3-Collider":   "colliders",
        "4-Instrument": "instruments",
    }

    dag_roles = {}
    for _, row in df.iterrows():
        t_name = str(row["Treatment_T"]).strip()
        role   = str(row["Role"]).strip()
        node   = str(row["Node_Name"]).strip()
        if role not in role_map:
            continue  # 跳过 0-Direct 等
        if t_name not in dag_roles:
            dag_roles[t_name] = {
                "confounders": set(),
                "mediators":   set(),
                "colliders":   set(),
                "instruments": set(),
            }
        dag_roles[t_name][role_map[role]].add(node)

    print(f"[DAG过滤] 已加载 {len(dag_roles)} 个操作变量的因果角色信息")
    return dag_roles


def build_safe_x_with_dag(op: str, df: pd.DataFrame, states: list,
                           dag_roles: dict) -> list:
    """
    构建控制变量集 safe_x，整合 DAG 因果角色过滤。

    策略：
      1. 先用原始相关性/滞后筛选得到候选集 candidate_x
      2. 如果 dag_roles 中有该操作变量的角色信息：
         a) 从候选集中剔除：instruments, colliders, mediators
         b) 只保留：confounders（以及 DAG 中未出现的变量，保守保留）
      3. 如果 dag_roles 为空（无 DAG 信息），回退到原始 get_safe_x()

    为什么 DAG 未覆盖的变量保守保留？
      DAG 分析的图可能不包含所有 329 个变量（因果发现算法可能只保留
      显著边），未出现在 DAG 中的变量无法判定角色，保留比剔除更安全。
    """
    # Step 1: 原始相关性筛选（与原版 get_safe_x 完全一致）
    candidate_x = get_safe_x(op, df, states)

    # Step 2: DAG 角色过滤
    if not dag_roles or op not in dag_roles:
        # 无 DAG 信息，回退
        return candidate_x

    roles = dag_roles[op]
    # 必须剔除的角色集合
    excluded_details = {"instrument": [], "collider": [], "mediator": []}

    filtered_x = []
    for var in candidate_x:
        if var in roles["instruments"]:
            excluded_details["instrument"].append(var)
        elif var in roles["colliders"]:
            excluded_details["collider"].append(var)
        elif var in roles["mediators"]:
            excluded_details["mediator"].append(var)
        else:
            # 是 confounder 或 DAG 中未出现（保守保留）
            filtered_x.append(var)

    n_excluded = len(candidate_x) - len(filtered_x)
    if n_excluded > 0:
        parts = []
        for role_name, vars_list in excluded_details.items():
            if vars_list:
                parts.append(f"{role_name}={len(vars_list)}")
        print(f"  [DAG过滤] {op}: 剔除 {n_excluded} 个变量 ({', '.join(parts)})，"
              f"保留 {len(filtered_x)} 个控制变量")

    return filtered_x


# ═══════════════════════════════════════════════════════════════════
#  模型定义
# ═══════════════════════════════════════════════════════════════════

class VAEEncoder(nn.Module):
    """
    VAE 编码器：X 序列 → (μ, logvar)
    推断时只用 μ（mean encoding），不采样，确保确定性。
    """
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, HIDDEN_DIM_ENC,
            batch_first=True,
            num_layers=NUM_LSTM_LAYERS,
            dropout=0.1 if NUM_LSTM_LAYERS > 1 else 0.0,
        )
        self.norm  = nn.LayerNorm(HIDDEN_DIM_ENC)
        self.fc    = nn.Sequential(nn.Linear(HIDDEN_DIM_ENC, 32), nn.SiLU())
        self.fc_mu     = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x: torch.Tensor):
        _, (h_n, _) = self.lstm(x)
        h = self.norm(h_n[-1])
        h = self.fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """推断专用：只返回 μ，不采样（确定性）"""
        mu, _ = self.forward(x)
        return mu


class VAEDecoder(nn.Module):
    """VAE 解码器：z → X 重建"""
    def __init__(self, latent_dim: int, seq_len: int, input_dim: int):
        super().__init__()
        self.seq_len   = seq_len
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, input_dim * seq_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, self.seq_len, self.input_dim)


class PredHead(nn.Module):
    """
    确定性预测头：μ → 标量预测
    Stage 2 中 head_Y 和 head_D 各自独立实例，互不共享梯度。
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM_HEAD), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM_HEAD, 1),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
#  阶段一训练：VAE 表征学习（encoder + decoder，不涉及 Y/D）
# ═══════════════════════════════════════════════════════════════════
def _train_vae_stage1(encoder: VAEEncoder, decoder: VAEDecoder,
                      X_train: torch.Tensor, device) -> None:
    """
    训练 VAE encoder + decoder：
      Loss = MSE_recon + β_anneal(epoch) × KL
    KL 退火：前 ANNEAL_EPOCHS 轮 β 从 0 线性升至 BETA_KL，
    防止早期 KL 惩罚过强导致 posterior collapse（latent 携带信息量下降）。
    训练完成后 encoder 和 decoder 的 requires_grad 已可被外部冻结。
    """
    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=0.002)
    loader    = DataLoader(TensorDataset(X_train), batch_size=256, shuffle=False)

    val_split = max(1, int(len(X_train) * 0.1))
    X_iv      = X_train[-val_split:]
    best_loss = float("inf")
    pat_cnt   = 0
    best_enc  = None
    best_dec  = None

    for epoch in range(MAX_EPOCHS_VAE):
        # KL 退火系数
        beta = BETA_KL * min(1.0, epoch / max(1, ANNEAL_EPOCHS))

        encoder.train(); decoder.train()
        for (bx,) in loader:
            optimizer.zero_grad()
            mu, logvar = encoder(bx)
            # 重参数化：训练时采样，推断时不用此路径
            std = torch.exp(0.5 * logvar)
            z   = mu + std * torch.randn_like(std)
            x_recon  = decoder(z)
            loss_rec = nn.functional.mse_loss(x_recon, bx)
            loss_kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss     = loss_rec + beta * loss_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()

        # 验证（也用采样 z，与训练一致）
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            mu_iv, lv_iv = encoder(X_iv)
            std_iv = torch.exp(0.5 * lv_iv)
            z_iv   = mu_iv + std_iv * torch.randn_like(std_iv)
            xr_iv  = decoder(z_iv)
            vl = (nn.functional.mse_loss(xr_iv, X_iv)
                  - 0.5 * beta * torch.mean(1 + lv_iv - mu_iv.pow(2) - lv_iv.exp())).item()

        if vl < best_loss - 1e-5:
            best_loss = vl
            pat_cnt   = 0
            best_enc  = {k: v.clone() for k, v in encoder.state_dict().items()}
            best_dec  = {k: v.clone() for k, v in decoder.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE:
                break

    if best_enc:
        encoder.load_state_dict(best_enc)
    if best_dec:
        decoder.load_state_dict(best_dec)


# ═══════════════════════════════════════════════════════════════════
#  阶段二训练：预测头（在冻结的 μ 上训练）
# ═══════════════════════════════════════════════════════════════════
def _train_head_stage2(head: PredHead, mu_train: torch.Tensor,
                       target_train: torch.Tensor, device) -> None:
    """
    在冻结 encoder 输出的 μ 上训练单个预测头（MSE loss）。
    mu_train 已在外部计算完毕（encoder.encode_mean(X_train)），
    梯度不会流回 encoder。
    """
    optimizer = optim.Adam(head.parameters(), lr=0.002)
    loader    = DataLoader(TensorDataset(mu_train, target_train),
                           batch_size=256, shuffle=False)
    val_split = max(1, int(len(mu_train) * 0.1))
    mu_iv     = mu_train[-val_split:]
    t_iv      = target_train[-val_split:]
    best_loss = float("inf")
    pat_cnt   = 0
    best_state = None

    for _ in range(MAX_EPOCHS_HEAD):
        head.train()
        for bmu, bt in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(head(bmu), bt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
            optimizer.step()

        head.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(head(mu_iv), t_iv).item()
        if vl < best_loss - 1e-5:
            best_loss  = vl
            pat_cnt    = 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE:
                break

    if best_state:
        head.load_state_dict(best_state)


# ═══════════════════════════════════════════════════════════════════
#  确定性哈希种子
# ═══════════════════════════════════════════════════════════════════
def _op_seed(op: str) -> int:
    return int(hashlib.md5(op.encode()).hexdigest()[:8], 16) % 100000


# ═══════════════════════════════════════════════════════════════════
#  核心训练函数：两阶段解耦 VAE-DML + R-Learner 非线性估计
# ═══════════════════════════════════════════════════════════════════
def train_one_op(op: str, df: pd.DataFrame, safe_x: list,
                 override_D=None, n_bootstrap: int = N_BOOTSTRAP):
    """
    两阶段解耦 VAE-DML + R-Learner 核心函数。

    在 v3 的基础上，每次 bootstrap 额外计算 R-Learner θ：
      1. 累积各 fold 的残差 (res_Y, res_D) 和潜表征 μ_vl
      2. 标准 DML θ = Cov(res_Y, res_D) / Var(res_D)
      3. R-Learner θ：
         - 伪结果 W = res_Y / (res_D + ε)
         - 权重   w = res_D²
         - 在 μ_vl 上拟合加权 GBM
         - θ_RL = 加权均值 Σ(w_i · τ̂(μ_i)) / Σ(w_i)

    返回：
      (theta_dml_med, theta_rl_med, p_dml, p_rl, SE_dml, SE_rl,
       n_avg, f_med, cv_dml, cv_rl, sign_rate_dml, sign_rate_rl)
    或 None（工具弱/样本不足/多数失败）
    """
    device = torch.device("cpu")

    # ── 标准化 ─────────────────────────────────────────────────────
    X_raw  = df[safe_x].values.astype(np.float32)
    X_mat  = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)

    Y_raw  = df["Y_grade"].values.astype(np.float32)
    D_raw  = (df[op].values if override_D is None
              else np.asarray(override_D, dtype=np.float32))
    Y_mean, Y_std = float(Y_raw.mean()), float(Y_raw.std()) + 1e-8
    D_mean, D_std = float(D_raw.mean()), float(D_raw.std()) + 1e-8
    Y_mat  = (Y_raw - Y_mean) / Y_std
    D_mat  = (D_raw - D_mean) / D_std

    # ── 滑动窗口序列 ────────────────────────────────────────────────
    seqs_X, tgt_Y, tgt_D = [], [], []
    for i in range(len(X_mat) - SEQ_LEN):
        seqs_X.append(X_mat[i: i + SEQ_LEN])
        tgt_Y.append(Y_mat[i + SEQ_LEN])
        tgt_D.append(D_mat[i + SEQ_LEN])
    seqs_X = np.array(seqs_X, dtype=np.float32)
    tgt_Y  = np.array(tgt_Y,  dtype=np.float32)
    tgt_D  = np.array(tgt_D,  dtype=np.float32)
    N          = len(seqs_X)
    block_size = N // K_FOLDS
    input_dim  = len(safe_x)

    # ── Bootstrap 外循环 ────────────────────────────────────────────
    op_base_seed = _op_seed(op)
    theta_dml_list, theta_rl_list = [], []
    f_list, n_list = [], []

    for boot_i in range(n_bootstrap):
        base_seed     = boot_i * 99991 + op_base_seed
        all_res_Y, all_res_D, all_mu_vl = [], [], []
        any_valid_fold = False

        for k in range(1, K_FOLDS):
            train_end = k * block_size - EMBARGO_GAP
            if train_end < MIN_TRAIN_SIZE:
                continue
            val_start = k * block_size
            val_end   = (k + 1) * block_size if k < K_FOLDS - 1 else N

            torch.manual_seed(base_seed * 100 + k)
            np.random.seed((base_seed * 100 + k) % (2**31))

            Xtr_np = seqs_X[:train_end]
            Xtr    = torch.tensor(Xtr_np).to(device)
            Ytr    = torch.tensor(tgt_Y[:train_end]).to(device)
            Dtr    = torch.tensor(tgt_D[:train_end]).to(device)
            Xvl    = torch.tensor(seqs_X[val_start:val_end]).to(device)
            Yvl    = tgt_Y[val_start:val_end]
            Dvl    = tgt_D[val_start:val_end]

            # ── Stage 1：训练 VAE（只重建 X，不涉及 Y/D）──────────
            encoder = VAEEncoder(input_dim, LATENT_DIM).to(device)
            decoder = VAEDecoder(LATENT_DIM, SEQ_LEN, input_dim).to(device)
            _train_vae_stage1(encoder, decoder, Xtr, device)

            # 冻结 encoder（Stage 2 的梯度不流回 encoder）
            for p in encoder.parameters():
                p.requires_grad_(False)
            encoder.eval()

            # ── 计算训练集的 μ（确定性，不采样）──────────────────
            with torch.no_grad():
                mu_tr = encoder.encode_mean(Xtr)   # shape: [train_end, LATENT_DIM]
                mu_vl = encoder.encode_mean(Xvl)   # shape: [val_size, LATENT_DIM]

            # ── Stage 2：独立训练 head_Y 和 head_D ───────────────
            head_Y = PredHead(LATENT_DIM).to(device)
            head_D = PredHead(LATENT_DIM).to(device)
            _train_head_stage2(head_Y, mu_tr, Ytr, device)
            _train_head_stage2(head_D, mu_tr, Dtr, device)

            # ── 推断：用 μ（确定性），计算残差 ────────────────────
            head_Y.eval(); head_D.eval()
            with torch.no_grad():
                pY = head_Y(mu_vl).cpu().numpy()
                pD = head_D(mu_vl).cpu().numpy()

            all_res_Y.extend(Yvl - pY)
            all_res_D.extend(Dvl - pD)
            # 累积潜表征用于 R-Learner
            all_mu_vl.append(mu_vl.cpu().numpy())
            any_valid_fold = True

        if not any_valid_fold or len(all_res_Y) < MIN_VALID_RESIDUALS:
            continue

        res_Y  = np.array(all_res_Y, dtype=np.float64)
        res_D  = np.array(all_res_D, dtype=np.float64)
        mu_all = np.concatenate(all_mu_vl, axis=0).astype(np.float64)

        # 去离群（3σ）
        mask  = ((np.abs(res_Y) < 3 * res_Y.std()) &
                 (np.abs(res_D) < 3 * res_D.std()))
        res_Y, res_D, mu_all = res_Y[mask], res_D[mask], mu_all[mask]
        if len(res_D) < MIN_VALID_RESIDUALS:
            continue

        # 强制去中心化
        res_Y -= res_Y.mean()
        res_D -= res_D.mean()
        n      = len(res_D)

        # F 统计量（工具强度）
        var_D  = np.var(res_D)
        f_stat = var_D / (D_std ** 2 + 1e-12) * n
        if f_stat < F_STAT_THRESHOLD:
            continue

        # ── Standard DML theta（协方差矩阵方式，与 v3 一致）────────
        cov_mat       = np.cov(res_D, res_Y)
        theta_std_dml = cov_mat[0, 1] / (cov_mat[0, 0] + 1e-12)
        theta_dml     = theta_std_dml * (Y_std / D_std)

        # ── R-Learner theta（非线性估计）───────────────────────────
        eps       = 1e-8
        # 安全除法：对绝对值过小的 res_D 用 eps 替换，防止除零
        safe_res_D = np.where(np.abs(res_D) < eps, eps, res_D)
        W_pseudo   = res_Y / safe_res_D
        w_weights  = res_D ** 2

        # 权重归一化（防止 GBM 数值问题）
        w_sum = w_weights.sum()
        if w_sum < eps:
            # 权重全为零，R-Learner 退化为 DML
            theta_rl = theta_dml
        else:
            w_normed = w_weights / w_sum * len(w_weights)
            gbm = GradientBoostingRegressor(
                n_estimators=RLEARNER_N_ESTIMATORS,
                max_depth=RLEARNER_MAX_DEPTH,
                min_samples_leaf=RLEARNER_MIN_SAMPLES_LEAF,
                learning_rate=RLEARNER_LR,
                subsample=RLEARNER_SUBSAMPLE,
                random_state=(base_seed + 7) % (2**31),
            )
            gbm.fit(mu_all, W_pseudo, sample_weight=w_normed)
            tau_hat = gbm.predict(mu_all)
            # θ_R-Learner = 加权均值
            theta_std_rl = np.sum(w_weights * tau_hat) / (w_sum + eps)
            # 反标准化
            theta_rl = theta_std_rl * (Y_std / D_std)

        theta_dml_list.append(theta_dml)
        theta_rl_list.append(theta_rl)
        f_list.append(f_stat)
        n_list.append(n)

    # ── Bootstrap 聚合 ──────────────────────────────────────────────
    min_success = max(1, n_bootstrap // 2)
    if len(theta_dml_list) < min_success:
        return None

    # DML 聚合
    arr_dml     = np.array(theta_dml_list)
    theta_dml_med = float(np.median(arr_dml))
    std_dml     = float(np.std(arr_dml))
    cv_dml      = std_dml / (abs(theta_dml_med) + 1e-8)
    sr_dml      = float(np.mean(np.sign(arr_dml) == np.sign(theta_dml_med)))
    SE_dml      = max(std_dml, 1e-8)
    t_dml       = theta_dml_med / SE_dml
    n_avg       = int(np.mean(n_list))
    p_dml       = 2 * (1 - stats.t.cdf(abs(t_dml), df=n_avg - 1))
    f_med       = float(np.median(f_list))

    # R-Learner 聚合
    arr_rl      = np.array(theta_rl_list)
    theta_rl_med = float(np.median(arr_rl))
    std_rl      = float(np.std(arr_rl))
    cv_rl       = std_rl / (abs(theta_rl_med) + 1e-8)
    sr_rl       = float(np.mean(np.sign(arr_rl) == np.sign(theta_rl_med)))
    SE_rl       = max(std_rl, 1e-8)
    t_rl        = theta_rl_med / SE_rl
    p_rl        = 2 * (1 - stats.t.cdf(abs(t_rl), df=n_avg - 1))

    return (theta_dml_med, theta_rl_med, p_dml, p_rl, SE_dml, SE_rl,
            n_avg, f_med, cv_dml, cv_rl, sr_dml, sr_rl)


# ═══════════════════════════════════════════════════════════════════
#  辅助：构建 safe_x（与原版完全一致）
# ═══════════════════════════════════════════════════════════════════
def get_safe_x(op: str, df: pd.DataFrame, states: list) -> list:
    y_vals = df["Y_grade"].values
    x_vals = df[op].values
    best_t_r, best_t_lag = 0.0, 0
    for lag in range(1, 15):
        r = abs(np.corrcoef(x_vals[:-lag], y_vals[lag:])[0, 1])
        if r > best_t_r:
            best_t_r, best_t_lag = r, lag
    safe_x = []
    for st in states:
        s_vals   = df[st].values
        best_s_r = 0.0
        best_s_l = 0
        for lag in range(1, 15):
            r = abs(np.corrcoef(s_vals[:-lag], y_vals[lag:])[0, 1])
            if r > best_s_r:
                best_s_r, best_s_l = r, lag
        if best_s_r > 0.05 and best_s_l >= best_t_lag:
            safe_x.append(st)
    return safe_x


# ═══════════════════════════════════════════════════════════════════
#  断点续传工具（与原版完全一致）
# ═══════════════════════════════════════════════════════════════════
def _load_checkpoint(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["_key"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done

def _append_checkpoint(path: str, rec: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _read_all_records(path: str) -> list:
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


# ═══════════════════════════════════════════════════════════════════
#  通用并行调度器（与原版完全一致）
# ═══════════════════════════════════════════════════════════════════
def _run_parallel(tasks: list, worker_fn, ckpt_path: str,
                  workers: int, desc: str = "任务") -> list:
    done_keys = _load_checkpoint(ckpt_path)
    pending   = [t for t in tasks if t["_key"] not in done_keys]
    skipped   = len(tasks) - len(pending)
    if skipped:
        print(f"  [断点续传] 跳过已完成 {skipped} 个，剩余 {len(pending)} 个")
    new_results = []
    if not pending:
        return new_results
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker_fn, t): t for t in pending}
        with tqdm(total=len(pending), desc=desc, ncols=80) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    task = futures[fut]
                    res  = {"_key": task["_key"], "_filtered": True,
                            "_reason": f"异常: {e}"}
                _append_checkpoint(ckpt_path, res)
                new_results.append(res)
                pbar.update(1)
    return new_results


# ═══════════════════════════════════════════════════════════════════
#  Worker 函数
# ═══════════════════════════════════════════════════════════════════

def _worker_placebo(task: dict) -> dict:
    op, perm_idx = task["op"], task["perm_idx"]
    df, states   = task["df"], task["states"]
    dag_roles    = task["dag_roles"]
    key          = task["_key"]
    if df[op].std() < 0.1:
        return {"_key": key, "_filtered": True, "_reason": "std<0.1"}
    safe_x = build_safe_x_with_dag(op, df, states, dag_roles)
    if len(safe_x) < 2:
        return {"_key": key, "_filtered": True, "_reason": "safe_x不足(DAG过滤后)"}
    rng       = np.random.default_rng(seed=perm_idx * 42 + _op_seed(op))
    D_placebo = rng.permutation(df[op].values.copy())
    result    = train_one_op(op, df, safe_x, override_D=D_placebo)
    if result is None:
        return {"_key": key, "_filtered": True, "_reason": "弱工具/样本不足"}
    (theta_dml, theta_rl, p_dml, p_rl, SE_dml, SE_rl,
     n, f, cv_dml, cv_rl, sr_dml, sr_rl) = result
    return {
        "_key":             key,   "_filtered": False,
        "操作节点":          op,    "排列索引":   perm_idx + 1,
        "θ_DML_安慰剂":     round(theta_dml, 5),
        "θ_RL_安慰剂":      round(theta_rl,  5),
        "P_DML":            round(p_dml, 4),
        "P_RL":             round(p_rl,  4),
        "SE_DML":           round(SE_dml, 5),
        "SE_RL":            round(SE_rl,  5),
        "CV_DML":           round(cv_dml, 4),
        "CV_RL":            round(cv_rl,  4),
        "符号一致率_DML":    round(sr_dml, 3),
        "符号一致率_RL":     round(sr_rl,  3),
        "有效残差数":        n,
        "F统计量":           round(f, 2),
        "显著_DML":          bool(p_dml < 0.05),
        "显著_RL":           bool(p_rl  < 0.05),
    }


def _worker_random_confounder(task: dict) -> dict:
    op, rep       = task["op"], task["rep"]
    n_confounders = task["n_confounders"]
    theta_dml_orig = task["theta_dml_orig"]
    theta_rl_orig  = task["theta_rl_orig"]
    SE_dml_orig   = task["SE_dml_orig"]
    SE_rl_orig    = task["SE_rl_orig"]
    safe_x_orig   = task["safe_x_orig"]
    df, key       = task["df"], task["_key"]

    rng        = np.random.default_rng(seed=rep * 1000 + _op_seed(op))
    df_noisy   = df.copy()
    noise_cols = []
    for nc in range(n_confounders):
        cname = f"__rc_{nc}__"
        df_noisy[cname] = rng.standard_normal(len(df_noisy))
        noise_cols.append(cname)
    safe_x_noisy = safe_x_orig + noise_cols
    result       = train_one_op(op, df_noisy, safe_x_noisy)
    if result is None:
        return {"_key": key, "_filtered": True, "_reason": "弱工具/样本不足"}

    (theta_dml_p, theta_rl_p, p_dml_p, p_rl_p, SE_dml_p, SE_rl_p,
     n_p, f_p, cv_dml_p, cv_rl_p, sr_dml_p, sr_rl_p) = result

    # DML 反驳判断（t_diff < 2.0）
    delta_dml       = abs(theta_dml_p - theta_dml_orig)
    se_comb_dml     = float(np.sqrt(SE_dml_p**2 + SE_dml_orig**2)) + 1e-12
    t_diff_dml      = delta_dml / se_comb_dml
    sign_ok_dml     = bool(np.sign(theta_dml_p) == np.sign(theta_dml_orig))
    near_zero_dml   = abs(theta_dml_orig) < 3 * SE_dml_orig
    passed_dml      = bool(t_diff_dml < 2.0 and (sign_ok_dml or near_zero_dml))
    rel_dev_dml     = delta_dml / (abs(theta_dml_orig) + 1e-8)

    # R-Learner 反驳判断（t_diff < 2.0）
    delta_rl       = abs(theta_rl_p - theta_rl_orig)
    se_comb_rl     = float(np.sqrt(SE_rl_p**2 + SE_rl_orig**2)) + 1e-12
    t_diff_rl      = delta_rl / se_comb_rl
    sign_ok_rl     = bool(np.sign(theta_rl_p) == np.sign(theta_rl_orig))
    near_zero_rl   = abs(theta_rl_orig) < 3 * SE_rl_orig
    passed_rl      = bool(t_diff_rl < 2.0 and (sign_ok_rl or near_zero_rl))
    rel_dev_rl     = delta_rl / (abs(theta_rl_orig) + 1e-8)

    return {
        "_key":              key,            "_filtered":   False,
        "操作节点":           op,             "重复索引":     rep + 1,
        "θ_DML_原始":        round(theta_dml_orig, 5),
        "θ_RL_原始":         round(theta_rl_orig,  5),
        "θ_DML_注入噪声":    round(theta_dml_p,    5),
        "θ_RL_注入噪声":     round(theta_rl_p,     5),
        "相对偏差_DML":      round(rel_dev_dml,    4),
        "相对偏差_RL":       round(rel_dev_rl,     4),
        "t_diff_DML":        round(t_diff_dml,     3),
        "t_diff_RL":         round(t_diff_rl,      3),
        "方向一致_DML":      sign_ok_dml,
        "方向一致_RL":       sign_ok_rl,
        "P_DML":             round(p_dml_p, 4),
        "P_RL":              round(p_rl_p,  4),
        "SE_DML":            round(SE_dml_p, 5),
        "SE_RL":             round(SE_rl_p,  5),
        "CV_DML":            round(cv_dml_p, 4),
        "CV_RL":             round(cv_rl_p,  4),
        "符号一致率_DML":    round(sr_dml_p, 3),
        "符号一致率_RL":     round(sr_rl_p,  3),
        "有效残差数":        n_p,
        "F统计量":           round(f_p, 2),
        "通过反驳_DML":      passed_dml,
        "通过反驳_RL":       passed_rl,
    }


def _worker_data_subset(task: dict) -> dict:
    op, sub_idx = task["op"], task["sub_idx"]
    start, end  = task["start"], task["end"]
    safe_x, df  = task["safe_x"], task["df"]
    key         = task["_key"]
    df_sub = df.iloc[start:end].copy()
    if len(df_sub) < SEQ_LEN + K_FOLDS * MIN_TRAIN_SIZE:
        return {"_key": key, "_filtered": True, "_reason": "样本不足"}
    result = train_one_op(op, df_sub, safe_x)
    if result is None:
        return {"_key": key, "_filtered": True, "_reason": "弱工具/样本不足"}
    (theta_dml, theta_rl, p_dml, p_rl, SE_dml, SE_rl,
     n, f, cv_dml, cv_rl, sr_dml, sr_rl) = result
    return {
        "_key":             key,          "_filtered": False,
        "操作节点":          op,           "子集索引":   sub_idx + 1,
        "时段起点":          start,        "时段终点":   end,
        "θ_DML_子集":       round(theta_dml, 5),
        "θ_RL_子集":        round(theta_rl,  5),
        "P_DML":            round(p_dml, 4),
        "P_RL":             round(p_rl,  4),
        "SE_DML":           round(SE_dml, 5),
        "SE_RL":            round(SE_rl,  5),
        "CV_DML":           round(cv_dml, 4),
        "CV_RL":            round(cv_rl,  4),
        "符号一致率_DML":    round(sr_dml, 3),
        "符号一致率_RL":     round(sr_rl,  3),
        "有效残差数":        n,
        "F统计量":           round(f, 2),
    }


# ═══════════════════════════════════════════════════════════════════
#  数据准备（对接 data_processing/ 新数据管线）
# ═══════════════════════════════════════════════════════════════════
def build_xin2_data(operability_csv: str = DEFAULT_OPERABILITY_CSV):
    """
    加载 XIN2 产线的建模数据。

    数据来源优先级：
      1. data/modeling_dataset_xin2_final.parquet（merge_final.py --line xin2 输出）
         → 已完成 X/Y/指标对齐，直接使用
      2. data/X_features_final.parquet + data/y_target_final.parquet
         → 手动对齐（与 merge_final.py 逻辑一致：±1min merge_asof）

    操作性分类（operable / observable）来自 classify_operability.py 输出的 CSV，
    包含 Group、Operability、Variable_Name 列。XIN2 对应 Group B+C。

    返回：
      (df_filtered, operable_in_df, observable_in_df)
    """

    # ── 读取操作性分类 ──────────────────────────────────────────────
    if not os.path.exists(operability_csv):
        raise FileNotFoundError(
            f"操作性分类文件不存在：{operability_csv}\n"
            f"请先运行 数据预处理/classify_operability.py 生成该文件，"
            f"或通过 --operability-csv 指定路径。"
        )
    op_df = pd.read_csv(operability_csv, encoding="utf-8-sig")
    op_df["Group"] = op_df["Group"].str.strip().str.upper()
    xin2_df = op_df[op_df["Group"].isin(["B", "C"])].copy()
    operable_set   = set(
        xin2_df[xin2_df["Operability"].str.strip() == "operable"]["Variable_Name"].str.strip()
    )
    observable_set = set(
        xin2_df[xin2_df["Operability"].str.strip() == "observable"]["Variable_Name"].str.strip()
    )
    print(f"[数据准备] Group B+C 共 {len(operable_set | observable_set)} 个变量，"
          f"operable={len(operable_set)}，observable={len(observable_set)}")

    # ── 路径 1：优先读取已对齐的建模宽表 ────────────────────────────
    if os.path.exists(MODELING_DATASET_XIN2):
        print(f"[数据准备] 读取已对齐建模宽表：{MODELING_DATASET_XIN2}")
        df = pd.read_parquet(MODELING_DATASET_XIN2)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "time"

        # Y 列：merge_final.py 输出列名为 y_fx_xin2，统一重命名为 Y_grade
        if "y_fx_xin2" in df.columns:
            df = df.rename(columns={"y_fx_xin2": "Y_grade"})
        elif "Y_grade" not in df.columns:
            raise KeyError(
                f"建模宽表 {MODELING_DATASET_XIN2} 中未找到 'y_fx_xin2' 或 'Y_grade' 列"
            )

        # 如果存在 y_fx_xin1 列（完整宽表而非 xin2 子集），也删除以避免混入
        if "y_fx_xin1" in df.columns:
            df = df.drop(columns=["y_fx_xin1"])

        # 剔除 Y_grade 为 NaN 的行
        df = df.dropna(subset=["Y_grade"])

    # ── 路径 2：回退到 X + Y 分别读取 ──────────────────────────────
    elif os.path.exists(X_PARQUET) and os.path.exists(Y_PARQUET):
        print(f"[数据准备] 未找到已对齐宽表，回退到分别读取 X + Y")
        print(f"  X: {X_PARQUET}")
        print(f"  Y: {Y_PARQUET}")

        X = pd.read_parquet(X_PARQUET)
        X.index = pd.to_datetime(X.index).tz_localize(None)
        X.index.name = "time"
        X = X.sort_index()

        y = pd.read_parquet(Y_PARQUET)
        y.index = pd.to_datetime(y.index).tz_localize(None)
        y.index.name = "time"
        y = y.sort_index()

        if "y_fx_xin2" not in y.columns:
            raise KeyError(f"Y 文件 {Y_PARQUET} 中未找到 'y_fx_xin2' 列")

        y_xin2 = y[["y_fx_xin2"]].dropna()

        # ±1min merge_asof 对齐（与 merge_final.py 一致）
        y_reset = y_xin2.reset_index().sort_values("time")
        X_reset = X.reset_index().rename(columns={"time": "_time_x"}).sort_values("_time_x")
        merged = pd.merge_asof(
            y_reset, X_reset,
            left_on="time", right_on="_time_x",
            direction="nearest", tolerance=pd.Timedelta("1min"),
        )
        merged = merged.drop(columns=["_time_x"])
        merged = merged.set_index("time")
        merged = merged.rename(columns={"y_fx_xin2": "Y_grade"})
        merged = merged.dropna(subset=["Y_grade"])
        df = merged
    else:
        raise FileNotFoundError(
            f"未找到数据文件。请先运行 data_processing/ 下的预处理脚本：\n"
            f"  已对齐宽表（推荐）: {MODELING_DATASET_XIN2}\n"
            f"  或分别: {X_PARQUET} + {Y_PARQUET}\n"
            f"详见 data_processing/README.md"
        )

    # ── 列过滤 + 低方差剔除 ────────────────────────────────────────
    df = df.loc[:, (df.std() > 1e-4)]
    all_known_vars = operable_set | observable_set
    valid_cols  = [c for c in df.columns
                   if c in all_known_vars or c == "Y_grade"]
    df_filtered = df[valid_cols]

    cols_in_df       = set(df_filtered.columns) - {"Y_grade"}
    operable_in_df   = operable_set   & cols_in_df
    observable_in_df = observable_set & cols_in_df
    print(f"[数据准备] 最终 DataFrame：{df_filtered.shape}，"
          f"operable={len(operable_in_df)}，observable={len(observable_in_df)}")
    return df_filtered, operable_in_df, observable_in_df


# ═══════════════════════════════════════════════════════════════════
#  实验零：稳定性诊断
# ═══════════════════════════════════════════════════════════════════
def run_stability_diagnosis(df, ops, states, workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验零：稳定性诊断（{N_BOOTSTRAP} 次 Bootstrap）")
    print(f" 架构：两阶段解耦 VAE-DML + R-Learner  |  推断用 μ（确定性）")
    print(f" 稳定标准：CV < {CV_WARN}  且  sign_rate ≥ {SIGN_RATE_MIN}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    states_list = list(states)
    rows = []
    n_stable_dml = 0
    n_stable_rl  = 0
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            continue
        result = train_one_op(op, df, safe_x)
        if result is None:
            print(f"  [跳过] {op:<30s}  估计失败（弱工具/样本不足）")
            continue
        (theta_dml, theta_rl, p_dml, p_rl, SE_dml, SE_rl,
         n, f, cv_dml, cv_rl, sr_dml, sr_rl) = result
        stable_dml = (cv_dml < CV_WARN and sr_dml >= SIGN_RATE_MIN)
        stable_rl  = (cv_rl  < CV_WARN and sr_rl  >= SIGN_RATE_MIN)
        if stable_dml:
            n_stable_dml += 1
        if stable_rl:
            n_stable_rl += 1
        flag_dml = "✓" if stable_dml else "⚠"
        flag_rl  = "✓" if stable_rl  else "⚠"
        print(f"  {op:<30s}  θ_DML={theta_dml:+.5f}({flag_dml})  "
              f"θ_RL={theta_rl:+.5f}({flag_rl})  "
              f"CV_DML={cv_dml:.3f}  CV_RL={cv_rl:.3f}  "
              f"p_DML={p_dml:.4f}  p_RL={p_rl:.4f}")
        rows.append({
            "操作节点": op,
            "θ_DML_中位数": round(theta_dml, 5),
            "θ_RL_中位数":  round(theta_rl,  5),
            "P_DML": round(p_dml, 4), "P_RL": round(p_rl, 4),
            "SE_DML": round(SE_dml, 5), "SE_RL": round(SE_rl, 5),
            "CV_DML": round(cv_dml, 4), "CV_RL": round(cv_rl, 4),
            "符号一致率_DML": round(sr_dml, 3),
            "符号一致率_RL":  round(sr_rl,  3),
            "F统计量": round(f, 2),
            "稳定_DML": stable_dml, "稳定_RL": stable_rl,
        })
    df_out   = pd.DataFrame(rows)
    out_path = os.path.join(STABILITY_OUT_DIR, "stability_diagnosis_nonlinear_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not df_out.empty:
        total = len(df_out)
        print(f"\n[稳定性诊断汇总]")
        print(f"  DML:       稳定 {n_stable_dml}/{total} 个操作变量")
        print(f"  R-Learner: 稳定 {n_stable_rl}/{total} 个操作变量")
        print(f"  CV 均值    DML={df_out['CV_DML'].mean():.3f}  RL={df_out['CV_RL'].mean():.3f}  （目标 < {CV_WARN}）")
        print(f"  sign_rate  DML={df_out['符号一致率_DML'].mean():.2f}  RL={df_out['符号一致率_RL'].mean():.2f}  （目标 ≥ {SIGN_RATE_MIN}）")
        if n_stable_dml / total < 0.5 and n_stable_rl / total < 0.5:
            print("  [⚠] 超过一半操作变量不稳定，建议继续调参（LATENT_DIM, BETA_KL, ANNEAL_EPOCHS）")
        else:
            print("  [✓] 多数操作变量稳定，可运行反驳实验")
    print(f"结果已保存：{out_path}")
    return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验一：安慰剂反驳
# ═══════════════════════════════════════════════════════════════════
def run_placebo(df, ops, states, n_permutations=5, workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(" 实验一：安慰剂反驳实验（随机排列操作变量 D）")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    ckpt_path   = os.path.join(PLACEBO_OUT_DIR, "checkpoint_placebo_nonlinear_v3.jsonl")
    states_list = list(states)
    tasks = []
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        for perm_idx in range(n_permutations):
            tasks.append({"_key": f"{op}__perm{perm_idx}", "op": op,
                          "perm_idx": perm_idx, "df": df, "states": states_list,
                          "dag_roles": dag_roles})
    _run_parallel(tasks, _worker_placebo, ckpt_path, workers, desc="安慰剂")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 安慰剂实验无有效结果"); return df_out
    sig_rate_dml = df_out["显著_DML"].mean()
    sig_rate_rl  = df_out["显著_RL"].mean()
    print(f"\n[安慰剂汇总]")
    print(f"  DML:       θ均值={df_out['θ_DML_安慰剂'].mean():+.5f}  显著率={sig_rate_dml:.1%}（期望≈5%）")
    print(f"  R-Learner: θ均值={df_out['θ_RL_安慰剂'].mean():+.5f}  显著率={sig_rate_rl:.1%}（期望≈5%）")
    pass_dml = sig_rate_dml <= 0.2
    pass_rl  = sig_rate_rl  <= 0.2
    print(f"  DML:       {'[✓] 通过' if pass_dml else '[⚠] 显著率偏高，可能存在虚假相关'}")
    print(f"  R-Learner: {'[✓] 通过' if pass_rl  else '[⚠] 显著率偏高，可能存在虚假相关'}")
    out_path = os.path.join(PLACEBO_OUT_DIR, "refutation_placebo_nonlinear_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验二：随机混杂变量反驳
# ═══════════════════════════════════════════════════════════════════
def run_random_confounder(df, ops, states, n_confounders=5, n_repeats=1, workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验二：随机混杂变量反驳（注入 {n_confounders} 个随机噪声列）")
    print(f" v3-nonlinear：Stage 1 仅重建 X，Stage 2 头不受噪声重建影响")
    print(f" 判断标准：t_diff = |Δθ| / √(SE_orig²+SE_noisy²) < 2.0")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    ckpt_path   = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "checkpoint_rc_nonlinear_v3.jsonl")
    states_list = list(states)
    print("[预计算原始 θ（DML + R-Learner）...]")
    orig_thetas = {}
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            continue
        result = train_one_op(op, df, safe_x)
        if result is None:
            print(f"  [跳过] {op:<30s}  原始估计失败"); continue
        (theta_dml, theta_rl, p_dml, p_rl, SE_dml, SE_rl,
         n, f, cv_dml, cv_rl, sr_dml, sr_rl) = result
        if p_dml > 0.05 and p_rl > 0.05 and cv_dml > CV_WARN and cv_rl > CV_WARN:
            print(f"  [跳过] {op:<30s}  不显著且不稳定"); continue
        orig_thetas[op] = (theta_dml, theta_rl, SE_dml, SE_rl, safe_x)
        flag_dml = "⚠" if cv_dml > CV_WARN else "✓"
        flag_rl  = "⚠" if cv_rl  > CV_WARN else "✓"
        print(f"  {op:<30s}  θ_DML={theta_dml:+.5f}({flag_dml})  "
              f"θ_RL={theta_rl:+.5f}({flag_rl})")
    tasks = []
    for op, (theta_dml_orig, theta_rl_orig, SE_dml_orig, SE_rl_orig, safe_x_orig) in orig_thetas.items():
        for rep in range(n_repeats):
            tasks.append({
                "_key": f"{op}__rep{rep}", "op": op, "rep": rep,
                "n_confounders": n_confounders,
                "theta_dml_orig": theta_dml_orig,
                "theta_rl_orig":  theta_rl_orig,
                "SE_dml_orig":    SE_dml_orig,
                "SE_rl_orig":     SE_rl_orig,
                "safe_x_orig":    safe_x_orig, "df": df,
            })
    _run_parallel(tasks, _worker_random_confounder, ckpt_path, workers, desc="随机混杂")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 随机混杂实验无有效结果"); return df_out
    pass_rate_dml = df_out["通过反驳_DML"].mean()
    pass_rate_rl  = df_out["通过反驳_RL"].mean()
    print(f"\n[随机混杂汇总]")
    print(f"  DML:       反驳通过率 = {pass_rate_dml:.1%}（期望 ≥ 80%）")
    print(f"  R-Learner: 反驳通过率 = {pass_rate_rl:.1%}（期望 ≥ 80%）")
    print(f"  DML:       {'[✓] 通过' if pass_rate_dml >= 0.8 else '[⚠] 通过率偏低，θ 对随机噪声注入仍然敏感'}")
    print(f"  R-Learner: {'[✓] 通过' if pass_rate_rl  >= 0.8 else '[⚠] 通过率偏低，θ 对随机噪声注入仍然敏感'}")
    out_path = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "refutation_rc_nonlinear_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验三：数据子集反驳
# ═══════════════════════════════════════════════════════════════════
def run_data_subset(df, ops, states, n_subsets=8, subset_frac=0.8, workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验三：数据子集反驳（{n_subsets} 个子集，每个取 {subset_frac:.0%} 数据）")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    ckpt_path   = os.path.join(DATA_SUBSET_OUT_DIR, "checkpoint_ds_nonlinear_v3.jsonl")
    states_list = list(states)
    T = len(df); subset_len = int(T * subset_frac)
    step = max(1, (T - subset_len) // max(1, n_subsets - 1))
    tasks = []
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            continue
        for sub_idx in range(n_subsets):
            start = min(sub_idx * step, T - subset_len); end = start + subset_len
            tasks.append({"_key": f"{op}__sub{sub_idx}", "op": op, "sub_idx": sub_idx,
                          "start": start, "end": end, "safe_x": safe_x, "df": df})
    _run_parallel(tasks, _worker_data_subset, ckpt_path, workers, desc="数据子集")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 数据子集实验无有效结果"); return df_out
    stable_dml = stable_rl = total_ops = 0
    for op, grp in df_out.groupby("操作节点"):
        if len(grp) < 3: continue
        total_ops += 1
        arr_dml = grp["θ_DML_子集"].values
        arr_rl  = grp["θ_RL_子集"].values
        cv_dml  = np.std(arr_dml) / (abs(np.mean(arr_dml)) + 1e-8)
        cv_rl   = np.std(arr_rl)  / (abs(np.mean(arr_rl))  + 1e-8)
        sc_dml  = np.mean(np.sign(arr_dml) == np.sign(np.median(arr_dml)))
        sc_rl   = np.mean(np.sign(arr_rl)  == np.sign(np.median(arr_rl)))
        if cv_dml < 0.30 and sc_dml >= 0.70:
            stable_dml += 1
        if cv_rl < 0.30 and sc_rl >= 0.70:
            stable_rl += 1
    if total_ops:
        gp_dml = stable_dml / total_ops
        gp_rl  = stable_rl  / total_ops
        print(f"\n[数据子集汇总]")
        print(f"  DML:       全局稳定通过率 = {gp_dml:.1%}（期望 ≥ 70%）")
        print(f"  R-Learner: 全局稳定通过率 = {gp_rl:.1%}（期望 ≥ 70%）")
        print(f"  DML:       {'[✓] 通过' if gp_dml >= 0.70 else '[⚠] θ 跨时段稳定性不足'}")
        print(f"  R-Learner: {'[✓] 通过' if gp_rl  >= 0.70 else '[⚠] θ 跨时段稳定性不足'}")
    out_path = os.path.join(DATA_SUBSET_OUT_DIR, "refutation_ds_nonlinear_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="XIN_2 因果推断反驳实验 v3-nonlinear（两阶段解耦 VAE-DML + R-Learner）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
建议执行顺序：
  # 1. 小样本快速验证稳定性改善（几分钟）
  python run_refutation_xin2_nonlinear_v3.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 2. 全量稳定性诊断（确认 CV 普遍下降）
  python run_refutation_xin2_nonlinear_v3.py --mode stability

  # 3. 反驳实验
  python run_refutation_xin2_nonlinear_v3.py --mode random_confounder --workers 4
  python run_refutation_xin2_nonlinear_v3.py --mode placebo --n_permutations 5 --workers 4
  python run_refutation_xin2_nonlinear_v3.py --mode data_subset --workers 4

  # 4. 全部一次跑
  python run_refutation_xin2_nonlinear_v3.py --mode all --workers 6

断点续传：检查点含 _nonlinear_v3 后缀，与 v1/v2/v3 不冲突。
调参建议：先改 LATENT_DIM（尝试 16/32/64），再改 BETA_KL（尝试 0.01/0.05/0.1），
         再改 ANNEAL_EPOCHS（尝试 10/20/40）。
R-Learner 调参：RLEARNER_N_ESTIMATORS, RLEARNER_MAX_DEPTH, RLEARNER_LR。
        """,
    )
    p.add_argument("--mode", required=True,
                   choices=["stability", "placebo", "random_confounder", "data_subset", "all"])
    p.add_argument("--n_permutations", type=int, default=5)
    p.add_argument("--n_confounders",  type=int, default=5)
    p.add_argument("--n_repeats",      type=int, default=1)
    p.add_argument("--n_subsets",      type=int, default=8)
    p.add_argument("--subset_frac",    type=float, default=0.8)
    p.add_argument("--workers",        type=int, default=4)
    p.add_argument("--n_bootstrap",    type=int, default=N_BOOTSTRAP,
                   help=f"Bootstrap 次数（默认 {N_BOOTSTRAP}；调参时用 3 加速）")
    p.add_argument("--sample_size",    type=int, default=0,
                   help="截取最近 N 条数据调参（0=全量）")
    p.add_argument("--dag-roles-csv", type=str, default="",
                   help="DAG 角色明细 CSV 路径（analyze_dag_causal_roles_v4_1.py 的输出）。"
                        "若不指定，回退到纯相关性筛选（与 v3 行为一致）。")
    p.add_argument("--operability-csv", type=str, default=DEFAULT_OPERABILITY_CSV,
                   help="操作性分类 CSV 路径（classify_operability.py 的输出，"
                        f"默认 {DEFAULT_OPERABILITY_CSV}）。")
    return p.parse_args()


def main():
    args = parse_args()
    global N_BOOTSTRAP
    N_BOOTSTRAP = args.n_bootstrap

    print("=" * 70)
    print(f" XIN_2 因果推断反驳实验 v3-nonlinear  |  模式: {args.mode.upper()}")
    print(f" 设备: {DEVICE}  |  并行线程: {args.workers}  |  Bootstrap: {N_BOOTSTRAP}")
    print(f" 架构: 两阶段解耦 VAE-DML + R-Learner 非线性因果效应估计")
    print(f"   Stage1: VAEEncoder(LATENT={LATENT_DIM}) + Decoder  [BETA_KL={BETA_KL}, 退火={ANNEAL_EPOCHS}轮]")
    print(f"   Stage2: 独立 head_Y + head_D（在冻结 μ 上训练）")
    print(f"   DML:    θ = Cov(res_Y, res_D) / Var(res_D)（线性）")
    print(f"   R-Learner: 加权 GBM on μ（非线性，n_est={RLEARNER_N_ESTIMATORS}, depth={RLEARNER_MAX_DEPTH}）")
    print("=" * 70)

    df, operable_in_df, observable_in_df = build_xin2_data(
        operability_csv=args.operability_csv,
    )
    if args.sample_size > 0:
        df = df.iloc[-args.sample_size:].copy()
        print(f"[调参模式] 截取最近 {args.sample_size} 条数据（共 {len(df)} 条）")

    ops    = sorted(operable_in_df   & set(df.columns))
    states = sorted(observable_in_df & set(df.columns))
    print(f"操作变量 {len(ops)} 个，状态变量 {len(states)} 个\n")

    # ── 加载 DAG 因果角色（v3.1 新增）────────────────────────────
    dag_csv = args.dag_roles_csv or DEFAULT_DAG_ROLES_CSV
    dag_roles = load_dag_roles(dag_csv)
    if not dag_roles:
        print("[注意] 未加载 DAG 角色信息，将使用纯相关性筛选（与 v3 行为一致）")
        print("       建议先运行 DAG 分析脚本生成角色表，再通过 --dag-roles-csv 传入")

    mode = args.mode
    if mode in ("stability", "all"):
        run_stability_diagnosis(df, set(ops), set(states),
                                workers=args.workers, dag_roles=dag_roles)
    if mode in ("placebo", "all"):
        run_placebo(df, set(ops), set(states),
                    n_permutations=args.n_permutations,
                    workers=args.workers, dag_roles=dag_roles)
    if mode in ("random_confounder", "all"):
        run_random_confounder(df, set(ops), set(states),
                              n_confounders=args.n_confounders,
                              n_repeats=args.n_repeats,
                              workers=args.workers, dag_roles=dag_roles)
    if mode in ("data_subset", "all"):
        run_data_subset(df, set(ops), set(states),
                        n_subsets=args.n_subsets, subset_frac=args.subset_frac,
                        workers=args.workers, dag_roles=dag_roles)

    print("\n" + "=" * 70)
    print(" 全部实验完成，结果保存至：")
    for label, path in [("稳定性诊断", STABILITY_OUT_DIR), ("安慰剂实验", PLACEBO_OUT_DIR),
                         ("随机混杂",  RANDOM_CONFOUNDER_OUT_DIR), ("数据子集", DATA_SUBSET_OUT_DIR)]:
        print(f"  {label:8s}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
