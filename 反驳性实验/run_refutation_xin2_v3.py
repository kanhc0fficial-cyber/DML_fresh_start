"""
run_refutation_xin2_v3.py
=================================
XIN_2 因果推断反驳实验 v3 ——「两阶段解耦 VAE-DML」架构

═══════════════════════════════════════════════════════════════════
  核心创新：两阶段解耦（保留 VAE，彻底修复不稳定根因）
═══════════════════════════════════════════════════════════════════

  【第一阶段】VAE 表征学习（只学 X 的因果状态空间）
    Loss_stage1 = MSE_recon + β_anneal(t) × KL
    输入：X 序列（329 个状态变量）
    输出：低维确定性均值 μ（latent_dim=32）
    训练完成后 encoder 权重全部冻结 (requires_grad=False)

  【第二阶段】双独立预测头（在冻结的 μ 上训练）
    Head_Y：μ → Ŷ = E[Y|X]    （与 Head_D 完全独立，无共享梯度）
    Head_D：μ → D̂ = E[D|X]
    残差：res_Y = Y - Ŷ，res_D = D - D̂
    θ = Cov(res_Y, res_D) / Var(res_D)  [Double ML]

  【推断时只用 μ，不采样】
    彻底消除 reparameterize 的随机性（eval 阶段 ε=0）
    保留 VAE 的随机训练过程（保留创新性），只在推断时确定性化

  创新性声明：
    VAE 的作用不是直接预测 Y/D，而是从高维共线工业传感器中
    学习紧凑的低维因果状态表征，再用 DML 估计因果效应。
    树模型 DML 无法利用时序结构和传感器间非线性关系；
    本方法将 VAE 时序表征学习与 Double ML 因果识别统一。

═══════════════════════════════════════════════════════════════════
  v3 相比 v2 的变化
═══════════════════════════════════════════════════════════════════
  1. 恢复 VAE 架构（encoder + decoder），保留创新点
  2. 两阶段解耦训练：Stage1 训 encoder，Stage2 训两个独立 head
  3. KL 退火（Annealing）：前 ANNEAL_EPOCHS 轮 β 从 0 线性升到 BETA_KL
     防止早期 KL 惩罚过强导致 posterior collapse
  4. 推断时使用 μ（mean encoding），不采样，彻底消除残差随机性
  5. 扩大 latent_dim：8→32，给 encoder 更多表征空间
  6. Bootstrap θ 聚合（保留 v2 策略）
  7. 修正反驳通过标准（保留 v2 的 t_diff < 2.0）
  8. 稳定性诊断实验零（保留 v2）
  9. LayerNorm + 梯度裁剪（保留 v2）

用法：
  python run_refutation_xin2_v3.py --mode stability --sample_size 2000 --n_bootstrap 3
  python run_refutation_xin2_v3.py --mode stability
  python run_refutation_xin2_v3.py --mode random_confounder --workers 4
  python run_refutation_xin2_v3.py --mode all --workers 6
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
#  核心训练函数：两阶段解耦 VAE-DML + Bootstrap θ 聚合
# ═══════════════════════════════════════════════════════════════════
def train_one_op(op: str, df: pd.DataFrame, safe_x: list,
                 override_D=None, n_bootstrap: int = N_BOOTSTRAP):
    """
    两阶段解耦 VAE-DML 核心函数。

    第一阶段（每次 bootstrap × 每个 fold）：
      - 训练 VAEEncoder + VAEDecoder（只重建 X）
      - KL 退火：β 从 0 线性升至 BETA_KL
      - 训练完成后 encoder 权重冻结

    第二阶段（在冻结 encoder 的 μ 上）：
      - 独立训练 head_Y（μ → Ŷ）和 head_D（μ → D̂）
      - 两个 head 完全独立，无共享梯度

    推断时：encoder.encode_mean()，只返回 μ，不采样
    → 彻底消除 reparameterize 随机性

    Bootstrap：n_bootstrap 次独立训练，θ 取中位数

    返回：
      (theta_med, p_val, SE_boot, n_avg, f_med, cv, sign_rate)
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
    theta_list, f_list, n_list = [], [], []

    for boot_i in range(n_bootstrap):
        base_seed     = boot_i * 99991 + op_base_seed
        all_res_Y, all_res_D = [], []
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
            any_valid_fold = True

        if not any_valid_fold or len(all_res_Y) < MIN_VALID_RESIDUALS:
            continue

        res_Y = np.array(all_res_Y, dtype=np.float64)
        res_D = np.array(all_res_D, dtype=np.float64)

        # 去离群（3σ）
        mask  = ((np.abs(res_Y) < 3 * res_Y.std()) &
                 (np.abs(res_D) < 3 * res_D.std()))
        res_Y, res_D = res_Y[mask], res_D[mask]
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

        # DML theta（协方差矩阵方式）
        cov_mat   = np.cov(res_D, res_Y)
        theta_std = cov_mat[0, 1] / (cov_mat[0, 0] + 1e-12)
        theta     = theta_std * (Y_std / D_std)

        theta_list.append(theta)
        f_list.append(f_stat)
        n_list.append(n)

    # ── Bootstrap 聚合 ──────────────────────────────────────────────
    min_success = max(1, n_bootstrap // 2)
    if len(theta_list) < min_success:
        return None

    theta_arr   = np.array(theta_list)
    theta_med   = float(np.median(theta_arr))
    theta_std_b = float(np.std(theta_arr))
    cv          = theta_std_b / (abs(theta_med) + 1e-8)
    sign_rate   = float(np.mean(np.sign(theta_arr) == np.sign(theta_med)))
    SE_boot     = max(theta_std_b, 1e-8)
    t_stat      = theta_med / SE_boot
    n_avg       = int(np.mean(n_list))
    p_val       = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_avg - 1))
    f_med       = float(np.median(f_list))

    return theta_med, p_val, SE_boot, n_avg, f_med, cv, sign_rate


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
    dag_roles    = task["dag_roles"]          # ← 新增
    key          = task["_key"]
    if df[op].std() < 0.1:
        return {"_key": key, "_filtered": True, "_reason": "std<0.1"}
    safe_x = build_safe_x_with_dag(op, df, states, dag_roles)  # ← 替换
    if len(safe_x) < 2:
        return {"_key": key, "_filtered": True, "_reason": "safe_x不足(DAG过滤后)"}
    rng       = np.random.default_rng(seed=perm_idx * 42 + _op_seed(op))
    D_placebo = rng.permutation(df[op].values.copy())
    result    = train_one_op(op, df, safe_x, override_D=D_placebo)
    if result is None:
        return {"_key": key, "_filtered": True, "_reason": "弱工具/样本不足"}
    theta_med, p_val, SE, n, f, cv, sr = result
    return {
        "_key":      key,   "_filtered": False,
        "操作节点":   op,    "排列索引":   perm_idx + 1,
        "θ_安慰剂":  round(theta_med, 5),
        "P_Value":   round(p_val, 4),    "SE_Boot":   round(SE, 5),
        "CV":        round(cv, 4),       "符号一致率": round(sr, 3),
        "有效残差数": n,                  "F统计量":   round(f, 2),
        "显著":      bool(p_val < 0.05),
    }


def _worker_random_confounder(task: dict) -> dict:
    op, rep       = task["op"], task["rep"]
    n_confounders = task["n_confounders"]
    theta_orig    = task["theta_orig"]
    SE_orig       = task["SE_orig"]
    safe_x_orig   = task["safe_x_orig"]
    df, key       = task["df"], task["_key"]

    rng        = np.random.default_rng(seed=rep * 1000 + _op_seed(op))
    df_noisy   = df.copy()
    noise_cols = []
    for nc in range(n_confounders):
        cname = f"__rc_{nc}__"
        df_noisy[cname] = rng.standard_normal(len(df_noisy))
        noise_cols.append(cname)
    # 噪声列加入 safe_x；Stage 1 会尝试重建它们（但随机噪声不可压缩，
    # 在 LATENT_DIM=32 的大容量下影响大幅减小），Stage 2 头不再受重建损失影响
    safe_x_noisy = safe_x_orig + noise_cols
    result       = train_one_op(op, df_noisy, safe_x_noisy)
    if result is None:
        return {"_key": key, "_filtered": True, "_reason": "弱工具/样本不足"}

    theta_p, p_p, SE_p, n_p, f_p, cv_p, sr_p = result

    # v3 反驳通过标准：t_diff < 2.0（双样本 t 检验）
    delta       = abs(theta_p - theta_orig)
    se_combined = float(np.sqrt(SE_p**2 + SE_orig**2)) + 1e-12
    t_diff      = delta / se_combined
    sign_ok     = bool(np.sign(theta_p) == np.sign(theta_orig))
    near_zero   = abs(theta_orig) < 3 * SE_orig
    passed      = bool(t_diff < 2.0 and (sign_ok or near_zero))
    rel_dev     = delta / (abs(theta_orig) + 1e-8)

    return {
        "_key":         key,            "_filtered":   False,
        "操作节点":      op,             "重复索引":     rep + 1,
        "θ_原始":       round(theta_orig, 5),
        "θ_注入噪声":   round(theta_p,    5),
        "相对偏差_ref": round(rel_dev,    4),
        "t_diff":       round(t_diff,     3),
        "方向一致":     sign_ok,
        "P_Value":      round(p_p, 4),   "SE_Boot":     round(SE_p, 5),
        "CV":           round(cv_p, 4),  "符号一致率":  round(sr_p, 3),
        "有效残差数":   n_p,             "F统计量":     round(f_p, 2),
        "通过反驳":     passed,
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
    theta_med, p_val, SE, n, f, cv, sr = result
    return {
        "_key":      key,          "_filtered": False,
        "操作节点":   op,           "子集索引":   sub_idx + 1,
        "时段起点":   start,        "时段终点":   end,
        "θ_子集":    round(theta_med, 5),
        "P_Value":   round(p_val, 4),    "SE_Boot":   round(SE, 5),
        "CV":        round(cv, 4),       "符号一致率": round(sr, 3),
        "有效残差数": n,                  "F统计量":   round(f, 2),
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
        if "_time_x" in merged.columns:
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
    valid_cols  = [c for c in df.columns
                   if c in (operable_set | observable_set) or c == "Y_grade"]
    df_filtered = df[valid_cols]

    cols_in_df       = set(df_filtered.columns) - {"Y_grade"}
    operable_in_df   = operable_set   & cols_in_df
    observable_in_df = observable_set & cols_in_df
    print(f"[数据准备] 最终 DataFrame：{df_filtered.shape}，"
          f"operable={len(operable_in_df)}，observable={len(observable_in_df)}")
    return df_filtered, operable_in_df, observable_in_df


# ═══════════════════════════════════════════════════════════════════
#  实验零：稳定性诊断（v3 新增 VAE 重建质量报告）
# ═══════════════════════════════════════════════════════════════════
def run_stability_diagnosis(df, ops, states, workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验零：稳定性诊断（{N_BOOTSTRAP} 次 Bootstrap）")
    print(f" 架构：两阶段解耦 VAE-DML  |  推断用 μ（确定性，不采样）")
    print(f" 稳定标准：CV < {CV_WARN}  且  sign_rate ≥ {SIGN_RATE_MIN}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    states_list = list(states)
    rows = []
    n_stable = 0
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
        theta_med, p_val, SE, n, f, cv, sr = result
        stable = (cv < CV_WARN and sr >= SIGN_RATE_MIN)
        if stable:
            n_stable += 1
        flag = "✓ 稳定" if stable else "⚠ 不稳定"
        print(f"  {op:<30s}  θ={theta_med:+.5f}  CV={cv:.3f}  "
              f"sign_rate={sr:.2f}  p={p_val:.4f}  [{flag}]")
        rows.append({
            "操作节点": op, "θ_中位数": round(theta_med,5),
            "P_Value": round(p_val,4), "SE_Boot": round(SE,5),
            "CV": round(cv,4), "符号一致率": round(sr,3),
            "F统计量": round(f,2), "稳定": stable,
        })
    df_out   = pd.DataFrame(rows)
    out_path = os.path.join(STABILITY_OUT_DIR, "stability_diagnosis_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not df_out.empty:
        total = len(df_out)
        print(f"\n[稳定性诊断汇总]  稳定 {n_stable}/{total} 个操作变量")
        print(f"  CV 均值      = {df_out['CV'].mean():.3f}  （目标 < {CV_WARN}）")
        print(f"  sign_rate 均值 = {df_out['符号一致率'].mean():.2f}  （目标 ≥ {SIGN_RATE_MIN}）")
        if n_stable / total < 0.5:
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
    ckpt_path   = os.path.join(PLACEBO_OUT_DIR, "checkpoint_placebo_v3.jsonl")
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
    sig_rate = df_out["显著"].mean()
    print(f"\n[安慰剂汇总]  θ均值={df_out['θ_安慰剂'].mean():+.5f}  显著率={sig_rate:.1%}（期望≈5%）")
    print(f"  {'[✓] 通过' if sig_rate <= 0.2 else '[⚠] 显著率偏高，可能存在虚假相关'}")
    out_path = os.path.join(PLACEBO_OUT_DIR, "refutation_placebo_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验二：随机混杂变量反驳
# ═══════════════════════════════════════════════════════════════════
def run_random_confounder(df, ops, states, n_confounders=5, n_repeats=1, workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验二：随机混杂变量反驳（注入 {n_confounders} 个随机噪声列）")
    print(f" v3 改进：Stage 1 仅重建 X，Stage 2 头不受噪声重建影响")
    print(f" 判断标准：t_diff = |Δθ| / √(SE_orig²+SE_noisy²) < 2.0")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    ckpt_path   = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "checkpoint_rc_v3.jsonl")
    states_list = list(states)
    print("[预计算原始 θ ...]")
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
        theta_med, p_val, SE, n, f, cv, sr = result
        if p_val > 0.05 and cv > CV_WARN:
            print(f"  [跳过] {op:<30s}  不显著且不稳定 (p={p_val:.3f}, CV={cv:.3f})"); continue
        orig_thetas[op] = (theta_med, SE, safe_x)
        flag = "⚠ 不稳定" if cv > CV_WARN else "✓"
        print(f"  {op:<30s}  θ={theta_med:+.5f}  SE={SE:.5f}  CV={cv:.3f}  {flag}")
    tasks = []
    for op, (theta_orig, SE_orig, safe_x_orig) in orig_thetas.items():
        for rep in range(n_repeats):
            tasks.append({"_key": f"{op}__rep{rep}", "op": op, "rep": rep,
                          "n_confounders": n_confounders, "theta_orig": theta_orig,
                          "SE_orig": SE_orig, "safe_x_orig": safe_x_orig, "df": df})
    _run_parallel(tasks, _worker_random_confounder, ckpt_path, workers, desc="随机混杂")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 随机混杂实验无有效结果"); return df_out
    pass_rate = df_out["通过反驳"].mean()
    print(f"\n[随机混杂汇总]  反驳通过率 = {pass_rate:.1%}（期望 ≥ 80%）")
    print(f"  {'[✓] 通过' if pass_rate >= 0.8 else '[⚠] 通过率偏低，θ 对随机噪声注入仍然敏感'}")
    out_path = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "refutation_rc_v3.csv")
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
    ckpt_path   = os.path.join(DATA_SUBSET_OUT_DIR, "checkpoint_ds_v3.jsonl")
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
    stable_ops = total_ops = 0
    for op, grp in df_out.groupby("操作节点"):
        if len(grp) < 3: continue
        total_ops += 1
        arr = grp["θ_子集"].values
        cv  = np.std(arr) / (abs(np.mean(arr)) + 1e-8)
        sc  = np.mean(np.sign(arr) == np.sign(np.median(arr)))
        if cv < 0.30 and sc >= 0.70:
            stable_ops += 1
    if total_ops:
        gp = stable_ops / total_ops
        print(f"\n[数据子集汇总]  全局稳定通过率 = {gp:.1%}（期望 ≥ 70%）")
        print(f"  {'[✓] 通过' if gp >= 0.70 else '[⚠] θ 跨时段稳定性不足'}")
    out_path = os.path.join(DATA_SUBSET_OUT_DIR, "refutation_ds_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="XIN_2 因果推断反驳实验 v3（两阶段解耦 VAE-DML）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
建议执行顺序：
  # 1. 小样本快速验证稳定性改善（几分钟）
  python run_refutation_xin2_v3.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 2. 全量稳定性诊断（确认 CV 普遍下降）
  python run_refutation_xin2_v3.py --mode stability

  # 3. 反驳实验
  python run_refutation_xin2_v3.py --mode random_confounder --workers 4
  python run_refutation_xin2_v3.py --mode placebo --n_permutations 5 --workers 4
  python run_refutation_xin2_v3.py --mode data_subset --workers 4

  # 4. 全部一次跑
  python run_refutation_xin2_v3.py --mode all --workers 6

断点续传：检查点含 _v3 后缀，与 v1/v2 不冲突。
调参建议：先改 LATENT_DIM（尝试 16/32/64），再改 BETA_KL（尝试 0.01/0.05/0.1），
         再改 ANNEAL_EPOCHS（尝试 10/20/40）。
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
    print(f" XIN_2 因果推断反驳实验 v3  |  模式: {args.mode.upper()}")
    print(f" 设备: {DEVICE}  |  并行线程: {args.workers}  |  Bootstrap: {N_BOOTSTRAP}")
    print(f" 架构: 两阶段解耦 VAE-DML")
    print(f"   Stage1: VAEEncoder(LATENT={LATENT_DIM}) + Decoder  [BETA_KL={BETA_KL}, 退火={ANNEAL_EPOCHS}轮]")
    print(f"   Stage2: 独立 head_Y + head_D（在冻结 μ 上训练）")
    print(f"   推断:   encode_mean()，不采样，确定性残差")
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
