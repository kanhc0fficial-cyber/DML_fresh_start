"""
run_refutation_xin2_v4.py
=================================
XIN_2 因果推断反驳实验 v4 ——「两阶段解耦 VAE-DML + 交叉拟合策略改进」

═══════════════════════════════════════════════════════════════════
  继承 v3 全部架构（两阶段解耦 VAE-DML + Bootstrap θ 聚合），
  在交叉拟合层面新增四项改进：
═══════════════════════════════════════════════════════════════════

【改进1】时序前向交叉拟合（v3 已有扩展窗口；v4 新增滑动窗口选项）
  --window_type expanding  (默认，与 v3 一致，扩展窗口)
  --window_type sliding    (仅用近期 block_size×SLIDING_WINDOW_RATIO 条训练)

  扩展窗口法（expanding）：
    fold k 训练集 = seqs_X[0 : k×block_size - embargo]
  滑动窗口法（sliding）：
    fold k 训练集 = seqs_X[max(0, train_end - W) : train_end]
    其中 W = block_size × SLIDING_WINDOW_RATIO

  两种方法均严格保证：估计第 k 折残差时只用该时段之前的数据训练，
  无未来信息泄露。

【改进2】分层交叉拟合（Stratified Cross-Fitting，v4 新增）
  --stratified  开关控制（默认关闭）

  适用于处理变量 D 为二元或严重不平衡的场景（如返工率 <15%）：
    在每个训练折中检查"高处理量"样本（D > 全局中位数）的数量；
    若高处理量样本不足 MIN_TREAT_SAMPLES 条，则跳过该折，
    避免倾向得分模型因极端不平衡而严重失准。

  实现方式：时序兼容版，不打乱顺序，只做折级别的平衡检查。

【改进3】重复交叉拟合折边随机化（Repeated Cross-Fitting，v4 改进）
  --fold_jitter_ratio 0.10  (默认 ±10% block_size 的折边抖动)
  设为 0 则退化为 v3 固定折边行为。

  v3 的 Bootstrap 重用固定折边（block_size = N // K_FOLDS）；
  v4 在每次 Bootstrap 迭代中对折边施加随机偏移：
    boundary_k = k × block_size + Uniform(-jitter_max, jitter_max)
  从而每次 Bootstrap 使用不同的数据切分，消除单次分组随机性，
  理论依据来自 Chernozhukov et al. (2018) 的重复交叉拟合建议。

【改进4】嵌套超参选择（Nested Cross-Fitting，v4 新增，可选）
  --nested_lr_search  开关控制（默认关闭，计算成本约 3×）

  在每个外层折的训练数据上做内层验证，比较候选学习率：
    NESTED_LR_CANDIDATES = [0.001, 0.003, 0.01]
  内层验证集 = 训练数据末 20%；外层测试集完全隔离，
  从未参与学习率选择，严格防止超参调优导致的数据泄露。

【改进5】交叉拟合策略对比实验（cf_compare，v4 新增）
  --mode cf_compare  对比四种策略在 ATE 估计方差和 CV 上的差异：
    策略A：标准扩展窗口（与 v3 一致）
    策略B：扩展窗口 + 分层折检查
    策略C：扩展窗口 + 折边随机化（重复交叉拟合）
    策略D：扩展窗口 + 嵌套学习率选择
  输出对比表 refutation_cf_compare_v4.csv，
  列出每种策略的 theta_med / SE_boot / CV / 计算耗时。

═══════════════════════════════════════════════════════════════════
  v4 相比 v3 的变化（架构层面不变）
═══════════════════════════════════════════════════════════════════
  1. 新增常量：FOLD_JITTER_RATIO, SLIDING_WINDOW_RATIO,
               NESTED_LR_SEARCH, NESTED_LR_CANDIDATES, MIN_TREAT_SAMPLES
  2. 新增函数：_select_best_lr_inner（嵌套LR选择的内层验证）
  3. _train_head_stage2：新增 nested_lr_search / latent_dim 参数
  4. train_one_op：新增 window_type / fold_jitter_ratio /
                       use_stratified / nested_lr_search 参数
  5. 所有实验函数接受 cf_cfg 字典，统一透传交叉拟合策略参数
  6. 新增 run_cf_compare() 对比不同策略
  7. 输出文件均带 _v4 后缀，不覆盖 v3 结果

用法：
  # 稳定性诊断（快速调参）
  python run_refutation_xin2_v4.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 开启折边随机化 + 分层检查
  python run_refutation_xin2_v4.py --mode stability --fold_jitter_ratio 0.1 --stratified

  # 开启滑动窗口
  python run_refutation_xin2_v4.py --mode stability --window_type sliding

  # 嵌套学习率搜索（计算开销较大）
  python run_refutation_xin2_v4.py --mode stability --nested_lr_search

  # 策略对比实验（核心新增实验）
  python run_refutation_xin2_v4.py --mode cf_compare --sample_size 3000

  # 全部反驳实验（带所有改进）
  python run_refutation_xin2_v4.py --mode all --fold_jitter_ratio 0.1 --stratified --workers 6
"""

import argparse
import hashlib
import json
import os
import time
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
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR  = os.path.join(REPO_ROOT, "data")

MODELING_DATASET_XIN2 = os.path.join(DATA_DIR, "modeling_dataset_xin2_final.parquet")
X_PARQUET             = os.path.join(DATA_DIR, "X_features_final.parquet")
Y_PARQUET             = os.path.join(DATA_DIR, "y_target_final.parquet")

DEFAULT_OPERABILITY_CSV = os.path.join(
    REPO_ROOT, "数据预处理",
    "数据与处理结果-分阶段-去共线性后",
    "non_collinear_representative_vars_operability.csv",
)

PLACEBO_OUT_DIR           = os.path.join(BASE_DIR, "安慰剂实验")
RANDOM_CONFOUNDER_OUT_DIR = os.path.join(BASE_DIR, "随机混杂变量实验")
DATA_SUBSET_OUT_DIR       = os.path.join(BASE_DIR, "数据子集实验")
STABILITY_OUT_DIR         = os.path.join(BASE_DIR, "稳定性诊断")
CF_COMPARE_OUT_DIR        = os.path.join(BASE_DIR, "交叉拟合策略对比")  # v4 新增

for _d in [PLACEBO_OUT_DIR, RANDOM_CONFOUNDER_OUT_DIR,
           DATA_SUBSET_OUT_DIR, STABILITY_OUT_DIR, CF_COMPARE_OUT_DIR]:
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
#  超参（与 v3 完全一致的部分）
# ═══════════════════════════════════════════════════════════════════
SEQ_LEN             = 6
EMBARGO_GAP         = 4
K_FOLDS             = 4
MAX_EPOCHS_VAE      = 60
MAX_EPOCHS_HEAD     = 40
PATIENCE            = 8
MIN_TRAIN_SIZE      = 100
MIN_VALID_RESIDUALS = 50
F_STAT_THRESHOLD    = 10.0

LATENT_DIM      = 32
BETA_KL         = 0.1
ANNEAL_EPOCHS   = 20
HIDDEN_DIM_ENC  = 64
NUM_LSTM_LAYERS = 2
HIDDEN_DIM_HEAD = 32
GRAD_CLIP       = 1.0
N_BOOTSTRAP     = 5
CV_WARN         = 0.30
SIGN_RATE_MIN   = 0.70

# ─── v4 交叉拟合策略改进参数 ──────────────────────────────────────
FOLD_JITTER_RATIO    = 0.10   # 折边抖动比例：每次 Bootstrap 随机偏移 ±10% block_size
                               # 设为 0 退化为 v3 固定折边行为
SLIDING_WINDOW_RATIO = 2      # 滑动窗口倍率：用最近 block_size×2 条数据作为训练集
NESTED_LR_SEARCH     = False  # 是否做嵌套学习率搜索（默认关闭，--nested_lr_search 开启）
NESTED_LR_CANDIDATES    = [0.001, 0.003, 0.01]  # 嵌套 LR 候选集
MIN_TREAT_SAMPLES       = 5   # 分层检查：训练折中"高处理量"样本少于此值时跳过该折
MIN_INNER_VAL_SAMPLES   = 60  # 嵌套LR选择最少样本数：低于此值时退化为默认LR避免内层验证集过小


# ══════════════════════════════════════════════════════════════════
#  DAG 因果角色过滤（与 v3 完全一致）
# ══════════════════════════════════════════════════════════════════
DEFAULT_DAG_ROLES_CSV = os.path.join(
    REPO_ROOT, "DAG图分析", "DAG解析结果", ""
)


def load_dag_roles(csv_path: str) -> dict:
    """加载 DAG 角色明细表（analyze_dag_causal_roles_v4_1.py 的输出）。"""
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
            continue
        if t_name not in dag_roles:
            dag_roles[t_name] = {k: set() for k in role_map.values()}
        dag_roles[t_name][role_map[role]].add(node)
    print(f"[DAG过滤] 已加载 {len(dag_roles)} 个操作变量的因果角色信息")
    return dag_roles


def build_safe_x_with_dag(op: str, df: pd.DataFrame, states: list,
                           dag_roles: dict) -> list:
    """构建控制变量集 safe_x，整合 DAG 因果角色过滤（与 v3 完全一致）。"""
    candidate_x = get_safe_x(op, df, states)
    if not dag_roles or op not in dag_roles:
        return candidate_x
    roles = dag_roles[op]
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
            filtered_x.append(var)
    n_excluded = len(candidate_x) - len(filtered_x)
    if n_excluded > 0:
        parts = [f"{r}={len(v)}" for r, v in excluded_details.items() if v]
        print(f"  [DAG过滤] {op}: 剔除 {n_excluded} 个变量 ({', '.join(parts)})，"
              f"保留 {len(filtered_x)} 个控制变量")
    return filtered_x


# ═══════════════════════════════════════════════════════════════════
#  模型定义（与 v3 完全一致）
# ═══════════════════════════════════════════════════════════════════

class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, HIDDEN_DIM_ENC,
            batch_first=True,
            num_layers=NUM_LSTM_LAYERS,
            dropout=0.1 if NUM_LSTM_LAYERS > 1 else 0.0,
        )
        self.norm      = nn.LayerNorm(HIDDEN_DIM_ENC)
        self.fc        = nn.Sequential(nn.Linear(HIDDEN_DIM_ENC, 32), nn.SiLU())
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
#  阶段一训练：VAE 表征学习（与 v3 完全一致）
# ═══════════════════════════════════════════════════════════════════
def _train_vae_stage1(encoder: VAEEncoder, decoder: VAEDecoder,
                      X_train: torch.Tensor, device) -> None:
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
        beta = BETA_KL * min(1.0, epoch / max(1, ANNEAL_EPOCHS))
        encoder.train(); decoder.train()
        for (bx,) in loader:
            optimizer.zero_grad()
            mu, logvar = encoder(bx)
            std    = torch.exp(0.5 * logvar)
            z      = mu + std * torch.randn_like(std)
            x_recon   = decoder(z)
            loss_rec  = nn.functional.mse_loss(x_recon, bx)
            loss_kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss      = loss_rec + beta * loss_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()

        encoder.eval(); decoder.eval()
        with torch.no_grad():
            mu_iv, lv_iv = encoder(X_iv)
            std_iv = torch.exp(0.5 * lv_iv)
            z_iv   = mu_iv + std_iv * torch.randn_like(std_iv)
            xr_iv  = decoder(z_iv)
            vl = (nn.functional.mse_loss(xr_iv, X_iv)
                  - 0.5 * beta * torch.mean(1 + lv_iv - mu_iv.pow(2) - lv_iv.exp())).item()

        if vl < best_loss - 1e-5:
            best_loss = vl; pat_cnt = 0
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
#  v4 新增：嵌套学习率内层选择
# ═══════════════════════════════════════════════════════════════════
def _select_best_lr_inner(mu_train: torch.Tensor, target_train: torch.Tensor,
                           latent_dim: int, device) -> float:
    """
    嵌套学习率选择（改进4的核心步骤）。

    在外层训练集内部再划分一个内层验证集（末 20%），
    对每个候选学习率跑 MAX_EPOCHS_HEAD//3 轮，选验证 MSE 最低的。
    外层测试集从未参与此过程，严格保证超参调优的数据独立性。

    返回：最优候选学习率（float）
    """
    if len(mu_train) < MIN_INNER_VAL_SAMPLES:
        # 数据太少，无法做内层验证，直接返回默认
        return 0.002

    inner_val_split = max(1, int(len(mu_train) * 0.20))
    mu_tr_i  = mu_train[:-inner_val_split]
    t_tr_i   = target_train[:-inner_val_split]
    mu_val_i = mu_train[-inner_val_split:]
    t_val_i  = target_train[-inner_val_split:]

    best_lr, best_val_mse = NESTED_LR_CANDIDATES[0], float("inf")
    inner_epochs = max(5, MAX_EPOCHS_HEAD // 3)

    for lr in NESTED_LR_CANDIDATES:
        head_trial = PredHead(latent_dim).to(device)
        opt_trial  = optim.Adam(head_trial.parameters(), lr=lr)
        loader_i   = DataLoader(
            TensorDataset(mu_tr_i, t_tr_i), batch_size=256, shuffle=False
        )
        for _ in range(inner_epochs):
            head_trial.train()
            for bmu, bt in loader_i:
                opt_trial.zero_grad()
                nn.functional.mse_loss(head_trial(bmu), bt).backward()
                torch.nn.utils.clip_grad_norm_(head_trial.parameters(), GRAD_CLIP)
                opt_trial.step()
        head_trial.eval()
        with torch.no_grad():
            val_mse = nn.functional.mse_loss(head_trial(mu_val_i), t_val_i).item()
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_lr = lr

    return best_lr


# ═══════════════════════════════════════════════════════════════════
#  阶段二训练：预测头（v4 新增 nested_lr_search 参数）
# ═══════════════════════════════════════════════════════════════════
def _train_head_stage2(head: PredHead, mu_train: torch.Tensor,
                       target_train: torch.Tensor, device,
                       nested_lr_search: bool = False,
                       latent_dim: int = LATENT_DIM) -> None:
    """
    在冻结 encoder 输出的 μ 上训练单个预测头（MSE loss）。

    v4 新增 nested_lr_search 参数：
      True  → 先调用 _select_best_lr_inner 选最优 LR（改进4）；
               外层测试集完全隔离，超参选择不依赖测试数据
      False → 固定 lr=0.002（与 v3 一致）
    """
    if nested_lr_search:
        lr = _select_best_lr_inner(mu_train, target_train, latent_dim, device)
    else:
        lr = 0.002

    optimizer  = optim.Adam(head.parameters(), lr=lr)
    loader     = DataLoader(TensorDataset(mu_train, target_train),
                            batch_size=256, shuffle=False)
    val_split  = max(1, int(len(mu_train) * 0.1))
    mu_iv      = mu_train[-val_split:]
    t_iv       = target_train[-val_split:]
    best_loss  = float("inf")
    pat_cnt    = 0
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
            best_loss  = vl; pat_cnt = 0
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
#  v4 新增：window_type / fold_jitter_ratio / use_stratified / nested_lr_search
# ═══════════════════════════════════════════════════════════════════
def train_one_op(op: str, df: pd.DataFrame, safe_x: list,
                 override_D=None, n_bootstrap: int = N_BOOTSTRAP,
                 window_type: str = "expanding",
                 fold_jitter_ratio: float = FOLD_JITTER_RATIO,
                 use_stratified: bool = False,
                 nested_lr_search: bool = False):
    """
    两阶段解耦 VAE-DML 核心函数（v4 版）。

    v4 新增参数说明：
    ─────────────────────────────────────────────────────────────
    window_type : str
        "expanding"（默认，与 v3 一致）
            train fold k = seqs_X[0 : k×block_size - embargo]
        "sliding"
            train fold k = seqs_X[max(0, train_end-W) : train_end]
            W = block_size × SLIDING_WINDOW_RATIO
        两种方式均严格前向（无未来泄露）。

    fold_jitter_ratio : float
        每次 Bootstrap 对折边施加 ±(ratio × block_size) 的随机偏移，
        实现「重复交叉拟合」——每轮使用不同的数据切分。
        设为 0 退化为 v3 固定折边。

    use_stratified : bool
        为 True 时，在每个折的训练集中检查高处理量样本数
        （D_raw > 全局中位数的样本），不足 MIN_TREAT_SAMPLES 则跳过该折。
        适用于 D 严重不平衡的场景（如返工率 <15%）。

    nested_lr_search : bool
        为 True 时，在每个折训练 head_Y/head_D 前，先用内层验证
        选出最优学习率（_select_best_lr_inner），再在完整训练集上
        用最优 LR 训练，外层测试集完全隔离。

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

    # ── 改进2：预计算全局 D 中位数（分层检查用）─────────────────
    d_global_median = float(np.median(D_raw))

    # ── 滑动窗口序列 ────────────────────────────────────────────────
    seqs_X, tgt_Y, tgt_D = [], [], []
    for i in range(len(X_mat) - SEQ_LEN):
        seqs_X.append(X_mat[i: i + SEQ_LEN])
        tgt_Y.append(Y_mat[i + SEQ_LEN])
        tgt_D.append(D_mat[i + SEQ_LEN])
    seqs_X = np.array(seqs_X, dtype=np.float32)
    tgt_Y  = np.array(tgt_Y,  dtype=np.float32)
    tgt_D  = np.array(tgt_D,  dtype=np.float32)
    # 对应原始 D_raw（对齐 seqs_X 的时间步，用于分层检查）
    D_raw_seq = D_raw[SEQ_LEN:]

    N          = len(seqs_X)
    block_size = N // K_FOLDS
    input_dim  = len(safe_x)

    # 改进1 滑动窗口大小
    sliding_window_size = int(block_size * SLIDING_WINDOW_RATIO)

    # ── Bootstrap 外循环 ────────────────────────────────────────────
    op_base_seed = _op_seed(op)
    theta_list, f_list, n_list = [], [], []
    folds_skipped_stratified = 0
    folds_total = 0

    for boot_i in range(n_bootstrap):
        base_seed      = boot_i * 99991 + op_base_seed
        all_res_Y, all_res_D = [], []
        any_valid_fold = False

        # ── 改进3：折边随机化（重复交叉拟合）──────────────────────
        # 每次 Bootstrap 用独立的 RNG 生成折边偏移量，
        # 使每轮使用不同的数据切分，消除单次分组随机性。
        if fold_jitter_ratio > 0:
            jitter_max = max(1, int(block_size * fold_jitter_ratio))
            rng_jitter = np.random.default_rng(base_seed + 77777)
            global_jitter = int(rng_jitter.integers(-jitter_max, jitter_max + 1))
        else:
            global_jitter = 0

        for k in range(1, K_FOLDS):
            folds_total += 1
            # 折边 = 标准位置 + 本次 Bootstrap 的全局偏移
            fold_boundary = k * block_size + global_jitter
            fold_boundary = max(
                MIN_TRAIN_SIZE + EMBARGO_GAP,
                min(N - MIN_VALID_RESIDUALS, fold_boundary)
            )
            train_end = fold_boundary - EMBARGO_GAP
            if train_end < MIN_TRAIN_SIZE:
                continue
            val_start = fold_boundary
            val_end   = (
                (k + 1) * block_size + global_jitter
                if k < K_FOLDS - 1 else N
            )
            val_end = max(val_start + MIN_VALID_RESIDUALS, min(N, val_end))
            if val_start >= val_end:
                continue

            torch.manual_seed(base_seed * 100 + k)
            np.random.seed((base_seed * 100 + k) % (2**31))

            # ── 改进2：分层折检查（时序兼容版）──────────────────
            if use_stratified:
                D_fold_train = D_raw_seq[:train_end]
                high_treat_count = int(np.sum(D_fold_train > d_global_median))
                if high_treat_count < MIN_TREAT_SAMPLES:
                    folds_skipped_stratified += 1
                    continue  # 该折处理样本不足，跳过

            # ── 改进1：窗口类型选择 ──────────────────────────────
            if window_type == "sliding":
                train_start = max(0, train_end - sliding_window_size)
            else:  # expanding（默认，与 v3 一致）
                train_start = 0

            Xtr = torch.tensor(seqs_X[train_start:train_end]).to(device)
            Ytr = torch.tensor(tgt_Y[train_start:train_end]).to(device)
            Dtr = torch.tensor(tgt_D[train_start:train_end]).to(device)

            Xvl = torch.tensor(seqs_X[val_start:val_end]).to(device)
            Yvl = tgt_Y[val_start:val_end]
            Dvl = tgt_D[val_start:val_end]

            if len(Xtr) < MIN_TRAIN_SIZE:
                continue

            # ── Stage 1：训练 VAE（只重建 X，不涉及 Y/D）──────────
            encoder = VAEEncoder(input_dim, LATENT_DIM).to(device)
            decoder = VAEDecoder(LATENT_DIM, SEQ_LEN, input_dim).to(device)
            _train_vae_stage1(encoder, decoder, Xtr, device)

            # 冻结 encoder
            for p in encoder.parameters():
                p.requires_grad_(False)
            encoder.eval()

            # ── 计算训练集和验证集的 μ（确定性，不采样）──────────
            with torch.no_grad():
                mu_tr = encoder.encode_mean(Xtr)
                mu_vl = encoder.encode_mean(Xvl)

            # ── Stage 2：独立训练 head_Y 和 head_D ──────────────
            # 改进4：支持嵌套学习率选择
            head_Y = PredHead(LATENT_DIM).to(device)
            head_D = PredHead(LATENT_DIM).to(device)
            _train_head_stage2(head_Y, mu_tr, Ytr, device,
                               nested_lr_search=nested_lr_search,
                               latent_dim=LATENT_DIM)
            _train_head_stage2(head_D, mu_tr, Dtr, device,
                               nested_lr_search=nested_lr_search,
                               latent_dim=LATENT_DIM)

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

    # 分层跳过率日志（仅在有跳过时打印）
    if use_stratified and folds_skipped_stratified > 0:
        skip_rate = folds_skipped_stratified / max(1, folds_total)
        if skip_rate > 0.5:
            print(f"  [分层警告] {op}: {folds_skipped_stratified}/{folds_total} 个折"
                  f"因处理样本不足被跳过（跳过率 {skip_rate:.1%}）")

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
#  辅助：构建 safe_x（与 v3 完全一致）
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
#  断点续传工具（与 v3 完全一致）
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
#  通用并行调度器（与 v3 完全一致）
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
#  Worker 函数（v4：cf_cfg 参数透传）
# ═══════════════════════════════════════════════════════════════════

def _worker_placebo(task: dict) -> dict:
    op, perm_idx = task["op"], task["perm_idx"]
    df, states   = task["df"], task["states"]
    dag_roles    = task["dag_roles"]
    cf_cfg       = task.get("cf_cfg", {})
    key          = task["_key"]
    if df[op].std() < 0.1:
        return {"_key": key, "_filtered": True, "_reason": "std<0.1"}
    safe_x = build_safe_x_with_dag(op, df, states, dag_roles)
    if len(safe_x) < 2:
        return {"_key": key, "_filtered": True, "_reason": "safe_x不足(DAG过滤后)"}
    rng       = np.random.default_rng(seed=perm_idx * 42 + _op_seed(op))
    D_placebo = rng.permutation(df[op].values.copy())
    result    = train_one_op(op, df, safe_x, override_D=D_placebo, **cf_cfg)
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
    cf_cfg        = task.get("cf_cfg", {})

    rng        = np.random.default_rng(seed=rep * 1000 + _op_seed(op))
    df_noisy   = df.copy()
    noise_cols = []
    for nc in range(n_confounders):
        cname = f"__rc_{nc}__"
        df_noisy[cname] = rng.standard_normal(len(df_noisy))
        noise_cols.append(cname)
    safe_x_noisy = safe_x_orig + noise_cols
    result       = train_one_op(op, df_noisy, safe_x_noisy, **cf_cfg)
    if result is None:
        return {"_key": key, "_filtered": True, "_reason": "弱工具/样本不足"}

    theta_p, p_p, SE_p, n_p, f_p, cv_p, sr_p = result
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
    cf_cfg      = task.get("cf_cfg", {})
    key         = task["_key"]
    df_sub = df.iloc[start:end].copy()
    if len(df_sub) < SEQ_LEN + K_FOLDS * MIN_TRAIN_SIZE:
        return {"_key": key, "_filtered": True, "_reason": "样本不足"}
    result = train_one_op(op, df_sub, safe_x, **cf_cfg)
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
#  数据准备（与 v3 完全一致）
# ═══════════════════════════════════════════════════════════════════
def build_xin2_data(operability_csv: str = DEFAULT_OPERABILITY_CSV):
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

    if os.path.exists(MODELING_DATASET_XIN2):
        print(f"[数据准备] 读取已对齐建模宽表：{MODELING_DATASET_XIN2}")
        df = pd.read_parquet(MODELING_DATASET_XIN2)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "time"
        if "y_fx_xin2" in df.columns:
            df = df.rename(columns={"y_fx_xin2": "Y_grade"})
        elif "Y_grade" not in df.columns:
            raise KeyError(
                f"建模宽表 {MODELING_DATASET_XIN2} 中未找到 'y_fx_xin2' 或 'Y_grade' 列"
            )
        if "y_fx_xin1" in df.columns:
            df = df.drop(columns=["y_fx_xin1"])
        df = df.dropna(subset=["Y_grade"])

    elif os.path.exists(X_PARQUET) and os.path.exists(Y_PARQUET):
        print(f"[数据准备] 未找到已对齐宽表，回退到分别读取 X + Y")
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
        y_xin2  = y[["y_fx_xin2"]].dropna()
        y_reset = y_xin2.reset_index().sort_values("time")
        X_reset = X.reset_index().rename(columns={"time": "_time_x"}).sort_values("_time_x")
        merged  = pd.merge_asof(
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
            f"  或分别: {X_PARQUET} + {Y_PARQUET}"
        )

    df = df.loc[:, (df.std() > 1e-4)]
    all_known_vars = operable_set | observable_set
    valid_cols  = [c for c in df.columns if c in all_known_vars or c == "Y_grade"]
    df_filtered = df[valid_cols]
    cols_in_df       = set(df_filtered.columns) - {"Y_grade"}
    operable_in_df   = operable_set   & cols_in_df
    observable_in_df = observable_set & cols_in_df
    print(f"[数据准备] 最终 DataFrame：{df_filtered.shape}，"
          f"operable={len(operable_in_df)}，observable={len(observable_in_df)}")
    return df_filtered, operable_in_df, observable_in_df


# ═══════════════════════════════════════════════════════════════════
#  实验零：稳定性诊断（v4：透传 cf_cfg）
# ═══════════════════════════════════════════════════════════════════
def run_stability_diagnosis(df, ops, states, workers=4, dag_roles: dict = None,
                            cf_cfg: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验零：稳定性诊断（{N_BOOTSTRAP} 次 Bootstrap）")
    print(f" 架构：两阶段解耦 VAE-DML  |  推断用 μ（确定性，不采样）")
    print(f" 交叉拟合策略：{_fmt_cf_cfg(cf_cfg)}")
    print(f" 稳定标准：CV < {CV_WARN}  且  sign_rate ≥ {SIGN_RATE_MIN}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    if cf_cfg is None:
        cf_cfg = {}
    states_list = list(states)
    rows = []
    n_stable = 0
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            continue
        result = train_one_op(op, df, safe_x, **cf_cfg)
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
            "操作节点": op, "θ_中位数": round(theta_med, 5),
            "P_Value": round(p_val, 4), "SE_Boot": round(SE, 5),
            "CV": round(cv, 4), "符号一致率": round(sr, 3),
            "F统计量": round(f, 2), "稳定": stable,
        })
    df_out   = pd.DataFrame(rows)
    out_path = os.path.join(STABILITY_OUT_DIR, "stability_diagnosis_v4.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not df_out.empty:
        total = len(df_out)
        print(f"\n[稳定性诊断汇总]  稳定 {n_stable}/{total} 个操作变量")
        print(f"  CV 均值      = {df_out['CV'].mean():.3f}  （目标 < {CV_WARN}）")
        print(f"  sign_rate 均值 = {df_out['符号一致率'].mean():.2f}  （目标 ≥ {SIGN_RATE_MIN}）")
        if n_stable / total < 0.5:
            print("  [⚠] 超过一半操作变量不稳定，建议调整策略参数")
        else:
            print("  [✓] 多数操作变量稳定，可运行反驳实验")
    print(f"结果已保存：{out_path}")
    return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验一：安慰剂反驳（v4：透传 cf_cfg）
# ═══════════════════════════════════════════════════════════════════
def run_placebo(df, ops, states, n_permutations=5, workers=4,
                dag_roles: dict = None, cf_cfg: dict = None):
    print("\n" + "=" * 70)
    print(" 实验一：安慰剂反驳实验（随机排列操作变量 D）")
    print(f" 交叉拟合策略：{_fmt_cf_cfg(cf_cfg)}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    if cf_cfg is None:
        cf_cfg = {}
    ckpt_path   = os.path.join(PLACEBO_OUT_DIR, "checkpoint_placebo_v4.jsonl")
    states_list = list(states)
    tasks = []
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        for perm_idx in range(n_permutations):
            tasks.append({"_key": f"{op}__perm{perm_idx}", "op": op,
                          "perm_idx": perm_idx, "df": df, "states": states_list,
                          "dag_roles": dag_roles, "cf_cfg": cf_cfg})
    _run_parallel(tasks, _worker_placebo, ckpt_path, workers, desc="安慰剂")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 安慰剂实验无有效结果"); return df_out
    sig_rate = df_out["显著"].mean()
    print(f"\n[安慰剂汇总]  θ均值={df_out['θ_安慰剂'].mean():+.5f}  显著率={sig_rate:.1%}（期望≈5%）")
    print(f"  {'[✓] 通过' if sig_rate <= 0.2 else '[⚠] 显著率偏高，可能存在虚假相关'}")
    out_path = os.path.join(PLACEBO_OUT_DIR, "refutation_placebo_v4.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验二：随机混杂变量反驳（v4：透传 cf_cfg）
# ═══════════════════════════════════════════════════════════════════
def run_random_confounder(df, ops, states, n_confounders=5, n_repeats=1,
                          workers=4, dag_roles: dict = None, cf_cfg: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验二：随机混杂变量反驳（注入 {n_confounders} 个随机噪声列）")
    print(f" 判断标准：t_diff = |Δθ| / √(SE_orig²+SE_noisy²) < 2.0")
    print(f" 交叉拟合策略：{_fmt_cf_cfg(cf_cfg)}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    if cf_cfg is None:
        cf_cfg = {}
    ckpt_path   = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "checkpoint_rc_v4.jsonl")
    states_list = list(states)
    print("[预计算原始 θ ...]")
    orig_thetas = {}
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            continue
        result = train_one_op(op, df, safe_x, **cf_cfg)
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
                          "SE_orig": SE_orig, "safe_x_orig": safe_x_orig,
                          "df": df, "cf_cfg": cf_cfg})
    _run_parallel(tasks, _worker_random_confounder, ckpt_path, workers, desc="随机混杂")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 随机混杂实验无有效结果"); return df_out
    pass_rate = df_out["通过反驳"].mean()
    print(f"\n[随机混杂汇总]  反驳通过率 = {pass_rate:.1%}（期望 ≥ 80%）")
    print(f"  {'[✓] 通过' if pass_rate >= 0.8 else '[⚠] 通过率偏低，θ 对随机噪声注入仍然敏感'}")
    out_path = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "refutation_rc_v4.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验三：数据子集反驳（v4：透传 cf_cfg）
# ═══════════════════════════════════════════════════════════════════
def run_data_subset(df, ops, states, n_subsets=8, subset_frac=0.8,
                    workers=4, dag_roles: dict = None, cf_cfg: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验三：数据子集反驳（{n_subsets} 个子集，每个取 {subset_frac:.0%} 数据）")
    print(f" 交叉拟合策略：{_fmt_cf_cfg(cf_cfg)}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    if cf_cfg is None:
        cf_cfg = {}
    ckpt_path   = os.path.join(DATA_SUBSET_OUT_DIR, "checkpoint_ds_v4.jsonl")
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
                          "start": start, "end": end, "safe_x": safe_x, "df": df,
                          "cf_cfg": cf_cfg})
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
    out_path = os.path.join(DATA_SUBSET_OUT_DIR, "refutation_ds_v4.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  v4 新增：实验四 — 交叉拟合策略对比实验（cf_compare）
# ═══════════════════════════════════════════════════════════════════
def run_cf_compare(df, ops, states, dag_roles: dict = None, n_ops: int = 5):
    """
    对比四种交叉拟合策略在 ATE 估计质量上的差异。

    策略对比：
      A：标准扩展窗口（与 v3 一致，折边固定）
      B：扩展窗口 + 分层折检查（use_stratified=True）
      C：扩展窗口 + 折边随机化（fold_jitter_ratio=0.10，重复交叉拟合）
      D：扩展窗口 + 嵌套学习率选择（nested_lr_search=True）

    指标：
      - theta_med    : ATE 中位数估计
      - SE_boot      : Bootstrap 标准误（反映估计方差）
      - CV           : 变异系数（越小越稳定）
      - sign_rate    : 符号一致率
      - 耗时(s)      : 各策略的计算时间

    输出：CF_COMPARE_OUT_DIR/refutation_cf_compare_v4.csv
    """
    print("\n" + "=" * 70)
    print(" 实验四（v4 新增）：交叉拟合策略对比实验")
    print(" 策略 A：标准扩展窗口（baseline，同 v3）")
    print(" 策略 B：扩展窗口 + 分层折检查")
    print(" 策略 C：扩展窗口 + 折边随机化（重复交叉拟合）")
    print(" 策略 D：扩展窗口 + 嵌套LR选择（计算成本较高）")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}

    strategies = {
        "A_standard":    dict(window_type="expanding", fold_jitter_ratio=0.0,
                              use_stratified=False, nested_lr_search=False),
        "B_stratified":  dict(window_type="expanding", fold_jitter_ratio=0.0,
                              use_stratified=True,  nested_lr_search=False),
        "C_jitter":      dict(window_type="expanding", fold_jitter_ratio=0.10,
                              use_stratified=False, nested_lr_search=False),
        "D_nested_lr":   dict(window_type="expanding", fold_jitter_ratio=0.0,
                              use_stratified=False, nested_lr_search=True),
    }

    states_list = list(states)
    # 选取前 n_ops 个操作变量（按字母序，跳过低方差）
    candidate_ops = [op for op in sorted(ops) if df[op].std() >= 0.1][:n_ops]
    if not candidate_ops:
        print("[警告] 无有效操作变量，跳过对比实验"); return pd.DataFrame()

    print(f"  选取 {len(candidate_ops)} 个操作变量参与对比：{candidate_ops}")
    rows = []

    for op in candidate_ops:
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            print(f"  [跳过] {op}  safe_x 不足")
            continue
        print(f"\n  操作变量：{op}  (safe_x 数量={len(safe_x)})")

        for strat_name, cfg in strategies.items():
            t0     = time.perf_counter()
            result = train_one_op(op, df, safe_x, **cfg)
            elapsed = time.perf_counter() - t0

            if result is None:
                print(f"    策略{strat_name[0]}：估计失败")
                rows.append({
                    "操作节点": op, "策略": strat_name,
                    "theta_med": None, "SE_boot": None,
                    "CV": None, "符号一致率": None,
                    "F统计量": None, "耗时_s": round(elapsed, 1),
                    "状态": "失败",
                })
            else:
                theta_med, p_val, SE, n, f, cv, sr = result
                stable = cv < CV_WARN and sr >= SIGN_RATE_MIN
                flag   = "✓" if stable else "⚠"
                print(f"    策略{strat_name[0]}：θ={theta_med:+.5f}  SE={SE:.5f}  "
                      f"CV={cv:.3f}  sign_rate={sr:.2f}  [{flag}]  耗时={elapsed:.1f}s")
                rows.append({
                    "操作节点":  op,       "策略":      strat_name,
                    "theta_med": round(theta_med, 5),
                    "SE_boot":   round(SE, 5),
                    "CV":        round(cv, 4),
                    "符号一致率": round(sr, 3),
                    "P_Value":   round(p_val, 4),
                    "F统计量":   round(f, 2),
                    "有效残差数": n,
                    "耗时_s":    round(elapsed, 1),
                    "稳定":      stable,
                    "状态":      "成功",
                })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print("[警告] 对比实验无有效结果"); return df_out

    # ── 汇总对比表 ────────────────────────────────────────────────
    print("\n[策略对比汇总]")
    print(f"  {'策略':<20s}  {'CV均值':>8s}  {'SE均值':>9s}  {'稳定率':>7s}  {'耗时均值(s)':>11s}")
    for strat_name in strategies.keys():
        sub = df_out[(df_out["策略"] == strat_name) & (df_out["状态"] == "成功")]
        if sub.empty:
            print(f"  {strat_name:<20s}  （无有效结果）")
            continue
        print(f"  {strat_name:<20s}  "
              f"{sub['CV'].mean():8.3f}  "
              f"{sub['SE_boot'].mean():9.5f}  "
              f"{sub['稳定'].mean():7.1%}  "
              f"{sub['耗时_s'].mean():11.1f}")

    out_path = os.path.join(CF_COMPARE_OUT_DIR, "refutation_cf_compare_v4.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")
    return df_out


# ═══════════════════════════════════════════════════════════════════
#  辅助：格式化 cf_cfg 用于日志打印
# ═══════════════════════════════════════════════════════════════════
def _fmt_cf_cfg(cf_cfg: dict) -> str:
    if not cf_cfg:
        return "默认（扩展窗口，固定折边，无分层，无嵌套LR）"
    parts = []
    wt = cf_cfg.get("window_type", "expanding")
    parts.append(f"window={wt}")
    jr = cf_cfg.get("fold_jitter_ratio", 0.0)
    parts.append(f"jitter={jr:.0%}")
    parts.append("分层=✓" if cf_cfg.get("use_stratified") else "分层=✗")
    parts.append("嵌套LR=✓" if cf_cfg.get("nested_lr_search") else "嵌套LR=✗")
    return "  |  ".join(parts)


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="XIN_2 因果推断反驳实验 v4（两阶段解耦 VAE-DML + 交叉拟合策略改进）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
建议执行顺序：
  # 1. 快速验证（小样本）
  python run_refutation_xin2_v4.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 2. 开启折边随机化 + 分层检查（推荐默认改进组合）
  python run_refutation_xin2_v4.py --mode stability \\
    --fold_jitter_ratio 0.1 --stratified

  # 3. 策略对比实验（毕设核心对比实验，约用 5×n_ops 次完整训练）
  python run_refutation_xin2_v4.py --mode cf_compare --sample_size 3000 --cf_compare_n_ops 5

  # 4. 反驳实验（带改进策略）
  python run_refutation_xin2_v4.py --mode all \\
    --fold_jitter_ratio 0.1 --stratified --workers 4

  # 5. 嵌套LR搜索（计算开销约 3×，建议单独运行）
  python run_refutation_xin2_v4.py --mode stability --nested_lr_search

v4 输出文件带 _v4 后缀，不覆盖 v3 结果。
        """,
    )
    p.add_argument("--mode", required=True,
                   choices=["stability", "placebo", "random_confounder",
                             "data_subset", "cf_compare", "all"])

    # ── 反驳实验参数（与 v3 一致）────────────────────────────────
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

    # ── v4 新增：交叉拟合策略参数 ────────────────────────────────
    p.add_argument("--window_type", type=str, default="expanding",
                   choices=["expanding", "sliding"],
                   help="交叉拟合窗口类型：expanding（扩展窗口，默认）"
                        " 或 sliding（滑动窗口，仅用近期数据训练）")
    p.add_argument("--fold_jitter_ratio", type=float, default=FOLD_JITTER_RATIO,
                   help=f"折边随机化比例（默认 {FOLD_JITTER_RATIO}，"
                        "每次 Bootstrap 折边偏移 ±ratio×block_size；"
                        "设为 0 退化为 v3 固定折边行为）")
    p.add_argument("--stratified", action="store_true", default=False,
                   help="开启分层折检查（适用于 D 严重不平衡场景）："
                        "训练折中高处理量样本不足 MIN_TREAT_SAMPLES 时跳过该折")
    p.add_argument("--nested_lr_search", action="store_true", default=False,
                   help="开启嵌套学习率搜索（改进4）：在每个外层折的训练集上"
                        "做内层验证选最优LR，计算开销约 3×，外层测试集完全隔离")
    p.add_argument("--cf_compare_n_ops", type=int, default=5,
                   help="策略对比实验（cf_compare）参与对比的操作变量数（默认 5）")

    # ── DAG / 操作性分类（与 v3 一致）───────────────────────────
    p.add_argument("--dag-roles-csv", type=str, default="",
                   help="DAG 角色明细 CSV 路径（analyze_dag_causal_roles_v4_1.py 的输出）")
    p.add_argument("--operability-csv", type=str, default=DEFAULT_OPERABILITY_CSV,
                   help=f"操作性分类 CSV 路径（默认 {DEFAULT_OPERABILITY_CSV}）")
    return p.parse_args()


def main():
    args = parse_args()
    global N_BOOTSTRAP
    N_BOOTSTRAP = args.n_bootstrap

    # ── 构建交叉拟合策略配置字典 ──────────────────────────────────
    cf_cfg = dict(
        window_type       = args.window_type,
        fold_jitter_ratio = args.fold_jitter_ratio,
        use_stratified    = args.stratified,
        nested_lr_search  = args.nested_lr_search,
    )

    print("=" * 70)
    print(f" XIN_2 因果推断反驳实验 v4  |  模式: {args.mode.upper()}")
    print(f" 设备: {DEVICE}  |  并行线程: {args.workers}  |  Bootstrap: {N_BOOTSTRAP}")
    print(f" 架构: 两阶段解耦 VAE-DML（Stage1=VAE表征，Stage2=独立预测头）")
    print(f" 交叉拟合策略: {_fmt_cf_cfg(cf_cfg)}")
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

    dag_csv   = args.dag_roles_csv or DEFAULT_DAG_ROLES_CSV
    dag_roles = load_dag_roles(dag_csv)
    if not dag_roles:
        print("[注意] 未加载 DAG 角色信息，将使用纯相关性筛选")

    mode = args.mode
    if mode in ("stability", "all"):
        run_stability_diagnosis(df, set(ops), set(states),
                                workers=args.workers,
                                dag_roles=dag_roles,
                                cf_cfg=cf_cfg)
    if mode in ("placebo", "all"):
        run_placebo(df, set(ops), set(states),
                    n_permutations=args.n_permutations,
                    workers=args.workers,
                    dag_roles=dag_roles,
                    cf_cfg=cf_cfg)
    if mode in ("random_confounder", "all"):
        run_random_confounder(df, set(ops), set(states),
                              n_confounders=args.n_confounders,
                              n_repeats=args.n_repeats,
                              workers=args.workers,
                              dag_roles=dag_roles,
                              cf_cfg=cf_cfg)
    if mode in ("data_subset", "all"):
        run_data_subset(df, set(ops), set(states),
                        n_subsets=args.n_subsets,
                        subset_frac=args.subset_frac,
                        workers=args.workers,
                        dag_roles=dag_roles,
                        cf_cfg=cf_cfg)
    if mode == "cf_compare":
        run_cf_compare(df, set(ops), set(states),
                       dag_roles=dag_roles,
                       n_ops=args.cf_compare_n_ops)

    print("\n" + "=" * 70)
    print(" 全部实验完成，结果保存至：")
    for label, path in [
        ("稳定性诊断",   STABILITY_OUT_DIR),
        ("安慰剂实验",   PLACEBO_OUT_DIR),
        ("随机混杂",     RANDOM_CONFOUNDER_OUT_DIR),
        ("数据子集",     DATA_SUBSET_OUT_DIR),
        ("策略对比",     CF_COMPARE_OUT_DIR),
    ]:
        print(f"  {label:8s}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
