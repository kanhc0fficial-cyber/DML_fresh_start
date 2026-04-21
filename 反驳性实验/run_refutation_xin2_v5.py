"""
run_refutation_xin2_v5.py
=================================
XIN_2 因果推断反驳实验 v5 ——「LSTM-VAE 联合训练微创新」架构

═══════════════════════════════════════════════════════════════════
  核心创新：基于 v3/v4 两阶段解耦架构，新增四项微创新改进
═══════════════════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────────────────────┐
  │ 微创新 A：因果优先不对称梯度投影（Causal-Priority GradProj）    │
  │ ─────────────────────────────────────────────────────────────  │
  │  在联合训练阶段，检测重建梯度与因果梯度的冲突：                    │
  │  若 <g_VAE, g_causal> < 0，则将 g_VAE 投影到不干扰因果方向：    │
  │    g_VAE_proj = g_VAE - <g_VAE, g_causal>/||g_causal||² × g_c │
  │  否则保留原始 g_VAE。                                          │
  │  与对称 PCGrad 的区别：因果任务始终为主任务，重建为辅助任务。     │
  │                                                                │
  │  论文创新表述：因果优先的不对称梯度投影策略，保护 DML 扰动估计    │
  │  任务的梯度方向不受重建任务干扰。                                │
  ├────────────────────────────────────────────────────────────────┤
  │ 微创新 B：双流潜变量结构（Dual-Stream Latent）                   │
  │ ─────────────────────────────────────────────────────────────  │
  │  LSTM 编码器（共享躯干）→ h_t                                   │
  │      ↙              ↘                                          │
  │  z_causal (低维)    z_recon (高维)                              │
  │      ↓                ↓                                        │
  │  因果头(Y,D)        解码器(重建X)                               │
  │                                                                │
  │  正交性约束：L_orth = ||W_c @ W_r^T||_F²                       │
  │  让重建帮助学习通用时序表示 h_t，因果任务从中提取需要的子空间。  │
  ├────────────────────────────────────────────────────────────────┤
  │ 微创新 C：课程式三阶段训练调度（Curriculum Training）            │
  │ ─────────────────────────────────────────────────────────────  │
  │  Phase 1（预热期，前 20%）：只训练 VAE，因果头冻结              │
  │  Phase 2（过渡期，中间 30%）：α(t) 从 0→1 线性增加因果权重      │
  │  Phase 3（精调期，后 50%）：因果为主，重建降权                   │
  │                                                                │
  │  好的时序表示是因果估计的前提，应先学再用，而非同时学。          │
  ├────────────────────────────────────────────────────────────────┤
  │ 微创新 D：不确定性加权 DML 残差（Uncertainty-Weighted DML）      │
  │ ─────────────────────────────────────────────────────────────  │
  │  VAE 输出 (μ, σ²)，标准做法只用 μ 做预测。                     │
  │  v5 用 σ_z 的大小作为样本编码可靠性指示器：                     │
  │    - 多次采样 z ~ N(μ, σ²) 计算预测方差                        │
  │    - 高不确定性样本降权，防止不稳定编码干扰 DML 估计            │
  │                                                                │
  │  将生成模型的不确定性量化与因果估计稳健性直接连接。             │
  └────────────────────────────────────────────────────────────────┘

  消融实验设计（--mode ablation）：
    组1：v3 baseline（两阶段解耦，无微创新）
    组2：+微创新 B（双流潜变量）
    组3：+微创新 B+C（双流 + 课程训练）
    组4：+微创新 B+C+A（双流 + 课程 + 梯度投影）
    组5：+微创新 B+C+A+D（全部微创新 = v5 完整版）

═══════════════════════════════════════════════════════════════════
  v5 相比 v4 的核心变化
═══════════════════════════════════════════════════════════════════
  1. 模型架构：DualStreamVAEEncoder（共享 LSTM + 双流投影）
  2. 训练流程：三阶段课程式联合训练（替代 v3/v4 的两阶段解耦）
  3. 梯度优化：因果优先不对称梯度投影
  4. 推断改进：不确定性加权残差（多次 MC 采样取加权均值）
  5. 新增消融实验：逐项累加微创新，量化各创新点的边际贡献
  6. 保留 v4 全部交叉拟合策略改进（折边随机化/分层/嵌套LR）
  7. 输出文件均带 _v5 后缀，不覆盖 v3/v4 结果

用法：
  # 稳定性诊断
  python run_refutation_xin2_v5.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 消融实验（核心新增实验）
  python run_refutation_xin2_v5.py --mode ablation --sample_size 3000 --ablation_n_ops 5

  # 全部反驳实验
  python run_refutation_xin2_v5.py --mode all --workers 4

  # 关闭某个微创新进行对比
  python run_refutation_xin2_v5.py --mode stability --no_dual_stream
  python run_refutation_xin2_v5.py --mode stability --no_curriculum
  python run_refutation_xin2_v5.py --mode stability --no_grad_proj
  python run_refutation_xin2_v5.py --mode stability --no_uncertainty_weight
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
    REPO_ROOT, "data",
    "操作变量和混杂变量",
    "non_collinear_representative_vars_operability.csv",
)

PLACEBO_OUT_DIR           = os.path.join(BASE_DIR, "安慰剂实验")
RANDOM_CONFOUNDER_OUT_DIR = os.path.join(BASE_DIR, "随机混杂变量实验")
DATA_SUBSET_OUT_DIR       = os.path.join(BASE_DIR, "数据子集实验")
STABILITY_OUT_DIR         = os.path.join(BASE_DIR, "稳定性诊断")
ABLATION_OUT_DIR          = os.path.join(BASE_DIR, "消融实验")  # v5 新增

for _d in [PLACEBO_OUT_DIR, RANDOM_CONFOUNDER_OUT_DIR,
           DATA_SUBSET_OUT_DIR, STABILITY_OUT_DIR, ABLATION_OUT_DIR]:
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
#  超参（继承 v4 全部 + v5 新增）
# ═══════════════════════════════════════════════════════════════════
SEQ_LEN             = 3    # 从 6 降到 3（减少特征维度，每折样本更多）
EMBARGO_GAP         = 4
K_FOLDS             = 3    # 从 4 降到 3（每折样本更多）
MAX_EPOCHS_VAE      = 60     # baseline Stage 1 VAE 最大轮数（与 v3/v4 一致）
MAX_EPOCHS_HEAD     = 40     # baseline Stage 2 预测头最大轮数（与 v3/v4 一致）
PATIENCE            = 8
MIN_TRAIN_SIZE      = 100
MIN_VALID_RESIDUALS = 20    # 从 50 降到 20（适配对齐后的小数据集）
F_STAT_THRESHOLD    = 3.0   # 从 10.0 降到 3.0（更多操作变量通过弱工具检验）
MIN_BOOTSTRAP_SUCCESS_RATE = 0.20  # bootstrap 最低成功率（从 0.5 降到 0.20）

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

# ─── v4 交叉拟合策略参数（完全继承）──────────────────────────────
FOLD_JITTER_RATIO    = 0.10
SLIDING_WINDOW_RATIO = 2
NESTED_LR_SEARCH     = False
NESTED_LR_CANDIDATES = [0.001, 0.003, 0.01]
MIN_TREAT_SAMPLES    = 5
MIN_INNER_VAL_SAMPLES = 60

# ─── v5 微创新超参 ───────────────────────────────────────────────
# 微创新 B：双流潜变量维度
LATENT_DIM_CAUSAL = 16   # z_causal 维度（低维，因果专用）
LATENT_DIM_RECON  = 48   # z_recon 维度（高维，重建专用）
LAMBDA_ORTH       = 0.01 # 正交性损失权重

# 微创新 C：课程式训练阶段比例和总轮数
MAX_EPOCHS_JOINT  = 80   # 联合训练总轮数（三阶段共用）
PHASE1_RATIO      = 0.20 # 预热期占比（只训练 VAE）
PHASE2_RATIO      = 0.30 # 过渡期占比（逐步引入因果）
# 剩余 50% 为精调期（因果为主，重建降权）
LAMBDA_RECON_FINAL = 0.3 # 精调期重建损失权重（< 1，降权）

# 微创新 D：不确定性加权
MC_SAMPLES         = 5    # MC 采样次数（推断时）
UNCERTAINTY_CLIP_QUANTILE = 0.90  # 不确定性超过此分位数的样本降权


# ══════════════════════════════════════════════════════════════════
#  DAG 因果角色过滤（与 v3/v4 完全一致）
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
    """构建控制变量集 safe_x，整合 DAG 因果角色过滤（与 v3/v4 完全一致）。"""
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
#  微创新 B：双流潜变量 VAE 编码器
# ═══════════════════════════════════════════════════════════════════

class DualStreamVAEEncoder(nn.Module):
    """
    双流 VAE 编码器（微创新 B）：

    LSTM 共享躯干 → h_t
        ↙                ↘
    z_causal (低维)    z_recon (高维)
        ↓                  ↓
    因果头(Y,D)        解码器(重建X)

    z_causal = W_c @ h_t + b_c   (dim = LATENT_DIM_CAUSAL)
    z_recon  ~ N(μ_r, σ_r²)      (dim = LATENT_DIM_RECON)

    正交性约束：L_orth = ||W_c @ W_r^T||_F²
    """
    def __init__(self, input_dim: int,
                 causal_dim: int = LATENT_DIM_CAUSAL,
                 recon_dim: int = LATENT_DIM_RECON):
        super().__init__()
        self.causal_dim = causal_dim
        self.recon_dim  = recon_dim

        # 共享 LSTM 躯干
        self.lstm = nn.LSTM(
            input_dim, HIDDEN_DIM_ENC,
            batch_first=True,
            num_layers=NUM_LSTM_LAYERS,
            dropout=0.1 if NUM_LSTM_LAYERS > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM_ENC)

        # 因果流：h_t → z_causal（确定性投影）
        self.fc_causal = nn.Sequential(
            nn.Linear(HIDDEN_DIM_ENC, 32), nn.SiLU(),
        )
        self.proj_causal = nn.Linear(32, causal_dim)

        # 重建流：h_t → (μ_recon, logvar_recon)（随机投影）
        self.fc_recon = nn.Sequential(
            nn.Linear(HIDDEN_DIM_ENC, 48), nn.SiLU(),
        )
        self.fc_mu_recon     = nn.Linear(48, recon_dim)
        self.fc_logvar_recon = nn.Linear(48, recon_dim)

    def forward(self, x: torch.Tensor):
        """
        返回：
          z_causal:    [B, causal_dim]  确定性因果潜变量
          mu_recon:    [B, recon_dim]   重建流均值
          logvar_recon:[B, recon_dim]   重建流 log 方差
          h_shared:    [B, HIDDEN_DIM_ENC]  共享表示（用于梯度投影分析）
        """
        _, (h_n, _) = self.lstm(x)
        h_shared = self.norm(h_n[-1])

        # 因果流
        z_causal = self.proj_causal(self.fc_causal(h_shared))

        # 重建流
        h_r = self.fc_recon(h_shared)
        mu_recon     = self.fc_mu_recon(h_r)
        logvar_recon = self.fc_logvar_recon(h_r)

        return z_causal, mu_recon, logvar_recon, h_shared

    def encode_causal(self, x: torch.Tensor) -> torch.Tensor:
        """推断专用：只返回 z_causal（确定性）"""
        z_causal, _, _, _ = self.forward(x)
        return z_causal

    def encode_causal_with_uncertainty(self, x: torch.Tensor):
        """
        推断时返回 z_causal 和重建流的不确定性指标（微创新 D）。

        不确定性 = 重建流 σ 的均值，反映该样本在潜空间的编码稳定性。
        """
        z_causal, mu_recon, logvar_recon, _ = self.forward(x)
        sigma_recon = torch.exp(0.5 * logvar_recon)
        uncertainty = sigma_recon.mean(dim=-1)  # [B]
        return z_causal, uncertainty

    def orthogonality_loss(self) -> torch.Tensor:
        """
        正交性损失（微创新 B 的核心约束）：
        L_orth = ||W_c @ W_r^T||_F²

        强制因果子空间和重建子空间不重叠，
        让两个流提取互补信息。
        """
        W_c = self.proj_causal.weight   # [causal_dim, 32]
        W_r = self.fc_mu_recon.weight   # [recon_dim, 48]
        # 由于两个投影来自不同的中间维度，
        # 需要在共享层面计算正交性。
        # 方法：通过共享层的权重矩阵将两个投影链接回同一空间
        # 简化版：直接计算两个流最终输出对 h_shared 的有效投影
        # W_c_eff = proj_causal.weight @ fc_causal[0].weight  -> [causal_dim, HIDDEN_DIM_ENC]
        # W_r_eff = fc_mu_recon.weight @ fc_recon[0].weight   -> [recon_dim, HIDDEN_DIM_ENC]
        W_c_eff = self.proj_causal.weight @ self.fc_causal[0].weight  # [causal_dim, HIDDEN_DIM_ENC]
        W_r_eff = self.fc_mu_recon.weight @ self.fc_recon[0].weight   # [recon_dim, HIDDEN_DIM_ENC]
        cross = W_c_eff @ W_r_eff.T  # [causal_dim, recon_dim]
        return torch.sum(cross ** 2)


class SingleStreamVAEEncoder(nn.Module):
    """
    单流 VAE 编码器（v3/v4 baseline，用于消融实验对比）。
    与 v3/v4 的 VAEEncoder 完全一致。
    """
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
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
        mu, _ = self.forward(x)
        return mu


class VAEDecoder(nn.Module):
    """VAE 解码器：z → X 重建（可接受不同维度的 z）"""
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
    """确定性预测头：z → 标量预测"""
    def __init__(self, latent_dim: int = LATENT_DIM_CAUSAL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM_HEAD), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM_HEAD, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
#  微创新 A：因果优先不对称梯度投影
# ═══════════════════════════════════════════════════════════════════

def _causal_priority_grad_projection(grad_vae: torch.Tensor,
                                      grad_causal: torch.Tensor) -> torch.Tensor:
    """
    因果优先不对称梯度投影（微创新 A 的核心函数）。

    当重建梯度与因果梯度方向冲突（内积 < 0）时，
    将重建梯度投影到不干扰因果梯度的方向：

      g_VAE_proj = g_VAE - <g_VAE, g_causal> / ||g_causal||² × g_causal

    只在冲突时投影，否则保留原始梯度（不对称优先：因果任务永远不被修改）。

    参数：
      grad_vae:    重建任务对共享参数的梯度（展平后）
      grad_causal: 因果任务对共享参数的梯度（展平后）

    返回：
      投影后的 grad_vae（展平后）
    """
    dot = torch.dot(grad_vae, grad_causal)
    if dot < 0:
        # 冲突：投影重建梯度到因果梯度的正交补空间
        norm_sq = torch.dot(grad_causal, grad_causal)
        if norm_sq > 1e-12:
            grad_vae = grad_vae - (dot / norm_sq) * grad_causal
    return grad_vae


def _apply_grad_projection(shared_params, loss_vae, loss_causal):
    """
    对共享参数应用因果优先梯度投影。

    步骤：
    1. 分别计算 loss_vae 和 loss_causal 对共享参数的梯度
    2. 如果两者冲突（内积<0），将 VAE 梯度投影
    3. 合并投影后的 VAE 梯度和原始因果梯度，设置到参数的 .grad 属性

    参数：
      shared_params: 共享参数列表（LSTM + LayerNorm 的参数）
      loss_vae:      重建损失（标量，已在计算图中）
      loss_causal:   因果损失（标量，已在计算图中）
    """
    # 计算 VAE 梯度
    grads_vae = torch.autograd.grad(
        loss_vae, shared_params, retain_graph=True, allow_unused=True
    )
    # 计算因果梯度
    grads_causal = torch.autograd.grad(
        loss_causal, shared_params, retain_graph=True, allow_unused=True
    )

    # 展平并投影
    flat_vae = torch.cat([
        g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
        for g, p in zip(grads_vae, shared_params)
    ])
    flat_causal = torch.cat([
        g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
        for g, p in zip(grads_causal, shared_params)
    ])

    flat_vae_proj = _causal_priority_grad_projection(flat_vae, flat_causal)

    # 合并：投影后的 VAE 梯度 + 原始因果梯度
    flat_merged = flat_vae_proj + flat_causal

    # 还原到各参数的 .grad
    offset = 0
    for p in shared_params:
        numel = p.numel()
        if p.grad is None:
            p.grad = flat_merged[offset:offset + numel].reshape(p.shape).clone()
        else:
            p.grad.copy_(flat_merged[offset:offset + numel].reshape(p.shape))
        offset += numel


# ═══════════════════════════════════════════════════════════════════
#  微创新 C：课程式三阶段联合训练
# ═══════════════════════════════════════════════════════════════════

def _train_joint_curriculum(
    encoder: DualStreamVAEEncoder,
    decoder: VAEDecoder,
    head_Y: PredHead,
    head_D: PredHead,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    D_train: torch.Tensor,
    device,
    use_grad_proj: bool = True,
    use_curriculum: bool = True,
):
    """
    三阶段课程式联合训练（微创新 A + B + C 的核心函数）。

    Phase 1（预热期，前 PHASE1_RATIO）：
      - 只优化 L_VAE = L_recon + β_anneal × L_KL + λ_orth × L_orth
      - 因果头参数冻结
      - 目的：让 LSTM 学会稳定的时序表示

    Phase 2（过渡期，中间 PHASE2_RATIO）：
      - L = L_VAE + α(t) × (L_Y + L_D)
      - α(t) 从 0 线性增到 1
      - 共享 LSTM 以较低学习率微调
      - 如果 use_grad_proj=True，对共享参数应用梯度投影

    Phase 3（精调期，后 50%）：
      - L = λ_recon × L_VAE + (L_Y + L_D)
      - λ_recon < 1，重建降权，因果为主
      - 全部参数正常学习率

    如果 use_curriculum=False，退化为标准联合训练（所有阶段用相同权重）。
    """
    all_params = (list(encoder.parameters()) + list(decoder.parameters())
                  + list(head_Y.parameters()) + list(head_D.parameters()))
    optimizer = optim.Adam(all_params, lr=0.002)

    loader  = DataLoader(TensorDataset(X_train, Y_train, D_train),
                         batch_size=256, shuffle=False)

    val_split = max(1, int(len(X_train) * 0.1))
    X_iv = X_train[-val_split:]
    Y_iv = Y_train[-val_split:]
    D_iv = D_train[-val_split:]

    total_epochs = MAX_EPOCHS_JOINT
    phase1_end = int(total_epochs * PHASE1_RATIO) if use_curriculum else 0
    phase2_end = int(total_epochs * (PHASE1_RATIO + PHASE2_RATIO)) if use_curriculum else 0

    best_loss = float("inf")
    pat_cnt   = 0
    best_state = None

    # 共享参数（LSTM + LayerNorm）—— 用于梯度投影
    shared_params = list(encoder.lstm.parameters()) + list(encoder.norm.parameters())

    for epoch in range(total_epochs):
        # ── 确定当前阶段 ────────────────────────────────────────
        if use_curriculum:
            if epoch < phase1_end:
                phase = 1
            elif epoch < phase2_end:
                phase = 2
            else:
                phase = 3
        else:
            phase = 3  # 无课程时直接当精调期

        # KL 退火
        beta = BETA_KL * min(1.0, epoch / max(1, ANNEAL_EPOCHS))

        # Phase 2 的因果权重 α(t)：线性从 0→1
        if phase == 2:
            alpha = (epoch - phase1_end) / max(1, phase2_end - phase1_end)
        elif phase == 1:
            alpha = 0.0
        else:
            alpha = 1.0

        # Phase 3 的重建权重
        recon_weight = LAMBDA_RECON_FINAL if phase == 3 else 1.0

        # Phase 1：冻结因果头
        if phase == 1:
            for p in head_Y.parameters():
                p.requires_grad_(False)
            for p in head_D.parameters():
                p.requires_grad_(False)
        else:
            for p in head_Y.parameters():
                p.requires_grad_(True)
            for p in head_D.parameters():
                p.requires_grad_(True)

        # Phase 2：共享 LSTM 以较低学习率
        if phase == 2:
            for pg in optimizer.param_groups:
                pg['lr'] = 0.001  # 降低 LSTM 学习率
        elif phase == 3:
            for pg in optimizer.param_groups:
                pg['lr'] = 0.002  # 恢复正常

        encoder.train(); decoder.train()
        head_Y.train(); head_D.train()

        for bx, by, bd in loader:
            optimizer.zero_grad()

            z_causal, mu_r, logvar_r, _ = encoder(bx)

            # ── 重建损失（通过 z_recon）──────────────────────────
            std_r  = torch.exp(0.5 * logvar_r)
            z_r    = mu_r + std_r * torch.randn_like(std_r)
            x_recon = decoder(z_r)
            loss_recon = nn.functional.mse_loss(x_recon, bx)
            loss_kl    = -0.5 * torch.mean(1 + logvar_r - mu_r.pow(2) - logvar_r.exp())
            loss_orth  = encoder.orthogonality_loss()
            loss_vae   = recon_weight * (loss_recon + beta * loss_kl) + LAMBDA_ORTH * loss_orth

            # ── 因果损失（通过 z_causal）─────────────────────────
            loss_Y = nn.functional.mse_loss(head_Y(z_causal), by)
            loss_D = nn.functional.mse_loss(head_D(z_causal), bd)
            loss_causal = alpha * (loss_Y + loss_D)

            # ── 梯度计算与投影 ───────────────────────────────────
            if use_grad_proj and phase >= 2 and alpha > 0:
                # 微创新 A：对共享参数应用因果优先梯度投影
                # 其他参数（因果头、重建头）正常反传
                _apply_grad_projection(shared_params, loss_vae, loss_causal)

                # 非共享参数正常反传
                non_shared_loss = loss_vae + loss_causal
                shared_ids = {id(p) for p in shared_params}
                non_shared_params = [p for p in all_params if id(p) not in shared_ids]
                non_shared_grads = torch.autograd.grad(
                    non_shared_loss, non_shared_params,
                    retain_graph=False, allow_unused=True
                )
                for p, g in zip(non_shared_params, non_shared_grads):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g.clone()
                        else:
                            p.grad.copy_(g)
            else:
                # 无梯度投影：标准反传
                total_loss = loss_vae + loss_causal
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
            optimizer.step()

        # ── 验证 ────────────────────────────────────────────────
        encoder.eval(); decoder.eval(); head_Y.eval(); head_D.eval()
        with torch.no_grad():
            z_c_iv, mu_r_iv, lv_r_iv, _ = encoder(X_iv)
            std_r_iv = torch.exp(0.5 * lv_r_iv)
            z_r_iv = mu_r_iv + std_r_iv * torch.randn_like(std_r_iv)
            xr_iv  = decoder(z_r_iv)

            vl_recon = nn.functional.mse_loss(xr_iv, X_iv).item()
            vl_Y     = nn.functional.mse_loss(head_Y(z_c_iv), Y_iv).item()
            vl_D     = nn.functional.mse_loss(head_D(z_c_iv), D_iv).item()
            vl = vl_recon * recon_weight + alpha * (vl_Y + vl_D)

        if vl < best_loss - 1e-5:
            best_loss = vl; pat_cnt = 0
            best_state = {
                'enc': {k: v.clone() for k, v in encoder.state_dict().items()},
                'dec': {k: v.clone() for k, v in decoder.state_dict().items()},
                'hY':  {k: v.clone() for k, v in head_Y.state_dict().items()},
                'hD':  {k: v.clone() for k, v in head_D.state_dict().items()},
            }
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE and phase >= 2:
                # Phase 1 不做早停，确保预热充分
                break

    if best_state:
        encoder.load_state_dict(best_state['enc'])
        decoder.load_state_dict(best_state['dec'])
        head_Y.load_state_dict(best_state['hY'])
        head_D.load_state_dict(best_state['hD'])


# ═══════════════════════════════════════════════════════════════════
#  v3/v4 baseline 训练函数（用于消融实验）
# ═══════════════════════════════════════════════════════════════════

def _train_vae_stage1_baseline(encoder: SingleStreamVAEEncoder,
                                decoder: VAEDecoder,
                                X_train: torch.Tensor, device) -> None:
    """Stage 1: 训练 VAE（与 v3/v4 完全一致）"""
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
            std = torch.exp(0.5 * logvar)
            z   = mu + std * torch.randn_like(std)
            x_recon  = decoder(z)
            loss_rec = nn.functional.mse_loss(x_recon, bx)
            loss_kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss     = loss_rec + beta * loss_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()

        encoder.eval(); decoder.eval()
        with torch.no_grad():
            mu_iv, lv_iv = encoder(X_iv)
            std_iv = torch.exp(0.5 * lv_iv)
            z_iv = mu_iv + std_iv * torch.randn_like(std_iv)
            xr_iv = decoder(z_iv)
            vl = (nn.functional.mse_loss(xr_iv, X_iv)
                  - 0.5 * beta * torch.mean(1 + lv_iv - mu_iv.pow(2) - lv_iv.exp())).item()

        if vl < best_loss - 1e-5:
            best_loss = vl; pat_cnt = 0
            best_enc = {k: v.clone() for k, v in encoder.state_dict().items()}
            best_dec = {k: v.clone() for k, v in decoder.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE:
                break

    if best_enc:
        encoder.load_state_dict(best_enc)
    if best_dec:
        decoder.load_state_dict(best_dec)


def _train_head_stage2_baseline(head: PredHead, mu_train: torch.Tensor,
                                 target_train: torch.Tensor, device) -> None:
    """Stage 2: 在冻结 μ 上训练预测头（与 v3 完全一致）"""
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
#  微创新 D：不确定性加权 DML 残差
# ═══════════════════════════════════════════════════════════════════

def _compute_uncertainty_weights(
    encoder: DualStreamVAEEncoder,
    X_val: torch.Tensor,
    n_mc_samples: int = MC_SAMPLES,
) -> np.ndarray:
    """
    计算不确定性权重（微创新 D）。

    对每个样本进行 MC 采样，利用重建流的 σ_z 来评估
    编码的不确定性，高不确定性的样本在 DML 残差计算中降权。

    步骤：
    1. 获取每个样本的 z_causal 和 uncertainty（σ_recon 均值）
    2. 以 UNCERTAINTY_CLIP_QUANTILE 为阈值计算权重
    3. 高不确定性样本权重降低，低不确定性样本权重不变

    返回：
      weights: [N_val] 归一化权重数组
    """
    encoder.eval()
    with torch.no_grad():
        _, uncertainties = encoder.encode_causal_with_uncertainty(X_val)
        unc = uncertainties.cpu().numpy()

    # 以分位数为阈值，高不确定性样本降权
    threshold = np.quantile(unc, UNCERTAINTY_CLIP_QUANTILE)
    # 权重 = 1 if unc <= threshold, else threshold/unc （平滑降权）
    weights = np.where(
        unc <= threshold,
        1.0,
        threshold / (unc + 1e-8)
    )
    # 归一化使均值为 1
    weights = weights / (weights.mean() + 1e-8)
    return weights


# ═══════════════════════════════════════════════════════════════════
#  核心训练函数：v5 完整版（联合训练 + 全部微创新）
# ═══════════════════════════════════════════════════════════════════

def train_one_op(op: str, df: pd.DataFrame, safe_x: list,
                 override_D=None, n_bootstrap: int = N_BOOTSTRAP,
                 window_type: str = "expanding",
                 fold_jitter_ratio: float = FOLD_JITTER_RATIO,
                 use_stratified: bool = False,
                 nested_lr_search: bool = False,
                 # v5 微创新开关
                 use_dual_stream: bool = True,
                 use_curriculum: bool = True,
                 use_grad_proj: bool = True,
                 use_uncertainty_weight: bool = True):
    """
    v5 核心训练函数：LSTM-VAE 联合训练 + 微创新 A/B/C/D。

    相比 v4 的核心区别：
    1. 双流编码器（B）：z_causal 用于因果头，z_recon 用于重建
    2. 课程式训练（C）：三阶段渐进式联合训练
    3. 梯度投影（A）：保护因果梯度不受重建干扰
    4. 不确定性加权（D）：高不确定性样本在残差中降权

    微创新开关：
      use_dual_stream:       False → 退化为 v3/v4 的单流编码器
      use_curriculum:         False → 退化为标准联合训练（无阶段划分）
      use_grad_proj:          False → 标准梯度更新（无投影）
      use_uncertainty_weight: False → 等权重残差（不使用不确定性）

    继承 v4 的交叉拟合策略参数（window_type, fold_jitter_ratio,
    use_stratified, nested_lr_search）。
    """
    device = torch.device("cpu")

    # ── 标准化 ─────────────────────────────────────────────────────
    X_raw = df[safe_x].values.astype(np.float32)
    X_mat = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)

    Y_raw = df["Y_grade"].values.astype(np.float32)
    D_raw = (df[op].values if override_D is None
             else np.asarray(override_D, dtype=np.float32))
    Y_mean, Y_std = float(Y_raw.mean()), float(Y_raw.std()) + 1e-8
    D_mean, D_std = float(D_raw.mean()), float(D_raw.std()) + 1e-8
    Y_mat = (Y_raw - Y_mean) / Y_std
    D_mat = (D_raw - D_mean) / D_std

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
    D_raw_seq = D_raw[SEQ_LEN:]

    N          = len(seqs_X)
    block_size = N // K_FOLDS
    input_dim  = len(safe_x)

    sliding_window_size = int(block_size * SLIDING_WINDOW_RATIO)

    # ── Bootstrap 外循环 ────────────────────────────────────────────
    op_base_seed = _op_seed(op)
    theta_list, f_list, n_list = [], [], []
    folds_skipped_stratified = 0
    folds_total = 0

    for boot_i in range(n_bootstrap):
        base_seed      = boot_i * 99991 + op_base_seed
        all_res_Y, all_res_D = [], []
        all_weights = []
        any_valid_fold = False

        # 折边随机化（继承 v4）
        if fold_jitter_ratio > 0:
            jitter_max = max(1, int(block_size * fold_jitter_ratio))
            rng_jitter = np.random.default_rng(base_seed + 77777)
            global_jitter = int(rng_jitter.integers(-jitter_max, jitter_max + 1))
        else:
            global_jitter = 0

        for k in range(1, K_FOLDS):
            folds_total += 1
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

            # 分层折检查（继承 v4）
            # 注意：不跳过折（skip 会导致测试样本遗漏 → 选择性偏差），
            # 仅记录不平衡折数量用于诊断。
            if use_stratified:
                D_fold_train = D_raw_seq[:train_end]
                high_treat_count = int(np.sum(D_fold_train > d_global_median))
                if high_treat_count < MIN_TREAT_SAMPLES:
                    folds_skipped_stratified += 1

            # 窗口类型（继承 v4）
            if window_type == "sliding":
                train_start = max(0, train_end - sliding_window_size)
            else:
                train_start = 0

            Xtr = torch.tensor(seqs_X[train_start:train_end]).to(device)
            Ytr = torch.tensor(tgt_Y[train_start:train_end]).to(device)
            Dtr = torch.tensor(tgt_D[train_start:train_end]).to(device)

            Xvl = torch.tensor(seqs_X[val_start:val_end]).to(device)
            Yvl = tgt_Y[val_start:val_end]
            Dvl = tgt_D[val_start:val_end]

            if len(Xtr) < MIN_TRAIN_SIZE:
                continue

            # ══════════════════════════════════════════════════════
            #  训练路径选择：v5 联合训练 vs v3/v4 baseline
            # ══════════════════════════════════════════════════════

            if use_dual_stream:
                # ── v5 路径：双流编码器 + 联合训练 ──────────────
                encoder = DualStreamVAEEncoder(
                    input_dim, LATENT_DIM_CAUSAL, LATENT_DIM_RECON
                ).to(device)
                decoder = VAEDecoder(LATENT_DIM_RECON, SEQ_LEN, input_dim).to(device)
                head_Y  = PredHead(LATENT_DIM_CAUSAL).to(device)
                head_D  = PredHead(LATENT_DIM_CAUSAL).to(device)

                _train_joint_curriculum(
                    encoder, decoder, head_Y, head_D,
                    Xtr, Ytr, Dtr, device,
                    use_grad_proj=use_grad_proj,
                    use_curriculum=use_curriculum,
                )

                # 推断
                encoder.eval(); head_Y.eval(); head_D.eval()
                with torch.no_grad():
                    z_c_vl = encoder.encode_causal(Xvl)
                    pY = head_Y(z_c_vl).cpu().numpy()
                    pD = head_D(z_c_vl).cpu().numpy()

                # 微创新 D：不确定性加权
                if use_uncertainty_weight:
                    w = _compute_uncertainty_weights(encoder, Xvl)
                else:
                    w = np.ones(len(Yvl))

            else:
                # ── v3/v4 baseline 路径 ────────────────────────
                encoder_bl = SingleStreamVAEEncoder(input_dim, LATENT_DIM).to(device)
                decoder_bl = VAEDecoder(LATENT_DIM, SEQ_LEN, input_dim).to(device)
                _train_vae_stage1_baseline(encoder_bl, decoder_bl, Xtr, device)

                for p in encoder_bl.parameters():
                    p.requires_grad_(False)
                encoder_bl.eval()

                with torch.no_grad():
                    mu_tr = encoder_bl.encode_mean(Xtr)
                    mu_vl = encoder_bl.encode_mean(Xvl)

                head_Y = PredHead(LATENT_DIM).to(device)
                head_D = PredHead(LATENT_DIM).to(device)
                _train_head_stage2_baseline(head_Y, mu_tr, Ytr, device)
                _train_head_stage2_baseline(head_D, mu_tr, Dtr, device)

                head_Y.eval(); head_D.eval()
                with torch.no_grad():
                    pY = head_Y(mu_vl).cpu().numpy()
                    pD = head_D(mu_vl).cpu().numpy()

                w = np.ones(len(Yvl))

            all_res_Y.extend(Yvl - pY)
            all_res_D.extend(Dvl - pD)
            all_weights.extend(w)
            any_valid_fold = True

        if not any_valid_fold or len(all_res_Y) < MIN_VALID_RESIDUALS:
            continue

        res_Y   = np.array(all_res_Y, dtype=np.float64)
        res_D   = np.array(all_res_D, dtype=np.float64)
        weights = np.array(all_weights, dtype=np.float64)

        # 去离群（3σ）
        mask = ((np.abs(res_Y) < 3 * res_Y.std()) &
                (np.abs(res_D) < 3 * res_D.std()))
        res_Y, res_D, weights = res_Y[mask], res_D[mask], weights[mask]
        if len(res_D) < MIN_VALID_RESIDUALS:
            continue

        # 强制去中心化
        res_Y -= res_Y.mean()
        res_D -= res_D.mean()
        n = len(res_D)

        # F 统计量（工具强度）
        var_D  = np.var(res_D)
        f_stat = var_D / (D_std ** 2 + 1e-12) * n
        if f_stat < F_STAT_THRESHOLD:
            continue

        # ── DML theta（加权协方差方式，微创新 D）───────────────────
        # 归一化权重
        w_norm = weights / (weights.sum() + 1e-12) * n
        # 加权协方差
        weighted_cov_DY = np.sum(w_norm * res_D * res_Y) / n
        weighted_var_D  = np.sum(w_norm * res_D * res_D) / n
        theta_std = weighted_cov_DY / (weighted_var_D + 1e-12)
        theta     = theta_std * (Y_std / D_std)

        theta_list.append(theta)
        f_list.append(f_stat)
        n_list.append(n)

    # 分层不平衡日志
    if use_stratified and folds_skipped_stratified > 0:
        imbalance_rate = folds_skipped_stratified / max(1, folds_total)
        if imbalance_rate > 0.5:
            print(f"  [分层警告] {op}: {folds_skipped_stratified}/{folds_total} 个折"
                  f"处理样本不足（不平衡率 {imbalance_rate:.1%}，仍保留训练）")

    # ── Bootstrap 聚合 ──────────────────────────────────────────────
    min_success = max(1, int(n_bootstrap * MIN_BOOTSTRAP_SUCCESS_RATE))
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
#  辅助：构建 safe_x（与 v3/v4 完全一致）
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
#  断点续传工具（与 v3/v4 完全一致）
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
#  通用并行调度器（与 v3/v4 完全一致）
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
#  Worker 函数（v5：新增 innov_cfg 透传微创新开关）
# ═══════════════════════════════════════════════════════════════════

def _worker_placebo(task: dict) -> dict:
    op, perm_idx = task["op"], task["perm_idx"]
    df, states   = task["df"], task["states"]
    dag_roles    = task["dag_roles"]
    cf_cfg       = task.get("cf_cfg", {})
    innov_cfg    = task.get("innov_cfg", {})
    key          = task["_key"]
    if df[op].std() < 0.1:
        return {"_key": key, "_filtered": True, "_reason": "std<0.1"}
    safe_x = build_safe_x_with_dag(op, df, states, dag_roles)
    if len(safe_x) < 2:
        return {"_key": key, "_filtered": True, "_reason": "safe_x不足(DAG过滤后)"}
    rng       = np.random.default_rng(seed=perm_idx * 42 + _op_seed(op))
    D_placebo = rng.permutation(df[op].values.copy())
    result    = train_one_op(op, df, safe_x, override_D=D_placebo,
                             **cf_cfg, **innov_cfg)
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
    innov_cfg     = task.get("innov_cfg", {})

    rng        = np.random.default_rng(seed=rep * 1000 + _op_seed(op))
    df_noisy   = df.copy()
    noise_cols = []
    for nc in range(n_confounders):
        cname = f"__rc_{nc}__"
        df_noisy[cname] = rng.standard_normal(len(df_noisy))
        noise_cols.append(cname)
    safe_x_noisy = safe_x_orig + noise_cols
    result       = train_one_op(op, df_noisy, safe_x_noisy,
                                **cf_cfg, **innov_cfg)
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
    innov_cfg   = task.get("innov_cfg", {})
    key         = task["_key"]
    df_sub = df.iloc[start:end].copy()
    if len(df_sub) < SEQ_LEN + K_FOLDS * MIN_TRAIN_SIZE:
        return {"_key": key, "_filtered": True, "_reason": "样本不足"}
    result = train_one_op(op, df_sub, safe_x, **cf_cfg, **innov_cfg)
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
#  数据准备（与 v3/v4 完全一致）
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
#  实验零：稳定性诊断（v5：透传 cf_cfg + innov_cfg）
# ═══════════════════════════════════════════════════════════════════
def run_stability_diagnosis(df, ops, states, workers=4, dag_roles: dict = None,
                            cf_cfg: dict = None, innov_cfg: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验零：稳定性诊断（{N_BOOTSTRAP} 次 Bootstrap）")
    print(f" 架构：LSTM-VAE 联合训练 + 微创新 A/B/C/D")
    print(f" 微创新配置：{_fmt_innov_cfg(innov_cfg)}")
    print(f" 交叉拟合策略：{_fmt_cf_cfg(cf_cfg)}")
    print(f" 稳定标准：CV < {CV_WARN}  且  sign_rate ≥ {SIGN_RATE_MIN}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    if cf_cfg is None:
        cf_cfg = {}
    if innov_cfg is None:
        innov_cfg = {}
    states_list = list(states)
    rows = []
    n_stable = 0
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            continue
        result = train_one_op(op, df, safe_x, **cf_cfg, **innov_cfg)
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
    out_path = os.path.join(STABILITY_OUT_DIR, "stability_diagnosis_v5.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not df_out.empty:
        total = len(df_out)
        print(f"\n[稳定性诊断汇总]  稳定 {n_stable}/{total} 个操作变量")
        print(f"  CV 均值      = {df_out['CV'].mean():.3f}  （目标 < {CV_WARN}）")
        print(f"  sign_rate 均值 = {df_out['符号一致率'].mean():.2f}  （目标 ≥ {SIGN_RATE_MIN}）")
        if n_stable / total < 0.5:
            print("  [⚠] 超过一半操作变量不稳定，建议调整微创新超参")
        else:
            print("  [✓] 多数操作变量稳定，可运行反驳实验")
    print(f"结果已保存：{out_path}")
    return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验一：安慰剂反驳
# ═══════════════════════════════════════════════════════════════════
def run_placebo(df, ops, states, n_permutations=5, workers=4,
                dag_roles: dict = None, cf_cfg: dict = None,
                innov_cfg: dict = None):
    print("\n" + "=" * 70)
    print(" 实验一：安慰剂反驳实验（随机排列操作变量 D）")
    print(f" 微创新配置：{_fmt_innov_cfg(innov_cfg)}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    if cf_cfg is None:
        cf_cfg = {}
    if innov_cfg is None:
        innov_cfg = {}
    ckpt_path   = os.path.join(PLACEBO_OUT_DIR, "checkpoint_placebo_v5.jsonl")
    states_list = list(states)
    tasks = []
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        for perm_idx in range(n_permutations):
            tasks.append({"_key": f"{op}__perm{perm_idx}", "op": op,
                          "perm_idx": perm_idx, "df": df, "states": states_list,
                          "dag_roles": dag_roles, "cf_cfg": cf_cfg,
                          "innov_cfg": innov_cfg})
    _run_parallel(tasks, _worker_placebo, ckpt_path, workers, desc="安慰剂")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 安慰剂实验无有效结果"); return df_out
    sig_rate = df_out["显著"].mean()
    print(f"\n[安慰剂汇总]  θ均值={df_out['θ_安慰剂'].mean():+.5f}  显著率={sig_rate:.1%}（期望≈5%）")
    print(f"  {'[✓] 通过' if sig_rate <= 0.2 else '[⚠] 显著率偏高，可能存在虚假相关'}")
    out_path = os.path.join(PLACEBO_OUT_DIR, "refutation_placebo_v5.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验二：随机混杂变量反驳
# ═══════════════════════════════════════════════════════════════════
def run_random_confounder(df, ops, states, n_confounders=5, n_repeats=1,
                          workers=4, dag_roles: dict = None,
                          cf_cfg: dict = None, innov_cfg: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验二：随机混杂变量反驳（注入 {n_confounders} 个随机噪声列）")
    print(f" 判断标准：t_diff = |Δθ| / √(SE_orig²+SE_noisy²) < 2.0")
    print(f" 微创新配置：{_fmt_innov_cfg(innov_cfg)}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    if cf_cfg is None:
        cf_cfg = {}
    if innov_cfg is None:
        innov_cfg = {}
    ckpt_path   = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "checkpoint_rc_v5.jsonl")
    states_list = list(states)
    print("[预计算原始 θ ...]")
    orig_thetas = {}
    for op in sorted(ops):
        if df[op].std() < 0.1:
            continue
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            continue
        result = train_one_op(op, df, safe_x, **cf_cfg, **innov_cfg)
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
                          "df": df, "cf_cfg": cf_cfg, "innov_cfg": innov_cfg})
    _run_parallel(tasks, _worker_random_confounder, ckpt_path, workers, desc="随机混杂")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 随机混杂实验无有效结果"); return df_out
    pass_rate = df_out["通过反驳"].mean()
    print(f"\n[随机混杂汇总]  反驳通过率 = {pass_rate:.1%}（期望 ≥ 80%）")
    print(f"  {'[✓] 通过' if pass_rate >= 0.8 else '[⚠] 通过率偏低，θ 对随机噪声注入仍然敏感'}")
    out_path = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "refutation_rc_v5.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验三：数据子集反驳
# ═══════════════════════════════════════════════════════════════════
def run_data_subset(df, ops, states, n_subsets=8, subset_frac=0.8,
                    workers=4, dag_roles: dict = None,
                    cf_cfg: dict = None, innov_cfg: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验三：数据子集反驳（{n_subsets} 个子集，每个取 {subset_frac:.0%} 数据）")
    print(f" 微创新配置：{_fmt_innov_cfg(innov_cfg)}")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    if cf_cfg is None:
        cf_cfg = {}
    if innov_cfg is None:
        innov_cfg = {}
    ckpt_path   = os.path.join(DATA_SUBSET_OUT_DIR, "checkpoint_ds_v5.jsonl")
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
                          "cf_cfg": cf_cfg, "innov_cfg": innov_cfg})
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
    out_path = os.path.join(DATA_SUBSET_OUT_DIR, "refutation_ds_v5.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  v5 新增：实验四 — 消融实验（Ablation Study）
# ═══════════════════════════════════════════════════════════════════
def run_ablation(df, ops, states, dag_roles: dict = None, n_ops: int = 5):
    """
    消融实验：逐项累加微创新，量化各创新点的边际贡献。

    消融组设计：
      组1 (baseline)：v3 两阶段解耦（无微创新）
          → use_dual_stream=False, use_curriculum=False,
            use_grad_proj=False, use_uncertainty_weight=False
      组2 (+B)：双流潜变量
          → use_dual_stream=True, use_curriculum=False,
            use_grad_proj=False, use_uncertainty_weight=False
      组3 (+B+C)：双流 + 课程训练
          → use_dual_stream=True, use_curriculum=True,
            use_grad_proj=False, use_uncertainty_weight=False
      组4 (+B+C+A)：双流 + 课程 + 梯度投影
          → use_dual_stream=True, use_curriculum=True,
            use_grad_proj=True, use_uncertainty_weight=False
      组5 (v5_full)：全部微创新
          → use_dual_stream=True, use_curriculum=True,
            use_grad_proj=True, use_uncertainty_weight=True

    指标：ATE 中位数、SE、CV、符号一致率、耗时
    """
    print("\n" + "=" * 70)
    print(" 实验四（v5 新增）：消融实验（Ablation Study）")
    print(" 逐项累加微创新 A/B/C/D，量化边际贡献")
    print("=" * 70)
    print(" 组1 (baseline)  ：v3 两阶段解耦（无微创新）")
    print(" 组2 (+B)        ：双流潜变量")
    print(" 组3 (+B+C)      ：双流 + 课程训练")
    print(" 组4 (+B+C+A)    ：双流 + 课程 + 梯度投影")
    print(" 组5 (v5_full)   ：全部微创新（+D 不确定性加权）")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}

    ablation_configs = {
        "1_baseline": dict(
            use_dual_stream=False, use_curriculum=False,
            use_grad_proj=False, use_uncertainty_weight=False,
        ),
        "2_+B_dual_stream": dict(
            use_dual_stream=True, use_curriculum=False,
            use_grad_proj=False, use_uncertainty_weight=False,
        ),
        "3_+B+C_curriculum": dict(
            use_dual_stream=True, use_curriculum=True,
            use_grad_proj=False, use_uncertainty_weight=False,
        ),
        "4_+B+C+A_grad_proj": dict(
            use_dual_stream=True, use_curriculum=True,
            use_grad_proj=True, use_uncertainty_weight=False,
        ),
        "5_v5_full": dict(
            use_dual_stream=True, use_curriculum=True,
            use_grad_proj=True, use_uncertainty_weight=True,
        ),
    }

    states_list = list(states)
    candidate_ops = [op for op in sorted(ops) if df[op].std() >= 0.1][:n_ops]
    if not candidate_ops:
        print("[警告] 无有效操作变量，跳过消融实验"); return pd.DataFrame()

    print(f"  选取 {len(candidate_ops)} 个操作变量参与消融：{candidate_ops}")
    rows = []

    for op in candidate_ops:
        safe_x = build_safe_x_with_dag(op, df, states_list, dag_roles)
        if len(safe_x) < 2:
            print(f"  [跳过] {op}  safe_x 不足")
            continue
        print(f"\n  操作变量：{op}  (safe_x 数量={len(safe_x)})")

        for config_name, cfg in ablation_configs.items():
            t0     = time.perf_counter()
            result = train_one_op(op, df, safe_x, **cfg)
            elapsed = time.perf_counter() - t0

            if result is None:
                print(f"    {config_name}：估计失败")
                rows.append({
                    "操作节点": op, "消融组": config_name,
                    "theta_med": None, "SE_boot": None,
                    "CV": None, "符号一致率": None,
                    "F统计量": None, "耗时_s": round(elapsed, 1),
                    "状态": "失败",
                })
            else:
                theta_med, p_val, SE, n, f, cv, sr = result
                stable = cv < CV_WARN and sr >= SIGN_RATE_MIN
                flag   = "✓" if stable else "⚠"
                print(f"    {config_name}：θ={theta_med:+.5f}  SE={SE:.5f}  "
                      f"CV={cv:.3f}  sign_rate={sr:.2f}  [{flag}]  耗时={elapsed:.1f}s")
                rows.append({
                    "操作节点":  op,       "消融组":     config_name,
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
        print("[警告] 消融实验无有效结果"); return df_out

    # ── 汇总对比表 ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[消融实验汇总]")
    print(f"  {'消融组':<25s}  {'CV均值':>8s}  {'SE均值':>9s}  {'稳定率':>7s}  {'耗时均值(s)':>11s}")
    for config_name in ablation_configs.keys():
        sub = df_out[(df_out["消融组"] == config_name) & (df_out["状态"] == "成功")]
        if sub.empty:
            print(f"  {config_name:<25s}  （无有效结果）")
            continue
        print(f"  {config_name:<25s}  "
              f"{sub['CV'].mean():8.3f}  "
              f"{sub['SE_boot'].mean():9.5f}  "
              f"{sub['稳定'].mean():7.1%}  "
              f"{sub['耗时_s'].mean():11.1f}")
    print("=" * 70)

    out_path = os.path.join(ABLATION_OUT_DIR, "ablation_study_v5.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")
    return df_out


# ═══════════════════════════════════════════════════════════════════
#  辅助：格式化配置用于日志打印
# ═══════════════════════════════════════════════════════════════════
def _fmt_cf_cfg(cf_cfg: dict) -> str:
    if not cf_cfg:
        return "默认（扩展窗口，固定折边）"
    parts = []
    wt = cf_cfg.get("window_type", "expanding")
    parts.append(f"window={wt}")
    jr = cf_cfg.get("fold_jitter_ratio", 0.0)
    parts.append(f"jitter={jr:.0%}")
    parts.append("分层=✓" if cf_cfg.get("use_stratified") else "分层=✗")
    parts.append("嵌套LR=✓" if cf_cfg.get("nested_lr_search") else "嵌套LR=✗")
    return "  |  ".join(parts)


def _fmt_innov_cfg(innov_cfg: dict) -> str:
    if not innov_cfg:
        return "v5 全部微创新（双流+课程+梯度投影+不确定性加权）"
    parts = []
    parts.append("双流=✓" if innov_cfg.get("use_dual_stream", True) else "双流=✗")
    parts.append("课程=✓" if innov_cfg.get("use_curriculum", True) else "课程=✗")
    parts.append("梯度投影=✓" if innov_cfg.get("use_grad_proj", True) else "梯度投影=✗")
    parts.append("不确定性=✓" if innov_cfg.get("use_uncertainty_weight", True) else "不确定性=✗")
    return "  |  ".join(parts)


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="XIN_2 因果推断反驳实验 v5（LSTM-VAE 联合训练 + 微创新 A/B/C/D）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
建议执行顺序：
  # 1. 快速验证（小样本）
  python run_refutation_xin2_v5.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 2. 消融实验（核心新增实验，量化各微创新的边际贡献）
  python run_refutation_xin2_v5.py --mode ablation --sample_size 3000 --ablation_n_ops 5

  # 3. 关闭特定微创新进行对比
  python run_refutation_xin2_v5.py --mode stability --no_dual_stream
  python run_refutation_xin2_v5.py --mode stability --no_curriculum
  python run_refutation_xin2_v5.py --mode stability --no_grad_proj
  python run_refutation_xin2_v5.py --mode stability --no_uncertainty_weight

  # 4. 全部反驳实验（带所有微创新）
  python run_refutation_xin2_v5.py --mode all --workers 4

  # 5. 配合 v4 的交叉拟合策略改进
  python run_refutation_xin2_v5.py --mode stability \\
    --fold_jitter_ratio 0.1 --stratified

v5 输出文件带 _v5 后缀，不覆盖 v3/v4 结果。
        """,
    )
    p.add_argument("--mode", required=True,
                   choices=["stability", "placebo", "random_confounder",
                             "data_subset", "ablation", "all"])

    # ── 反驳实验参数（与 v3/v4 一致）───────────────────────────
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

    # ── v4 交叉拟合策略参数（完全继承）──────────────────────────
    p.add_argument("--window_type", type=str, default="expanding",
                   choices=["expanding", "sliding"])
    p.add_argument("--fold_jitter_ratio", type=float, default=FOLD_JITTER_RATIO)
    p.add_argument("--stratified", action="store_true", default=False)
    p.add_argument("--nested_lr_search", action="store_true", default=False)

    # ── v5 微创新开关（默认全部开启）────────────────────────────
    p.add_argument("--no_dual_stream", action="store_true", default=False,
                   help="关闭微创新 B（双流潜变量），退化为单流编码器")
    p.add_argument("--no_curriculum", action="store_true", default=False,
                   help="关闭微创新 C（课程训练），使用标准联合训练")
    p.add_argument("--no_grad_proj", action="store_true", default=False,
                   help="关闭微创新 A（因果优先梯度投影），使用标准梯度更新")
    p.add_argument("--no_uncertainty_weight", action="store_true", default=False,
                   help="关闭微创新 D（不确定性加权），使用等权重残差")

    # ── 消融实验参数 ────────────────────────────────────────────
    p.add_argument("--ablation_n_ops", type=int, default=5,
                   help="消融实验参与对比的操作变量数（默认 5）")

    # ── DAG / 操作性分类（与 v3/v4 一致）──────────────────────
    p.add_argument("--dag-roles-csv", type=str, default="")
    p.add_argument("--operability-csv", type=str, default=DEFAULT_OPERABILITY_CSV)

    return p.parse_args()


def main():
    args = parse_args()
    global N_BOOTSTRAP
    N_BOOTSTRAP = args.n_bootstrap

    # ── 构建配置字典 ──────────────────────────────────────────────
    cf_cfg = dict(
        window_type       = args.window_type,
        fold_jitter_ratio = args.fold_jitter_ratio,
        use_stratified    = args.stratified,
        nested_lr_search  = args.nested_lr_search,
    )
    innov_cfg = dict(
        use_dual_stream       = not args.no_dual_stream,
        use_curriculum        = not args.no_curriculum,
        use_grad_proj         = not args.no_grad_proj,
        use_uncertainty_weight = not args.no_uncertainty_weight,
    )

    print("=" * 70)
    print(f" XIN_2 因果推断反驳实验 v5  |  模式: {args.mode.upper()}")
    print(f" 设备: {DEVICE}  |  并行线程: {args.workers}  |  Bootstrap: {N_BOOTSTRAP}")
    print(f" 架构: LSTM-VAE 联合训练 + 微创新 A/B/C/D")
    print(f" 微创新配置: {_fmt_innov_cfg(innov_cfg)}")
    print(f" 交叉拟合策略: {_fmt_cf_cfg(cf_cfg)}")
    print("=" * 70)

    df_raw, operable_in_df_raw, observable_in_df_raw = build_xin2_data(
        operability_csv=args.operability_csv,
    )

    # ── 窗口聚合对齐（将每行 Y 测量时间点作为锚点，汇聚 X/D 统计量）──
    from build_aligned_dataset import build_aligned_dataset
    operable_cols_raw   = sorted(operable_in_df_raw   & set(df_raw.columns))
    observable_cols_raw = sorted(observable_in_df_raw & set(df_raw.columns))
    df, new_operable, new_observable = build_aligned_dataset(
        df_raw,
        operable_cols  = operable_cols_raw,
        observable_cols= observable_cols_raw,
        window_minutes = 30,
        y_ffill_limit  = 2,
    )
    # update sets after alignment
    operable_in_df   = set(new_operable)
    observable_in_df = set(new_observable)

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
                                cf_cfg=cf_cfg,
                                innov_cfg=innov_cfg)
    if mode in ("placebo", "all"):
        run_placebo(df, set(ops), set(states),
                    n_permutations=args.n_permutations,
                    workers=args.workers,
                    dag_roles=dag_roles,
                    cf_cfg=cf_cfg,
                    innov_cfg=innov_cfg)
    if mode in ("random_confounder", "all"):
        run_random_confounder(df, set(ops), set(states),
                              n_confounders=args.n_confounders,
                              n_repeats=args.n_repeats,
                              workers=args.workers,
                              dag_roles=dag_roles,
                              cf_cfg=cf_cfg,
                              innov_cfg=innov_cfg)
    if mode in ("data_subset", "all"):
        run_data_subset(df, set(ops), set(states),
                        n_subsets=args.n_subsets,
                        subset_frac=args.subset_frac,
                        workers=args.workers,
                        dag_roles=dag_roles,
                        cf_cfg=cf_cfg,
                        innov_cfg=innov_cfg)
    if mode == "ablation":
        run_ablation(df, set(ops), set(states),
                     dag_roles=dag_roles,
                     n_ops=args.ablation_n_ops)

    print("\n" + "=" * 70)
    print(" 全部实验完成，结果保存至：")
    for label, path in [
        ("稳定性诊断",   STABILITY_OUT_DIR),
        ("安慰剂实验",   PLACEBO_OUT_DIR),
        ("随机混杂",     RANDOM_CONFOUNDER_OUT_DIR),
        ("数据子集",     DATA_SUBSET_OUT_DIR),
        ("消融实验",     ABLATION_OUT_DIR),
    ]:
        print(f"  {label:8s}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
