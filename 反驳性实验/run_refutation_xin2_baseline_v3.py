"""
run_refutation_xin2_baseline_v3.py
====================================
XIN_2 因果推断反驳实验 — 基线方法 V3「非线性双重机器学习（RF + GBM + R-Learner）」

═══════════════════════════════════════════════════════════════════
  基线定位：当线性工作点假设不成立时，提供可对比的非线性传统机器学习基线，
  与创新版 v3/v4/v5（LSTM-VAE-DML）以及基线 V1（RF-DML）、V2（GBM-DML）
  形成多维对照。本脚本融合 RF 与 GBM 的 nuisance 集成预测，并引入 R-Learner
  作为额外的非线性 CATE 估计器。

  方法：集成 Double ML（RF+GBM nuisance）+ R-Learner
  ─────────────────────────────────────────────────────────────────
  核心思路：
    1. 构建滞后特征矩阵：将过去 SEQ_LEN=6 个时间步的状态变量展平为
       特征向量，显式捕捉滞后效应（工业场景滞后效应强，必须纳入）
    2. 交叉拟合（Cross-Fitting）：
         - 在训练折分别用 RF 和 GBM 拟合 Ŷ 和 D̂
         - nuisance 预测取 RF 和 GBM 的均值（集成），降低模型偏倚
         - 在验证折计算残差 res_Y = Y - Ŷ_ensemble，res_D = D - D̂_ensemble
    3. DML 估计（标准路径）：θ_DML = Cov(res_Y, res_D) / Var(res_D)
    4. R-Learner 估计（非线性 CATE 路径）：
         - 构造伪结果 W = res_Y / (res_D + ε)
         - 以 res_D² 为样本权重，用 GBM 拟合 W ~ X（即学习 τ(X)）
         - θ_R-Learner = 验证折上 τ(X) 的加权均值
    5. Bootstrap 聚合：N_BOOTSTRAP 次独立拟合，θ 取中位数

  为何选 RF+GBM 集成 + R-Learner 作 V3？
    - RF+GBM 集成：两类模型偏差-方差特性互补（RF 方差低、GBM 偏差低），
      集成后 nuisance 估计更稳健，残差更干净
    - R-Learner（Nie & Wager 2021）：直接学习异质处理效应函数 τ(X)，
      不依赖线性工作点假设，适合处理效应随协变量非线性变化的场景
    - DML + R-Learner 同时输出：若两者一致 → 线性假设成立；若 R-Learner
      明显不同 → 存在异质性，线性近似可能有偏

  与创新版的对比：
    - V3/V4/V5 使用 LSTM-VAE 学习时序潜变量表征（SEQ 结构内化于模型）
    - 本基线用手工滞后展平（SEQ 结构外化为特征工程），适合对照研究

═══════════════════════════════════════════════════════════════════

用法：
  # 快速稳定性诊断（截取 2000 条数据，3 次 Bootstrap，几分钟内完成）
  python run_refutation_xin2_baseline_v3.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 全量稳定性诊断
  python run_refutation_xin2_baseline_v3.py --mode stability

  # 安慰剂反驳
  python run_refutation_xin2_baseline_v3.py --mode placebo --n_permutations 5 --workers 4

  # 随机混杂变量反驳
  python run_refutation_xin2_baseline_v3.py --mode random_confounder --workers 4

  # 数据子集反驳
  python run_refutation_xin2_baseline_v3.py --mode data_subset --workers 4

  # 全部实验一次跑
  python run_refutation_xin2_baseline_v3.py --mode all --workers 4
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        """tqdm 不可用时的兼容性 stub，支持 with 语句和 .update() 调用。"""
        def __init__(self, iterable=None, **kw):
            self._it = list(iterable) if iterable is not None else []
            self._desc = kw.get("desc", "")
            self._total = kw.get("total", len(self._it))
            print(f"[{self._desc}] 共 {self._total} 个任务（建议 pip install tqdm）")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            return iter(self._it)
        def update(self, n=1):
            pass


# ═══════════════════════════════════════════════════════════════════
#  路径配置（与 v3/v4/v5/baseline_v1/baseline_v2 完全一致）
# ═══════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))   # 反驳性实验/
REPO_ROOT = os.path.dirname(BASE_DIR)                    # 仓库根目录
DATA_DIR  = os.path.join(REPO_ROOT, "data")

MODELING_DATASET_XIN2 = os.path.join(DATA_DIR, "modeling_dataset_xin2_final.parquet")
X_PARQUET             = os.path.join(DATA_DIR, "X_features_final.parquet")
Y_PARQUET             = os.path.join(DATA_DIR, "y_target_final.parquet")

DEFAULT_OPERABILITY_CSV = os.path.join(
    REPO_ROOT, "数据预处理",
    "数据与处理结果-分阶段-去共线性后",
    "non_collinear_representative_vars_operability.csv",
)

# ── 实验结果输出目录（共享目录，文件名含 _baseline_v3 后缀以区分）──
PLACEBO_OUT_DIR           = os.path.join(BASE_DIR, "安慰剂实验")
RANDOM_CONFOUNDER_OUT_DIR = os.path.join(BASE_DIR, "随机混杂变量实验")
DATA_SUBSET_OUT_DIR       = os.path.join(BASE_DIR, "数据子集实验")
STABILITY_OUT_DIR         = os.path.join(BASE_DIR, "稳定性诊断")

for _d in [PLACEBO_OUT_DIR, RANDOM_CONFOUNDER_OUT_DIR,
           DATA_SUBSET_OUT_DIR, STABILITY_OUT_DIR]:
    os.makedirs(_d, exist_ok=True)

DEFAULT_DAG_ROLES_CSV = os.path.join(
    REPO_ROOT, "DAG图分析", "DAG解析结果", ""
)


# ═══════════════════════════════════════════════════════════════════
#  超参
# ═══════════════════════════════════════════════════════════════════
SEQ_LEN             = 6    # 滞后窗口长度（与 v3 LSTM 序列长度一致）
EMBARGO_GAP         = 4    # 训练/验证折之间的时间间隔（防数据泄露）
K_FOLDS             = 4    # 交叉拟合折数
N_BOOTSTRAP         = 5    # Bootstrap 重复次数
MIN_TRAIN_SIZE      = 100
MIN_VALID_RESIDUALS = 50
F_STAT_THRESHOLD    = 10.0
CV_WARN             = 0.30  # CV 超过此值视为不稳定
SIGN_RATE_MIN       = 0.70  # 符号一致率低于此值视为不可信

# ── RF 超参（与 baseline_v1 一致）───────────────────────────────────
RF_N_ESTIMATORS     = 100
RF_MAX_DEPTH        = 5     # 控制过拟合（小数据集）
RF_MIN_SAMPLES_LEAF = 5     # 进一步防止过拟合
RF_MAX_FEATURES     = "sqrt"  # DML 文献推荐

# ── GBM 超参（与 baseline_v2 一致）──────────────────────────────────
GBM_N_ESTIMATORS     = 100
GBM_MAX_DEPTH        = 3      # GBM 推荐浅树（防止过拟合，每轮专注修残差）
GBM_LEARNING_RATE    = 0.1    # 标准学习率
GBM_SUBSAMPLE        = 0.8    # 随机抽样（Stochastic GBM，减少方差、加快收敛）
GBM_MIN_SAMPLES_LEAF = 10     # 叶节点最小样本数（工业小样本场景下保守设置）

# ── R-Learner 超参 ──────────────────────────────────────────────────
RLEARNER_N_ESTIMATORS     = 100
RLEARNER_MAX_DEPTH        = 5
RLEARNER_MIN_SAMPLES_LEAF = 5
RLEARNER_MIN_WEIGHT       = 1e-10  # 伪结果权重阈值（res_D² < 此值的样本不参与拟合）


# ═══════════════════════════════════════════════════════════════════
#  DAG 因果角色加载（与 v3/baseline_v1/baseline_v2 完全一致）
# ═══════════════════════════════════════════════════════════════════
def load_dag_roles(csv_path: str) -> dict:
    """加载 DAG 角色明细表。格式：Treatment_T, Role, Node_Name"""
    if not csv_path or not os.path.isfile(csv_path):
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


def get_safe_x(op: str, df: pd.DataFrame, states: list) -> list:
    """与 v3/baseline_v1/baseline_v2 完全一致：按滞后相关性筛选控制变量"""
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


def build_safe_x_with_dag(op: str, df: pd.DataFrame, states: list,
                           dag_roles: dict) -> list:
    """与 v3/baseline_v1/baseline_v2 完全一致：相关性筛选后叠加 DAG 角色过滤"""
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
        parts = [f"{k}={len(v)}" for k, v in excluded_details.items() if v]
        print(f"  [DAG过滤] {op}: 剔除 {n_excluded} 个变量 ({', '.join(parts)})，"
              f"保留 {len(filtered_x)} 个控制变量")
    return filtered_x


# ═══════════════════════════════════════════════════════════════════
#  确定性哈希种子（与 v3/baseline_v1/baseline_v2 完全一致）
# ═══════════════════════════════════════════════════════════════════
def _op_seed(op: str) -> int:
    return int(hashlib.md5(op.encode()).hexdigest()[:8], 16) % 100000


# ═══════════════════════════════════════════════════════════════════
#  滞后特征矩阵构建（与 baseline_v1/baseline_v2 完全一致）
# ═══════════════════════════════════════════════════════════════════
def _build_lag_features(X_norm: np.ndarray, n_lags: int) -> np.ndarray:
    """
    将归一化状态矩阵 X_norm (T × d) 展平为滞后特征矩阵。

    对时间点 t（t >= n_lags），特征向量 = flatten(X_norm[t-n_lags : t])，
    shape = (n_lags × d,)，对应 LSTM 的输入窗口。

    返回：shape (T - n_lags, n_lags * d) 的 float32 矩阵。
    显式捕捉工业场景中的强滞后效应，与 v3 的 LSTM 序列输入等价。
    """
    T, d = X_norm.shape
    rows = []
    for t in range(n_lags, T):
        rows.append(X_norm[t - n_lags:t].flatten())
    return np.array(rows, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  核心训练函数：集成 DML（RF+GBM）+ R-Learner + Bootstrap θ 聚合
# ═══════════════════════════════════════════════════════════════════
def train_one_op(op: str, df: pd.DataFrame, safe_x: list,
                 override_D=None, n_bootstrap: int = N_BOOTSTRAP):
    """
    集成 DML + R-Learner 核心函数。

    对每个 bootstrap × fold：
      1. 构建滞后特征矩阵 X_lag（T-SEQ_LEN 行，SEQ_LEN×d 列）
      2. 分别用 RF 和 GBM 拟合 nuisance 函数：
           Ŷ_rf, Ŷ_gbm = E[Y | X_lag]
           D̂_rf, D̂_gbm = E[D | X_lag]
      3. 集成 nuisance：取 RF 和 GBM 预测的算术均值
           Ŷ = (Ŷ_rf + Ŷ_gbm) / 2
           D̂ = (D̂_rf + D̂_gbm) / 2
      4. 在验证折计算残差：res_Y = Y - Ŷ，res_D = D - D̂
      5. DML theta：θ_std = Cov(res_D, res_Y) / Var(res_D)
         θ_DML = θ_std × (Y_std / D_std)（反归一化）
      6. R-Learner theta：
         - 伪结果 W = res_Y / (res_D + ε)
         - 样本权重 = res_D²
         - 用加权 GBM 拟合 W ~ X_val 得到 τ̂(X)
         - θ_R-Learner = Σ τ̂(X_i) * res_D_i² / Σ res_D_i²（加权均值）
         - 反归一化：θ_R-Learner × (Y_std / D_std)

    Bootstrap N_BOOTSTRAP 次，θ 取中位数。

    返回：(theta_dml_med, theta_rl_med, p_dml, p_rl, SE_dml, SE_rl,
           n_avg, f_med, cv_dml, cv_rl, sign_rate_dml, sign_rate_rl)
    或 None（工具弱/样本不足/多数折失败）
    """
    # ── 标准化 ────────────────────────────────────────────────────
    X_df   = df[safe_x].ffill().bfill()
    X_raw  = X_df.values.astype(np.float32)
    X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)

    Y_raw  = df["Y_grade"].values.astype(np.float32)
    if override_D is None:
        D_series = df[op].ffill().bfill()
        D_raw    = D_series.fillna(D_series.mean()).values.astype(np.float32)
    else:
        D_raw = np.asarray(override_D, dtype=np.float32)
    Y_mean, Y_std = float(Y_raw.mean()), float(Y_raw.std()) + 1e-8
    D_mean, D_std = float(D_raw.mean()), float(D_raw.std()) + 1e-8
    Y_norm = (Y_raw - Y_mean) / Y_std
    D_norm = (D_raw - D_mean) / D_std

    # ── 滞后特征展平 ──────────────────────────────────────────────
    X_lag  = _build_lag_features(X_norm, SEQ_LEN)   # (N, SEQ_LEN*d)
    Y_vec  = Y_norm[SEQ_LEN:]                        # (N,)
    D_vec  = D_norm[SEQ_LEN:]                        # (N,)
    N          = len(X_lag)
    block_size = N // K_FOLDS

    op_base_seed = _op_seed(op)
    theta_dml_list, theta_rl_list = [], []
    f_list, n_list = [], []

    for boot_i in range(n_bootstrap):
        base_seed    = boot_i * 99991 + op_base_seed
        all_res_Y, all_res_D, all_X_vl = [], [], []
        any_valid_fold = False

        for k in range(1, K_FOLDS):
            train_end = k * block_size - EMBARGO_GAP
            if train_end < MIN_TRAIN_SIZE:
                continue
            val_start = k * block_size
            val_end   = (k + 1) * block_size if k < K_FOLDS - 1 else N

            rng_seed = (base_seed * 100 + k) % (2 ** 31)

            X_tr = X_lag[:train_end]
            X_vl = X_lag[val_start:val_end]
            Y_tr = Y_vec[:train_end]
            Y_vl = Y_vec[val_start:val_end]
            D_tr = D_vec[:train_end]
            D_vl = D_vec[val_start:val_end]

            if len(X_tr) < MIN_TRAIN_SIZE or len(X_vl) == 0:
                continue

            # ── RF nuisance ──────────────────────────────────────
            rf_Y = RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                max_features=RF_MAX_FEATURES,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                n_jobs=1,
                random_state=rng_seed,
            )
            rf_Y.fit(X_tr, Y_tr)

            rf_D = RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                max_features=RF_MAX_FEATURES,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                n_jobs=1,
                random_state=rng_seed + 1,
            )
            rf_D.fit(X_tr, D_tr)

            # ── GBM nuisance ─────────────────────────────────────
            gbm_Y = GradientBoostingRegressor(
                n_estimators=GBM_N_ESTIMATORS,
                max_depth=GBM_MAX_DEPTH,
                learning_rate=GBM_LEARNING_RATE,
                subsample=GBM_SUBSAMPLE,
                min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
                random_state=rng_seed + 2,
            )
            gbm_Y.fit(X_tr, Y_tr)

            gbm_D = GradientBoostingRegressor(
                n_estimators=GBM_N_ESTIMATORS,
                max_depth=GBM_MAX_DEPTH,
                learning_rate=GBM_LEARNING_RATE,
                subsample=GBM_SUBSAMPLE,
                min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
                random_state=rng_seed + 3,
            )
            gbm_D.fit(X_tr, D_tr)

            # ── 集成 nuisance 预测（RF+GBM 均值）─────────────────
            pred_Y_ensemble = (rf_Y.predict(X_vl) + gbm_Y.predict(X_vl)) / 2.0
            pred_D_ensemble = (rf_D.predict(X_vl) + gbm_D.predict(X_vl)) / 2.0

            res_Y_k = Y_vl - pred_Y_ensemble
            res_D_k = D_vl - pred_D_ensemble

            all_res_Y.extend(res_Y_k)
            all_res_D.extend(res_D_k)
            all_X_vl.append(X_vl)
            any_valid_fold = True

        if not any_valid_fold or len(all_res_Y) < MIN_VALID_RESIDUALS:
            continue

        res_Y = np.array(all_res_Y, dtype=np.float64)
        res_D = np.array(all_res_D, dtype=np.float64)
        X_vl_all = np.concatenate(all_X_vl, axis=0)

        # 去离群（3σ）
        mask  = ((np.abs(res_Y) < 3 * res_Y.std()) &
                 (np.abs(res_D) < 3 * res_D.std()))
        res_Y, res_D = res_Y[mask], res_D[mask]
        X_vl_all = X_vl_all[mask]
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

        # ── DML θ（协方差矩阵方式，与 v3/baseline_v1/v2 完全一致）──
        cov_mat       = np.cov(res_D, res_Y)
        theta_dml_std = cov_mat[0, 1] / (cov_mat[0, 0] + 1e-12)
        theta_dml     = theta_dml_std * (Y_std / D_std)

        # ── R-Learner θ ─────────────────────────────────────────
        # 伪结果 W = res_Y / (res_D + ε)，权重 = res_D²
        # 拟合加权 GBM：W ~ X，得到 CATE 函数 τ̂(X)
        theta_rl = None
        try:
            eps = 1e-6
            # 安全除法：保留 res_D 符号，对绝对值过小的用带符号 eps 替换
            safe_res_D = np.where(np.abs(res_D) < eps, np.sign(res_D) * eps, res_D)
            # 处理 res_D 恰好为 0 的情况（sign=0），赋正 eps
            safe_res_D = np.where(safe_res_D == 0, eps, safe_res_D)
            w_pseudo   = res_Y / safe_res_D
            w_weights  = res_D ** 2

            # 排除权重接近零的样本（res_D ≈ 0 意味着无信息）
            w_mask = w_weights > RLEARNER_MIN_WEIGHT
            if w_mask.sum() >= MIN_VALID_RESIDUALS:
                X_rl     = X_vl_all[w_mask]
                w_ps     = w_pseudo[w_mask]
                w_wt     = w_weights[w_mask]

                # 截断伪结果中的极端值（防止数值爆炸）
                w_clip_lo = np.percentile(w_ps, 1)
                w_clip_hi = np.percentile(w_ps, 99)
                w_ps = np.clip(w_ps, w_clip_lo, w_clip_hi)

                rl_model = GradientBoostingRegressor(
                    n_estimators=RLEARNER_N_ESTIMATORS,
                    max_depth=RLEARNER_MAX_DEPTH,
                    min_samples_leaf=RLEARNER_MIN_SAMPLES_LEAF,
                    learning_rate=GBM_LEARNING_RATE,
                    subsample=GBM_SUBSAMPLE,
                    random_state=(base_seed + 777) % (2 ** 31),
                )
                rl_model.fit(X_rl, w_ps, sample_weight=w_wt)

                # τ̂(X) 在全体验证集上的加权均值
                tau_pred  = rl_model.predict(X_vl_all)
                w_all     = res_D ** 2
                theta_rl_std = float(np.average(tau_pred, weights=w_all))
                theta_rl = theta_rl_std * (Y_std / D_std)
        except Exception:
            theta_rl = None

        theta_dml_list.append(theta_dml)
        if theta_rl is not None:
            theta_rl_list.append(theta_rl)
        f_list.append(f_stat)
        n_list.append(n)

    # ── Bootstrap 聚合 ────────────────────────────────────────────
    min_success = max(1, n_bootstrap // 2)
    if len(theta_dml_list) < min_success:
        return None

    # DML 聚合
    dml_arr     = np.array(theta_dml_list)
    theta_dml_med = float(np.median(dml_arr))
    dml_std_b   = float(np.std(dml_arr, ddof=1)) if len(dml_arr) > 1 else 0.0
    cv_dml      = dml_std_b / (abs(theta_dml_med) + 1e-8)
    sr_dml      = float(np.mean(np.sign(dml_arr) == np.sign(theta_dml_med)))
    SE_dml      = max(dml_std_b, 1e-8)
    t_dml       = theta_dml_med / SE_dml
    n_avg       = int(np.mean(n_list))
    p_dml       = 2 * (1 - stats.t.cdf(abs(t_dml), df=n_avg - 1))
    f_med       = float(np.median(f_list))

    # R-Learner 聚合（可能部分 bootstrap 失败，允许少于 DML 成功数）
    if len(theta_rl_list) >= max(1, min_success // 2):
        rl_arr      = np.array(theta_rl_list)
        theta_rl_med = float(np.median(rl_arr))
        rl_std_b    = float(np.std(rl_arr, ddof=1)) if len(rl_arr) > 1 else 0.0
        cv_rl       = rl_std_b / (abs(theta_rl_med) + 1e-8)
        sr_rl       = float(np.mean(np.sign(rl_arr) == np.sign(theta_rl_med)))
        SE_rl       = max(rl_std_b, 1e-8)
        t_rl        = theta_rl_med / SE_rl
        p_rl        = 2 * (1 - stats.t.cdf(abs(t_rl), df=n_avg - 1))
    else:
        theta_rl_med = None
        p_rl = SE_rl = cv_rl = sr_rl = None

    return (theta_dml_med, theta_rl_med, p_dml, p_rl,
            SE_dml, SE_rl, n_avg, f_med,
            cv_dml, cv_rl, sr_dml, sr_rl)


# ═══════════════════════════════════════════════════════════════════
#  断点续传工具（与 v3/baseline_v1/baseline_v2 完全一致）
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
#  通用并行调度器（与 v3/baseline_v1/baseline_v2 完全一致）
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
#  辅助：从 train_one_op 结果中提取 DML / R-Learner 字段
# ═══════════════════════════════════════════════════════════════════
def _unpack_result(result):
    """将 train_one_op 的 12-tuple 解包为命名值。"""
    (theta_dml, theta_rl, p_dml, p_rl,
     SE_dml, SE_rl, n_avg, f_med,
     cv_dml, cv_rl, sr_dml, sr_rl) = result
    return {
        "theta_dml": theta_dml, "theta_rl": theta_rl,
        "p_dml": p_dml, "p_rl": p_rl,
        "SE_dml": SE_dml, "SE_rl": SE_rl,
        "n": n_avg, "f": f_med,
        "cv_dml": cv_dml, "cv_rl": cv_rl,
        "sr_dml": sr_dml, "sr_rl": sr_rl,
    }


# ═══════════════════════════════════════════════════════════════════
#  Worker 函数（输出 DML + R-Learner 双通道结果）
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
        return {"_key": key, "_filtered": True, "_reason": "safe_x不足"}
    rng       = np.random.default_rng(seed=perm_idx * 42 + _op_seed(op))
    D_placebo = rng.permutation(df[op].values.copy())
    result    = train_one_op(op, df, safe_x, override_D=D_placebo)
    if result is None:
        return {"_key": key, "_filtered": True, "_reason": "弱工具/样本不足"}
    r = _unpack_result(result)
    rec = {
        "_key": key, "_filtered": False,
        "操作节点": op, "排列索引": perm_idx + 1,
        "θ_DML_安慰剂": round(r["theta_dml"], 5),
        "P_DML":        round(r["p_dml"], 4),
        "SE_DML":       round(r["SE_dml"], 5),
        "CV_DML":       round(r["cv_dml"], 4),
        "符号一致率_DML": round(r["sr_dml"], 3),
        "有效残差数":   r["n"],
        "F统计量":      round(r["f"], 2),
        "显著_DML":     bool(r["p_dml"] < 0.05),
    }
    if r["theta_rl"] is not None:
        rec.update({
            "θ_RL_安慰剂":  round(r["theta_rl"], 5),
            "P_RL":          round(r["p_rl"], 4),
            "SE_RL":         round(r["SE_rl"], 5),
            "CV_RL":         round(r["cv_rl"], 4),
            "符号一致率_RL": round(r["sr_rl"], 3),
            "显著_RL":       bool(r["p_rl"] < 0.05),
        })
    return rec


def _worker_random_confounder(task: dict) -> dict:
    op, rep       = task["op"], task["rep"]
    n_confounders = task["n_confounders"]
    theta_dml_orig = task["theta_dml_orig"]
    SE_dml_orig    = task["SE_dml_orig"]
    theta_rl_orig  = task.get("theta_rl_orig")
    SE_rl_orig     = task.get("SE_rl_orig")
    safe_x_orig   = task["safe_x_orig"]
    df, key       = task["df"], task["_key"]

    rng = np.random.default_rng(seed=rep * 1000 + _op_seed(op))
    df_noisy   = df.copy()
    noise_cols = []
    for nc in range(n_confounders):
        cname = f"__rc_{nc}__"
        df_noisy[cname] = rng.standard_normal(len(df_noisy))
        noise_cols.append(cname)
    safe_x_noisy = safe_x_orig + noise_cols
    result = train_one_op(op, df_noisy, safe_x_noisy)
    if result is None:
        return {"_key": key, "_filtered": True, "_reason": "弱工具/样本不足"}
    r = _unpack_result(result)

    # DML 偏差检验
    delta_dml     = abs(r["theta_dml"] - theta_dml_orig)
    se_comb_dml   = float(np.sqrt(r["SE_dml"] ** 2 + SE_dml_orig ** 2)) + 1e-12
    t_diff_dml    = delta_dml / se_comb_dml
    sign_ok_dml   = bool(np.sign(r["theta_dml"]) == np.sign(theta_dml_orig))
    near_zero_dml = abs(theta_dml_orig) < 3 * SE_dml_orig
    passed_dml    = bool(t_diff_dml < 2.0 and (sign_ok_dml or near_zero_dml))
    rel_dev_dml   = delta_dml / (abs(theta_dml_orig) + 1e-8)

    rec = {
        "_key": key, "_filtered": False,
        "操作节点": op, "重复索引": rep + 1,
        "θ_DML_原始":     round(theta_dml_orig, 5),
        "θ_DML_注入噪声": round(r["theta_dml"], 5),
        "相对偏差_DML":   round(rel_dev_dml, 4),
        "t_diff_DML":     round(t_diff_dml, 3),
        "方向一致_DML":   sign_ok_dml,
        "P_DML":          round(r["p_dml"], 4),
        "SE_DML":         round(r["SE_dml"], 5),
        "CV_DML":         round(r["cv_dml"], 4),
        "符号一致率_DML": round(r["sr_dml"], 3),
        "有效残差数":     r["n"],
        "F统计量":        round(r["f"], 2),
        "通过反驳_DML":   passed_dml,
    }

    # R-Learner 偏差检验（若可用）
    if r["theta_rl"] is not None and theta_rl_orig is not None and SE_rl_orig is not None:
        delta_rl     = abs(r["theta_rl"] - theta_rl_orig)
        se_comb_rl   = float(np.sqrt(r["SE_rl"] ** 2 + SE_rl_orig ** 2)) + 1e-12
        t_diff_rl    = delta_rl / se_comb_rl
        sign_ok_rl   = bool(np.sign(r["theta_rl"]) == np.sign(theta_rl_orig))
        near_zero_rl = abs(theta_rl_orig) < 3 * SE_rl_orig
        passed_rl    = bool(t_diff_rl < 2.0 and (sign_ok_rl or near_zero_rl))
        rel_dev_rl   = delta_rl / (abs(theta_rl_orig) + 1e-8)
        rec.update({
            "θ_RL_原始":     round(theta_rl_orig, 5),
            "θ_RL_注入噪声": round(r["theta_rl"], 5),
            "相对偏差_RL":   round(rel_dev_rl, 4),
            "t_diff_RL":     round(t_diff_rl, 3),
            "方向一致_RL":   sign_ok_rl,
            "P_RL":          round(r["p_rl"], 4),
            "SE_RL":         round(r["SE_rl"], 5),
            "CV_RL":         round(r["cv_rl"], 4),
            "符号一致率_RL": round(r["sr_rl"], 3),
            "通过反驳_RL":   passed_rl,
        })

    return rec


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
    r = _unpack_result(result)
    rec = {
        "_key": key, "_filtered": False,
        "操作节点": op, "子集索引": sub_idx + 1,
        "时段起点": start, "时段终点": end,
        "θ_DML_子集":   round(r["theta_dml"], 5),
        "P_DML":         round(r["p_dml"], 4),
        "SE_DML":        round(r["SE_dml"], 5),
        "CV_DML":        round(r["cv_dml"], 4),
        "符号一致率_DML": round(r["sr_dml"], 3),
        "有效残差数":    r["n"],
        "F统计量":       round(r["f"], 2),
    }
    if r["theta_rl"] is not None:
        rec.update({
            "θ_RL_子集":    round(r["theta_rl"], 5),
            "P_RL":          round(r["p_rl"], 4),
            "SE_RL":         round(r["SE_rl"], 5),
            "CV_RL":         round(r["cv_rl"], 4),
            "符号一致率_RL": round(r["sr_rl"], 3),
        })
    return rec


# ═══════════════════════════════════════════════════════════════════
#  数据准备（与 v3/baseline_v1/baseline_v2 完全一致）
# ═══════════════════════════════════════════════════════════════════
def build_xin2_data(operability_csv: str = DEFAULT_OPERABILITY_CSV):
    """加载 XIN2 产线建模数据（与 v3 build_xin2_data 完全一致）"""
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
#  实验零：稳定性诊断
# ═══════════════════════════════════════════════════════════════════
def run_stability_diagnosis(df, ops, states, workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验零：稳定性诊断（{N_BOOTSTRAP} 次 Bootstrap）")
    print(f" 架构：集成 DML（RF+GBM nuisance）+ R-Learner 基线 V3")
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
        r = _unpack_result(result)
        stable_dml = (r["cv_dml"] < CV_WARN and r["sr_dml"] >= SIGN_RATE_MIN)
        if stable_dml:
            n_stable_dml += 1
        flag_dml = "✓" if stable_dml else "⚠"

        rl_info = ""
        stable_rl = False
        if r["theta_rl"] is not None:
            stable_rl = (r["cv_rl"] < CV_WARN and r["sr_rl"] >= SIGN_RATE_MIN)
            if stable_rl:
                n_stable_rl += 1
            flag_rl = "✓" if stable_rl else "⚠"
            rl_info = (f"  θ_RL={r['theta_rl']:+.5f} CV_RL={r['cv_rl']:.3f} "
                       f"sr_RL={r['sr_rl']:.2f} [{flag_rl}]")
        else:
            rl_info = "  θ_RL=N/A"

        print(f"  {op:<30s}  θ_DML={r['theta_dml']:+.5f}  CV_DML={r['cv_dml']:.3f}  "
              f"sr_DML={r['sr_dml']:.2f}  p_DML={r['p_dml']:.4f}  [{flag_dml}]"
              f"{rl_info}")

        row = {
            "操作节点": op,
            "θ_DML": round(r["theta_dml"], 5),
            "P_DML": round(r["p_dml"], 4),
            "SE_DML": round(r["SE_dml"], 5),
            "CV_DML": round(r["cv_dml"], 4),
            "符号一致率_DML": round(r["sr_dml"], 3),
            "F统计量": round(r["f"], 2),
            "稳定_DML": stable_dml,
        }
        if r["theta_rl"] is not None:
            row.update({
                "θ_RL": round(r["theta_rl"], 5),
                "P_RL": round(r["p_rl"], 4),
                "SE_RL": round(r["SE_rl"], 5),
                "CV_RL": round(r["cv_rl"], 4),
                "符号一致率_RL": round(r["sr_rl"], 3),
                "稳定_RL": stable_rl,
            })
        rows.append(row)

    df_out   = pd.DataFrame(rows)
    out_path = os.path.join(STABILITY_OUT_DIR, "stability_diagnosis_baseline_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not df_out.empty:
        total = len(df_out)
        print(f"\n[稳定性诊断汇总 — DML]  稳定 {n_stable_dml}/{total} 个操作变量")
        print(f"  CV_DML 均值      = {df_out['CV_DML'].mean():.3f}  （目标 < {CV_WARN}）")
        print(f"  sign_rate_DML 均值 = {df_out['符号一致率_DML'].mean():.2f}  （目标 ≥ {SIGN_RATE_MIN}）")
        if "θ_RL" in df_out.columns:
            rl_valid = df_out.dropna(subset=["θ_RL"])
            if not rl_valid.empty:
                total_rl = len(rl_valid)
                print(f"\n[稳定性诊断汇总 — R-Learner]  稳定 {n_stable_rl}/{total_rl} 个操作变量")
                print(f"  CV_RL 均值      = {rl_valid['CV_RL'].mean():.3f}  （目标 < {CV_WARN}）")
                print(f"  sign_rate_RL 均值 = {rl_valid['符号一致率_RL'].mean():.2f}  （目标 ≥ {SIGN_RATE_MIN}）")
        if total > 0 and n_stable_dml / total < 0.5:
            print("  [⚠] 超过一半操作变量 DML 不稳定，建议调整集成超参")
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
    ckpt_path   = os.path.join(PLACEBO_OUT_DIR, "checkpoint_placebo_baseline_v3.jsonl")
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
    print(f"\n[安慰剂汇总 — DML]  θ均值={df_out['θ_DML_安慰剂'].mean():+.5f}  "
          f"显著率={sig_rate_dml:.1%}（期望≈5%）")
    print(f"  {'[✓] 通过' if sig_rate_dml <= 0.2 else '[⚠] 显著率偏高，可能存在虚假相关'}")

    if "显著_RL" in df_out.columns:
        rl_valid = df_out.dropna(subset=["θ_RL_安慰剂"])
        if not rl_valid.empty:
            sig_rate_rl = rl_valid["显著_RL"].mean()
            print(f"\n[安慰剂汇总 — R-Learner]  θ均值={rl_valid['θ_RL_安慰剂'].mean():+.5f}  "
                  f"显著率={sig_rate_rl:.1%}（期望≈5%）")
            print(f"  {'[✓] 通过' if sig_rate_rl <= 0.2 else '[⚠] 显著率偏高，可能存在虚假相关'}")

    out_path = os.path.join(PLACEBO_OUT_DIR, "refutation_placebo_baseline_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验二：随机混杂变量反驳
# ═══════════════════════════════════════════════════════════════════
def run_random_confounder(df, ops, states, n_confounders=5, n_repeats=1,
                          workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验二：随机混杂变量反驳（注入 {n_confounders} 个随机噪声列）")
    print(f" 判断标准：t_diff = |Δθ| / √(SE_orig²+SE_noisy²) < 2.0")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    ckpt_path   = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "checkpoint_rc_baseline_v3.jsonl")
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
        r = _unpack_result(result)
        if r["p_dml"] > 0.05 and r["cv_dml"] > CV_WARN:
            print(f"  [跳过] {op:<30s}  不显著且不稳定 "
                  f"(p_DML={r['p_dml']:.3f}, CV_DML={r['cv_dml']:.3f})"); continue
        orig_thetas[op] = {
            "theta_dml": r["theta_dml"], "SE_dml": r["SE_dml"],
            "theta_rl": r["theta_rl"], "SE_rl": r["SE_rl"],
            "safe_x": safe_x,
        }
        rl_str = f"  θ_RL={r['theta_rl']:+.5f}" if r["theta_rl"] is not None else "  θ_RL=N/A"
        flag = "⚠ 不稳定" if r["cv_dml"] > CV_WARN else "✓"
        print(f"  {op:<30s}  θ_DML={r['theta_dml']:+.5f}  SE_DML={r['SE_dml']:.5f}  "
              f"CV_DML={r['cv_dml']:.3f}{rl_str}  {flag}")

    tasks = []
    for op, orig in orig_thetas.items():
        for rep in range(n_repeats):
            tasks.append({
                "_key": f"{op}__rep{rep}", "op": op, "rep": rep,
                "n_confounders": n_confounders,
                "theta_dml_orig": orig["theta_dml"],
                "SE_dml_orig": orig["SE_dml"],
                "theta_rl_orig": orig["theta_rl"],
                "SE_rl_orig": orig["SE_rl"],
                "safe_x_orig": orig["safe_x"], "df": df,
            })
    _run_parallel(tasks, _worker_random_confounder, ckpt_path, workers, desc="随机混杂")
    recs   = [r for r in _read_all_records(ckpt_path) if not r.get("_filtered")]
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    if df_out.empty:
        print("[警告] 随机混杂实验无有效结果"); return df_out

    pass_rate_dml = df_out["通过反驳_DML"].mean()
    print(f"\n[随机混杂汇总 — DML]  反驳通过率 = {pass_rate_dml:.1%}（期望 ≥ 80%）")
    print(f"  {'[✓] 通过' if pass_rate_dml >= 0.8 else '[⚠] 通过率偏低，θ_DML 对随机噪声注入仍然敏感'}")

    if "通过反驳_RL" in df_out.columns:
        rl_valid = df_out.dropna(subset=["通过反驳_RL"])
        if not rl_valid.empty:
            pass_rate_rl = rl_valid["通过反驳_RL"].mean()
            print(f"\n[随机混杂汇总 — R-Learner]  反驳通过率 = {pass_rate_rl:.1%}（期望 ≥ 80%）")
            print(f"  {'[✓] 通过' if pass_rate_rl >= 0.8 else '[⚠] 通过率偏低，θ_RL 对随机噪声注入仍然敏感'}")

    out_path = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "refutation_rc_baseline_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  实验三：数据子集反驳
# ═══════════════════════════════════════════════════════════════════
def run_data_subset(df, ops, states, n_subsets=8, subset_frac=0.8,
                    workers=4, dag_roles: dict = None):
    print("\n" + "=" * 70)
    print(f" 实验三：数据子集反驳（{n_subsets} 个子集，每个取 {subset_frac:.0%} 数据）")
    print("=" * 70)
    if dag_roles is None:
        dag_roles = {}
    ckpt_path   = os.path.join(DATA_SUBSET_OUT_DIR, "checkpoint_ds_baseline_v3.jsonl")
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

    # DML 稳定性评估
    stable_ops_dml = total_ops = 0
    stable_ops_rl  = total_ops_rl = 0
    for op, grp in df_out.groupby("操作节点"):
        if len(grp) < 3:
            continue
        total_ops += 1
        arr_dml = grp["θ_DML_子集"].values
        cv_dml  = np.std(arr_dml) / (abs(np.mean(arr_dml)) + 1e-8)
        sc_dml  = np.mean(np.sign(arr_dml) == np.sign(np.median(arr_dml)))
        if cv_dml < 0.30 and sc_dml >= 0.70:
            stable_ops_dml += 1

        if "θ_RL_子集" in grp.columns:
            rl_vals = grp["θ_RL_子集"].dropna().values
            if len(rl_vals) >= 3:
                total_ops_rl += 1
                cv_rl = np.std(rl_vals) / (abs(np.mean(rl_vals)) + 1e-8)
                sc_rl = np.mean(np.sign(rl_vals) == np.sign(np.median(rl_vals)))
                if cv_rl < 0.30 and sc_rl >= 0.70:
                    stable_ops_rl += 1

    if total_ops:
        gp_dml = stable_ops_dml / total_ops
        print(f"\n[数据子集汇总 — DML]  全局稳定通过率 = {gp_dml:.1%}（期望 ≥ 70%）")
        print(f"  {'[✓] 通过' if gp_dml >= 0.70 else '[⚠] θ_DML 跨时段稳定性不足'}")
    if total_ops_rl:
        gp_rl = stable_ops_rl / total_ops_rl
        print(f"\n[数据子集汇总 — R-Learner]  全局稳定通过率 = {gp_rl:.1%}（期望 ≥ 70%）")
        print(f"  {'[✓] 通过' if gp_rl >= 0.70 else '[⚠] θ_RL 跨时段稳定性不足'}")

    out_path = os.path.join(DATA_SUBSET_OUT_DIR, "refutation_ds_baseline_v3.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="XIN_2 因果推断反驳实验 — 基线 V3（RF+GBM 集成 DML + R-Learner）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
建议执行顺序：
  # 1. 小样本快速验证（几分钟内）
  python run_refutation_xin2_baseline_v3.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 2. 全量稳定性诊断
  python run_refutation_xin2_baseline_v3.py --mode stability

  # 3. 反驳实验
  python run_refutation_xin2_baseline_v3.py --mode placebo --n_permutations 5 --workers 4
  python run_refutation_xin2_baseline_v3.py --mode random_confounder --workers 4
  python run_refutation_xin2_baseline_v3.py --mode data_subset --workers 4

  # 4. 全部一次跑
  python run_refutation_xin2_baseline_v3.py --mode all --workers 4

断点续传：检查点含 _baseline_v3 后缀，与 v3/v4/v5/baseline_v1/baseline_v2 不冲突。
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
                   help="DAG 角色明细 CSV 路径。若不指定，回退到纯相关性筛选。")
    p.add_argument("--operability-csv", type=str, default=DEFAULT_OPERABILITY_CSV,
                   help=f"操作性分类 CSV 路径（默认 {DEFAULT_OPERABILITY_CSV}）。")
    return p.parse_args()


def main():
    args = parse_args()
    global N_BOOTSTRAP
    N_BOOTSTRAP = args.n_bootstrap

    print("=" * 70)
    print(f" XIN_2 因果推断反驳实验 — 基线 V3（RF+GBM 集成 DML + R-Learner）")
    print(f" 模式: {args.mode.upper()}  |  并行线程: {args.workers}  |  Bootstrap: {N_BOOTSTRAP}")
    print(f" 架构: 集成 DML（RF+GBM nuisance 均值）+ R-Learner + 滞后特征（SEQ_LEN={SEQ_LEN}）")
    print(f"   RF(n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, "
          f"min_samples_leaf={RF_MIN_SAMPLES_LEAF})")
    print(f"   GBM(n_estimators={GBM_N_ESTIMATORS}, max_depth={GBM_MAX_DEPTH}, "
          f"lr={GBM_LEARNING_RATE}, subsample={GBM_SUBSAMPLE})")
    print(f"   R-Learner(n_estimators={RLEARNER_N_ESTIMATORS}, max_depth={RLEARNER_MAX_DEPTH}, "
          f"min_samples_leaf={RLEARNER_MIN_SAMPLES_LEAF})")
    print(f"   交叉拟合：K_FOLDS={K_FOLDS}, EMBARGO_GAP={EMBARGO_GAP}")
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
