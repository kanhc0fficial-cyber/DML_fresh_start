"""
run_refutation_xin2_baseline_v2.py
====================================
XIN_2 因果推断反驳实验 — 基线方法 V2「梯度提升 DML（GBM-DML）」

═══════════════════════════════════════════════════════════════════
  基线定位：提供可对比的传统机器学习基线，与创新版 v3/v4/v5（LSTM-VAE-DML）
  以及基线 V1（RF-DML）形成多维对照。本脚本使用梯度提升树（GBM），
  在非线性建模能力和偏差-方差折中上与 RF 有所不同。

  方法：梯度提升 Double ML（GBM-DML）
  ─────────────────────────────────────────────────────────────────
  核心思路（Chernozhukov et al. 2018 的 DML-PLM 框架）：
    1. 构建滞后特征矩阵：将过去 SEQ_LEN=6 个时间步的状态变量展平为
       特征向量，显式捕捉滞后效应（工业场景滞后效应强，必须纳入）
    2. 交叉拟合（Cross-Fitting）：
         - 在训练折用 GBM 拟合 Ŷ = E[Y|X_lag] 和 D̂ = E[D|X_lag]
         - 在验证折计算残差 res_Y = Y - Ŷ，res_D = D - D̂
    3. DML 估计：θ = Cov(res_Y, res_D) / Var(res_D)
       （在原始单位上反归一化）
    4. Bootstrap 聚合：N_BOOTSTRAP 次独立拟合，θ 取中位数

  为何选 GBM 作 V2（对比 RF 的 V1）？
    - GBM 是序贯 Boosting，每轮只修正上一轮的残差，偏差更低
    - 在中等维度特征下，GBM 通常比 RF 精度更高（但训练略慢）
    - subsample < 1 提供随机性（类似 RF 的 bagging），增加多样性
    - 与 RF 形成互补基线：二者同时通过反驳 → 结论更可信

  GBM vs RF 的关键差异（期待在实验中体现）：
    - GBM 偏差更低 → 残差更干净 → θ 估计方差可能更小（CV 更低）
    - GBM 更容易过拟合小样本 → min_samples_leaf 设大一点以保守
    - 两者输出一致 → 对 v3/v4/v5 的佐证更强

  与创新版的对比：
    - V3/V4/V5 使用 LSTM-VAE 学习时序潜变量表征（SEQ 结构内化于模型）
    - 本基线用手工滞后展平（SEQ 结构外化为特征工程），适合对照研究

═══════════════════════════════════════════════════════════════════

用法：
  # 快速稳定性诊断（截取 2000 条数据，3 次 Bootstrap，几分钟内完成）
  python run_refutation_xin2_baseline_v2.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 全量稳定性诊断
  python run_refutation_xin2_baseline_v2.py --mode stability

  # 安慰剂反驳
  python run_refutation_xin2_baseline_v2.py --mode placebo --n_permutations 5 --workers 4

  # 随机混杂变量反驳
  python run_refutation_xin2_baseline_v2.py --mode random_confounder --workers 4

  # 数据子集反驳
  python run_refutation_xin2_baseline_v2.py --mode data_subset --workers 4

  # 全部实验一次跑
  python run_refutation_xin2_baseline_v2.py --mode all --workers 4
"""

import argparse
import concurrent.futures
import glob as _glob_module
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
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
#  路径配置（与 v3/v4/v5/baseline_v1 完全一致）
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

# ── 实验结果输出目录（与 v3/baseline_v1 共享目录，文件名含 _baseline_v2 后缀）──
PLACEBO_OUT_DIR           = os.path.join(BASE_DIR, "安慰剂实验")
RANDOM_CONFOUNDER_OUT_DIR = os.path.join(BASE_DIR, "随机混杂变量实验")
DATA_SUBSET_OUT_DIR       = os.path.join(BASE_DIR, "数据子集实验")
STABILITY_OUT_DIR         = os.path.join(BASE_DIR, "稳定性诊断")

for _d in [PLACEBO_OUT_DIR, RANDOM_CONFOUNDER_OUT_DIR,
           DATA_SUBSET_OUT_DIR, STABILITY_OUT_DIR]:
    os.makedirs(_d, exist_ok=True)

DEFAULT_DAG_ROLES_CSV = next(
    iter(sorted(
        _glob_module.glob(os.path.join(REPO_ROOT, "DAG图分析", "DAG解析结果", "*_Roles_Table.csv")),
        key=os.path.getmtime,
        reverse=True,
    )),
    "",
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

# ── GBM 超参 ───────────────────────────────────────────────────────
GBM_N_ESTIMATORS   = 100
GBM_MAX_DEPTH      = 3      # GBM 推荐浅树（防止过拟合，每轮专注修残差）
GBM_LEARNING_RATE  = 0.1    # 标准学习率
GBM_SUBSAMPLE      = 0.8    # 随机抽样（Stochastic GBM，减少方差、加快收敛）
GBM_MIN_SAMPLES_LEAF = 10   # 叶节点最小样本数（工业小样本场景下保守设置）


# ═══════════════════════════════════════════════════════════════════
#  DAG 因果角色加载（与 v3/baseline_v1 完全一致）
# ═══════════════════════════════════════════════════════════════════
def load_dag_roles(csv_path: str) -> dict:
    """加载 DAG 角色明细表。格式：Treatment_T, Role, Node_Name"""
    if not csv_path:
        print("[DAG过滤] 未找到角色表 CSV（DAG图分析/DAG解析结果/ 下无 *_Roles_Table.csv），"
              "将回退到纯相关性筛选")
        return {}
    if not os.path.isfile(csv_path):
        print(f"[DAG过滤] 角色表文件不存在: {csv_path}，将回退到纯相关性筛选")
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
    """与 v3/baseline_v1 完全一致：按滞后相关性筛选控制变量"""
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
    """与 v3/baseline_v1 完全一致：相关性筛选后叠加 DAG 角色过滤"""
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
#  确定性哈希种子（与 v3/baseline_v1 完全一致）
# ═══════════════════════════════════════════════════════════════════
def _op_seed(op: str) -> int:
    return int(hashlib.md5(op.encode()).hexdigest()[:8], 16) % 100000


# ═══════════════════════════════════════════════════════════════════
#  滞后特征矩阵构建（与 baseline_v1 完全一致）
# ═══════════════════════════════════════════════════════════════════
def _build_lag_features(X_norm: np.ndarray, n_lags: int) -> np.ndarray:
    """
    将归一化状态矩阵 X_norm (T × d) 展平为滞后特征矩阵。

    对时间点 t（t >= n_lags），特征向量 = flatten(X_norm[t-n_lags : t])，
    shape = (n_lags × d,)，对应 LSTM 的输入窗口。

    返回：shape (T - n_lags, n_lags * d) 的 float32 矩阵。
    显式捕捉工业场景中的强滞后效应，与 v3 的 LSTM 序列输入等价。

    [优化] 用向量化高级索引替代 Python 循环 + list.append，
    对大数据集（数万行 × 数十维 × SEQ_LEN=6）速度提升 10-50x。
    """
    T, d = X_norm.shape
    n_samples = T - n_lags
    # row_idx[i, :] = [i, i+1, ..., i+n_lags-1]，即从时间步 i 开始的 n_lags 个行索引
    row_idx = np.arange(n_samples)[:, None] + np.arange(n_lags)[None, :]  # (n_samples, n_lags)
    windows = X_norm[row_idx]  # (n_samples, n_lags, d)
    return windows.reshape(n_samples, n_lags * d).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  核心训练函数：GBM-DML + Bootstrap θ 聚合
# ═══════════════════════════════════════════════════════════════════
def train_one_op(op: str, df: pd.DataFrame, safe_x: list,
                 override_D=None, n_bootstrap: int = N_BOOTSTRAP):
    """
    GBM Double ML 核心函数。

    对每个 bootstrap × fold：
      1. 构建滞后特征矩阵 X_lag（T-SEQ_LEN 行，SEQ_LEN×d 列）
      2. 在训练折用 GradientBoostingRegressor 拟合：
           Ŷ = E[Y | X_lag]   (head_Y)
           D̂ = E[D | X_lag]   (head_D)
      3. 在验证折计算 DML 残差：res_Y = Y - Ŷ，res_D = D - D̂
      4. DML theta：θ_std = Cov(res_D, res_Y) / Var(res_D)
         θ = θ_std × (Y_std / D_std)（反归一化）
    Bootstrap N_BOOTSTRAP 次，θ 取中位数。

    GBM 相比 RF（baseline_v1）：
      - 序贯拟合，每轮只修正残差，偏差更低
      - subsample=0.8 提供随机性，增加模型多样性
      - max_depth=3（浅树），learning_rate=0.1（标准设置）

    返回：(theta_med, p_val, SE_boot, n_avg, f_med, cv, sign_rate)
    或 None（工具弱/样本不足/多数折失败）
    """
    # ── 标准化 ────────────────────────────────────────────────────
    # ffill X 列（DCS exception-based 录制，只在变化时记录，ffill 是正确的）
    X_df   = df[safe_x].ffill().bfill()
    X_raw  = X_df.values.astype(np.float32)
    X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)

    Y_raw  = df["Y_grade"].values.astype(np.float32)
    if override_D is None:
        # ffill 操作变量（同 X 的记录方式），再用列均值填充残余 NaN
        D_series = df[op].ffill().bfill()
        D_raw    = D_series.fillna(D_series.mean()).values.astype(np.float32)
    else:
        D_raw = np.asarray(override_D, dtype=np.float32)
    Y_mean, Y_std = float(Y_raw.mean()), float(Y_raw.std()) + 1e-8
    D_mean, D_std = float(D_raw.mean()), float(D_raw.std()) + 1e-8
    Y_norm = (Y_raw - Y_mean) / Y_std
    D_norm = (D_raw - D_mean) / D_std

    # ── 滞后特征展平 ──────────────────────────────────────────────
    # X_lag[i] = flatten(X_norm[i : i+SEQ_LEN])，对应目标 Y[i+SEQ_LEN]
    X_lag  = _build_lag_features(X_norm, SEQ_LEN)   # (N, SEQ_LEN*d)
    Y_vec  = Y_norm[SEQ_LEN:]                        # (N,)
    D_vec  = D_norm[SEQ_LEN:]                        # (N,)
    N          = len(X_lag)
    block_size = N // K_FOLDS

    op_base_seed = _op_seed(op)
    theta_list, f_list, n_list = [], [], []

    for boot_i in range(n_bootstrap):
        base_seed    = boot_i * 99991 + op_base_seed
        all_res_Y, all_res_D = [], []
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

            # GBM 拟合 Y 的期望（E[Y|X_lag]）
            model_Y = GradientBoostingRegressor(
                n_estimators=GBM_N_ESTIMATORS,
                max_depth=GBM_MAX_DEPTH,
                learning_rate=GBM_LEARNING_RATE,
                subsample=GBM_SUBSAMPLE,
                min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
                random_state=rng_seed,
            )
            model_Y.fit(X_tr, Y_tr)

            # GBM 拟合 D 的期望（E[D|X_lag]）
            model_D = GradientBoostingRegressor(
                n_estimators=GBM_N_ESTIMATORS,
                max_depth=GBM_MAX_DEPTH,
                learning_rate=GBM_LEARNING_RATE,
                subsample=GBM_SUBSAMPLE,
                min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
                random_state=rng_seed + 1,
            )
            model_D.fit(X_tr, D_tr)

            res_Y_k = Y_vl - model_Y.predict(X_vl)
            res_D_k = D_vl - model_D.predict(X_vl)

            all_res_Y.extend(res_Y_k)
            all_res_D.extend(res_D_k)
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
        n = len(res_D)

        # F 统计量（工具强度）
        var_D  = np.var(res_D)
        f_stat = var_D / (D_std ** 2 + 1e-12) * n
        if f_stat < F_STAT_THRESHOLD:
            continue

        # DML θ（协方差矩阵方式，与 v3/baseline_v1 完全一致）
        cov_mat   = np.cov(res_D, res_Y)
        theta_std = cov_mat[0, 1] / (cov_mat[0, 0] + 1e-12)
        theta     = theta_std * (Y_std / D_std)

        theta_list.append(theta)
        f_list.append(f_stat)
        n_list.append(n)

    # ── Bootstrap 聚合 ────────────────────────────────────────────
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
#  断点续传工具（与 v3/baseline_v1 完全一致）
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
#  通用并行调度器（与 v3/baseline_v1 完全一致）
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
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
#  Worker 函数（与 baseline_v1 一致，调用本脚本的 train_one_op）
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
    theta_med, p_val, SE, n, f, cv, sr = result
    return {
        "_key": key, "_filtered": False,
        "操作节点": op, "排列索引": perm_idx + 1,
        "θ_安慰剂": round(theta_med, 5),
        "P_Value":  round(p_val, 4), "SE_Boot":  round(SE, 5),
        "CV":       round(cv, 4),    "符号一致率": round(sr, 3),
        "有效残差数": n,              "F统计量":  round(f, 2),
        "显著":     bool(p_val < 0.05),
    }


def _worker_random_confounder(task: dict) -> dict:
    op, rep       = task["op"], task["rep"]
    n_confounders = task["n_confounders"]
    theta_orig    = task["theta_orig"]
    SE_orig       = task["SE_orig"]
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
    theta_p, p_p, SE_p, n_p, f_p, cv_p, sr_p = result

    delta       = abs(theta_p - theta_orig)
    se_combined = float(np.sqrt(SE_p ** 2 + SE_orig ** 2)) + 1e-12
    t_diff      = delta / se_combined
    sign_ok     = bool(np.sign(theta_p) == np.sign(theta_orig))
    near_zero   = abs(theta_orig) < 3 * SE_orig
    passed      = bool(t_diff < 2.0 and (sign_ok or near_zero))
    rel_dev     = delta / (abs(theta_orig) + 1e-8)

    return {
        "_key": key, "_filtered": False,
        "操作节点": op, "重复索引": rep + 1,
        "θ_原始":      round(theta_orig, 5),
        "θ_注入噪声":  round(theta_p,    5),
        "相对偏差_ref": round(rel_dev,   4),
        "t_diff":      round(t_diff,     3),
        "方向一致":    sign_ok,
        "P_Value":     round(p_p, 4),  "SE_Boot":   round(SE_p, 5),
        "CV":          round(cv_p, 4), "符号一致率": round(sr_p, 3),
        "有效残差数":  n_p,            "F统计量":   round(f_p, 2),
        "通过反驳":    passed,
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
        "_key": key, "_filtered": False,
        "操作节点": op, "子集索引": sub_idx + 1,
        "时段起点": start, "时段终点": end,
        "θ_子集":   round(theta_med, 5),
        "P_Value":  round(p_val, 4),  "SE_Boot":  round(SE, 5),
        "CV":       round(cv, 4),     "符号一致率": round(sr, 3),
        "有效残差数": n,               "F统计量":  round(f, 2),
    }


# ═══════════════════════════════════════════════════════════════════
#  数据准备（与 v3/baseline_v1 完全一致）
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
    print(f" 架构：GBM-DML 基线 V2（梯度提升 nuisance + 滞后特征）")
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
            "操作节点": op, "θ_中位数": round(theta_med, 5),
            "P_Value":  round(p_val, 4), "SE_Boot": round(SE, 5),
            "CV":       round(cv, 4),    "符号一致率": round(sr, 3),
            "F统计量":  round(f, 2),     "稳定": stable,
        })
    df_out   = pd.DataFrame(rows)
    out_path = os.path.join(STABILITY_OUT_DIR, "stability_diagnosis_baseline_v2.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not df_out.empty:
        total = len(df_out)
        print(f"\n[稳定性诊断汇总]  稳定 {n_stable}/{total} 个操作变量")
        print(f"  CV 均值      = {df_out['CV'].mean():.3f}  （目标 < {CV_WARN}）")
        print(f"  sign_rate 均值 = {df_out['符号一致率'].mean():.2f}  （目标 ≥ {SIGN_RATE_MIN}）")
        if total > 0 and n_stable / total < 0.5:
            print("  [⚠] 超过一半操作变量不稳定，建议调整 GBM 超参（max_depth, n_estimators, learning_rate）")
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
    ckpt_path   = os.path.join(PLACEBO_OUT_DIR, "checkpoint_placebo_baseline_v2.jsonl")
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
    out_path = os.path.join(PLACEBO_OUT_DIR, "refutation_placebo_baseline_v2.csv")
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
    ckpt_path   = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "checkpoint_rc_baseline_v2.jsonl")
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
    out_path = os.path.join(RANDOM_CONFOUNDER_OUT_DIR, "refutation_rc_baseline_v2.csv")
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
    ckpt_path   = os.path.join(DATA_SUBSET_OUT_DIR, "checkpoint_ds_baseline_v2.jsonl")
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
    out_path = os.path.join(DATA_SUBSET_OUT_DIR, "refutation_ds_baseline_v2.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out_path}"); return df_out


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="XIN_2 因果推断反驳实验 — 基线 V2（GBM-DML）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
建议执行顺序：
  # 1. 小样本快速验证（几分钟内）
  python run_refutation_xin2_baseline_v2.py --mode stability --sample_size 2000 --n_bootstrap 3

  # 2. 全量稳定性诊断
  python run_refutation_xin2_baseline_v2.py --mode stability

  # 3. 反驳实验
  python run_refutation_xin2_baseline_v2.py --mode placebo --n_permutations 5 --workers 4
  python run_refutation_xin2_baseline_v2.py --mode random_confounder --workers 4
  python run_refutation_xin2_baseline_v2.py --mode data_subset --workers 4

  # 4. 全部一次跑
  python run_refutation_xin2_baseline_v2.py --mode all --workers 4

断点续传：检查点含 _baseline_v2 后缀，与 v3/v4/v5/baseline_v1 不冲突。
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
    print(f" XIN_2 因果推断反驳实验 — 基线 V2（GBM-DML）")
    print(f" 模式: {args.mode.upper()}  |  并行线程: {args.workers}  |  Bootstrap: {N_BOOTSTRAP}")
    print(f" 架构: Gradient Boosting DML + 滞后特征（SEQ_LEN={SEQ_LEN}）")
    print(f"   GBM(n_estimators={GBM_N_ESTIMATORS}, max_depth={GBM_MAX_DEPTH}, "
          f"lr={GBM_LEARNING_RATE}, subsample={GBM_SUBSAMPLE})")
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
