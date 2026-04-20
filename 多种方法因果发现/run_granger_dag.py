"""
run_granger_dag.py
==================
基于 Granger 因果检验在真实工业数据上进行时序因果发现，
输出 GraphML 供 DAG 解析器下游使用。

算法原理：
  Granger 因果检验：若在 X_j 的自回归模型中加入 X_i 的滞后值后，
  预测误差显著降低（F 检验 p < 阈值），则认为 X_i Granger-引起 X_j。

  对每对有序变量 (i→j)（满足物理拓扑约束）：
    完整模型：X_j,t = α_0 + Σ_{k=1}^{K} α_k X_j,{t-k} + Σ_{k=1}^{K} β_k X_i,{t-k} + ε
    限制模型：X_j,t = α_0 + Σ_{k=1}^{K} α_k X_j,{t-k} + ε
    F 统计量：[(RSS_r - RSS_u)/p] / [RSS_u/(n - 2p - 1)]

  优点：
    - 无需深度学习，计算速度快（CPU 即可处理 100+ 变量）
    - 统计显著性检验，可解释性强
    - 作为深度学习方法的互补验证基线

  适配点（与 run_innovation_real_data.py 一致）：
    1. 物理拓扑掩码过滤：仅对 can_cause=True 的配对进行检验，加速且防止伪发现
    2. y_grade 并入数据，物理方向限制 y→其他全部禁止
    3. 多重检验校正：Benjamini-Hochberg FDR 校正（替代 Bonferroni，保留更多真实边）
    4. 边权重 = -log10(p_val)（越大表示越显著，便于可视化和阈值调优）
    5. 输出图强制 DAG 后处理（按权重移除最弱环边）

用法：
  python run_granger_dag.py [--line xin1|xin2|both] [--lags K]
                            [--alpha α] [--fdr-alpha α_fdr]
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config import prepare_data, can_cause

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 超参数 ──────────────────────────────────────────────────────────────────
# 滞后阶数：10min 采样 × 5 = 50min，覆盖浮选工序主要响应时间
MAX_LAGS = 5

# 原始 p 值阈值（BH 校正前的单次检验显著性水平）
RAW_ALPHA = 0.05

# BH FDR 校正后的目标 FDR 水平（控制假发现率）
FDR_ALPHA = 0.05

# 最小边权重阈值（-log10 p 值，低于此值不加入图）
# log10(0.05) ≈ 1.30，即 p < 0.05 对应 -log10(p) > 1.30
MIN_NEG_LOG_P = 1.30

ALGO_NAME = "granger"


# ─── 物理拓扑掩码 ─────────────────────────────────────────────────────────────

def build_topology_mask(valid_vars, var_to_stage, var_to_group, line):
    """
    构建 (N+1, N+1) 物理因果可行性掩码（含 y_grade 末列/末行）。
    mask[i, j] = 1 表示 i→j 物理上可行。
    """
    N = len(valid_vars)
    mask = np.zeros((N + 1, N + 1), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i != j and can_cause(
                var_to_stage[valid_vars[i]], var_to_stage[valid_vars[j]],
                var_to_group.get(valid_vars[i]), var_to_group.get(valid_vars[j]),
                line,
            ):
                mask[i, j] = 1.0
        # X_i → y_grade
        if can_cause(
            var_to_stage[valid_vars[i]], "Y",
            var_to_group.get(valid_vars[i]), None,
            line,
        ):
            mask[i, N] = 1.0

    mask[N, :] = 0.0
    np.fill_diagonal(mask, 0.0)
    return mask


# ─── OLS 工具函数 ─────────────────────────────────────────────────────────────

def _ols_rss(X_design, y):
    """
    普通最小二乘（OLS）拟合，返回残差平方和 RSS。

    X_design: (T, p) 设计矩阵（含截距列）
    y:        (T,) 目标向量
    """
    # lstsq 直接求最小二乘解（比手写矩阵逆更稳健，处理近奇异情形）
    coef, _, _, _ = np.linalg.lstsq(X_design, y, rcond=-1)
    resid = y - X_design @ coef
    return float(np.dot(resid, resid))


def granger_f_test(x_cause, x_target, max_lags):
    """
    对 x_cause → x_target 方向进行单次 Granger F 检验。

    参数：
      x_cause:  (T,) 候选原因变量（已标准化）
      x_target: (T,) 目标变量（已标准化）
      max_lags: 滞后阶数 K

    返回：
      p_val (float): F 检验 p 值（越小表示越显著）
      f_stat (float): F 统计量
    """
    T = len(x_target)
    K = max_lags
    n_use = T - K  # 有效样本数

    if n_use < 2 * K + 5:
        # 样本量不足以拟合模型，返回最大 p 值
        return 1.0, 0.0

    # 构造限制模型设计矩阵：截距 + x_target 的 K 个滞后
    X_r = np.ones((n_use, 1 + K), dtype=np.float64)
    for k in range(1, K + 1):
        X_r[:, k] = x_target[K - k: T - k]

    # 构造完整模型设计矩阵：限制模型 + x_cause 的 K 个滞后
    X_u = np.zeros((n_use, 1 + 2 * K), dtype=np.float64)
    X_u[:, :1 + K] = X_r
    for k in range(1, K + 1):
        X_u[:, 1 + K + k - 1] = x_cause[K - k: T - k]

    y = x_target[K:].astype(np.float64)

    rss_r = _ols_rss(X_r, y)
    rss_u = _ols_rss(X_u, y)

    # 防止数值问题导致 rss_u > rss_r
    if rss_u >= rss_r or rss_u < 1e-12:
        return 1.0, 0.0

    # 自由度
    df_num = K                        # 新增参数数量
    df_den = n_use - 2 * K - 1       # 完整模型残差自由度

    if df_den < 1:
        return 1.0, 0.0

    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    p_val = float(1.0 - stats.f.cdf(f_stat, df_num, df_den))
    return p_val, f_stat


# ─── BH FDR 校正 ─────────────────────────────────────────────────────────────

def bh_correction(p_values, alpha=FDR_ALPHA):
    """
    Benjamini-Hochberg FDR 校正。

    返回：bool 数组，True 表示显著（通过 FDR 校正）。
    """
    p_arr = np.array(p_values, dtype=np.float64)
    n = len(p_arr)
    if n == 0:
        return np.array([], dtype=bool)

    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha

    # 找最大的 k 使得 p_{(k)} <= k/m * alpha
    cummax = np.zeros(n, dtype=bool)
    last_sig = -1
    for i in range(n):
        if sorted_p[i] <= thresholds[i]:
            last_sig = i

    if last_sig >= 0:
        cummax[:last_sig + 1] = True

    # 还原原始顺序
    result = np.zeros(n, dtype=bool)
    result[order] = cummax
    return result


# ─── 主检验流程 ───────────────────────────────────────────────────────────────

def run_granger(line="xin1", max_lags=MAX_LAGS, fdr_alpha=FDR_ALPHA):
    """
    对单条产线运行 Granger 因果检验并输出 GraphML。

    参数：
      line:     'xin1' 或 'xin2'
      max_lags: 最大滞后阶数
      fdr_alpha: BH FDR 校正目标 FDR 水平
    """
    print(f"\n{'='*70}")
    print(f"Granger 因果检验  [产线={line}]  滞后阶={max_lags}  FDR_α={fdr_alpha}")
    print(f"{'='*70}")

    t0 = time.time()
    df, valid_vars, var_to_stage, var_to_group = prepare_data(line)
    N = len(valid_vars)
    print(f"变量数: {N}（+y_grade=1）  样本数: {len(df)}")

    all_vars = valid_vars + ["y_grade"]
    d = len(all_vars)

    # 全局标准化
    X_raw = df[all_vars].values.astype(np.float64)
    X_norm = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)

    topo_mask = build_topology_mask(valid_vars, var_to_stage, var_to_group, line)
    n_feasible = int(topo_mask.sum())
    total_possible = d * d - d
    print(f"物理可行有向边数: {n_feasible} / {total_possible}  "
          f"({n_feasible / total_possible * 100:.1f}%)")

    # ── 遍历所有物理可行方向，执行 Granger F 检验 ──
    print(f"\n开始 Granger 因果检验（共 {n_feasible} 对）...")
    pairs = []       # [(i, j)] 待检验的有序变量对
    p_vals = []      # p 值列表（对应 pairs）
    f_stats = []     # F 统计量列表

    for i in range(d):
        for j in range(d):
            if topo_mask[i, j] > 0.5:
                p_val, f_stat = granger_f_test(X_norm[:, i], X_norm[:, j], max_lags)
                pairs.append((i, j))
                p_vals.append(p_val)
                f_stats.append(f_stat)

    if not pairs:
        print(f"  [警告] 无物理可行边，请检查拓扑掩码配置")
        return

    # ── BH FDR 校正 ──
    is_sig = bh_correction(p_vals, alpha=fdr_alpha)
    n_sig = int(is_sig.sum())
    print(f"  检验对数: {len(pairs)}  通过 BH FDR 校正（FDR={fdr_alpha}）: {n_sig} 对")

    # ── 构建 DiGraph ──
    G = nx.DiGraph()
    for var in all_vars:
        G.add_node(var)

    edge_count = 0
    for idx, (i, j) in enumerate(pairs):
        if is_sig[idx]:
            p_val = p_vals[idx]
            # 权重：-log10(p_val)，p_val 最小取 1e-300 防止 log(0)
            weight = -np.log10(max(p_val, 1e-300))
            if weight >= MIN_NEG_LOG_P:
                G.add_edge(all_vars[i], all_vars[j],
                           weight=float(weight),
                           p_value=float(p_val),
                           f_stat=float(f_stats[idx]))
                edge_count += 1

    print(f"  阈值过滤后边数: {edge_count}（-log10(p) ≥ {MIN_NEG_LOG_P:.2f}）")

    # ── DAG 后处理：移除最弱环边 ──
    removed = 0
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda e: G[e[0]][e[1]].get("weight", 0.0))
        G.remove_edge(weakest[0], weakest[1])
        edge_count -= 1
        removed += 1
    if removed:
        print(f"  [Granger] DAG 后处理：移除 {removed} 条环边")

    out_path = os.path.join(OUT_DIR, f"{ALGO_NAME}_real_dag_{line}.graphml")
    nx.write_graphml(G, out_path)
    elapsed = time.time() - t0
    print(f"  ✓ [Granger] [{line}]: {edge_count} 条边 → {out_path}  耗时 {elapsed:.1f}s")

    # ── 输出对 y_grade 影响最显著的变量排行 ──
    y_edges = [
        (all_vars[i], -np.log10(max(p_vals[k], 1e-300)), f_stats[k])
        for k, (i, j) in enumerate(pairs)
        if j == d - 1 and is_sig[k]
    ]
    y_edges.sort(key=lambda x: -x[1])
    if y_edges:
        print(f"\n  对 y_grade 影响最显著的变量 Top-10 [{line}]：")
        print(f"  {'变量名':<40} {'-log10(p)':>12} {'F统计量':>10}")
        print(f"  {'-'*40} {'-'*12} {'-'*10}")
        for var, neg_log_p, f_s in y_edges[:10]:
            print(f"  {var:<40} {neg_log_p:>12.3f} {f_s:>10.3f}")

    return G


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Granger 因果检验时序因果发现（双产线支持，输出 GraphML）"
    )
    parser.add_argument(
        "--line",
        choices=["xin1", "xin2", "both"],
        default="both",
        help="产线选择（默认: both）",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=MAX_LAGS,
        help=f"最大滞后阶数（默认: {MAX_LAGS}）",
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=FDR_ALPHA,
        dest="fdr_alpha",
        help=f"BH FDR 校正目标 FDR 水平（默认: {FDR_ALPHA}）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=RAW_ALPHA,
        help=f"原始 p 值显著性阈值（默认: {RAW_ALPHA}，仅作参考，BH 校正以 --fdr-alpha 为准）",
    )
    args = parser.parse_args()

    lines = ["xin1", "xin2"] if args.line == "both" else [args.line]
    for ln in lines:
        run_granger(ln, max_lags=args.lags, fdr_alpha=args.fdr_alpha)
