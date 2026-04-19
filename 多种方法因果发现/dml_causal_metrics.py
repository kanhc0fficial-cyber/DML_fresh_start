"""
dml_causal_metrics.py
=====================
面向双重机器学习（DML）的因果发现评估指标

核心动机
--------
传统指标（SHD/F1/TPR/FDR）将所有边一视同仁，但在 DML 场景下误判代价高度不对称：

  ┌──────────────────────┬──────────────────────────────────────────────┐
  │ 误判类型             │ DML 后果                                     │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 中介变量 → 控制集    │ 阻断因果路径，ATE 向 0 偏（过控制偏差）     │
  │ 对撞变量 → 控制集    │ 打开虚假路径，引入新混杂（碰撞器偏差）      │
  │ 混杂变量 ← 漏掉      │ 经典混杂偏差，ATE 不一致                    │
  │ 工具变量 → 误判混杂  │ 浪费有效工具，降低估计效率                  │
  └──────────────────────┴──────────────────────────────────────────────┘

新指标：DML 控制质量得分（DML-CQS）
------------------------------------
对每个(处理 T, 结果 Y)对，将其余变量分四类，然后评估：

  CIS  (Confounder Inclusion Score)    : 混杂纳入 F1，衡量"是否纳入了该纳入的"
  BCES (Bad Control Exclusion Score)   : 坏控制排除 F1，衡量"是否排除了不该纳入的"
  IVP  (Instrument Variable Precision) : 工具识别精度，衡量"找到的工具靠不靠谱"

  DML-CQS = 0.4 × CIS + 0.4 × BCES + 0.2 × IVP

图遍历说明
----------
本模块只依赖 numpy，不引入额外图论库，所有路径搜索用矩阵幂次法实现。
使用方前提：输入邻接矩阵已二值化（threshold 后的 0/1 矩阵）。
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# 一、图基础操作
# ══════════════════════════════════════════════════════════════════════════════

def binarize(adj: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """将软权重邻接矩阵二值化"""
    b = (np.abs(adj) > threshold).astype(np.float32)
    np.fill_diagonal(b, 0)
    return b


def get_descendants(adj_bin: np.ndarray, node: int) -> Set[int]:
    """
    返回节点 node 的所有后代（不含自身）。
    方法：对邻接矩阵做矩阵幂次累积，直到不动点。
    adj_bin[i, j] = 1 表示 i → j。
    """
    d = adj_bin.shape[0]
    reachable = np.zeros(d, dtype=bool)
    frontier = np.zeros(d, dtype=bool)
    frontier[node] = True

    for _ in range(d):
        # 从 frontier 出发，走一步
        next_step = (adj_bin[frontier].sum(axis=0) > 0)
        new = next_step & ~reachable & ~frontier
        reachable |= frontier
        if not new.any():
            break
        frontier = new

    reachable[node] = False  # 排除自身
    return set(np.where(reachable)[0].tolist())


def get_ancestors(adj_bin: np.ndarray, node: int) -> Set[int]:
    """
    返回节点 node 的所有祖先（不含自身）。
    等价于在转置图上求后代。
    """
    return get_descendants(adj_bin.T, node)


def has_directed_path(adj_bin: np.ndarray, src: int, dst: int) -> bool:
    """判断 src → ... → dst 是否存在有向路径"""
    return dst in get_descendants(adj_bin, src)


# ══════════════════════════════════════════════════════════════════════════════
# 二、因果角色分类
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CausalRoles:
    """给定(T, Y)对，图中其余变量的因果角色分类结果"""
    confounders:  FrozenSet[int] = field(default_factory=frozenset)
    mediators:    FrozenSet[int] = field(default_factory=frozenset)
    colliders:    FrozenSet[int] = field(default_factory=frozenset)
    instruments:  FrozenSet[int] = field(default_factory=frozenset)
    others:       FrozenSet[int] = field(default_factory=frozenset)

    @property
    def bad_controls(self) -> FrozenSet[int]:
        """坏控制变量 = 中介 + 对撞"""
        return self.mediators | self.colliders

    @property
    def valid_controls(self) -> FrozenSet[int]:
        """有效控制变量 = 混杂（DML 需要纳入的）"""
        return self.confounders

    def summary(self) -> str:
        return (
            f"Confounders={len(self.confounders)}, "
            f"Mediators={len(self.mediators)}, "
            f"Colliders={len(self.colliders)}, "
            f"Instruments={len(self.instruments)}, "
            f"Others={len(self.others)}"
        )


def identify_causal_roles(
    adj_bin: np.ndarray,
    treatment: int,
    outcome: int
) -> CausalRoles:
    """
    基于二值化 DAG，识别所有变量（除 T 和 Y 外）的因果角色。

    定义（基于 Pearl 因果框架）
    ─────────────────────────────
    混杂变量 (Confounder) :
        既是 T 的祖先，也是 Y 的祖先，且不在 T→Y 路径上。
        → DML 必须控制。

    中介变量 (Mediator) :
        T 的后代，且是 Y 的祖先（即在 T→Y 的有向路径上）。
        → 坏控制：控制它会阻断部分因果效应。

    对撞变量 (Collider) :
        既是 T 的后代，也是 Y 的后代（T→V←Y 结构，或被其后代激活）。
        → 坏控制：控制它会打开虚假关联。
        注：此处采用"共同后代"近似定义，捕捉最常见情形。

    工具变量 (Instrument) :
        T 的祖先，但非 Y 的祖先，也不是 T 的后代。
        → 可用于 IV 估计，不应纳入控制集。

    其他 (Others) :
        与 T/Y 均无直接因果关联。

    参数
    ----
    adj_bin   : 二值化有向邻接矩阵 (d×d)，adj[i,j]=1 表示 i→j
    treatment : 处理变量索引
    outcome   : 结果变量索引

    返回
    ----
    CausalRoles 数据类
    """
    d = adj_bin.shape[0]
    all_vars = frozenset(range(d)) - {treatment, outcome}

    t_desc = get_descendants(adj_bin, treatment)   # T 的后代
    t_anc  = get_ancestors(adj_bin, treatment)     # T 的祖先
    y_anc  = get_ancestors(adj_bin, outcome)       # Y 的祖先
    y_desc = get_descendants(adj_bin, outcome)     # Y 的后代

    # 中介：在 T→Y 有向路径上（T 的后代 ∩ Y 的祖先）
    mediators = frozenset((t_desc & y_anc) & all_vars)

    # 对撞（近似）：T 和 Y 的共同后代
    colliders = frozenset((t_desc & y_desc) & all_vars)

    # ── 后门路径祖先（关键修正）──────────────────────────────────────────
    # 工具变量 Z 的特征：Z 是 T 的祖先，但对 Y 的所有影响都经过 T（无后门路径）。
    # 判断方法：在"删除 T 节点所有出边"的子图中，检查 Z 是否还能到达 Y。
    # 能到达 → Z 有 T 以外的途径影响 Y → 混杂变量
    # 不能到达 → Z 只能通过 T 影响 Y → 工具变量
    adj_no_t_out = adj_bin.copy()
    adj_no_t_out[treatment, :] = 0    # 删除 T 的所有出边
    y_backdoor_anc = get_ancestors(adj_no_t_out, outcome)  # Y 的"后门路径"祖先

    # 混杂：T 和 Y 的共同祖先，且在删除 T 出边后仍能到达 Y（= 有后门路径）
    # 注意：Python 运算符优先级 `-` > `&`，加显式括号确保可读性与正确性一致
    confounders = frozenset(((t_anc & y_backdoor_anc) & all_vars) - mediators - colliders)

    # 工具：T 的祖先，且在删除 T 出边后无法到达 Y（= 只能通过 T 影响 Y）
    instruments = frozenset((t_anc - y_backdoor_anc) & all_vars)

    others = frozenset(all_vars - confounders - mediators - colliders - instruments)

    return CausalRoles(
        confounders=confounders,
        mediators=mediators,
        colliders=colliders,
        instruments=instruments,
        others=others
    )


# ══════════════════════════════════════════════════════════════════════════════
# 三、DML 控制质量得分（核心新指标）
# ══════════════════════════════════════════════════════════════════════════════

def _f1(tp: int, fp: int, fn: int) -> float:
    """计算 F1，处理零分母"""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


@dataclass
class DMLCQSResult:
    """单个(T, Y)对的 DML 控制质量得分详情"""
    treatment: int
    outcome:   int

    # 子指标
    cis:  float   # Confounder Inclusion Score
    bces: float   # Bad Control Exclusion Score
    ivp:  float   # Instrument Variable Precision

    # 综合得分
    dml_cqs: float  # = 0.4*CIS + 0.4*BCES + 0.2*IVP

    # 计数细节（供调试）
    n_true_confounders:  int = 0
    n_true_bad_controls: int = 0
    n_true_instruments:  int = 0
    n_pred_confounders:  int = 0
    n_pred_bad_controls: int = 0
    n_pred_instruments:  int = 0

    # 角色分类（供可视化）
    true_roles: Optional[CausalRoles] = field(default=None)
    pred_roles: Optional[CausalRoles] = field(default=None)


def compute_dml_cqs(
    adj_true: np.ndarray,
    adj_pred: np.ndarray,
    treatment: int,
    outcome:   int,
    threshold: float = 0.05,
    w_cis:  float = 0.4,
    w_bces: float = 0.4,
    w_ivp:  float = 0.2,
) -> DMLCQSResult:
    """
    计算单个(处理T, 结果Y)对的 DML 控制质量得分。

    参数
    ----
    adj_true  : 真实 DAG 邻接矩阵
    adj_pred  : 预测 DAG 邻接矩阵（软权重，内部自动二值化）
    treatment : 处理变量索引
    outcome   : 结果变量索引
    threshold : 二值化阈值
    w_cis/w_bces/w_ivp : 各子指标权重（默认 0.4/0.4/0.2，须和为 1）

    返回
    ----
    DMLCQSResult 数据类，含三个子指标和综合得分
    """
    adj_true_bin = binarize(adj_true, threshold=threshold)  # 真实图二值化（与预测图使用相同阈值）
    adj_pred_bin = binarize(adj_pred, threshold=threshold)

    true_roles = identify_causal_roles(adj_true_bin, treatment, outcome)
    pred_roles = identify_causal_roles(adj_pred_bin, treatment, outcome)

    # ── 子指标一：混杂纳入得分（CIS） ─────────────────────────────────────
    # 真正应该纳入的：true_roles.confounders
    # 算法认为应纳入的（= 预测为混杂）：pred_roles.confounders
    true_conf  = true_roles.confounders
    pred_conf  = pred_roles.confounders
    tp_conf    = len(true_conf & pred_conf)
    fp_conf    = len(pred_conf - true_conf)
    fn_conf    = len(true_conf - pred_conf)
    cis        = _f1(tp_conf, fp_conf, fn_conf)

    # ── 子指标二：坏控制排除得分（BCES） ──────────────────────────────────
    # 真正的坏控制：true_roles.bad_controls（中介 + 对撞）
    # 算法正确排除了多少（= 预测中没有把坏控制误判为混杂）
    #
    # 正例 = 真正的坏控制；算法"发现"坏控制 = pred_roles.bad_controls
    # TP = 真实坏控制 且 预测也是坏控制（成功识别）
    # FP = 真实不是坏控制 但预测为坏控制（误报普通变量为坏控制）
    # FN = 真实坏控制 但预测为混杂（最严重错误：把坏控制放进了控制集）
    true_bad  = true_roles.bad_controls
    pred_bad  = pred_roles.bad_controls
    tp_bad    = len(true_bad & pred_bad)
    fp_bad    = len(pred_bad - true_bad)
    fn_bad    = len(true_bad - pred_bad)
    bces      = _f1(tp_bad, fp_bad, fn_bad)

    # ── 子指标三：工具变量精度（IVP） ─────────────────────────────────────
    # 只看 Precision（宁缺毋滥：找错工具比没找到危害更大）
    true_iv   = true_roles.instruments
    pred_iv   = pred_roles.instruments
    tp_iv     = len(true_iv & pred_iv)
    fp_iv     = len(pred_iv - true_iv)
    ivp       = _precision(tp_iv, fp_iv)

    # ── 综合 DML-CQS ──────────────────────────────────────────────────────
    dml_cqs = w_cis * cis + w_bces * bces + w_ivp * ivp

    return DMLCQSResult(
        treatment=treatment,
        outcome=outcome,
        cis=cis,
        bces=bces,
        ivp=ivp,
        dml_cqs=dml_cqs,
        n_true_confounders=len(true_conf),
        n_true_bad_controls=len(true_bad),
        n_true_instruments=len(true_iv),
        n_pred_confounders=len(pred_conf),
        n_pred_bad_controls=len(pred_bad),
        n_pred_instruments=len(pred_iv),
        true_roles=true_roles,
        pred_roles=pred_roles,
    )


def compute_dml_cqs_multi(
    adj_true: np.ndarray,
    adj_pred: np.ndarray,
    treatment_indices: List[int],
    outcome_idx:       int,
    threshold: float = 0.05,
) -> Dict:
    """
    对多个处理变量分别计算 DML-CQS，返回汇总统计。

    返回字典键：
      'per_treatment' : List[DMLCQSResult]
      'mean_cis'      : float
      'mean_bces'     : float
      'mean_ivp'      : float
      'mean_dml_cqs'  : float  ← 最终汇报指标
    """
    per_treatment = []
    for t in treatment_indices:
        if t == outcome_idx:
            continue
        result = compute_dml_cqs(adj_true, adj_pred, t, outcome_idx, threshold)
        per_treatment.append(result)

    if not per_treatment:
        return {
            'per_treatment': [],
            'mean_cis': 0.0, 'mean_bces': 0.0,
            'mean_ivp': 0.0, 'mean_dml_cqs': 0.0,
        }

    return {
        'per_treatment': per_treatment,
        'mean_cis':      float(np.mean([r.cis      for r in per_treatment])),
        'mean_bces':     float(np.mean([r.bces     for r in per_treatment])),
        'mean_ivp':      float(np.mean([r.ivp      for r in per_treatment])),
        'mean_dml_cqs':  float(np.mean([r.dml_cqs  for r in per_treatment])),
    }
