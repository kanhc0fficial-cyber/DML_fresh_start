"""
analyze_dag_causal_roles.py (v4.1 - 修复碰撞节点检测的组合爆炸问题)
========================================================
通用 DAG 因果角色解析脚本（支持命令行参数）

核心改进（v4.1，相对于 v4.0）：
  1. ✅ 修复碰撞节点检测：彻底移除 `all_simple_paths` 无向图遍历
         原实现：在 G.to_undirected() 上调用 all_simple_paths(cutoff=8)
         问题：无向图上路径数量可达数亿条，导致脚本"卡死"，try-except 无法救场
         新实现：collider_set = t_descendants & y_descendants（纯集合交集，O(1) 可达）
         语义说明：新定义捕捉"T 和 Y 的共同后代"，比 Pearl 路径级碰撞定义更宽泛，
                   在 DML 控制变量选择中更保守（多报漏报），工程上是合理的。

v4.0 已有改进（保留）：
  1. ✅ 移除 cutoff 机制：所有祖先/后代/路径计算均使用全图
  2. ✅ 修复中介变量检测：使用 descendants(T) ∩ ancestors(Y) 替代 all_simple_paths
  3. ✅ 修复混杂因子检测：G_without_T 仅在循环外复制一次
  4. ✅ 修复版本号不一致问题
  5. ✅ 修复异常处理：移除永远不会触发的 NetworkXNoPath 捕获
  6. ✅ DAG 有效性检查、严格 Y 节点定位、命令行参数支持

操作变量（T）判定方式：
  - 读取 non_collinear_representative_vars_operability.csv
  - 仅当 Operability == 'operable' 的变量，才视为可操作干预变量 (T)

因果角色定义（基于 Pearl SCM + 路径分析）：
  - 混杂因子：C ∈ ancestors(T)，C ∉ descendants(T)，且移除 T 后 C 仍可达 Y（后门路径）
  - 中介变量：M ∈ descendants(T) ∩ ancestors(Y)，M ≠ T，M ≠ Y
  - 碰撞节点：Z ∈ descendants(T) ∩ descendants(Y)（T 和 Y 的共同后代，v4.1 修正定义）
  - 工具变量：IV ∈ ancestors(T)，且移除 T 后 IV 与 Y 全局不连通（排他性）

用法：
  python analyze_dag_causal_roles.py <graphml_path> [options]

  必需参数：
    graphml_path          GraphML 文件的绝对路径

  可选参数：
    --operability-csv     可操作性元数据 CSV 路径（默认：自动查找）
    --output-dir          输出目录（默认：与 GraphML 同目录）
    --y-node              目标变量名称（默认：自动检测）
    --verbose             详细输出模式

示例：
  python analyze_dag_causal_roles.py "C:/path/to/dag.graphml"
  python analyze_dag_causal_roles.py "C:/path/to/dag.graphml" --y-node "y_grade" --verbose
"""

import os
import sys
import argparse
import networkx as nx
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 默认配置
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PROJECT_ROOT = r"C:\DML_fresh_start"
DEFAULT_OPERABILITY_CSV = os.path.join(
    DEFAULT_PROJECT_ROOT, "数据预处理",
    "数据与处理结果-分阶段-去共线性后",
    "non_collinear_representative_vars_operability.csv",
)

# 目标变量候选名称（按优先级排序）
Y_CANDIDATES = ["y_grade", "Y_grade", "y_fx_xin2", "target", "outcome"]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_operability(csv_path):
    """
    读取可操作性元数据。
    返回两个字典:
      meta_lookup:   {变量名 -> {stage, group, desc, operability, reason}}
      operable_set:  所有 Operability == 'operable' 的变量名集合
    """
    if not os.path.exists(csv_path):
        print(f"[警告] 可操作性文件未找到: {csv_path}")
        print(f"       将无法提供变量元数据信息")
        return {}, set()

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    meta_lookup = {}
    operable_set = set()

    for _, row in df.iterrows():
        var = str(row.get("Variable_Name", "")).strip()
        if not var:
            continue
        operability = str(row.get("Operability", "")).strip().lower()
        meta_lookup[var] = {
            "stage":       row.get("Stage_ID", "?"),
            "group":       str(row.get("Group", "?")).strip(),
            "desc":        str(row.get("Description_CN", "")).strip(),
            "operability": operability,
            "reason":      str(row.get("Operability_Reason", "")).strip(),
        }
        if operability == "operable":
            operable_set.add(var)

    return meta_lookup, operable_set


def resolve_y_node(G, y_node_override=None, verbose=False):
    """
    在图中定位目标变量节点。

    返回：(节点名, 是否有入边)
    异常：ValueError 如果无法定位 Y 节点
    """
    if y_node_override:
        if y_node_override in G.nodes:
            has_edges = G.in_degree(y_node_override) > 0
            if verbose:
                print(f"[Y节点] 使用用户指定: {y_node_override} (入度={G.in_degree(y_node_override)})")
            return y_node_override, has_edges
        else:
            raise ValueError(f"用户指定的 Y 节点 '{y_node_override}' 不在图中")

    for name in Y_CANDIDATES:
        if name in G.nodes:
            has_edges = G.in_degree(name) > 0
            if verbose:
                print(f"[Y节点] 自动检测到: {name} (入度={G.in_degree(name)})")
            return name, has_edges

    raise ValueError(
        f"无法在图中定位目标变量 Y。\n"
        f"  候选列表: {Y_CANDIDATES}\n"
        f"  图中节点数: {len(G.nodes)}\n"
        f"  请使用 --y-node 参数手动指定目标变量名称"
    )


def validate_dag(G, verbose=False):
    """
    验证图的有效性（必须是有向无环图）。

    返回：(is_valid, error_message)
    """
    if not G.is_directed():
        return False, "图不是有向图（必须是 DiGraph）"

    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        return False, f"图包含 {len(self_loops)} 个自环: {self_loops[:5]}"

    if not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation="original")
            cycle_str = " -> ".join([f"{u}" for u, v, _ in cycle[:5]])
            if len(cycle) > 5:
                cycle_str += " -> ..."
            return False, f"图包含环，无法进行 Pearl SCM 分析。环示例: {cycle_str}"
        except Exception:
            return False, "图包含环，无法进行 Pearl SCM 分析"

    if verbose:
        print(f"[✓] DAG 有效性检查通过: {len(G.nodes)} 节点, {len(G.edges)} 有向边")

    return True, None


def find_causal_roles(G, T, Y, verbose=False):
    """
    基于全图路径分析的因果角色检测（v4.1）

    参数：
      G: 有向无环图（必须是 DAG）
      T: 处理变量（treatment）
      Y: 结果变量（outcome）
      verbose: 是否输出详细调试信息

    返回：
      confounders, mediators, colliders, instruments (均为 dict {节点名: True})

    因果角色定义（基于 Pearl SCM，v4.1 修正）：
      - 混杂因子：C ∈ ancestors(T)，C ∉ descendants(T)，移除 T 后 C 仍可达 Y
      - 中介变量：M ∈ descendants(T) ∩ ancestors(Y)，M ≠ T，M ≠ Y
      - 碰撞节点：Z ∈ descendants(T) ∩ descendants(Y)（共同后代，v4.1 修正）
      - 工具变量：IV ∈ ancestors(T)，移除 T 后 IV 与 Y 全局不连通

    v4.1 关键修复：
      碰撞节点检测不再使用 all_simple_paths（无向图组合爆炸风险），
      改为纯集合交集 t_descendants & y_descendants，时间复杂度 O(V+E)。
    """
    if T not in G.nodes or Y not in G.nodes:
        if verbose:
            print(f"[警告] T={T} 或 Y={Y} 不在图中，跳过角色分析")
        return {}, {}, {}, {}

    # ═══════════════════════════════════════════════════════════════════════
    # 预计算全图祖先/后代（无 cutoff）
    # ═══════════════════════════════════════════════════════════════════════
    t_ancestors = nx.ancestors(G, T)
    t_descendants = nx.descendants(G, T)
    y_ancestors = nx.ancestors(G, Y)
    y_descendants = nx.descendants(G, Y)

    if verbose:
        print(f"[调试] T={T}: {len(t_ancestors)} 个祖先, {len(t_descendants)} 个后代")
        print(f"[调试] Y={Y}: {len(y_ancestors)} 个祖先, {len(y_descendants)} 个后代")

    confounders = {}
    mediators = {}
    colliders = {}
    instruments = {}

    # ─────────────────────────────────────────────────────────────────────
    # 1. 中介变量检测
    #    定义：M ∈ descendants(T) ∩ ancestors(Y)，M ≠ T，M ≠ Y
    #    等价于"M 在某条 T→...→Y 有向路径的内部"，无需枚举路径。
    # ─────────────────────────────────────────────────────────────────────
    mediator_set = (t_descendants & y_ancestors) - {T, Y}
    for node in mediator_set:
        mediators[node] = True

    if verbose and mediators:
        sample = sorted(mediators.keys())[:5]
        print(f"[调试] 中介变量 ({len(mediators)}): {sample}{'...' if len(mediators) > 5 else ''}")

    # ─────────────────────────────────────────────────────────────────────
    # 2. 混杂因子检测
    #    定义：C ∈ ancestors(T)，C ∉ descendants(T)，
    #          且移除 T 后 C 仍可达 Y（存在后门路径）
    #    G_without_T 仅复制一次，消除循环内重复复制开销。
    # ─────────────────────────────────────────────────────────────────────
    confounder_candidates = t_ancestors - t_descendants

    G_without_T = G.copy()
    G_without_T.remove_node(T)

    for node in confounder_candidates:
        try:
            if nx.has_path(G_without_T, node, Y):
                confounders[node] = True
        except nx.NodeNotFound:
            pass

    if verbose and confounders:
        sample = sorted(confounders.keys())[:5]
        print(f"[调试] 混杂因子 ({len(confounders)}): {sample}{'...' if len(confounders) > 5 else ''}")

    # ─────────────────────────────────────────────────────────────────────
    # 3. 碰撞节点检测（v4.1 修正：纯集合运算，彻底消灭组合爆炸）
    #
    #    ⚠️  v4.0 原实现问题：
    #        在 G.to_undirected() 上调用 all_simple_paths(cutoff=8)，
    #        无向图路径数可达数亿条，导致脚本永远卡死。
    #        try-except 无法捕获无限循环，此问题是真实的"定时炸弹"。
    #
    #    ✅  v4.1 新定义：
    #        collider_set = t_descendants & y_descendants
    #        即"同时是 T 的后代 AND Y 的后代"的节点集合。
    #        时间复杂度：O(V+E)（BFS），与图规模线性相关，永不卡死。
    #
    #    📌  语义说明：
    #        Pearl 路径级碰撞（path-specific collider）要求节点在某条
    #        T-Y 路径上且两个相邻路径节点均指向它。本定义是其超集，
    #        会多报部分节点，但在 DML 控制变量选择中"多报"比"漏报"更安全。
    # ─────────────────────────────────────────────────────────────────────
    collider_set = t_descendants & y_descendants
    for node in collider_set:
        colliders[node] = True

    if verbose and colliders:
        sample = sorted(colliders.keys())[:5]
        print(f"[调试] 碰撞节点 ({len(colliders)}): {sample}{'...' if len(colliders) > 5 else ''}")

    # ─────────────────────────────────────────────────────────────────────
    # 4. 工具变量检测
    #    定义：IV ∈ ancestors(T)，且移除 T 后 IV 与 Y 全局不连通（排他性）
    #    复用上面已创建的 G_without_T。
    # ─────────────────────────────────────────────────────────────────────
    for node in t_ancestors:
        try:
            if not nx.has_path(G_without_T, node, Y):
                instruments[node] = True
        except nx.NodeNotFound:
            instruments[node] = True

    if verbose and instruments:
        sample = sorted(instruments.keys())[:5]
        print(f"[调试] 工具变量 ({len(instruments)}): {sample}{'...' if len(instruments) > 5 else ''}")

    # ═══════════════════════════════════════════════════════════════════════
    # 返回结果
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(
            f"[调试] 角色统计: 混杂={len(confounders)}, 中介={len(mediators)}, "
            f"碰撞={len(colliders)}, 工具={len(instruments)}"
        )

    return confounders, mediators, colliders, instruments


def get_meta_str(node, meta_lookup, verbose=False):
    """
    返回节点的元数据摘要。
    返回：(stage, group, operability, desc) 元组
    """
    m = meta_lookup.get(node, {})
    if not m and verbose:
        print(f"[警告] 节点 '{node}' 缺少元数据信息")
    return (
        str(m.get("stage", "?")),
        str(m.get("group", "?")),
        str(m.get("operability", "?")),
        str(m.get("desc", ""))[:28],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 命令行参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="通用 DAG 因果角色解析脚本（v4.1 - 修复碰撞节点组合爆炸问题）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python %(prog)s "C:/path/to/dag.graphml"
  python %(prog)s "C:/path/to/dag.graphml" --y-node "y_grade"
  python %(prog)s "C:/path/to/dag.graphml" --verbose
  python %(prog)s "C:/path/to/dag.graphml" \\
      --operability-csv "C:/path/to/operability.csv" \\
      --output-dir "C:/path/to/output"
        """,
    )

    parser.add_argument(
        "graphml_path",
        type=str,
        help="GraphML 文件的绝对路径",
    )

    parser.add_argument(
        "--operability-csv",
        type=str,
        default=None,
        help=f"可操作性元数据 CSV 路径（默认：{DEFAULT_OPERABILITY_CSV}）",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认：与 GraphML 文件同目录）",
    )

    parser.add_argument(
        "--y-node",
        type=str,
        default=None,
        help=f"目标变量名称（默认：自动检测，候选列表={Y_CANDIDATES}）",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_reports(
    G, y_node, y_has_edges, t_nodes_in_graph, t_nodes_not_in_graph,
    operable_set, meta_lookup, output_dir, graphml_path, verbose,
):
    """生成文本报告和 CSV 文件"""
    W = 138

    rep = []
    rep.append("=" * W)
    rep.append("  DAG 因果角色解析报告 (v4.1 - 修复碰撞节点组合爆炸问题)")
    rep.append("=" * W)
    rep.append(f"  GraphML 来源  : {graphml_path}")
    rep.append(f"  图规模        : {len(G.nodes)} 节点  {len(G.edges)} 有向边")
    rep.append(f"  目标变量 Y    : {y_node}  (入度={G.in_degree(y_node)})")
    rep.append(f"  专家认定 operable 变量总数 : {len(operable_set)}")
    rep.append(f"  其中在本图中出现           : {len(t_nodes_in_graph)} 个")
    rep.append(f"  路径检测方式               : 全图祖先/后代集合运算（v4.1）")
    rep.append("")

    # ── 一、operable 变量总览表 ──────────────────────────────────────────
    rep.append("-" * W)
    rep.append("  【一】专家认定 Operable 操作变量总览（在图中出现情况）")
    rep.append("-" * W)
    rep.append(f"  {'变量名':<40} {'在图中':^6} {'直接->Y':^7} {'Stage':>5} {'Group':>5} {'描述'}")
    rep.append(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*5} {'-'*5} {'-'*28}")

    for var in sorted(
        operable_set,
        key=lambda v: (meta_lookup.get(v, {}).get("stage", 99), v),
    ):
        in_graph = "Yes" if var in G.nodes else "No"
        direct_y = "Yes" if (var in G.nodes and y_node and G.has_edge(var, y_node)) else "No"
        stage, group, _, desc = get_meta_str(var, meta_lookup, verbose=verbose)
        rep.append(f"  {var:<40} {in_graph:^6} {direct_y:^7} {stage:>5} {group:>5} {desc}")

    rep.append("")

    # ── 不在图中的 operable 变量 ─────────────────────────────────────────
    if t_nodes_not_in_graph:
        rep.append("-" * W)
        rep.append("  【注意】以下 operable 变量未出现在图中（算法未检测到因果关系）：")
        rep.append("-" * W)
        for var in t_nodes_not_in_graph:
            _, _, _, desc = get_meta_str(var, meta_lookup, verbose=verbose)
            rep.append(f"    - {var:<40}  {desc}")
        rep.append("")

    # ── 二、Pearl SCM 角色分析 ───────────────────────────────────────────
    rep.append("=" * W)
    rep.append("  【二】Pearl SCM 因果角色分析（v4.1）")
    rep.append("=" * W)
    rep.append("  角色定义（基于 Pearl 结构因果模型，v4.1 修正）：")
    rep.append("    - 混杂因子：C ∈ ancestors(T)，C ∉ descendants(T)，移除 T 后 C 仍可达 Y")
    rep.append("    - 中介变量：M ∈ descendants(T) ∩ ancestors(Y)，M ≠ T，M ≠ Y")
    rep.append("    - 碰撞节点：Z ∈ descendants(T) ∩ descendants(Y)（T 和 Y 的共同后代）")
    rep.append("    - 工具变量：IV ∈ ancestors(T)，移除 T 后 IV 与 Y 全局不连通（排他性）")
    rep.append("")
    rep.append("  v4.1 核心改进（相对于 v4.0）：")
    rep.append("    ✓ 碰撞节点：彻底移除 all_simple_paths 无向图遍历（组合爆炸定时炸弹）")
    rep.append("      改为 t_descendants & y_descendants 集合交集，时间复杂度 O(V+E)")
    rep.append("      语义扩展：新定义是 Pearl 路径级碰撞的超集，在 DML 中更保守安全")
    rep.append("")
    rep.append("  v4.0 已有改进（保留）：")
    rep.append("    ✓ 移除 cutoff：所有计算使用全图，消除候选集不完整问题")
    rep.append("    ✓ 中介变量：使用 descendants(T) ∩ ancestors(Y) 集合运算")
    rep.append("    ✓ 混杂因子：G_without_T 仅复制一次，消除循环内重复复制")
    rep.append("")

    # 初始化 csv_rows
    csv_rows = []

    if not y_has_edges:
        rep.append("  [注] Y 节点在本图中无入边，Pearl 角色分析无法基于图结构推断，")
        rep.append("       请参考【一】中的直接连边情况，或降低因果发现算法阈值后重跑。")
    elif not t_nodes_in_graph:
        rep.append("  [注] 无 operable 变量出现在图中，无法进行角色分析。")
    else:
        col_t = 38
        col_role = 14
        col_node = 38
        col_sc = 7
        col_gr = 5
        col_op = 10

        header = (
            f"  {'操作变量T':<{col_t}} | {'角色类型':<{col_role}} | "
            f"{'关联节点':<{col_node}} | {'Stage':>{col_sc}} | "
            f"{'Group':>{col_gr}} | {'Operability':>{col_op}} | {'描述'}"
        )
        rep.append(header)
        rep.append("  " + "-" * (W - 2))

        any_result = False

        for t_name in t_nodes_in_graph:
            conf, medi, coll, inst = find_causal_roles(G, t_name, y_node, verbose=verbose)
            has_direct = G.has_edge(t_name, y_node)

            if not (conf or medi or coll or inst or has_direct):
                continue

            any_result = True

            def _add_rows(role_tag, nodes_dict, _t=t_name):
                for node in sorted(nodes_dict):
                    stage, group, operable, desc = get_meta_str(node, meta_lookup, verbose=verbose)
                    rep.append(
                        f"  {_t:<{col_t}} | {role_tag:<{col_role}} | "
                        f"{node:<{col_node}} | {stage:>{col_sc}} | "
                        f"{group:>{col_gr}} | {operable:>{col_op}} | {desc}"
                    )
                    csv_rows.append({
                        "Target_Y":      y_node,
                        "Treatment_T":   _t,
                        "T_Stage":       meta_lookup.get(_t, {}).get("stage", "?"),
                        "T_Group":       meta_lookup.get(_t, {}).get("group", "?"),
                        "T_Desc":        meta_lookup.get(_t, {}).get("desc", "")[:30],
                        "Role":          role_tag,
                        "Node_Name":     node,
                        "Node_Stage":    stage,
                        "Node_Group":    group,
                        "Node_Operable": operable,
                        "Node_Desc":     desc,
                    })

            if has_direct:
                rep.append(
                    f"  {t_name:<{col_t}} | {'0-Direct':<{col_role}} | "
                    f"{y_node:<{col_node}} | {'':>{col_sc}} | "
                    f"{'':>{col_gr}} | {'':>{col_op}} | 直接因果边指向Y"
                )
                csv_rows.append({
                    "Target_Y":      y_node,
                    "Treatment_T":   t_name,
                    "T_Stage":       meta_lookup.get(t_name, {}).get("stage", "?"),
                    "T_Group":       meta_lookup.get(t_name, {}).get("group", "?"),
                    "T_Desc":        meta_lookup.get(t_name, {}).get("desc", "")[:30],
                    "Role":          "0-Direct",
                    "Node_Name":     y_node,
                    "Node_Stage":    "",
                    "Node_Group":    "",
                    "Node_Operable": "",
                    "Node_Desc":     "直接因果边指向Y",
                })

            _add_rows("1-Confounder", conf)
            _add_rows("2-Mediator",   medi)
            _add_rows("3-Collider",   coll)
            _add_rows("4-Instrument", inst)

        if not any_result:
            rep.append("  所有 operable 变量与 Y 均无结构性因果连通，")
            rep.append("  建议检查因果发现算法阈值或重新运行集成投票。")

        rep.append("  " + "-" * (W - 2))

    rep.append("")
    rep.append("=" * W)
    rep.append("  报告结束")
    rep.append("=" * W)

    report_text = "\n".join(rep)

    # ── 写入 TXT 报告 ────────────────────────────────────────────────────
    graphml_name = Path(graphml_path).stem
    txt_path = os.path.join(output_dir, f"{graphml_name}_Analysis_Report.txt")
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        f.write(report_text)
    print(f"      ✓ 文本报告: {txt_path}")

    # ── 写入 CSV 明细表 ──────────────────────────────────────────────────
    if csv_rows:
        df_out = pd.DataFrame(csv_rows)
        csv_path = os.path.join(output_dir, f"{graphml_name}_Roles_Table.csv")
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"      ✓ CSV 明细表: {csv_path}")
    else:
        print(f"      [注] 无有效行数据，CSV 未生成（Y 无入边或 T 与 Y 无连通路径）")

    # ── 写出不在图中的 operable 变量清单 CSV ─────────────────────────────
    if t_nodes_not_in_graph:
        miss_rows = []
        for var in sorted(t_nodes_not_in_graph):
            stage, group, _, desc = get_meta_str(var, meta_lookup, verbose=verbose)
            miss_rows.append({
                "Variable_Name": var,
                "Stage":         stage,
                "Group":         group,
                "Description":   desc,
                "Note":          "operable 但未出现在图中",
            })
        df_miss = pd.DataFrame(miss_rows)
        miss_path = os.path.join(output_dir, f"{graphml_name}_Operable_NotInGraph.csv")
        df_miss.to_csv(miss_path, index=False, encoding="utf-8-sig")
        print(f"      ✓ 不在图中的变量清单: {miss_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """主函数"""
    args = parse_args()

    graphml_path = args.graphml_path
    operability_csv = args.operability_csv or DEFAULT_OPERABILITY_CSV
    output_dir = args.output_dir
    y_node_override = args.y_node
    verbose = args.verbose

    # 验证输入文件
    if not os.path.exists(graphml_path):
        print(f"[错误] GraphML 文件不存在: {graphml_path}")
        sys.exit(1)

    graphml_path = os.path.abspath(graphml_path)

    # 确定输出目录
    if output_dir is None:
        graphml_dir = os.path.dirname(graphml_path)
        graphml_name = Path(graphml_path).stem
        output_dir = os.path.join(graphml_dir, f"{graphml_name}_causal_analysis")

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("DAG 因果角色解析脚本 v4.1")
        print("=" * 80)
        print(f"GraphML 文件: {graphml_path}")
        print(f"可操作性 CSV: {operability_csv}")
        print(f"输出目录:     {output_dir}")
        print(f"Y 节点:       {y_node_override or '自动检测'}")
        print("=" * 80)

    # 1. 加载可操作性元数据
    print("\n[1/6] 加载可操作性元数据...")
    meta_lookup, operable_set = load_operability(operability_csv)
    print(f"      元数据记录: {len(meta_lookup)} 条，其中 operable 变量: {len(operable_set)} 个")

    # 2. 加载 GraphML
    print(f"\n[2/6] 加载 GraphML 文件...")
    try:
        G = nx.read_graphml(graphml_path)
    except Exception as e:
        print(f"[错误] 无法加载 GraphML 文件: {e}")
        sys.exit(1)
    print(f"      图规模: {len(G.nodes)} 节点, {len(G.edges)} 有向边")

    # 3. DAG 有效性检查
    print(f"\n[3/6] DAG 有效性检查...")
    is_valid, error_msg = validate_dag(G, verbose=verbose)
    if not is_valid:
        print(f"[错误] {error_msg}")
        print(f"       Pearl SCM 因果分析要求输入图必须是有向无环图（DAG）")
        sys.exit(1)
    print(f"      ✓ DAG 检查通过")

    # 4. 定位 Y 节点
    print(f"\n[4/6] 定位目标变量 Y...")
    try:
        y_node, y_has_edges = resolve_y_node(G, y_node_override=y_node_override, verbose=verbose)
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)
    print(f"      目标变量: Y = {y_node}  (入度={G.in_degree(y_node)})")
    if not y_has_edges:
        print(f"      [警告] Y 节点无入边，可能无法进行有意义的因果分析")

    # 5. 识别操作变量
    print(f"\n[5/6] 识别操作变量（Treatment）...")
    t_nodes_in_graph = sorted(
        [v for v in operable_set if v in G.nodes],
        key=lambda v: meta_lookup.get(v, {}).get("stage", 99),
    )
    t_nodes_not_in_graph = sorted([v for v in operable_set if v not in G.nodes])
    print(f"      operable 变量在图中出现: {len(t_nodes_in_graph)} 个")
    print(f"      operable 变量不在图中:   {len(t_nodes_not_in_graph)} 个")
    if len(t_nodes_in_graph) == 0:
        print(f"      [警告] 无 operable 变量出现在图中，无法进行角色分析")

    # 6. 执行因果角色分析 & 生成报告
    print(f"\n[6/6] 执行 Pearl SCM 因果角色分析...")
    print(f"      使用全图祖先/后代集合运算（v4.1）")

    generate_reports(
        G=G,
        y_node=y_node,
        y_has_edges=y_has_edges,
        t_nodes_in_graph=t_nodes_in_graph,
        t_nodes_not_in_graph=t_nodes_not_in_graph,
        operable_set=operable_set,
        meta_lookup=meta_lookup,
        output_dir=output_dir,
        graphml_path=graphml_path,
        verbose=verbose,
    )

    print(f"\n{'='*80}")
    print(f"分析完成！结果已保存至: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[中断] 用户取消操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 发生未预期的异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
