"""
analyze_tcdf_xin2_operable.py
==============================
针对 tcdf_space_time_dag_xin2.graphml 的专项解析脚本。

核心特点（区别于通用脚本）：
  1. 仅分析单一 GraphML：tcdf_space_time_dag_xin2.graphml
  2. 操作变量（T）判定方式为工艺专家知识：
       读取 non_collinear_representative_vars_operability.csv，
       仅当 Operability == 'operable' 的变量，才视为可操作干预变量 (T)。
  3. 所有其他节点（observable）作为潜在中间/混杂变量参与角色分析。
  4. 输出为表格形式，覆盖全部 operable T 与 Y 之间的四类角色。
  5. 结果存入独立子文件夹，不覆盖通用脚本的输出。

输出目录：C:\\DML_fresh_start\\DAG图分析\\DAG解析结果\\工艺判断操作变量解析结果\\
"""

import os
import networkx as nx
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = r"C:\DML_fresh_start"
GRAPHML_PATH  = os.path.join(PROJECT_ROOT, "多种方法因果发现", "因果发现结果",
                             "tcdf_space_time_dag_xin2.graphml")
OPERABILITY_CSV = os.path.join(PROJECT_ROOT, "数据预处理",
                               "数据与处理结果-分阶段-去共线性后",
                               "non_collinear_representative_vars_operability.csv")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "DAG图分析", "DAG解析结果",
                             "工艺判断操作变量解析结果")

# xin2 线目标变量（在 TCDF 图中以 y_grade 命名）
Y_CANDIDATES  = ["y_grade", "Y_grade", "y_fx_xin2"]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_operability():
    """
    读取可操作性元数据。
    返回两个字典:
      meta_lookup:   {变量名 -> {stage, group, desc, operability, reason}}
      operable_set:  所有 Operability == 'operable' 的变量名集合
    """
    if not os.path.exists(OPERABILITY_CSV):
        print(f"[错误] 可操作性文件未找到: {OPERABILITY_CSV}")
        return {}, set()

    df = pd.read_csv(OPERABILITY_CSV, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    meta_lookup  = {}
    operable_set = set()

    for _, row in df.iterrows():
        var = str(row.get("Variable_Name", "")).strip()
        if not var:
            continue
        operability = str(row.get("Operability", "")).strip().lower()
        meta_lookup[var] = {
            "stage":       row.get("Stage_ID",       "?"),
            "group":       str(row.get("Group",       "?")).strip(),
            "desc":        str(row.get("Description_CN", "")).strip(),
            "operability": operability,
            "reason":      str(row.get("Operability_Reason", "")).strip(),
        }
        if operability == "operable":
            operable_set.add(var)

    return meta_lookup, operable_set


def resolve_y_node(G):
    """在图中定位目标变量节点，返回 (节点名, 是否有入边)。"""
    for name in Y_CANDIDATES:
        if name in G.nodes:
            has_edges = G.in_degree(name) > 0
            return name, has_edges
    # 兜底：最大入度节点
    best = max(G.nodes, key=lambda n: G.in_degree(n), default=None)
    return best, (G.in_degree(best) > 0 if best else False)


def find_causal_roles(G, T, Y):
    """
    基于 Pearl SCM，把图中所有节点按照其相对于 T -> Y 路径的角色分类。
    返回: confounders, mediators, colliders, instruments (均为 dict {节点名: True})
    """
    if T not in G.nodes or Y not in G.nodes:
        return {}, {}, {}, {}

    t_parents  = set(G.predecessors(T))
    t_children = set(G.successors(T))
    y_parents  = set(G.predecessors(Y))
    y_children = set(G.successors(Y))

    confounders = {}
    mediators   = {}
    colliders   = {}
    instruments = {}

    candidates = set(G.nodes) - {T, Y}
    for node in candidates:
        is_pt = node in t_parents
        is_ct = node in t_children
        is_py = node in y_parents
        is_cy = node in y_children

        if is_pt and is_py:               confounders[node] = True   # C <- T, C -> Y
        if is_ct and is_py:               mediators[node]   = True   # T -> M -> Y
        if is_ct and is_cy:               colliders[node]   = True   # T -> Z <- Y
        if is_pt and not is_py and not is_cy:
                                          instruments[node] = True   # IV -> T only

    return confounders, mediators, colliders, instruments


def get_meta_str(node, meta_lookup):
    """返回节点的元数据摘要字符串（stage/group/operable/desc）。"""
    m = meta_lookup.get(node, {})
    return (str(m.get("stage", "?")),
            str(m.get("group", "?")),
            str(m.get("operability", "?")),
            str(m.get("desc", ""))[:28])


# ─────────────────────────────────────────────────────────────────────────────
# 主分析流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载可操作性元数据
    print("加载可操作性元数据...")
    meta_lookup, operable_set = load_operability()
    print(f"  元数据记录: {len(meta_lookup)} 条，其中 operable 变量: {len(operable_set)} 个")

    # 2. 加载 GraphML
    if not os.path.exists(GRAPHML_PATH):
        print(f"[错误] GraphML 文件不存在: {GRAPHML_PATH}")
        return
    G = nx.read_graphml(GRAPHML_PATH)
    print(f"图加载完成: {len(G.nodes)} 节点, {len(G.edges)} 有向边")

    # 3. 定位 Y 节点
    y_node, y_has_edges = resolve_y_node(G)
    print(f"目标变量 Y = {y_node}  (有入边: {y_has_edges})")

    # 4. 确定本图中实际存在的 operable T 变量
    t_nodes_in_graph = sorted(
        [v for v in operable_set if v in G.nodes],
        key=lambda v: meta_lookup.get(v, {}).get("stage", 99)
    )
    t_nodes_not_in_graph = sorted([v for v in operable_set if v not in G.nodes])
    print(f"  operable 变量在图中出现: {len(t_nodes_in_graph)} 个")
    print(f"  operable 变量不在图中:   {len(t_nodes_not_in_graph)} 个")

    # ──────────────────── 5. 构建文本报告 ────────────────────────────────
    W = 138  # 表格宽度

    rep = []
    rep.append("=" * W)
    rep.append(f"  [XIN2 / TCDF] 工艺专家操作变量解析报告")
    rep.append("=" * W)
    rep.append(f"  GraphML 来源  : {GRAPHML_PATH}")
    rep.append(f"  图规模        : {len(G.nodes)} 节点  {len(G.edges)} 有向边")
    rep.append(f"  目标变量 Y    : {y_node}  (有入边={y_has_edges})")
    rep.append(f"  专家认定 operable 变量总数 : {len(operable_set)}")
    rep.append(f"  其中在本图中出现           : {len(t_nodes_in_graph)} 个")
    rep.append("")

    # ── 5a. operable 变量总览表 ──────────────────────────────────────────
    rep.append("-" * W)
    rep.append("  【一】专家认定 Operable 操作变量总览（在图中出现情况）")
    rep.append("-" * W)
    rep.append(f"  {'变量名':<40} {'在图中':^6} {'直接->Y':^7} {'Stage':>5} {'Group':>5} {'描述'}")
    rep.append(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*5} {'-'*5} {'-'*28}")

    for var in sorted(operable_set, key=lambda v: (meta_lookup.get(v, {}).get("stage", 99), v)):
        in_graph  = "Yes" if var in G.nodes else "No"
        direct_y  = "Yes" if (var in G.nodes and y_node and G.has_edge(var, y_node)) else "No"
        stage, group, _, desc = get_meta_str(var, meta_lookup)
        rep.append(f"  {var:<40} {in_graph:^6} {direct_y:^7} {stage:>5} {group:>5} {desc}")

    rep.append("")

    # ── 5b. 不在图中的 operable 变量（注意事项）─────────────────────────
    if t_nodes_not_in_graph:
        rep.append("-" * W)
        rep.append("  【注意】以下 operable 变量未出现在 TCDF xin2 图中（算法未检测到因果关系）：")
        rep.append("-" * W)
        for var in t_nodes_not_in_graph:
            _, _, _, desc = get_meta_str(var, meta_lookup)
            rep.append(f"    - {var:<40}  {desc}")
        rep.append("")

    # ── 5c. Pearl SCM 角色分析全局表格 ───────────────────────────────────
    rep.append("=" * W)
    rep.append("  【二】Pearl SCM 因果角色分析（仅以 operable 变量为 T）")
    rep.append("=" * W)

    if not y_has_edges:
        rep.append("  [注] Y 节点在本图中无入边，Pearl 角色分析无法基于图结构推断，")
        rep.append("       请参考【一】中的直接连边情况，或降低 TCDF 阈值后重跑。")
    elif not t_nodes_in_graph:
        rep.append("  [注] 无 operable 变量出现在图中，无法进行角色分析。")
    else:
        col_t    = 38
        col_role = 14
        col_node = 38
        col_sc   = 7
        col_st   = 5
        col_gr   = 5
        col_op   = 10
        col_desc = 26

        header = (f"  {'操作变量T':<{col_t}} | {'角色类型':<{col_role}} | "
                  f"{'关联节点':<{col_node}} | {'Stage':>{col_sc}} | "
                  f"{'Group':>{col_gr}} | {'Operability':>{col_op}} | {'描述'}")
        rep.append(header)
        rep.append("  " + "-" * (W - 2))

        any_result = False
        for t_name in t_nodes_in_graph:
            conf, medi, coll, inst = find_causal_roles(G, t_name, y_node)
            has_direct = G.has_edge(t_name, y_node)

            if not (conf or medi or coll or inst or has_direct):
                continue  # 与 Y 完全无结构关联，跳过

            any_result = True

            def _add_rows(role_tag, nodes_dict):
                for node in sorted(nodes_dict):
                    stage, group, operable, desc = get_meta_str(node, meta_lookup)
                    rep.append(
                        f"  {t_name:<{col_t}} | {role_tag:<{col_role}} | "
                        f"{node:<{col_node}} | {stage:>{col_sc}} | "
                        f"{group:>{col_gr}} | {operable:>{col_op}} | {desc:<{col_desc}}"
                    )

            if has_direct:
                rep.append(
                    f"  {t_name:<{col_t}} | {'0-Direct':<{col_role}} | "
                    f"{y_node:<{col_node}} | {'':>{col_sc}} | "
                    f"{'':>{col_gr}} | {'':>{col_op}} | 直接因果边指向Y"
                )
            _add_rows("1-Confounder", conf)
            _add_rows("2-Mediator",   medi)
            _add_rows("3-Collider",   coll)
            _add_rows("4-Instrument", inst)

        if not any_result:
            rep.append("  所有 operable 变量与 Y 均无结构性因果连通，")
            rep.append("  建议检查 TCDF 阈值或重新运行集成投票。")

        rep.append("  " + "-" * (W - 2))

    rep.append("")
    rep.append("=" * W)
    rep.append("  报告结束")
    rep.append("=" * W)

    report_text = "\n".join(rep)

    # ── 6. 写入 TXT 报告 ─────────────────────────────────────────────────
    txt_path = os.path.join(OUTPUT_DIR, "XIN2_TCDF_Operable_Analysis_Report.txt")
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        f.write(report_text)
    print(f"\n[OK] 文本报告已保存至: {txt_path}")

    # ── 7. 写入 CSV 明细表 ───────────────────────────────────────────────
    csv_rows = []
    if y_has_edges and t_nodes_in_graph:
        for t_name in t_nodes_in_graph:
            conf, medi, coll, inst = find_causal_roles(G, t_name, y_node)
            has_direct = G.has_edge(t_name, y_node)
            if not (conf or medi or coll or inst or has_direct):
                continue

            def _collect(role_tag, nodes_dict):
                for node in sorted(nodes_dict):
                    stage, group, operable, desc = get_meta_str(node, meta_lookup)
                    csv_rows.append({
                        "Target_Y":      y_node,
                        "Treatment_T":   t_name,
                        "T_Stage":       meta_lookup.get(t_name, {}).get("stage", "?"),
                        "T_Group":       meta_lookup.get(t_name, {}).get("group", "?"),
                        "T_Desc":        meta_lookup.get(t_name, {}).get("desc", "")[:30],
                        "Role":          role_tag,
                        "Node_Name":     node,
                        "Node_Stage":    stage,
                        "Node_Group":    group,
                        "Node_Operable": operable,
                        "Node_Desc":     desc,
                    })

            if has_direct:
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
            _collect("1-Confounder", conf)
            _collect("2-Mediator",   medi)
            _collect("3-Collider",   coll)
            _collect("4-Instrument", inst)

    if csv_rows:
        df_out = pd.DataFrame(csv_rows)
        csv_path = os.path.join(OUTPUT_DIR, "XIN2_TCDF_Operable_Roles_Table.csv")
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] CSV 明细表已保存至: {csv_path}")
    else:
        print("[注] 无有效行数据，CSV 未生成（Y 无入边或 T 与 Y 无连通路径）。")

    # ── 8. 写出不在图中的 operable 变量清单 CSV ──────────────────────────
    if t_nodes_not_in_graph:
        miss_rows = []
        for var in sorted(t_nodes_not_in_graph):
            stage, group, _, desc = get_meta_str(var, meta_lookup)
            miss_rows.append({
                "Variable_Name": var,
                "Stage":         stage,
                "Group":         group,
                "Description":   desc,
                "Note":          "operable 但未出现在 TCDF xin2 图中",
            })
        df_miss = pd.DataFrame(miss_rows)
        miss_path = os.path.join(OUTPUT_DIR, "XIN2_TCDF_Operable_NotInGraph.csv")
        df_miss.to_csv(miss_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 不在图中的 operable 变量清单: {miss_path}")

    print("\n所有分析完成。")


if __name__ == "__main__":
    main()
