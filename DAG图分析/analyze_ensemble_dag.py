"""
analyze_ensemble_dag.py
=======================
针对 xin1 / xin2 双产线集成 DAG 的结构因果解析脚本。

核心修正（相对于旧版 run_dag_analysis_refined.py）：
  1. 严格遵守物理隔离原则：xin1 使用 Group A+C，xin2 使用 Group B+C。
  2. 正确读取各产线独立的集成 GraphML 文件（而非单一合并图）。
  3. 从特征元数据表加载变量分组信息，提供有物理意义的角色注解。
  4. 综合读取 TCDF / NTS-NOTEARS / CUTS+ / Ensemble 各方法的因果强度 CSV，
     输出带定量分数的混杂/中介/工具变量报告。
  5. 所有路径均基于项目根目录的绝对路径，无需手动修改即可运行。

输出目录：C:\\DML_fresh_start\\DAG图分析\\DAG解析结果\\
"""

import os
import sys
import networkx as nx
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置（全部使用绝对路径，不依赖 cwd）
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"C:\DML_fresh_start"
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "多种方法因果发现", "因果发现结果")
META_CSV       = os.path.join(PROJECT_ROOT, "数据预处理",
                              "数据与处理结果-分阶段-去共线性后",
                              "non_collinear_representative_vars_annotated.csv")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "DAG图分析", "DAG解析结果")

# ─────────────────────────────────────────────────────────────────────────────
# 产线配置
# ─────────────────────────────────────────────────────────────────────────────
LINE_CONFIG = {
    "xin1": {
        "allowed_groups": {"A", "C"},    # Group A (新1专有) + Group C (公用)
        "y_target":       "y_fx_xin1",   # 理论目标变量名
        "y_fallback":     ["y_grade", "Y_grade"],  # GraphML 中实际存在的节点名
        "graphml":        "ensemble_integrated_dag_xin1.graphml",
        "effects_files": {               # 各方法因果强度文件
            "ensemble":       "ensemble_effects_on_y_xin1.csv",
            "tcdf":           "tcdf_space_time_effects_on_y_xin1.csv",
            "nts_notears":    "nts_notears_effects_on_y_xin1.csv",
            "cuts_plus":      "cuts_plus_effects_on_y_xin1.csv",
        },
    },
    "xin2": {
        "allowed_groups": {"B", "C"},    # Group B (新2专有) + Group C (公用)
        "y_target":       "y_fx_xin2",
        "y_fallback":     ["y_grade", "Y_grade"],
        "graphml":        "ensemble_integrated_dag_xin2.graphml",
        "effects_files": {
            "ensemble":       "ensemble_effects_on_y_xin2.csv",
            "tcdf":           "tcdf_space_time_effects_on_y_xin2.csv",
            "nts_notears":    "nts_notears_effects_on_y_xin2.csv",
            "cuts_plus":      "cuts_plus_effects_on_y_xin2.csv",
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata():
    """加载特征元数据表，返回 variable → {Stage_ID, Group, Description_CN} 字典。"""
    if not os.path.exists(META_CSV):
        print(f"[警告] 元数据文件未找到: {META_CSV}")
        return {}
    meta = pd.read_csv(META_CSV, encoding="utf-8-sig")
    # 兼容可能的编码差异
    meta.columns = [c.strip() for c in meta.columns]
    lookup = {}
    for _, row in meta.iterrows():
        var = str(row.get("Variable_Name", "")).strip()
        if var:
            lookup[var] = {
                "stage":    row.get("Stage_ID", "?"),
                "group":    str(row.get("Group", "?")).strip(),
                "desc":     str(row.get("Description_CN", "")).strip(),
                "keep":     str(row.get("Keep_Remove", "keep")).strip().lower(),
            }
    return lookup


def load_effects(line_id, cfg, meta_lookup):
    """
    汇总该产线所有方法的因果强度 CSV，返回
      dict: { var_name → {"score": float, "method": str, "stage": ?, "group": str} }
    优先级：ensemble > tcdf > nts_notears > cuts_plus
    """
    score_map = {}
    priority_order = ["cuts_plus", "nts_notears", "tcdf", "ensemble"]  # 低优先先写，后写覆盖

    for method in priority_order:
        fname = cfg["effects_files"].get(method)
        if not fname:
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [警告] 读取 {fname} 失败: {e}")
            continue

        # 兼容不同列名约定
        var_col   = next((c for c in df.columns if "source" in c.lower() or "var" in c.lower()), None)
        score_col = next((c for c in df.columns if "score" in c.lower() or "attention" in c.lower() or "causal" in c.lower()), None)

        if var_col is None or score_col is None:
            print(f"  [警告] {fname} 列名无法识别: {df.columns.tolist()}")
            continue

        for _, row in df.iterrows():
            var   = str(row[var_col]).strip()
            score = float(row[score_col]) if pd.notna(row[score_col]) else 0.0
            # 从元数据补充 stage / group（优先文件自带）
            stage = row.get("Stage", row.get("stage", meta_lookup.get(var, {}).get("stage", "?")))
            group = row.get("Group", row.get("group", meta_lookup.get(var, {}).get("group", "?")))
            score_map[var] = {
                "score":  score,
                "method": method,
                "stage":  stage,
                "group":  str(group).strip(),
            }

    return score_map


def resolve_y_node(G, cfg):
    """
    在 GraphML 中定位 Y 目标节点：
      先检查 y_target (如 y_fx_xin1)，再依次尝试 y_fallback。
    若所有候选均无入边，则返回得分 / 在-degree 最高的节点作为近似目标。
    """
    # 1. 直接匹配
    for y_name in [cfg["y_target"]] + cfg["y_fallback"]:
        if y_name in G.nodes:
            if G.in_degree(y_name) > 0:
                return y_name, True   # (name, has_edges)
            else:
                return y_name, False  # 节点存在但无入边

    # 2. 最高入度的候选节点（兜底）
    best = max(G.nodes, key=lambda n: G.in_degree(n), default=None)
    return best, (G.in_degree(best) > 0 if best else False)


def find_causal_roles(G, T, Y, allowed_vars):
    """
    基于 Pearl SCM 定义，在有向图 G 中为 T→Y 路径分类所有中间节点。
    仅考虑 allowed_vars 集合内的节点（物理隔离约束）。
    """
    if T not in G.nodes or Y not in G.nodes:
        return {}, {}, {}, {}

    t_parents  = set(G.predecessors(T)) & allowed_vars
    t_children = set(G.successors(T))   & allowed_vars
    y_parents  = set(G.predecessors(Y)) & allowed_vars
    y_children = set(G.successors(Y))   & allowed_vars

    confounders, mediators, colliders, instruments = {}, {}, {}, {}

    for node in (allowed_vars - {T, Y}):
        is_pt = node in t_parents
        is_ct = node in t_children
        is_py = node in y_parents
        is_cy = node in y_children

        # 混杂: C → T  且  C → Y
        if is_pt and is_py:
            confounders[node] = True
        # 中介: T → M  且  M → Y
        if is_ct and is_py:
            mediators[node] = True
        # 对撞: T → Z  且  Y → Z
        if is_ct and is_cy:
            colliders[node] = True
        # 工具: IV → T  且  IV ≠ Y 的祖先（无直接路径到 Y）
        if is_pt and not is_py and not is_cy:
            instruments[node] = True

    return confounders, mediators, colliders, instruments


def format_var_info(var, score_map, meta_lookup, indent="      "):
    """返回变量的格式化单行注解字符串。"""
    score_info = score_map.get(var, {})
    meta_info  = meta_lookup.get(var, {})
    score_str  = f"{score_info.get('score', 0):.4f}" if score_info else " N/A "
    method_str = score_info.get("method", "N/A")
    stage_str  = str(score_info.get("stage", meta_info.get("stage", "?")))
    group_str  = score_info.get("group", meta_info.get("group", "?"))
    desc_str   = meta_info.get("desc", "")[:30]
    return (f"{indent}● {var:<40s} "
            f"[Score={score_str}, Method={method_str}, "
            f"Stage={stage_str}, Group={group_str}] {desc_str}")


# ─────────────────────────────────────────────────────────────────────────────
# 核心分析流程
# ─────────────────────────────────────────────────────────────────────────────

def analyze_line(line_id, cfg, meta_lookup):
    """对单条产线执行完整的 DAG 结构分析，生成文本报告。"""
    print(f"\n{'='*60}")
    print(f"  开始处理产线: {line_id.upper()}")
    print(f"{'='*60}")

    # ── 1. 载入集成 GraphML ──────────────────────────────────────────────
    gml_path = os.path.join(RESULTS_DIR, cfg["graphml"])
    if not os.path.exists(gml_path):
        print(f"[错误] GraphML 文件不存在: {gml_path}")
        return None

    G = nx.read_graphml(gml_path)
    print(f"  图加载完成：{len(G.nodes)} 节点，{len(G.edges)} 有向边")

    # ── 2. 物理隔离：筛选该产线允许的变量 ──────────────────────────────
    allowed_groups = cfg["allowed_groups"]
    allowed_vars   = set()
    # Y 目标节点名集合（不受分组约束）
    y_candidates = set([cfg["y_target"]] + cfg["y_fallback"])
    for node in G.nodes:
        # 检查元数据中的 Group 归属
        info = meta_lookup.get(node, {})
        g    = info.get("group", "?")
        if g in allowed_groups:
            allowed_vars.add(node)
        elif node in y_candidates:
            # Y 目标节点不受分组限制，始终保留
            allowed_vars.add(node)
        # 其他无法查到 Group 的非 Y 节点不加入 allowed_vars（物理隔离严格执行）

    print(f"  物理隔离后允许变量数: {len(allowed_vars)} "
          f"(Groups={allowed_groups}，共 {len(G.nodes)} 节点)")

    # ── 3. 定位 Y 目标节点 ───────────────────────────────────────────────
    y_node, y_has_edges = resolve_y_node(G, cfg)
    print(f"  目标变量 Y = {y_node}  (有入边: {y_has_edges})")
    if not y_has_edges:
        print(f"  [注意] Y 节点无入边，将基于因果强度 CSV 而非图结构分析直接影响变量。")

    # ── 4. 载入各方法因果强度 ────────────────────────────────────────────
    score_map = load_effects(line_id, cfg, meta_lookup)
    print(f"  有定量分数的变量数: {len(score_map)}")

    # ── 5. 确定 T 候选集（在允许组内的非 Y 节点）──────────────
    # 我们将所有属于本产线的变量都视为潜在的“操作变量 (T)”来进行全景表格梳理
    t_candidates = sorted(
        [v for v in allowed_vars if v not in [cfg["y_target"]] + cfg["y_fallback"]],
        key=lambda v: score_map.get(v, {}).get("score", 0),
        reverse=True,
    )

    print(f"  参与角色分析的处理变量 (T) 数: {len(t_candidates)}")

    # ── 6. 生成文本报告 ──────────────────────────────────────────────────
    lines = []
    lines.append("=" * 72)
    lines.append(f"  [{line_id.upper()}] 集成 DAG 结构因果解析报告")
    lines.append("=" * 72)
    lines.append(f"  GraphML 文件  : {gml_path}")
    lines.append(f"  图规模        : {len(G.nodes)} 节点  {len(G.edges)} 有向边")
    lines.append(f"  物理隔离分组  : Groups = {sorted(allowed_groups)}  -> {len(allowed_vars)} 个有效变量")
    lines.append(f"  目标变量 Y    : {y_node}  (有入边={y_has_edges})")
    lines.append(f"  有定量分数变量: {len(score_map)} 个")
    lines.append("")

    # ── 6a. 因果强度排名总表 ─────────────────────────────────────────────
    lines.append("─" * 72)
    lines.append("  【因果强度 Top 排名（直接/综合影响 Y）】")
    lines.append("─" * 72)
    ranked_vars = sorted(
        [v for v in score_map if v in allowed_vars],
        key=lambda v: score_map[v].get("score", 0),
        reverse=True,
    )
    if ranked_vars:
        lines.append(f"  {'排名':<4} {'变量名':<42} {'Score':>8}  {'Method':<12} {'Stage':<6} {'Group':<6} 描述")
        lines.append(f"  {'-'*4} {'-'*42} {'-'*8}  {'-'*12} {'-'*6} {'-'*6} {'-'*20}")
        for rank, var in enumerate(ranked_vars[:50], 1):
            info  = score_map[var]
            mdesc = meta_lookup.get(var, {}).get("desc", "")[:24]
            lines.append(
                f"  {rank:<4d} {var:<42s} {info['score']:>8.4f}  "
                f"{info['method']:<12s} {str(info['stage']):<6} {info['group']:<6} {mdesc}"
            )
    else:
        lines.append("  （无有效因果强度数据，请先运行各算法脚本）")
    lines.append("")

    # ── 6b. Pearl 结构角色分析（全局角色表格）────────────────────────
    if y_has_edges and t_candidates:
        lines.append("─" * 130)
        lines.append("  【Pearl SCM 节点角色分析（全局操作变量因果明细表）】")
        lines.append("─" * 130)
        lines.append(f"  {'操作变量 (T)':<35} | {'角色类型':<14} | {'关联变量 (Node)':<35} | {'Score':>6} | {'Stage':>5} | {'Group':>5} | {'描述'}")
        lines.append("─" * 130)

        for t_name in t_candidates:
            confounders, mediators, colliders, instruments = find_causal_roles(
                G, t_name, y_node, allowed_vars
            )
            if not (confounders or mediators or colliders or G.has_edge(t_name, y_node)):
                continue  # 如果该变量与 Y 完全没有因果通路，则跳过

            def _append_role_nodes(role_name, nodes_dict):
                for node in sorted(nodes_dict):
                    score_info = score_map.get(node, {})
                    meta_info = meta_lookup.get(node, {})
                    score_str = f"{score_info.get('score', 0):.4f}" if score_info else "   N/A"
                    stage_str = str(score_info.get("stage", meta_info.get("stage", "?")))
                    group_str = str(score_info.get("group", meta_info.get("group", "?")))
                    desc_str = meta_info.get("desc", "")[:24]
                    lines.append(f"  {t_name:<35} | {role_name:<14} | {node:<35} | {score_str:>6} | {stage_str:>5} | {group_str:>5} | {desc_str}")

            if G.has_edge(t_name, y_node):
                lines.append(f"  {t_name:<35} | {'0-Direct':<14} | {y_node:<35} | {'-':>6} | {'-':>5} | {'-':>5} | 直接影响Y")
            
            _append_role_nodes("1-Confounder", confounders)
            _append_role_nodes("2-Mediator", mediators)
            _append_role_nodes("3-Collider", colliders)
            _append_role_nodes("4-Instrument", instruments)

        lines.append("─" * 130)
        lines.append("")

        # 附带写出一份纯 CSV 文件，以备做研究和报表数据挖掘
        csv_path = os.path.join(OUTPUT_DIR, f"{line_id.upper()}_DAG_Roles_Table.csv")
        try:
            with open(csv_path, 'w', encoding='utf-8-sig') as f:
                f.write("Target_Y,Treatment_T,Role,Node_Name,Causal_Score,Stage,Group,Description\n")
                for t_name in t_candidates:
                    c, m, col, iv = find_causal_roles(G, t_name, y_node, allowed_vars)
                    if not (c or m or col or G.has_edge(t_name, y_node)): continue
                    if G.has_edge(t_name, y_node):
                        f.write(f"{y_node},{t_name},0-Direct,{y_node},,-,-,直接影响Y\n")
                    def _write_csv_role(role_name, nodes_dict):
                        for node in sorted(nodes_dict):
                            score_info = score_map.get(node, {})
                            meta_info = meta_lookup.get(node, {})
                            score_str = f"{score_info.get('score', 0):.4f}" if score_info else ""
                            stage_str = str(score_info.get("stage", meta_info.get("stage", "?")))
                            group_str = str(score_info.get("group", meta_info.get("group", "?")))
                            desc_str = meta_info.get("desc", "").replace(",", "，")
                            f.write(f"{y_node},{t_name},{role_name},{node},{score_str},{stage_str},{group_str},{desc_str}\n")
                    _write_csv_role("1-Confounder", c)
                    _write_csv_role("2-Mediator", m)
                    _write_csv_role("3-Collider", col)
                    _write_csv_role("4-Instrument", iv)
            lines.append(f"  [OK] 角色明细表已额外导出为CSV: {csv_path}\n")
        except Exception as e:
            pass
    else:
        lines.append("─" * 72)
        lines.append("  【注】集成图中 Y 节点无入边，Pearl 角色分析已跳过。")
        lines.append("       因果关系请直接参考上方[因果强度 Top 排名]表格。")
        lines.append("       建议重新运行 ensemble_dag_voting.py 以修复 Y 连接。")
        lines.append("")

    lines.append("=" * 72)
    lines.append(f"  {line_id.upper()} 报告结束")
    lines.append("=" * 72)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载特征元数据…")
    meta_lookup = load_metadata()
    print(f"元数据加载完成：{len(meta_lookup)} 个变量记录")

    for line_id, cfg in LINE_CONFIG.items():
        report_text = analyze_line(line_id, cfg, meta_lookup)
        if report_text is None:
            print(f"[跳过] {line_id.upper()} 分析失败，请检查 GraphML 文件是否存在。")
            continue

        out_file = os.path.join(OUTPUT_DIR, f"{line_id.upper()}_DAG_Analysis_Report.txt")
        with open(out_file, "w", encoding="utf-8-sig") as f:
            f.write(report_text)
        print(f"\n  [OK] {line_id.upper()} 报告已保存至: {out_file}")

    # ── 汇总对比两产线排名 ───────────────────────────────────────────────
    print("\n生成双产线对比汇总 CSV…")
    all_dfs = []
    meta_lookup_copy = meta_lookup  # alias for clarity

    for line_id, cfg in LINE_CONFIG.items():
        effects_csv_priority = ["ensemble", "tcdf", "nts_notears", "cuts_plus"]
        for method in effects_csv_priority:
            fname = cfg["effects_files"].get(method)
            if not fname:
                continue
            fpath = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            var_col   = next((c for c in df.columns if "source" in c.lower() or "var" in c.lower()), None)
            score_col = next((c for c in df.columns if "score" in c.lower() or "attention" in c.lower() or "causal" in c.lower()), None)
            if var_col is None or score_col is None:
                continue
            df2 = df[[var_col, score_col]].copy()
            df2.columns = ["Variable_Name", "Causal_Score"]
            df2["Line"]   = line_id
            df2["Method"] = method
            # 补充元数据
            df2["Stage"]  = df2["Variable_Name"].map(lambda v: meta_lookup_copy.get(v, {}).get("stage", "?"))
            df2["Group"]  = df2["Variable_Name"].map(lambda v: meta_lookup_copy.get(v, {}).get("group", "?"))
            df2["Desc"]   = df2["Variable_Name"].map(lambda v: meta_lookup_copy.get(v, {}).get("desc",  ""))
            all_dfs.append(df2)
            break  # 每条产线只取最高优先的可用文件

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True).sort_values(
            ["Line", "Causal_Score"], ascending=[True, False]
        )
        combined_path = os.path.join(OUTPUT_DIR, "dual_line_causal_comparison.csv")
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"  [OK] 双产线对比汇总已保存至: {combined_path}")
    else:
        print("  [警告] 未能读取任何有效的因果强度 CSV，对比汇总未生成。")

    print("\n所有分析完成。")


if __name__ == "__main__":
    main()
