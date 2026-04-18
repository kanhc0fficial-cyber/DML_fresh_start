import networkx as nx
import pandas as pd
import os
import re

def parse_dag():
    # Load treatment variables from CSVs and normalize (remove suffix like _AO, _I, etc)
    df1 = pd.read_csv(r'C:\backup\doubleml\操作变量\output_MC_纯操作变量合集.csv')
    df2 = pd.read_csv(r'C:\backup\doubleml\操作变量\output_纯操作变量合集.csv')
    csv_t_vars = set(df1.iloc[:, 0].tolist() + df2.iloc[:, 0].tolist())
    
    # Prefix mapping: prefix -> original_csv_name
    t_prefixes = {}
    for v in csv_t_vars:
        # Match base name (e.g., MC2_QC510_FCSJGSJ from MC2_QC510_FCSJGSJ_AO)
        # Usually suffixes are _AO, _I, _F, _AI, _AI6, _SP, _W, _DL, _HZ
        base = re.sub(r'(_[A-Z0-9]+)+$', '', v)
        if base not in t_prefixes: t_prefixes[base] = []
        t_prefixes[base].append(v)

    # Load GraphML
    graph_path = r'C:\DML_fresh_start\多种方法因果发现\因果发现结果\ensemble_integrated_dag.graphml'
    if not os.path.exists(graph_path):
        print(f"Error: GraphML file not found at {graph_path}")
        return

    G = nx.read_graphml(graph_path)
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")

    # Match T-vars in graph
    # We'll consider a node as a T-var if it starts with any of our prefixes
    matched_t_nodes = []
    for node in G.nodes:
        # Check if node name or its prefix matches our CSV vars
        for prefix in t_prefixes:
            if node.startswith(prefix):
                matched_t_nodes.append(node)
                break
    
    # Potential Outcomes: prioritize high in-degree nodes starting with PJ or y
    outcomes = []
    candidates = [n for n in G.nodes if n.lower().startswith('y') or n.startswith('PJ')]
    candidates = sorted(candidates, key=lambda n: G.in_degree(n), reverse=True)
    
    # Filter for nodes that actually have connections
    active_outcomes = [n for n in candidates if G.in_degree(n) > 10 or G.out_degree(n) > 10]
    
    if not active_outcomes:
        # Fallback to the requested names even if empty
        active_outcomes = ['y_grade', 'PJ_YT']
        
    print(f"Target Outcomes identified: {active_outcomes[:5]}")

    def find_roles(T, Y):
        confounders, mediators, colliders, instruments = [], [], [], []
        
        # Directed connections
        t_parents = set(G.predecessors(T))
        t_children = set(G.successors(T))
        y_parents = set(G.predecessors(Y))
        y_children = set(G.successors(Y))
        
        for node in G.nodes:
            if node == T or node == Y:
                continue
            
            is_parent_t = node in t_parents
            is_child_t = node in t_children
            is_parent_y = node in y_parents
            is_child_y = node in y_children
            
            # 1. Confounder: T <- C -> Y
            if is_parent_t and is_parent_y:
                confounders.append(node)
            
            # 2. Mediator: T -> M -> Y
            if is_child_t and is_parent_y:
                mediators.append(node)
                
            # 3. Collider: T -> Z <- Y
            if is_child_t and is_child_y:
                colliders.append(node)
                
            # 4. Instrumental Variable: IV -> T and no direct path to Y
            if is_parent_t and not is_parent_y and not is_child_y:
                instruments.append(node)
                
        return confounders, mediators, colliders, instruments

    report = []
    report.append("========================================================")
    report.append("           DAG 结构因果解析报告 (集成图解析)")
    report.append("========================================================\n")
    report.append(f"输入文件: {graph_path}")
    report.append(f"图中匹配的操作变量数: {len(matched_t_nodes)}")
    report.append(f"主要目标变量: {', '.join(active_outcomes[:3])}\n")

    # Limit to top 3 outcomes to avoid huge file
    for y_name in active_outcomes[:2]:
        report.append(f"\n########################################################")
        report.append(f"   【分析目标变量 (Outcome)】: {y_name}")
        report.append(f"########################################################\n")
        
        for t in sorted(matched_t_nodes):
            c, m, col, iv = find_roles(t, y_name)
            
            # Only report if there's any causal connection (not just instruments)
            # or if it's a strongly connected T-var
            if not (c or m or col):
                # Check if it has an edge to Y
                if not G.has_edge(t, y_name):
                    continue
            
            report.append(f"========================================================")
            report.append(f"   目标动作 (Treatment / T) = {t}")
            report.append(f"   最终结果 (Outcome / Y) = {y_name}")
            report.append(f"========================================================")
            
            report.append("\n【1】混杂变量 (Confounders, C) [ 路径: T <- C -> Y ]")
            report.append(f"  > 找到了 {len(c)} 个: ")
            for item in c: report.append(f"      ● {item}")
                
            report.append("\n【2】中介变量 (Mediators, M) [ 路径: T -> M -> Y ]")
            report.append(f"  > 找到了 {len(m)} 个: ")
            for item in m: report.append(f"      ● {item}")
                
            report.append("\n【3】对撞变量 (Colliders, Z) [ 路径: T -> Z <- Y ]")
            report.append(f"  > 找到了 {len(col)} 个: ")
            for item in col: report.append(f"      ● {item}")
                
            report.append("\n【4】备选工具变量 (Instrumental Variables, IV) [ 路径: IV -> T (且无直接流向Y) ]")
            report.append(f"  > 找到了 {len(iv)} 个: ")
            # Only list first 20 instruments to save space
            for item in iv[:20]: report.append(f"      ● {item}")
            if len(iv) > 20: report.append(f"      ... (还有 {len(iv)-20} 个)")
            report.append("\n")

    out_dir = "DAG解析结果"
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "DAG_Structures_Report_Refined.txt")
    
    with open(report_file, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(report))
    
    print(f"解析完成！精简报告已保存至: {report_file}")

if __name__ == "__main__":
    parse_dag()
