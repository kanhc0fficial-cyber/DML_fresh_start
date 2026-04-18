"""
ensemble_dag_voting.py
======================
多方法 DAG 集成投票脚本。
【双产线版】：分别对 xin1 和 xin2 两条产线的 DAG 进行集成。

集成策略:
  - 3 个算法（NTS-NOTEARS, CUTS+, TCDF）的结果各贡献投票
  - 按归一化得分排序，用贪心插入保证 DAG 无环
  - threshold_eta 控制保留门槛（0.5 = 简单多数）
"""
import networkx as nx
import numpy as np
import pandas as pd
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")


def ensemble_dag_voting_for_line(line: str, threshold_eta: float = 0.4):
    """对指定产线进行集成投票"""
    print(f"\n{'='*60}")
    print(f"  集成投票 [产线 = {line}]")
    print(f"{'='*60}")

    dag_files = {
        "cuts_plus":   f"cuts_plus_dag_{line}.graphml",
        "tcdf":        f"tcdf_space_time_dag_{line}.graphml",
        "nts_notears": f"nts_notears_dag_{line}.graphml",
    }

    combined_edges = {}
    all_nodes = set()
    active_dags = 0

    print("正在聚合各算法生成的局部 DAG 结构...")
    for name, filename in dag_files.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"  - 跳过 {name}: 文件不存在 ({filename})")
            continue

        try:
            G = nx.read_graphml(path)
            if len(G.edges()) == 0:
                print(f"  - {name}: 图中无边，跳过")
                continue

            active_dags += 1
            all_nodes.update(G.nodes())

            edges = list(G.edges(data=True))
            max_w = max([abs(d.get('weight', 1.0)) for u, v, d in edges]) if edges else 1.0

            for u, v, d in edges:
                w = abs(d.get('weight', 1.0)) / (max_w + 1e-9)
                edge_score = 1.0 + w
                combined_edges[(u, v)] = combined_edges.get((u, v), 0.0) + edge_score

            print(f"  + {name}: {len(edges)} 条边")
        except Exception as e:
            print(f"  - 读取 {name} 失败: {e}")

    if active_dags == 0:
        print(f"[{line}] 未找到任何有效 DAG 文件，跳过集成。")
        return

    # 归一化总分
    max_total_score = max(combined_edges.values()) if combined_edges else 1.0
    for edge in combined_edges:
        combined_edges[edge] /= max_total_score

    # 构建集成 DAG（贪心破环）
    final_G = nx.DiGraph()
    final_G.add_nodes_from(list(all_nodes))

    sorted_edges = sorted(combined_edges.items(), key=lambda x: x[1], reverse=True)
    print(f"\n聚合完成，共识别出 {len(sorted_edges)} 条候选边。执行 DAG 破环重构...")

    added_count = 0
    rejected_count = 0
    cycle_prevented = 0

    for (u, v), score in sorted_edges:
        # 特赦规则：如果目标节点是最终的精矿品位 Y，无视 0.4 阈值强制候选
        is_target_y = (v in ['y_grade', 'Y_grade'])
        
        if score < threshold_eta and not is_target_y:
            rejected_count += 1
            continue
            
        final_G.add_edge(u, v, weight=score)
        if not nx.is_directed_acyclic_graph(final_G):
            final_G.remove_edge(u, v)
            cycle_prevented += 1
        else:
            added_count += 1

    output_graphml = os.path.join(RESULTS_DIR, f"ensemble_integrated_dag_{line}.graphml")
    nx.write_graphml(final_G, output_graphml)

    print("-" * 50)
    print(f"[{line}] 集成 DAG 构建完成！")
    print(f"  最终保留边数: {added_count}")
    print(f"  低分剔除边数: {rejected_count}")
    print(f"  破环保护剔除: {cycle_prevented}")
    print(f"  结果保存至: {output_graphml}")

    # 统计指向 Y_grade 的高分边
    y_edges = [(u, v, d['weight'])
               for u, v, d in final_G.edges(data=True)
               if v in ['y_grade', 'Y_grade']]
    if y_edges:
        y_df = pd.DataFrame(y_edges, columns=['Source', 'Target', 'Score'])
        y_df = y_df.sort_values('Score', ascending=False)
        print(f"\n  TOP 影响 Y_grade 的因果变量 ({line}):")
        print(y_df.head(15).to_string(index=False))
        out_csv = os.path.join(RESULTS_DIR, f"ensemble_effects_on_y_{line}.csv")
        y_df.to_csv(out_csv, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="集成 DAG 投票 (双产线)")
    parser.add_argument("--line", choices=["xin1", "xin2", "both"], default="both",
                        help="集成哪条产线: xin1/xin2/both (默认 both)")
    parser.add_argument("--eta", type=float, default=0.4,
                        help="投票保留阈值 (0.33=并集, 0.5=多数决, 0.66=交集)")
    args = parser.parse_args()

    lines = ["xin1", "xin2"] if args.line == "both" else [args.line]
    for ln in lines:
        ensemble_dag_voting_for_line(ln, args.eta)
