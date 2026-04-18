"""
parse_dag_structures.py
=======================
读取 PCMCI 计算后保存的高维 P-Value 矩阵，将其塌缩为“摘要因果图 (Summary Graph)”。
然后，基于严谨的 Pearl 结构因果模型 (SCM)，为指定的核心处理变量 (T)
自动寻址并打印出标准的四种因果链路：
1. 混杂变量 (Confounder)
2. 中介变量 (Mediator)
3. 对撞变量 (Collider)
4. 工具变量 (Instrumental Variable)
"""

import os
import numpy as np
import sys

# 路径配置
BASE_DIR = r"C:\backup\doubleml\VAE_LSTM_DML\结果\large_pcmci"
ALPHA = 0.05

def load_pcmci_data(line_id="xin1"):
    suffix = f"_{line_id}" if line_id == "xin2" else ""
    p_file = os.path.join(BASE_DIR, f"pcmci_p_matrix{suffix}.npy")
    val_file = os.path.join(BASE_DIR, f"pcmci_val_matrix{suffix}.npy")
    var_file = os.path.join(BASE_DIR, f"pcmci_var_names{suffix}.txt")
    
    if not os.path.exists(p_file):
        print(f"未找到 {p_file}，请确保该管线的 PCMCI 已计算完成！")
        return None, None, None
        
    p_matrix = np.load(p_file)
    val_matrix = np.load(val_file)
    with open(var_file, "r", encoding="utf-8") as f:
        var_names = [line.strip() for line in f.readlines()]
        
    return p_matrix, val_matrix, var_names

def build_summary_graph(p_matrix, var_names):
    N = len(var_names)
    G = np.zeros((N, N), dtype=bool)
    max_lag = p_matrix.shape[2]
    
    for i in range(N):
        for j in range(N):
            for lag in range(1, max_lag):
                if p_matrix[i, j, lag] <= ALPHA:
                    G[i, j] = True
                    break
    return G

def find_causal_roles(G, var_names, T_name, Y_name="Y_grade"):
    try:
        t_idx = var_names.index(T_name)
        y_idx = var_names.index(Y_name)
    except ValueError:
        return ""
        
    confounders, mediators, colliders, instruments = [], [], [], []
    
    for i, node in enumerate(var_names):
        if i == t_idx or i == y_idx:
            continue
            
        if G[i, t_idx] and G[i, y_idx]: confounders.append(node)
        if G[t_idx, i] and G[i, y_idx]: mediators.append(node)
        if G[t_idx, i] and G[y_idx, i]: colliders.append(node)
        if G[i, t_idx] and not G[i, y_idx]: instruments.append(node)
            
    output_lines = []
    output_lines.append(f"\n========================================================")
    output_lines.append(f"   目标动作 (Treatment / T) = {T_name}")
    output_lines.append(f"   最终结果 (Outcome / Y) = {Y_name}")
    output_lines.append(f"========================================================")
    
    output_lines.append("\n【1】混杂变量 (Confounders, C) [ 路径: T <- C -> Y ]")
    output_lines.append(f"  > 找到了 {len(confounders)} 个: ")
    for c in confounders: output_lines.append(f"      ● {c}")
        
    output_lines.append("\n【2】中介变量 (Mediators, M) [ 路径: T -> M -> Y ]")
    output_lines.append(f"  > 找到了 {len(mediators)} 个: ")
    for m in mediators: output_lines.append(f"      ● {m}")
        
    output_lines.append("\n【3】对撞变量 (Colliders, Z) [ 路径: T -> Z <- Y (Y的过去影响Z) ]")
    output_lines.append(f"  > 找到了 {len(colliders)} 个: ")
    for z in colliders: output_lines.append(f"      ● {z}")
        
    output_lines.append("\n【4】备选工具变量 (Instrumental Variables, IV) [ 路径: IV -> T (且无直接流向Y) ]")
    output_lines.append(f"  > 找到了 {len(instruments)} 个: ")
    for iv in instruments: output_lines.append(f"      ● {iv}")
    
    output_lines.append("========================================================\n")
    return "\n".join(output_lines)

def process_line(line_id):
    p_matrix, val_matrix, var_names = load_pcmci_data(line_id)
    if p_matrix is None: return
    
    G = build_summary_graph(p_matrix, var_names)
    test_ops = [v for v in var_names if ("_AO" in v or "_F_W" in v or "AI4" in v or "AI6" in v or "AI8" in v or "AI11" in v or "AI12" in v) and v != "Y_grade"]
    
    full_report = [f"========================================================",
                   f" {line_id.upper()} 全阵列 PCMCI 结构因果解析报告",
                   f"========================================================\n"]
    
    for op in test_ops:
        report_text = find_causal_roles(G, var_names, op)
        if report_text: full_report.append(report_text)
                
    out_dir = r"C:\backup\doubleml\VAE_LSTM_DML\结果\pure_vae_dml"
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, f"{line_id.upper()}_DAG_Structures_Report.txt")
    
    with open(report_file, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(full_report))
    print(f"完整报告已落盘至: {report_file}")

if __name__ == "__main__":
    for line in ["xin1", "xin2"]:
        process_line(line)
