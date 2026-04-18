"""
run_collinearity_detection_ao_first.py
======================================
基于 HCA 聚类的共线性检测脚本。
修改规则：在同一个共线性簇中，优先选择以 "AO" 结尾的操作变量作为代表。
若同为 AO 或同非 AO，则选择 LightGBM Gain 最高的变量。
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings("ignore")

import builtins
def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

# ─── 路径配置 ───
BASE_DIR = r"C:\backup\doubleml"
STAGE_DIR = r"C:\backup\gemini_clean\output_stages_classified_split_by_stage"
X_PARQUET = os.path.join(BASE_DIR, "X_features_new.parquet")
LGBM_IMP = r"c:\dml\lgbm和tcdf\结果\lgbm_xin1_all_vars.csv"

# 输出结果保存到当前目录下的“因果发现结果”文件夹
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")
os.makedirs(OUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUT_DIR, "Collinearity_Analysis_Report_AO_First.md")
CLEAN_VARS_PATH = os.path.join(OUT_DIR, "non_collinear_representative_vars.csv")

def load_stage_dict():
    var_to_stage = {}
    csv_files = glob.glob(os.path.join(STAGE_DIR, "*.csv"))
    for f in csv_files:
        basename = os.path.basename(f).lower()
        if "stage_unknown" in basename: continue
        import re
        match = re.search(r"stage_(\d+)", basename)
        if match:
            stage_num = int(match.group(1))
            try:
                df = pd.read_csv(f, usecols=["NAME"])
                for name in df["NAME"].dropna().str.strip():
                    var_to_stage[name] = stage_num
            except: pass
    return var_to_stage

def main():
    print("="*60)
    print("启动共线性检验流程 - 优先保留 AO 操作变量模式")
    print("="*60)
    
    var_to_stage = load_stage_dict()
    
    print("[1/5] 读取 LGBM 粗筛底表并对齐可用特征...")
    if not os.path.exists(LGBM_IMP):
        print(f"错误：找不到 LGBM 重要性文件 {LGBM_IMP}")
        return

    imp_df = pd.read_csv(LGBM_IMP)
    imp_df = imp_df[imp_df["Total_Gain"] > 0]
    valid_vars = [v for v in imp_df["Original_Var"] if v in var_to_stage]
    
    print(f"[2/5] 载入 Parquet 数据集 (维度 D={len(valid_vars)})...")
    X = pd.read_parquet(X_PARQUET)[valid_vars]
    
    print("[3/5] 数据重采样 (10min) 以平滑噪声...")
    X.index = pd.to_datetime(X.index).tz_localize(None)
    X_re = X.resample('10min').mean().ffill().bfill()
    
    if len(X_re) > 8000:
        X_re = X_re.tail(8000)
    
    print(f"数据矩阵准备完毕 (N={len(X_re)}, D={len(valid_vars)})。")
    
    print("[4/5] 计算斯皮尔曼秩相关矩阵并进行 HCA 聚类...")
    t0 = time.time()
    corr_matrix = X_re.corr(method='spearman').fillna(0)
    
    vif_pseudo = np.diag(np.linalg.pinv(corr_matrix.values))
    max_vif = np.max(vif_pseudo)
    
    dist_matrix = 1 - np.abs(corr_matrix.values)
    dist_matrix = np.clip(dist_matrix, 0, 2)
    np.fill_diagonal(dist_matrix, 0)
    
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='average')
    
    # 阈值 0.08 表示相关性 |rho| >= 0.92 的变量将被聚类在一起
    THRESHOLD = 0.08
    cluster_labels = fcluster(Z, t=THRESHOLD, criterion='distance')
    print(f"聚类完成，耗时 {time.time()-t0:.1f} 秒。")
    
    print("[5/5] 正在根据『AO优先规则』筛选代表变量并生成报告...")
    
    cluster_dict = {}
    for i, var in enumerate(valid_vars):
        cid = cluster_labels[i]
        if cid not in cluster_dict:
            cluster_dict[cid] = []
        cluster_dict[cid].append(var)
    
    gain_dict = dict(zip(imp_df["Original_Var"], imp_df["Total_Gain"]))
    
    representatives = []
    report_lines = []
    
    report_lines.append(f"# 工业控制变量共线性分析报告 (AO 优先模式)")
    report_lines.append(f"- **总探测量**: {len(valid_vars)} 个变量")
    report_lines.append(f"- **最大虚拟 VIF**: {max_vif:.2f}")
    report_lines.append(f"- **选择策略**: 优先选择以 `AO` 结尾的变量。若无 AO 变量，则选择 LGBM Gain 最高的变量。\n")
    
    redundant_clusters_sorted = sorted([g for g in cluster_dict.values() if len(g) >= 2], key=len, reverse=True)
    
    for idx, c_vars in enumerate(redundant_clusters_sorted):
        # 核心修改：排序规则 (是否AO结尾, Gain分数)
        c_vars.sort(key=lambda x: (x.upper().endswith('AO'), gain_dict.get(x, 0)), reverse=True)
        best_rep = c_vars[0]
        
        report_lines.append(f"### 族群 #{idx+1} [成员数: {len(c_vars)}]")
        report_lines.append(f"- **保留代表**: `{best_rep}` {'(操作变量 AO)' if best_rep.upper().endswith('AO') else '(非AO变量)'}")
        report_lines.append(f"- **LGBM Gain**: {gain_dict.get(best_rep,0):.2f}")
        sub_str = ", ".join([f"`{v}`" for v in c_vars[1:]])
        report_lines.append(f"- **已剔除成员**: {sub_str}\n")
            
    # 计算最终保留名单
    for c_vars in cluster_dict.values():
        c_vars.sort(key=lambda x: (x.upper().endswith('AO'), gain_dict.get(x, 0)), reverse=True)
        representatives.append(c_vars[0])
        
    with open(REPORT_PATH, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(report_lines))
        
    rep_df = pd.DataFrame({"Non_Collinear_Representative": representatives, "Mapped_Stage": [var_to_stage.get(v) for v in representatives]})
    rep_df.to_csv(CLEAN_VARS_PATH, index=False, encoding="utf-8-sig")
    
    print(f"\n处理完成！最终保留变量数: {len(representatives)}")
    print(f"报告已生成: {REPORT_PATH}")
    print(f"名单已导出: {CLEAN_VARS_PATH}")

if __name__ == "__main__":
    main()
