"""
run_collinearity_detection.py
=============================
使用斯皮尔曼秩相关 (Spearman Rank) 与 层次凝聚聚类 (HCA) 的先进抗共线性分析算法。
不同于传统的皮尔逊线性检验，斯皮尔曼秩相关更能捕捉工业传感器中非线性的“单调相关”共线特性。
脚本会自动扫描所有合法工艺变量，形成“高能共线组”，挑选代表变量并输出 Markdown 自动化报告和无共线清洗名单。
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
ABC_CSV = r"C:\backup\doubleml\操作变量和混杂变量\output_ABC_分类合集_带变化标签.csv"

# 修改为当前项目路径
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果")
os.makedirs(OUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUT_DIR, "Collinearity_Analysis_Report.md")
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

def load_abc_metadata():
    """读取 ABC 分类及变化统计信息，仅保留 change_status 为 Active 的变量"""
    try:
        df = pd.read_csv(ABC_CSV)
        # 建立 变量 -> {Group, change_count} 的映射，且仅保留 Active 变量
        meta = {}
        for _, row in df.iterrows():
            if str(row.get("change_status", "")).strip() != "Active":
                continue
            name = str(row.get("NAME", "")).strip()
            if not name: continue
            meta[name] = {
                "Group": str(row.get("Group", "C")).strip(),
                "change_count": float(row.get("change_count", 0))
            }
        print(f"  从 ABC 文件中识别到 {len(meta)} 个 Active 变量。")
        return meta
    except Exception as e:
        print(f"警告：无法读取 ABC 分类文件 {ABC_CSV}: {e}")
        return {}

def main():
    print("="*60)
    print("启动非线性共线性检验：全量 Active 变量 HCA 聚类提纯流程")
    print("="*60)
    
    var_to_stage = load_stage_dict()
    abc_meta = load_abc_metadata()
    
    print("[1/5] 正在从 ABC 数据库加载 Active 变量清单（绕过模型筛选）...")
    # 仅保留在工艺阶段分类库中且在 Parquet 数据集中存在的变量
    all_potential_vars = list(abc_meta.keys())
    
    # 预检数据列是否存在
    print(f"  Active 候选总数: {len(all_potential_vars)} 维")
    
    # 获取 Parquet 中的所有实际列名 (使用 pyarrow 引擎快速获取 schema)
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(X_PARQUET)
    available_cols = set(parquet_file.schema.names)
    
    valid_vars = [v for v in all_potential_vars if v in available_cols and v in var_to_stage]
    print(f"  对齐工艺库与数据文件后，最终参与聚类维度: D={len(valid_vars)}")
    
    print(f"[2/5] 载入全量时序数据矩阵 (D={len(valid_vars)})...")
    X = pd.read_parquet(X_PARQUET, columns=valid_vars)
    
    # 时序特征下采样到 10分钟/点 以滤除传感器毫秒级的高频瞬时噪声
    print("[3/5] 数据抗噪平滑化：应用 10min 重采样并前向填充...")
    X.index = pd.to_datetime(X.index).tz_localize(None)
    X_re = X.resample('10min').mean().ffill().bfill()
    
    if len(X_re) > 8000:
        X_re = X_re.tail(8000)
    
    print(f"数据矩阵准备完毕 (N={len(X_re)} 时间点)。")
    
    print("[4/5] 计算斯皮尔曼非线秩相关矩阵并进行凝聚多维聚类...")
    t0 = time.time()
    corr_matrix = X_re.corr(method='spearman').fillna(0)
    
    # 转化为对角线为 0 的绝对距离矩阵
    dist_matrix = 1 - np.abs(corr_matrix.values)
    
    # --- 跨组隔离：不同 Group (A/B/C) 之间严禁合并 ---
    print("  应用业务规则：跨组别变量隔离（不同组距离强制设为最大）...")
    for i in range(len(valid_vars)):
        g_i = abc_meta.get(valid_vars[i], {}).get("Group", "C")
        for j in range(i + 1, len(valid_vars)):
            g_j = abc_meta.get(valid_vars[j], {}).get("Group", "C")
            if g_i != g_j:
                dist_matrix[i, j] = 2.0
                dist_matrix[j, i] = 2.0

    dist_matrix = np.clip(dist_matrix, 0, 2)
    np.fill_diagonal(dist_matrix, 0)
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='average')
    
    # 阀值从 0.08 提高到 0.20 (即相关系数 |rho| >= 0.80 即可合并)
    # 显著增加聚类强度，大幅精简冗余变量
    THRESHOLD = 0.20
    cluster_labels = fcluster(Z, t=THRESHOLD, criterion='distance')
    print(f"聚类分析完成，计算耗时 {time.time()-t0:.1f} 秒！")
    
    # --- 整理聚合分析结果 ---
    print("[5/5] 正在生成共线集群识别清单与代表变量选择...")
    
    cluster_dict = {}
    for i, var in enumerate(valid_vars):
        cid = cluster_labels[i]
        if cid not in cluster_dict:
            cluster_dict[cid] = []
        cluster_dict[cid].append(var)
        
    num_total_clusters = len(cluster_dict)
    num_redundant_groups = sum([1 for g in cluster_dict.values() if len(g) > 1])
    
    representatives = []
    report_lines = []
    
    report_lines.append(f"# 工业控制变量共线性提纯分析报告 (全量业务变量版)")
    report_lines.append(f"*(本报告已绕过 LGBM 筛选，采用全量 ABC 变量池，确保滞后系统的因果完整性)*\n")
    report_lines.append("## 1. 宏观结论")
    report_lines.append(f"- **输入源**: 放弃了基于模型 Gain 的有偏筛选，直接采用 `ABC_分类合集` 提供的 **{len(all_potential_vars)}** 个全量工艺专家变量。")
    report_lines.append(f"- **业务约束**: 强制执行 **跨组隔离** (A/B/C 组绝不合并) 与 **AO 优先选择**。")
    report_lines.append(f"- **净化成果**: 从 **{len(valid_vars)}** 个原始维度 提纯至 **{num_total_clusters}** 维独立物理特征。\n")
    
    report_lines.append("## 2. 强冗余控制簇清单 (筛选逻辑：AO 优先 > 变化频率优先)\n")
    
    def representative_sort_key(var_name):
        """代表变量排序键：
        1. AO 字符优先 (1)
        2. 变化计数 (change_count) 优先 (反映物理活跃度)
        """
        has_ao = 1 if "AO" in var_name.upper() else 0
        active_score = abc_meta.get(var_name, {}).get("change_count", 0)
        return (has_ao, active_score)

    redundant_clusters_sorted = sorted([g for g in cluster_dict.values() if len(g) >= 2], key=len, reverse=True)
    
    for idx, c_vars in enumerate(redundant_clusters_sorted):
        c_vars.sort(key=representative_sort_key, reverse=True)
        best_rep = c_vars[0]
        grp = abc_meta.get(best_rep, {}).get("Group", "C")
        
        report_lines.append(f"### 簇群 #{idx+1} [成员数: {len(c_vars)}, 组别: {grp}]")
        report_lines.append(f"- **保留代表**: `{best_rep}` (活性评分: {abc_meta.get(best_rep,{}).get('change_count',0):.0f})")
        sub_vars = [f"`{v}` ({abc_meta.get(v,{}).get('change_count',0):.0f})" for v in c_vars[1:]]
        report_lines.append(f"- **已剔除成员**: {', '.join(sub_vars)}\n")
            
    for c_vars in cluster_dict.values():
        c_vars.sort(key=representative_sort_key, reverse=True)
        representatives.append(c_vars[0])
        
    with open(REPORT_PATH, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(report_lines))
        
    rep_df = pd.DataFrame({"Non_Collinear_Representative": representatives, "Mapped_Stage": [var_to_stage.get(v) for v in representatives]})
    rep_df.to_csv(CLEAN_VARS_PATH, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*60)
    print(f"大功告成！完美过滤得到了 {len(representatives)} 维最能打的无共线“铁血矩阵指纹”。")
    print(f"分析报告 (.md文稿) 已经写入: {REPORT_PATH}")
    print(f"最新清洗出来的提纯特征字段名单写入: {CLEAN_VARS_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()
