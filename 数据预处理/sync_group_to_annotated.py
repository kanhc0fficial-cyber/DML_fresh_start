"""
sync_group_to_annotated.py
==========================
将 ABC 分类合集中的 Group（A/B/C）和 change_status 字段
同步合并到 non_collinear_representative_vars_annotated.csv 中。

- A: 新1（Ⅰ系列）专线浮选设备
- B: 新2（Ⅱ系列）专线浮选设备
- C: 全厂公用设备（无系列标识，如鼓风机、浮选机电流等）
"""
import pandas as pd

ABC_CSV = r"C:\backup\doubleml\操作变量和混杂变量\output_ABC_分类合集_带变化标签.csv"
ANN_CSV = r"C:\DML_fresh_start\数据预处理\数据与处理结果-分阶段-去共线性后\non_collinear_representative_vars_annotated.csv"
OUT_CSV = ANN_CSV  # 直接覆盖更新

# 只读ABC文件中需要的列
abc = pd.read_csv(ABC_CSV, usecols=["NAME", "Group", "Group_Reason", "change_count", "change_status"])

# 读原annotated文件（仅保留非ABC衍生列，避免重复合并问题）
ann = pd.read_csv(ANN_CSV)
# 如果annotated已包含过这些列，先删掉
for drop_col in ["Group", "Group_Reason", "change_count", "change_status", "NAME"]:
    if drop_col in ann.columns:
        ann.drop(columns=[drop_col], inplace=True)

# 以 Variable_Name 为 key 左连接
merged = ann.merge(abc, left_on="Variable_Name", right_on="NAME", how="left")
merged.drop(columns=["NAME"], inplace=True)

# 对未匹配的（手工质检变量如 CXXY_PW 等），推断 Group 为公用
merged["Group"] = merged["Group"].fillna("C")
merged["change_status"] = merged["change_status"].fillna("Unknown")
merged["change_count"] = merged["change_count"].fillna(0)

# 整理列顺序
cols = ["Variable_Name", "Stage_ID", "Stage_Name", "Description_CN", "Unit",
        "Expert_Review", "Keep_Remove", "Group", "Group_Reason", "change_count", "change_status"]
merged = merged[[c for c in cols if c in merged.columns]]

merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] Group 同步完成，共 {len(merged)} 个变量写入 {OUT_CSV}")
print("\nGroup 分布:")
print(merged["Group"].value_counts())
print("\nchange_status 分布:")
print(merged["change_status"].value_counts())
