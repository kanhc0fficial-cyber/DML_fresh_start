import pandas as pd
import numpy as np
import re

file1 = r"C:\DML_fresh_start\双重机器学习\结果\joint_causal_xin1\joint_causal_dml_xin1_fixed.csv"
file2 = r"C:\DML_fresh_start\双重机器学习\结果\joint_causal_xin2\joint_causal_dml_xin2_fixed.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("==== 共享/同名设备因果效应一致性对比 (MC1: 塔磨分级, MC2: 磁选) ====")
# Filter shared
df1_shared = df1[df1['操作节点'].str.startswith('MC')].copy()
df2_shared = df2[df2['操作节点'].str.startswith('MC')].copy()

merged = pd.merge(df1_shared, df2_shared, on='操作节点', suffixes=('_XIN1', '_XIN2'))
merged['Diff'] = merged['θ_效应值_XIN1'] - merged['θ_效应值_XIN2']

print(merged[['操作节点', 'θ_效应值_XIN1', 'θ_效应值_XIN2', 'Diff']].to_string(index=False))

print("\n==== 独立设备因果效应分布对比 (FX: 浮选) ====")
def extract_fx_stage(name):
    match = re.search(r'FX_X\d([A-Z0-9]+)_', name)
    if match:
        return match.group(1)
    return 'Other'

df1_fx = df1[df1['操作节点'].str.startswith('FX')].copy()
df2_fx = df2[df2['操作节点'].str.startswith('FX')].copy()

df1_fx['Stage'] = df1_fx['操作节点'].apply(extract_fx_stage)
df2_fx['Stage'] = df2_fx['操作节点'].apply(extract_fx_stage)

agg1 = df1_fx.groupby('Stage')['θ_效应值'].agg(['mean', 'count', 'max', 'min']).reset_index()
agg2 = df2_fx.groupby('Stage')['θ_效应值'].agg(['mean', 'count', 'max', 'min']).reset_index()

agg_merged = pd.merge(agg1, agg2, on='Stage', suffixes=('_XIN1', '_XIN2'), how='outer').fillna(0)
print("按浮选阶段分组的效应值统计 (CX:粗选, JX:精选, SX:扫选)")
print(agg_merged.to_string(index=False))

print("\n==== XIN1 与 XIN2 异常极值检查 ====")
idx1 = df1['θ_效应值'].abs().idxmax()
idx2 = df2['θ_效应值'].abs().idxmax()
print(f"XIN1 绝对值最大效应节点: {df1.loc[idx1, '操作节点']} ({df1.loc[idx1, 'θ_效应值']:.5f})")
print(f"XIN2 绝对值最大效应节点: {df2.loc[idx2, '操作节点']} ({df2.loc[idx2, 'θ_效应值']:.5f})")
