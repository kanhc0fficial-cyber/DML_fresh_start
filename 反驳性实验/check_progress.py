"""
检查DML理论验证实验的进度
"""
import os
import json
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent / "DML理论验证"

print("=" * 70)
print(" DML理论验证实验进度检查")
print("=" * 70)

if not OUT_DIR.exists():
    print(f"\n输出目录不存在: {OUT_DIR}")
    exit(0)

# 检查所有生成的文件
files = list(OUT_DIR.glob("*"))
print(f"\n已生成 {len(files)} 个文件:")

for f in sorted(files):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<60s} {size_kb:>8.1f} KB")

# 读取CSV文件查看实验进度
csv_files = list(OUT_DIR.glob("*.csv"))
print(f"\n详细进度 (共 {len(csv_files)} 个CSV文件):")

for csv_file in sorted(csv_files):
    try:
        df = pd.read_csv(csv_file)
        print(f"\n  {csv_file.name}:")
        print(f"    实验次数: {len(df)}")
        if 'bias' in df.columns:
            print(f"    平均偏差: {df['bias'].mean():+.6f}")
            print(f"    RMSE: {df['bias'].std():.6f}")
        if 'covers_true' in df.columns:
            print(f"    覆盖率: {df['covers_true'].mean():.1%}")
    except Exception as e:
        print(f"  {csv_file.name}: 读取失败 ({e})")

# 读取JSON汇总文件
json_files = list(OUT_DIR.glob("*.json"))
print(f"\n汇总结果 (共 {len(json_files)} 个JSON文件):")

for json_file in sorted(json_files):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n  {json_file.name}:")
        for key, val in data.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.6f}")
            else:
                print(f"    {key}: {val}")
    except Exception as e:
        print(f"  {json_file.name}: 读取失败 ({e})")

print("\n" + "=" * 70)
