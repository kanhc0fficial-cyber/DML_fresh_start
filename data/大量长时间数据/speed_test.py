"""测试 calamine 引擎速度 vs openpyxl"""
import time, pandas as pd
from pathlib import Path
from python_calamine import CalamineWorkbook

NDATA   = Path(r"c:\backup\doubleml\大量长时间数据")
BIGFILE = NDATA / "10.22/10.22/100_20251022_111010_50tables.xlsx"

# 加载白名单
BASE = Path(r"c:\backup\doubleml")
fx = set(pd.read_csv(BASE/'output_最终可用变量合集.csv', usecols=['NAME'])['NAME'].dropna().str.strip().str.upper())
mg = set()
with open(NDATA/'磨矿.txt', encoding='utf-8') as f:
    for l in f:
        p=l.strip().split()
        if p: mg.add(p[0].upper())
cx = set()
with open(NDATA/'磁选.txt', encoding='utf-8') as f:
    for l in f:
        p=l.strip().split()
        if p: cx.add(p[0].upper())
whitelist = fx | mg | cx
whitelist.discard('')

size_mb = BIGFILE.stat().st_size / 1024 / 1024
print(f"测试文件: {BIGFILE.name} ({size_mb:.1f} MB)")

# === calamine ===
print("\n--- python-calamine ---")
t0 = time.time()
wb = CalamineWorkbook.from_path(str(BIGFILE))
sheet_names = wb.sheet_names
matching = [(s, s.upper()) for s in sheet_names if s.upper() in whitelist]
print(f"  Sheet列表获取: {time.time()-t0:.2f}s, 共{len(sheet_names)}个, 匹配{len(matching)}个")

for sheet_name, tag in matching[:3]:
    t1 = time.time()
    data = wb.get_sheet_by_name(sheet_name).to_python()
    if not data or len(data) < 2:
        continue
    headers = [str(h).lower().strip() for h in data[0]]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    print(f"  {tag}: {len(df)} 行, 用时{time.time()-t1:.3f}s")

print(f"\ncalamine 总测试用时: {time.time()-t0:.1f}s")
