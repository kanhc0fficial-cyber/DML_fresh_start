"""
validate_whitelist.py  (v2 - 新白名单版)
=========================================
验证新白名单（操作变量和混杂变量/）与数据文件 sheet 的匹配情况。
白名单来源：两个 CSV 的 NAME 列合并
  - output_MC_可用版.csv
  - output_最终可用变量合集_无digital版.csv
"""
import pandas as pd
from python_calamine import CalamineWorkbook
from pathlib import Path

BASE  = Path(r'c:\backup\doubleml')
NDATA = BASE / '大量长时间数据'

# ── 加载新白名单（操作变量和混杂变量/ 两个CSV合并） ──────────────────────────
wl_dir = BASE / '操作变量和混杂变量'

mc = set(pd.read_csv(wl_dir / 'output_MC_可用版.csv', usecols=['NAME'])
           ['NAME'].dropna().str.strip().str.upper())
fx = set(pd.read_csv(wl_dir / 'output_最终可用变量合集_无digital版.csv', usecols=['NAME'])
           ['NAME'].dropna().str.strip().str.upper())

whitelist = mc | fx
whitelist.discard('')

print(f'白名单总计: {len(whitelist)} 个')
print(f'  MC_可用版: {len(mc)} 个')
print(f'  最终可用_无digital版: {len(fx)} 个')
print(f'  两者交集: {len(mc & fx)} 个')

# ── 验证文件1（使用 calamine，更快） ─────────────────────────────────────────
path = NDATA / '10.22/10.22/1_20251022_093724_50tables.xlsx'
if path.exists():
    wb = CalamineWorkbook.from_path(str(path))
    sheet_names = wb.sheet_names
    matched     = [s for s in sheet_names if s.upper() in whitelist]
    not_matched = [s for s in sheet_names if s.upper() not in whitelist]
    print(f'\n文件1 (50sheets): 匹配={len(matched)}, 未匹配={len(not_matched)}')
    print('匹配的:', matched[:20])
    print('未匹配前5:', not_matched[:5])
else:
    print(f'\n[跳过] 文件不存在: {path}')

# ── 验证文件2（MC类sheet覆盖率） ─────────────────────────────────────────────
path2 = NDATA / '10.22/10.22/100_20251022_111010_50tables.xlsx'
if path2.exists():
    wb2 = CalamineWorkbook.from_path(str(path2))
    mc_sheets   = [s for s in wb2.sheet_names if s.upper().startswith('MC')]
    mc_matched  = [s for s in mc_sheets if s.upper() in whitelist]
    if mc_sheets:
        print(f'\n文件2 MC类sheet总数: {len(mc_sheets)}，白名单命中: {len(mc_matched)}')
        print('  命中示例:', mc_matched[:5])
    else:
        print('\n文件2 无MC类sheet')
else:
    print(f'\n[跳过] 文件不存在: {path2}')
