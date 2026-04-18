"""
annotate_variables.py
=====================
通用工具脚本：读取一个包含变量名的 CSV 文件，
为其中的每个变量名自动查找其所属工艺阶段和描述信息，
生成一份"带注释的变量名册"供领域专家快速审阅。

使用方法：
  python annotate_variables.py <输入CSV文件路径> [变量名列名]

示例：
  python annotate_variables.py "non_collinear_representative_vars.csv"
  python annotate_variables.py "my_vars.csv" "Variable_Name"

输入 CSV 中只需包含一列变量名即可。
输出：同目录下生成 <原文件名>_annotated.csv 和 <原文件名>_annotated.md
"""

import os
import sys
import glob
import re
import pandas as pd
import builtins

def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

# ─── 路径配置（阶段分类CSV所在目录）────────────────────────────────────────
STAGE_DIR = r"C:\backup\gemini_clean\output_stages_classified_split_by_stage"

# 两个权威变量表，中文描述均来自 COMMENT 列
VAR_EXCEL_FILES = [
    r"C:\backup\gemini_clean\variable_FX.xlsx",
    r"C:\backup\gemini_clean\T.磨磁\variable_MC.xlsx",
]

STAGE_NAMES = {
    0: "Stage 0 - 公共动力（旋流器、水泵等公用配套）",
    1: "Stage 1 - 药剂合成（捕收剂、起泡剂等注入系统）",
    2: "Stage 2 - 磁选分离（磁选机与粗选前端）",
    3: "Stage 3 - 塔磨分级（立磨分级旋流器）",
    4: "Stage 4 - 浓缩缓冲（浓密机缓冲矿浆池）",
    5: "Stage 5 - 调浆激发（浮选前调浆加药）",
    6: "Stage 6 - 浮选网络（浮选机组主体）",
    7: "Stage 7 - 尾矿收尾（尾矿浆输送废弃）",
    8: "Stage 8 - 精矿脱水（过滤脱水压滤机）",
}

def load_stage_info():
    """返回 {NAME: {stage_id, stage_name, description, unit}} 的映射。
    来源优先级：Excel COMMENT 列 > Stage CSV（仅尖2B填充阶段）
    """
    # Step 1: 从两个 Excel 读取 {NAME: {comment, unit}}
    excel_lookup = {}
    for xls_path in VAR_EXCEL_FILES:
        if not os.path.exists(xls_path):
            print(f"  警告: 找不到 {xls_path}，跳过")
            continue
        try:
            df_xls = pd.read_excel(xls_path, usecols=["NAME", "COMMENT", "ENG_UNITS"])
            for _, row in df_xls.iterrows():
                name = str(row.get("NAME", "")).strip()
                if not name:
                    continue
                comment = str(row.get("COMMENT", "")).strip()
                unit = str(row.get("ENG_UNITS", "")).strip()
                if comment in ["nan", ""]:
                    comment = ""
                if unit in ["nan", ""]:
                    unit = "—"
                excel_lookup[name] = {"comment": comment, "unit": unit}
            print(f"  找到 Excel：{os.path.basename(xls_path)}，共 {len(df_xls)} 条记录")
        except Exception as e:
            print(f"  Excel 读取失败 {xls_path}: {e}")

    # Step 2: 从 Stage CSV 读取 {NAME: stage_id}
    var_stage = {}
    csv_files = glob.glob(os.path.join(STAGE_DIR, "*.csv"))
    for f in csv_files:
        basename = os.path.basename(f).lower()
        if "stage_unknown" in basename:
            continue
        match = re.search(r"stage_(\d+)", basename)
        if not match:
            continue
        stage_num = int(match.group(1))
        try:
            df_csv = pd.read_csv(f, usecols=["NAME"])
            for name in df_csv["NAME"].dropna().astype(str).str.strip():
                var_stage[name] = stage_num
        except:
            pass

    # Step 3: 合并为最终 var_info
    all_names = set(excel_lookup) | set(var_stage)
    var_info = {}
    for name in all_names:
        stage_id = var_stage.get(name)
        ex = excel_lookup.get(name, {})
        comment = ex.get("comment", "")
        unit = ex.get("unit", "—")
        var_info[name] = {
            "Stage_ID": stage_id if stage_id is not None else "Unknown",
            "Stage_Name": STAGE_NAMES.get(stage_id, "未知阶段") if stage_id is not None else "未知阶段",
            "Description": comment if comment else "（需专家填写）",
            "Unit": unit,
        }
    return var_info

def annotate(input_csv: str, name_col: str = None):
    print("="*60)
    print("工控变量通用注释工具 - 专家审阅模式")
    print("="*60)
    
    if not os.path.exists(input_csv):
        print(f"错误：找不到输入文件 {input_csv}")
        sys.exit(1)
    
    print(f"[1/3] 读取输入文件: {input_csv}")
    df_in = pd.read_csv(input_csv, encoding="utf-8-sig",
                        encoding_errors="replace")
    
    # 自动识别变量名列
    if name_col and name_col in df_in.columns:
        var_col = name_col
    else:
        # 尝试自动匹配常见列名
        candidates = [c for c in df_in.columns
                      if any(k in c.lower() for k in
                             ["name", "var", "col", "feature", "represent", "variable"])]
        if candidates:
            var_col = candidates[0]
        else:
            var_col = df_in.columns[0]
    
    print(f"  检测到变量名列: '{var_col}'，共 {len(df_in)} 行")
    variables = df_in[var_col].dropna().astype(str).str.strip().unique().tolist()
    
    print(f"[2/3] 加载工艺阶段分类数据库（共 {len(variables)} 个待注释变量）...")
    var_info = load_stage_info()
    
    print(f"  阶段数据库覆盖 {len(var_info)} 个已知变量，"
          f"本批次覆盖率 = {sum(1 for v in variables if v in var_info)}/{len(variables)}")
    
    print("[3/3] 正在生成注释结果...")
    
    rows = []
    for v in variables:
        info = var_info.get(v, {})
        rows.append({
            "Variable_Name": v,
            "Stage_ID": info.get("Stage_ID", "Unknown"),
            "Stage_Name": info.get("Stage_Name", "未知阶段（不在分类库中）"),
            "Description_CN": info.get("Description", "（未找到中文描述，请专家填写）"),
            "Unit": info.get("Unit", "—"),
            "Expert_Review": "",      # 留白供领域专家填写审阅意见
            "Keep_Remove": "keep",    # 专家决定 keep / remove
        })
    
    df_out = pd.DataFrame(rows)
    
    base = os.path.splitext(input_csv)[0]
    out_csv = base + "_annotated.csv"
    out_md = base + "_annotated.md"
    
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  已输出带注释 CSV: {out_csv}")
    
    # ── Markdown 报告（分阶段显示，方便专家按工艺流程审阅）──────────────
    md_lines = ["# 工控变量注释表 - 领域专家审阅版\n"]
    md_lines.append(f"**共 {len(df_out)} 个变量，来源文件: `{os.path.basename(input_csv)}`**\n")
    md_lines.append("> 请在 `Expert_Review` 列填写意见，在 `Keep_Remove` 列填写 `keep` 或 `remove`\n")
    
    for stage_id in sorted([s for s in STAGE_NAMES] + [-1]):
        if stage_id == -1:
            subset = df_out[df_out["Stage_ID"] == "Unknown"]
            section_title = "Unknown - 未知阶段（不在分类库）"
        else:
            subset = df_out[df_out["Stage_ID"] == stage_id]
            section_title = STAGE_NAMES.get(stage_id, f"Stage {stage_id}")
        
        if subset.empty:
            continue
        
        md_lines.append(f"\n## {section_title}\n")
        md_lines.append(f"| # | 变量名 | 中文描述 | 单位 | 专家意见 | 保留/删除 |")
        md_lines.append(f"|---|--------|---------|------|---------|---------|")
        for i, (_, row) in enumerate(subset.iterrows(), 1):
            md_lines.append(
                f"| {i} | `{row['Variable_Name']}` | {row['Description_CN']} "
                f"| {row['Unit']} |  |  |"
            )
    
    with open(out_md, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(md_lines))
    print(f"  已输出 Markdown 审阅表: {out_md}")
    
    # 简要统计
    known = df_out[df_out["Stage_ID"] != "Unknown"]
    print(f"\n{'='*60}")
    print(f"注释完成！各阶段变量分布：")
    for sid, grp in known.groupby("Stage_ID"):
        print(f"  Stage {sid}: {len(grp)} 个变量")
    unknown_cnt = len(df_out[df_out["Stage_ID"] == "Unknown"])
    if unknown_cnt:
        print(f"  Unknown: {unknown_cnt} 个变量（未在分类数据库中找到）")
    print("="*60)

if __name__ == "__main__":
    # 命令行：python annotate_variables.py <csv路径> [列名]
    if len(sys.argv) < 2:
        # 默认直接注释共线性清洗后的代表变量名单
        default = r"c:\dml\lgbm和tcdf\with_time_space\结果\non_collinear_representative_vars.csv"
        print(f"未指定输入文件，使用默认路径: {default}")
        annotate(default, "Non_Collinear_Representative")
    elif len(sys.argv) == 2:
        annotate(sys.argv[1])
    else:
        annotate(sys.argv[1], sys.argv[2])
