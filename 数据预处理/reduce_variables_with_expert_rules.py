"""
结合 `non_collinear_representative_vars_operability.csv` 与专家知识文档，
逐变量给出保留/删除分析，并按专家规则输出降维后的 parquet。

默认输入：
  - 变量清单: data/操作变量和混杂变量/non_collinear_representative_vars_operability.csv
  - 专家知识: 东鞍山烧结厂选矿专家知识.txt
  - 建模宽表: data/modeling_dataset_final.parquet

默认输出：
  - 数据预处理/结果/expert_variable_reduction_analysis.csv
  - 数据预处理/结果/expert_variable_reduction_report.md
  - 数据预处理/结果/expert_reduced_variables_<line>.csv
  - data/modeling_dataset_final_expert_reduced.parquet
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = SCRIPT_DIR / "结果"
RESULT_DIR.mkdir(exist_ok=True)
MAX_PREVIEW_VARS = 10
EXPERT_EXCERPT_MAX_LENGTH = 1000

DEFAULT_VARIABLES_CSV = DATA_DIR / "操作变量和混杂变量" / "non_collinear_representative_vars_operability.csv"
DEFAULT_EXPERT_DOC = REPO_ROOT / "东鞍山烧结厂选矿专家知识.txt"
DEFAULT_INPUT_PARQUET = DATA_DIR / "modeling_dataset_final.parquet"
DEFAULT_ANALYSIS_CSV = RESULT_DIR / "expert_variable_reduction_analysis.csv"
DEFAULT_REPORT_MD = RESULT_DIR / "expert_variable_reduction_report.md"
DEFAULT_STAGE_MD = REPO_ROOT / "数据预处理" / "数据与处理结果-分阶段-去共线性后" / "non_collinear_representative_vars_annotated.md"

LINE_TO_GROUPS = {
    "all": {"A", "B", "C"},
    "xin1": {"A", "C"},
    "xin2": {"B", "C"},
}

STAGE_NAMES = {
    0: "公共动力/上游边界",
    1: "药剂合成",
    2: "磁选分离",
    3: "塔磨分级",
    4: "浓缩缓冲",
    5: "调浆激发",
    6: "浮选网络",
    7: "尾矿收尾",
    8: "精矿脱水",
}

STAGE_HEURISTICS = [
    (0, ("CXXY_", "FX_GFJ", "MC1_AH", "MC2_RC"), ()),
    (1, ("FX_LT_6", "FX_HV_63", "FX_JBC3", "FX_JBC19", "FX_LT_601"), ()),
    (2, ("MC2_QC", "MC2_PET", "MC2_JYB", "EY", "CXG_", "XHCJ_", "LHCJ_"), ()),
    (3, ("MC1_GKB", "MC1_FET", "MC1_FV", "MC1_LV", "MC1_TM", "MC1_LET", "MC2_YLB", "MC2_ZJB"), ()),
    (4, ("FX_P", "MC2_NSJ", "MC2_CQC"), ()),
    (5, ("FX_AT_", "FX_HGB", "FX_LGB2", "FX_TV_11", "FX_TV_21", "FX_FT_17", "FX_FT_27"), ()),
    (6, ("FX_X1", "FX_X2", "FX_FXJ", "FX_ZJB1", "FX_DT_", "FX_AH"), ()),
    (7, (), ("尾矿", "事故池", "尾矿泵")),
    (8, (), ("精矿", "压滤", "脱水")),
]

EXPERT_PRINCIPLES = [
    "原矿品位、磁性铁、亚铁、碳酸铁是前馈边界条件，应优先保留。",
    "磁选区的核心控制变量是励磁电压/电流、尾矿阀门、给矿/冲矿压力与液位；纯电气健康量可降维。",
    "塔磨/旋流器回路重点保留给矿流量、补水阀位、泵池液位、泵频/压力；轴承/减速机健康量可降维。",
    "浮选区重点保留分矿给矿量、药剂泵/阀、pH、矿浆浓度、气量、泡沫层厚度、关键泵池液位。",
    "报警、配电室频率/功率因数、泛化相电流、缺少语义的空描述变量优先删除。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="专家知识驱动的变量降维与 parquet 生成工具")
    parser.add_argument("--variables-csv", default=str(DEFAULT_VARIABLES_CSV))
    parser.add_argument("--expert-doc", default=str(DEFAULT_EXPERT_DOC))
    parser.add_argument("--annotated-md", default=str(DEFAULT_STAGE_MD))
    parser.add_argument("--input-parquet", default=str(DEFAULT_INPUT_PARQUET))
    parser.add_argument("--output-parquet", default="")
    parser.add_argument("--analysis-csv", default=str(DEFAULT_ANALYSIS_CSV))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--line", choices=["auto", "all", "xin1", "xin2"], default="auto")
    return parser.parse_args()


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(k in text for k in keywords)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def infer_line_mode(line_arg: str, parquet_path: Path) -> str:
    if line_arg != "auto":
        return line_arg
    name = parquet_path.name.lower()
    if "xin1" in name:
        return "xin1"
    if "xin2" in name:
        return "xin2"
    return "all"


def load_stage_mapping(md_path: Path) -> dict[str, int]:
    if not md_path.exists():
        return {}
    content = md_path.read_text(encoding="utf-8-sig", errors="replace")
    sections = re.split(r"## Stage (\S+)", content)
    mapping: dict[str, int] = {}
    idx = 1
    while idx + 1 < len(sections):
        stage_text = sections[idx]
        try:
            stage_id = int(stage_text)
        except ValueError:
            print(f"[warn] annotated markdown 中存在无法解析的 Stage 标题: {stage_text!r}")
            idx += 2
            continue
        body = sections[idx + 1]
        for match in re.finditer(r"\|\s*\d+\s*\|\s*`?(\S+?)`?\s*\|", body):
            mapping[match.group(1)] = stage_id
        idx += 2
    return mapping


def infer_stage(name: str, desc: str, known_stage: int | None) -> tuple[int | None, str]:
    if known_stage is not None:
        return known_stage, "annotated_md"

    upper = name.upper()
    text = f"{upper} {desc}"

    for stage_id, prefixes, keywords in STAGE_HEURISTICS:
        if prefixes and upper.startswith(prefixes):
            return stage_id, "heuristic"
        if keywords and contains_any(text, keywords):
            return stage_id, "heuristic"
    return None, "unknown"


def classify_signal(name: str, desc: str, stage_id: int | None) -> tuple[str, str, bool]:
    upper = name.upper()
    text = f"{upper} {desc}"

    if contains_any(text, ["报警", "_BJ", "_ZT", "状态"]):
        return "alarm_or_status", "报警/状态信号不直接表征工艺机理，优先删除。", False

    if contains_any(text, ["电网频率", "总功率因素", "总瞬时有功功率", "总瞬时无功功率", "A相电流", "B相电流", "C相电流"]):
        return "power_supply_health", "配电/供电健康量与主工艺因果链距离较远，优先删除。", False

    if contains_any(text, ["线圈温度", "滑动轴承", "减速机油池温度", "主电机电流"]) and stage_id == 3:
        return "equipment_health", "塔磨设备健康量可作为运维信号，但对工艺降维优先级较低。", False

    if contains_any(text, ["原矿", "品位", "磁性铁", "亚铁", "碳酸铁"]) and upper.startswith(("CXXY_", "EY", "CXG_", "XHCJ_", "LHCJ_")):
        return "boundary_quality", "专家知识明确指出该类品位/矿相指标是前馈边界条件。", True

    if contains_any(text, ["给定", "设定", "AO", "F_W"]) and contains_any(text, ["阀", "励磁", "加水", "NaOH", "CaO", "TD-II", "K6-1", "给矿"]):
        return "direct_control", "该变量属于可下发的控制指令或设定值，必须保留。", True

    if stage_id == 2 and contains_any(text, ["励磁电压值", "励磁电流值", "尾矿阀门实际开度", "冲矿水", "选矿液位", "出口管道压力"]):
        return "magnetic_core_state", "专家知识将其列为磁选区核心控制/状态量。", True

    if stage_id == 3 and contains_any(text, ["给矿管道流量", "加水管道流量", "泵池液位", "阀位给定", "给矿泵频率", "溢流泵频率", "沉砂加水"]):
        return "grinding_classification_state", "专家知识将其列为塔磨/旋流器主回路变量。", True

    if stage_id in {5, 6} and contains_any(text, ["PH值", "浓度", "泡沫层厚度", "气量", "给矿泵", "入矿管道流量", "药", "NaOH", "CaO", "TD-II", "K6-1", "泵池液位"]):
        return "flotation_core_state", "专家知识将其列为浮选回路关键状态/加药/气量变量。", True

    if contains_any(text, ["液位", "流量", "压力", "浓度", "PH值", "泡沫层厚度", "气量", "励磁电压值", "励磁电流值"]):
        return "process_state", "该变量直接表征矿浆流态、药剂环境或分选状态，建议保留。", True

    if contains_any(text, ["阀位反馈", "开度实际值", "蝶阀", "电动阀开度", "液位阀"]) and not contains_any(text, ["尾矿阀门实际开度", "阀位给定", "设定"]):
        return "actuator_feedback_duplicate", "阀位反馈多为执行器回显，可由设定/流量/液位间接表征，优先降维。", False

    if contains_any(text, ["浮选机", "渣浆泵", "螺杆泵", "化工泵", "鼓风机", "搅拌槽"]) and contains_any(text, ["电流", "频率反馈"]) and stage_id in {5, 6}:
        return "support_equipment_load", "设备电流/频率更多反映机组负荷，降维时优先让位于流量、液位、泡沫和药剂量。", False

    # classify_signal() 由 build_analysis() 调用；传入前 Description_CN 已经 normalize_text() 归一化。
    if desc == "":
        return "unidentified_signal", "缺少中文描述且无法从命名稳定识别工艺语义，保守删除。", False

    return "weak_aux_signal", "辅助信号与核心工艺链路关系较弱，建议删除。", False


def build_analysis(df: pd.DataFrame, stage_map: dict[str, int]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        name = normalize_text(row["Non_Collinear_Representative"])
        desc = normalize_text(row.get("Description_CN", ""))
        group = normalize_text(row.get("Group", "")) or "C"
        known_stage = stage_map.get(name)
        stage_id, stage_source = infer_stage(name, desc, known_stage)
        signal_class, decision_reason, keep_all = classify_signal(name, desc, stage_id)
        rows.append(
            {
                "Variable_Name": name,
                "Group": group,
                "Description_CN": desc,
                "Original_Operability": normalize_text(row.get("Operability", "")),
                "Stage_ID": stage_id if stage_id is not None else "",
                "Stage_Name": STAGE_NAMES.get(stage_id, "未识别"),
                "Stage_Source": stage_source,
                "Signal_Class": signal_class,
                "Keep_All": "keep" if keep_all else "drop",
                "Keep_xin1": "keep" if keep_all and group in LINE_TO_GROUPS["xin1"] else "drop",
                "Keep_xin2": "keep" if keep_all and group in LINE_TO_GROUPS["xin2"] else "drop",
                "Decision_Reason": decision_reason,
            }
        )
    analysis = pd.DataFrame(rows)
    return analysis.sort_values(["Keep_All", "Stage_ID", "Group", "Variable_Name"], ascending=[False, True, True, True])


def render_markdown(
    analysis: pd.DataFrame,
    expert_doc_path: Path,
    input_parquet: Path,
    output_parquet: Path,
    line_mode: str,
    expert_excerpt: str,
) -> str:
    keep_col = "Keep_All" if line_mode == "all" else f"Keep_{line_mode}"
    kept = analysis[analysis[keep_col] == "keep"]
    dropped = analysis[analysis[keep_col] == "drop"]

    lines = [
        "# 专家知识驱动的变量降维分析报告",
        "",
        f"- 专家文档：`{expert_doc_path}`",
        f"- 输入 parquet：`{input_parquet}`",
        f"- 输出 parquet：`{output_parquet}`",
        f"- 降维模式：`{line_mode}`",
        f"- 原始变量数：**{len(analysis)}**",
        f"- 保留变量数：**{len(kept)}**",
        f"- 删除变量数：**{len(dropped)}**",
        "",
        "## 专家知识抽取要点",
        "",
    ]
    lines.extend([f"- {item}" for item in EXPERT_PRINCIPLES])
    if expert_excerpt:
        lines.extend(["", "### 专家文档片段", "", expert_excerpt, ""])

    lines.extend(
        [
            "## 分阶段保留统计",
            "",
            "| Stage | 名称 | 保留数 | 删除数 |",
            "|---|---|---:|---:|",
        ]
    )

    for stage_id in sorted({sid for sid in analysis["Stage_ID"].tolist() if sid != ""}):
        sub = analysis[analysis["Stage_ID"] == stage_id]
        lines.append(
            f"| {stage_id} | {STAGE_NAMES.get(stage_id, '未识别')} | "
            f"{(sub[keep_col] == 'keep').sum()} | {(sub[keep_col] == 'drop').sum()} |"
        )

    lines.extend(
        [
            "",
            "## 逐变量分析",
            "",
            "| 变量名 | Group | Stage | 信号类型 | 结论 | 说明 |",
            "|---|---|---|---|---|---|",
        ]
    )

    for _, row in analysis.iterrows():
        stage_label = row["Stage_ID"] if row["Stage_ID"] != "" else "?"
        reason = str(row["Decision_Reason"]).replace("\n", " ").replace("|", "｜")
        lines.append(
            f"| `{row['Variable_Name']}` | {row['Group']} | {stage_label} | {row['Signal_Class']} | "
            f"{row[keep_col]} | {reason} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    variables_csv = Path(args.variables_csv)
    expert_doc = Path(args.expert_doc)
    annotated_md = Path(args.annotated_md)
    input_parquet = Path(args.input_parquet)
    output_parquet = Path(args.output_parquet) if args.output_parquet else input_parquet.with_name(
        f"{input_parquet.stem}_expert_reduced.parquet"
    )
    analysis_csv = Path(args.analysis_csv)
    report_md = Path(args.report_md)

    if not variables_csv.exists():
        raise FileNotFoundError(f"找不到变量清单：{variables_csv}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"找不到输入 parquet：{input_parquet}")

    line_mode = infer_line_mode(args.line, input_parquet)
    stage_map = load_stage_mapping(annotated_md)
    variables_df = pd.read_csv(variables_csv, encoding="utf-8-sig")
    analysis = build_analysis(variables_df, stage_map)
    keep_col = "Keep_All" if line_mode == "all" else f"Keep_{line_mode}"

    df_parquet = pd.read_parquet(input_parquet)
    candidate_vars = analysis.loc[analysis[keep_col] == "keep", "Variable_Name"].tolist()
    missing_vars = [v for v in candidate_vars if v not in df_parquet.columns]
    if missing_vars:
        preview = ", ".join(missing_vars[:MAX_PREVIEW_VARS])
        suffix = " ..." if len(missing_vars) > MAX_PREVIEW_VARS else ""
        print(f"[warn] {len(missing_vars)} 个保留变量未出现在输入 parquet 中：{preview}{suffix}")
    selected_vars = [v for v in candidate_vars if v in df_parquet.columns]
    target_cols = [c for c in df_parquet.columns if c.lower().startswith("y_")]
    reduced_cols = selected_vars + [c for c in target_cols if c not in selected_vars]
    reduced_df = df_parquet.loc[:, reduced_cols]

    analysis = analysis.copy()
    analysis["Present_In_Input_Parquet"] = analysis["Variable_Name"].isin(df_parquet.columns).map({True: "yes", False: "no"})

    expert_excerpt = ""
    if expert_doc.exists():
        text = expert_doc.read_text(encoding="utf-8", errors="replace")
        expert_excerpt = text[:EXPERT_EXCERPT_MAX_LENGTH].strip()
        if len(text) > EXPERT_EXCERPT_MAX_LENGTH:
            expert_excerpt += "..."

    analysis_csv.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    analysis.to_csv(analysis_csv, index=False, encoding="utf-8-sig")
    reduced_df.to_parquet(output_parquet)

    selected_vars_path = RESULT_DIR / f"expert_reduced_variables_{line_mode}.csv"
    pd.DataFrame({"Variable_Name": selected_vars}).to_csv(selected_vars_path, index=False, encoding="utf-8-sig")

    report_md.write_text(
        render_markdown(
            analysis=analysis,
            expert_doc_path=expert_doc,
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            line_mode=line_mode,
            expert_excerpt=expert_excerpt,
        ),
        encoding="utf-8-sig",
    )

    print(f"变量分析输出: {analysis_csv}")
    print(f"分析报告输出: {report_md}")
    print(f"保留变量清单: {selected_vars_path}")
    print(f"降维 parquet 输出: {output_parquet}")
    print(f"当前模式 `{line_mode}` 保留变量 {len(selected_vars)} 个，目标列 {len(target_cols)} 个。")


if __name__ == "__main__":
    main()
