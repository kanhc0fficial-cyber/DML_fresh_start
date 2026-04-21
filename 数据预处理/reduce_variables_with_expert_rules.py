"""
专家知识驱动的“聚合优先”降维脚本。

核心原则：
1. 先做工艺语义聚合，再做少量删除。
2. 不靠正则推断变量含义，而是基于：
   - 专家知识文档
   - 当前变量清单
   - 当前 parquet 中已存在的精确变量名 / 精确注释
3. 对用户明确要求保留的流程概念，即使当前数据源缺失，也在输出中保留占位列（全 NaN），
   以保证概念层 schema 不丢失。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = SCRIPT_DIR / "结果"
RESULT_DIR.mkdir(exist_ok=True)

DEFAULT_VARIABLES_CSV = DATA_DIR / "操作变量和混杂变量" / "non_collinear_representative_vars_operability.csv"
DEFAULT_OUTPUT_ABC_CSV = DATA_DIR / "操作变量和混杂变量" / "output_ABC_分类合集_带变化标签.csv"
DEFAULT_INPUT_PARQUET = DATA_DIR / "modeling_dataset_final.parquet"
DEFAULT_EXPERT_DOC = REPO_ROOT / "东鞍山烧结厂选矿专家知识.txt"
DEFAULT_ANALYSIS_CSV = RESULT_DIR / "expert_variable_reduction_analysis.csv"
DEFAULT_REPORT_MD = RESULT_DIR / "expert_variable_reduction_report.md"
DEFAULT_CONCEPT_CSV = RESULT_DIR / "expert_aggregation_concepts.csv"


@dataclass(frozen=True)
class ConceptSpec:
    feature_name: str
    feature_cn: str
    process: str
    role: str
    method: str
    source_vars: tuple[str, ...]
    required: bool
    reason: str
    missing_policy: str = "placeholder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="专家知识驱动的聚合优先降维脚本")
    parser.add_argument("--variables-csv", default=str(DEFAULT_VARIABLES_CSV))
    parser.add_argument("--abc-csv", default=str(DEFAULT_OUTPUT_ABC_CSV))
    parser.add_argument("--input-parquet", default=str(DEFAULT_INPUT_PARQUET))
    parser.add_argument("--output-parquet", default="")
    parser.add_argument("--analysis-csv", default=str(DEFAULT_ANALYSIS_CSV))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--concept-csv", default=str(DEFAULT_CONCEPT_CSV))
    parser.add_argument("--expert-doc", default=str(DEFAULT_EXPERT_DOC))
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def collect_exact_comment_vars(
    comment_to_names: dict[str, list[str]],
    parquet_cols: set[str],
    comments: list[str],
) -> list[str]:
    result: list[str] = []
    for comment in comments:
        for name in comment_to_names.get(comment, []):
            if name in parquet_cols and name not in result:
                result.append(name)
    return result


def build_comment_maps(abc_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    name_to_comment: dict[str, str] = {}
    comment_to_names: dict[str, list[str]] = {}
    name_to_group: dict[str, str] = {}
    for _, row in abc_df.iterrows():
        name = normalize_text(row.get("NAME", ""))
        if not name:
            continue
        comment = normalize_text(row.get("COMMENT", ""))
        group = normalize_text(row.get("Group", ""))
        if comment:
            name_to_comment[name] = comment
            comment_to_names.setdefault(comment, []).append(name)
        if group:
            name_to_group[name] = group
    return name_to_comment, comment_to_names, name_to_group


def aggregate_series(df: pd.DataFrame, vars_in_df: list[str], method: str) -> pd.Series:
    if not vars_in_df:
        return pd.Series(pd.NA, index=df.index, dtype="float64")

    sub = df[vars_in_df].apply(pd.to_numeric, errors="coerce")
    if method == "first":
        return sub.iloc[:, 0]
    if method == "mean":
        return sub.mean(axis=1)
    if method == "sum":
        return sub.sum(axis=1, min_count=1)
    if method == "max":
        return sub.max(axis=1)
    if method == "any":
        return sub.max(axis=1, skipna=True)
    if method == "delta_sum":
        delta = sub.diff().clip(lower=0)
        return delta.sum(axis=1, min_count=1)
    raise ValueError(f"未知聚合方法: {method}")


def build_concepts(
    parquet_cols: set[str],
    comment_to_names: dict[str, list[str]],
) -> list[ConceptSpec]:
    concepts: list[ConceptSpec] = []

    def add(
        feature_name: str,
        feature_cn: str,
        process: str,
        role: str,
        method: str,
        source_vars: list[str],
        reason: str,
        required: bool = True,
        missing_policy: str = "placeholder",
    ) -> None:
        concepts.append(
            ConceptSpec(
                feature_name=feature_name,
                feature_cn=feature_cn,
                process=process,
                role=role,
                method=method,
                source_vars=tuple(v for v in source_vars if v not in {"", None}),
                required=required,
                reason=reason,
                missing_policy=missing_policy,
            )
        )

    # 上游边界
    add("raw_ore_grade", "原矿品位", "边界条件", "原矿品位", "first", ["CXXY_PW"], "专家知识明确要求保留原矿品位。")
    add("raw_ore_magnetic_iron", "原矿磁性铁", "边界条件", "磁性铁", "first", ["CXXY_CXN"], "专家知识明确要求保留原矿磁性铁。")
    add("raw_ore_ferrous_iron", "原矿亚铁", "边界条件", "亚铁", "first", ["CXXY_YT"], "专家知识明确要求保留原矿亚铁。")
    add("raw_ore_siderite", "原矿碳酸铁", "边界条件", "碳酸铁", "first", ["CXXY_TSN"], "专家知识明确要求保留原矿碳酸铁。")

    # 破碎：当前数据源缺失，保留 schema
    for feature_name, feature_cn in [
        ("agg_crushing_belt_status", "破碎_皮带启停"),
        ("agg_crushing_belt_frequency", "破碎_皮带频率"),
        ("agg_crushing_level_height", "破碎_料位高度"),
        ("agg_crushing_feed_rate", "破碎_给矿量"),
        ("agg_crushing_cumulative_amount", "破碎_累积量"),
    ]:
        add(feature_name, feature_cn, "破碎", feature_cn.split("_", 1)[1], "mean", [], "用户要求必须保留，但当前 parquet 与变量清单中未发现对应源变量。")

    # 球磨：当前数据源同样缺关键变量，保留 schema
    for feature_name, feature_cn in [
        ("agg_grinding_cyclone_pressure", "球磨_旋流器压力"),
        ("agg_grinding_sand_add_water", "球磨_沉沙补加水"),
        ("agg_grinding_cyclone_pool_level", "球磨_旋流器泵池液位"),
        ("agg_grinding_cyclone_pump_frequency", "球磨_旋流器泵频"),
        ("agg_grinding_ball_mill_feed_rate", "球磨_球磨给矿量"),
        ("agg_grinding_pendulum_state", "球磨_摆式状态"),
    ]:
        add(feature_name, feature_cn, "球磨", feature_cn.split("_", 1)[1], "mean", [], "用户要求必须保留，但当前 parquet 与变量清单中未发现对应源变量。")

    # 磁选：使用精确注释归集
    add(
        "agg_magnetic_excitation_voltage",
        "磁选_励磁电压",
        "磁选",
        "励磁电压",
        "mean",
        collect_exact_comment_vars(comment_to_names, parquet_cols, ["励磁电压值"]),
        "多个强磁机的励磁电压属于并联设备同类控制量，按专家知识做概念层聚合。",
    )
    add(
        "agg_magnetic_excitation_current",
        "磁选_励磁电流",
        "磁选",
        "励磁电流",
        "mean",
        collect_exact_comment_vars(comment_to_names, parquet_cols, ["励磁电流值"]),
        "多个强磁机的励磁电流属于并联设备同类控制量，按专家知识做概念层聚合。",
    )
    add(
        "agg_magnetic_level",
        "磁选_液位",
        "磁选",
        "液位",
        "mean",
        collect_exact_comment_vars(comment_to_names, parquet_cols, ["选矿液位"]),
        "液位虽存在不准问题，但用户明确要求保留，因此按并联槽体做均值聚合。",
    )
    add(
        "agg_magnetic_coil_temperature",
        "磁选_线圈温度",
        "磁选",
        "线圈温度",
        "mean",
        collect_exact_comment_vars(comment_to_names, parquet_cols, ["线圈温度值"]),
        "用户要求保留线圈温度，按并联设备聚合为单一过程温度特征。",
    )
    add(
        "agg_magnetic_motor_current_proxy",
        "磁选_电机电流代理",
        "磁选",
        "电机电流",
        "mean",
        [
            "MC2_RC101_DL_AI", "MC2_RC102_DL_AI", "MC2_RC103_DL_AI", "MC2_RC104_DL_AI",
            "MC2_RC105_DL_AI", "MC2_RC106_DL_AI", "MC2_RC107_DL_AI", "MC2_RC109_DL_AI",
            "MC2_RC110_DL_AI", "MC2_RC111_DL_AI", "MC2_RC112_DL_AI", "MC2_JYB2_DL_AI",
        ],
        "当前数据中未找到明确命名为磁选电机电流的单列，使用现场已存在的相关 A 相电流信号做代理聚合。",
    )
    add(
        "agg_magnetic_tailings_valve_opening",
        "磁选_尾矿阀门开度",
        "磁选",
        "尾矿阀门",
        "mean",
        collect_exact_comment_vars(comment_to_names, parquet_cols, ["1#尾矿阀门实际开度反馈值", "2#尾矿阀门实际开度反馈值"]),
        "专家知识明确要求保留尾矿阀门调节信息，按并联磁选机统一聚合。",
        required=False,
    )
    add(
        "agg_magnetic_flush_water_pressure",
        "磁选_冲矿水压力",
        "磁选",
        "冲矿水压力",
        "mean",
        collect_exact_comment_vars(comment_to_names, parquet_cols, ["冲矿水压力值", "冲矿水入口压力值", "冲矿水出口压力值"]),
        "冲矿水压力是磁选区重要流体边界条件，按专家知识保留为辅助聚合特征。",
        required=False,
    )
    add(
        "agg_magnetic_blowdown_valve_opening",
        "磁选_排污阀门开度",
        "磁选",
        "排污阀门",
        "mean",
        collect_exact_comment_vars(comment_to_names, parquet_cols, ["排污阀门开度实际值"]),
        "排污阀门开度反映磁选回路排尾状态，作为辅助聚合特征保留。",
        required=False,
    )

    # 塔磨
    add(
        "agg_tower_pump_pool_makeup_water",
        "塔磨_泵池补加水量",
        "塔磨",
        "泵池补加水量",
        "mean",
        ["MC1_FET503_AI"],
        "当前数据中仅发现明确的泵池加水流量变量，按用户要求直接保留。",
    )
    add(
        "agg_tower_pump_pool_level",
        "塔磨_泵池液位",
        "塔磨",
        "泵池液位",
        "mean",
        ["MC1_LET101_AI", "MC1_LET301_AI", "MC1_LET501_AI"],
        "多个给矿泵池液位属于并行塔磨回路，按概念聚合。",
    )
    add(
        "agg_tower_cyclone_feed_pump_frequency",
        "塔磨_旋流器泵池泵频",
        "塔磨",
        "旋流器泵池泵频",
        "mean",
        [
            "MC1_GKB701_HZ", "MC1_GKB702_HZ", "MC1_GKB703_HZ", "MC1_GKB704_HZ",
            "MC1_GKB705_HZ", "MC1_GKB706_HZ", "MC1_GKB707_HZ", "MC1_GKB708_HZ",
            "MC1_GKB709_HZ", "MC1_GKB710_HZ", "MC1_GKB711_HZ", "MC1_GKB712_HZ",
        ],
        "塔磨区多个旋流器给矿泵频率属于同一控制层，按专家知识做均值聚合。",
    )
    add(
        "agg_tower_cyclone_overflow_pump_frequency",
        "塔磨_旋流器溢流泵频",
        "塔磨",
        "旋流器溢流泵频",
        "mean",
        [
            "MC2_YLB802_HZ", "MC2_YLB803_HZ", "MC2_YLB804_HZ", "MC2_YLB805_HZ",
            "MC2_YLB806_HZ", "MC2_YLB807_HZ", "MC2_YLB808_HZ", "MC2_YLB809_HZ",
            "MC2_YLB810_HZ", "MC2_YLB811_HZ", "MC2_YLB812_HZ",
        ],
        "溢流泵频率用于补强塔磨旋流器回路泵频表征。",
        required=False,
    )
    add(
        "agg_tower_cyclone_add_water",
        "塔磨_旋流器补加水量",
        "塔磨",
        "旋流器补加水量",
        "mean",
        ["MC1_FET101_AI", "MC1_FET201_AI", "MC1_FET301_AI", "MC1_FET401_AI", "MC1_FET501_AI", "MC1_FET601_AI"],
        "旋流器沉砂加水流量为用户指定必须保留项，按并联支路聚合。",
    )
    add(
        "agg_tower_cyclone_switch_state",
        "塔磨_旋流器开关状态",
        "塔磨",
        "旋流器开关状态",
        "any",
        [],
        "用户要求必须保留，但当前 parquet 与变量清单中未发现明确的旋流器开关状态列。",
    )
    add(
        "agg_tower_motor_current",
        "塔磨_主电机电流",
        "塔磨",
        "主电机电流",
        "mean",
        [
            "MC1_TM201_ZDJ_DL_AI", "MC1_TM202_ZDJ_DL_AI", "MC1_TM203_ZDJ_DL_AI",
            "MC1_TM204_ZDJ_DL_AI", "MC1_TM205_ZDJ_DL_AI", "MC1_TM206_ZDJ_DL_AI",
        ],
        "塔磨主电机电流能反映载荷水平，作为辅助聚合特征保留。",
        required=False,
    )

    # 浮选
    add(
        "agg_flotation_feed_rate",
        "浮选_给矿量",
        "浮选",
        "给矿量",
        "sum",
        ["FX_FT_1702", "FX_FT_2601", "FX_FT_2602", "FX_FT_2701", "FX_FT_2702"],
        "用户要求保留浮选给矿量，按各支路流量求和形成总给矿概念特征。",
    )
    add(
        "agg_flotation_reagent_cao_dose",
        "浮选_CaO加药量代理",
        "浮选",
        "CaO加药量",
        "mean",
        ["FX_HGB2201_F", "FX_HGB2201_F_W", "FX_HGB2202_F", "FX_HGB2203_F", "FX_HGB2204_F", "FX_HGB2204_F_W"],
        "CaO 当前可直接观测的是泵频/设定，按专家知识将其聚合为加药量代理特征。",
    )
    add(
        "agg_flotation_reagent_naoh_dose",
        "浮选_NaOH加药量代理",
        "浮选",
        "NaOH加药量",
        "mean",
        ["FX_LGB2701_F", "FX_LGB2702_F", "FX_LGB2702_F_W"],
        "NaOH 当前可直接观测的是泵频/设定，按专家知识将其聚合为加药量代理特征。",
    )
    add(
        "agg_flotation_reagent_tdii_dose",
        "浮选_TDII加药量代理",
        "浮选",
        "TD-II加药量",
        "mean",
        [
            "FX_LGB2301_F", "FX_LGB2301_F_W", "FX_LGB2302_F", "FX_LGB2302_F_W",
            "FX_LGB2303_F", "FX_LGB2303_F_W", "FX_LGB2304_F", "FX_LGB2304_F_W",
            "FX_LGB2401_F", "FX_LGB2401_F_W", "FX_LGB2402_F", "FX_LGB2402_F_W",
            "FX_LGB2403_F", "FX_LGB2403_F_W", "FX_LGB2404_F", "FX_LGB2404_F_W",
        ],
        "TD-II 是关键捕收剂，多个粗选/精选支路按泵频和设定做概念层聚合。",
    )
    add(
        "agg_flotation_reagent_k6_dose",
        "浮选_K6-1加药量代理",
        "浮选",
        "K6-1加药量",
        "mean",
        ["FX_LGB2502_F", "FX_LGB2502_F_W", "FX_LGB2602_F", "FX_LGB2602_F_W"],
        "K6-1 为关键抑制剂支路，按泵频和设定做概念层聚合。",
    )
    add(
        "agg_flotation_froth_thickness",
        "浮选_泡沫厚度",
        "浮选",
        "泡沫厚度",
        "mean",
        [
            "FX_X2CX3_AI1", "FX_X2JX_AI1", "FX_X2SX1_AI1", "FX_X2SX2_AI1", "FX_X2CX1_AI1", "FX_X2SX3_AI1",
            "FX_X1SX2_AI1", "FX_X1SX3_AI1", "FX_X1CX1_AI1", "FX_X1CX2_AI1", "FX_X1CX3_AI1", "FX_X1JX_AI1", "FX_X2CX2_AI1",
        ],
        "多个粗选/精选/扫选槽的泡沫厚度共同表征浮选泡沫态，按并联槽体聚合。",
    )
    add(
        "agg_flotation_ph",
        "浮选_PH值",
        "浮选",
        "PH值",
        "mean",
        ["FX_AT_2102", "FX_AT_2103", "FX_AT_2104", "FX_AT_1101", "FX_AT_1102", "FX_AT_1103", "FX_AT_1104"],
        "pH 是浮选药剂体系核心变量，按各测点均值聚合。",
    )
    add(
        "agg_flotation_airflow_actual",
        "浮选_充气量实际值",
        "浮选",
        "充气量",
        "mean",
        [
            "FX_X2CX3_AI5", "FX_X2CX3_AI9", "FX_X2JX_AI5", "FX_X2JX_AI9", "FX_X2JX_AI13",
            "FX_X2SX1_AI5", "FX_X2SX1_AI9", "FX_X2SX1_AI13", "FX_X2CX1_AI5", "FX_X2SX2_AI5", "FX_X2SX2_AI9",
            "FX_X2SX3_AI5", "FX_X2SX3_AI9", "FX_X1SX1_AI5", "FX_X1SX1_AI9", "FX_X1SX1_AI13",
            "FX_X1SX2_AI5", "FX_X1SX2_AI9", "FX_X1SX3_AI5", "FX_X1SX3_AI9", "FX_X1CX1_AI5",
            "FX_X1CX1_AI9", "FX_X1CX2_AI5", "FX_X1CX2_AI9", "FX_X1CX3_AI5", "FX_X1CX3_AI9",
            "FX_X1JX_AI5", "FX_X1JX_AI9", "FX_X1JX_AI13", "FX_X2CX1_AI9", "FX_X2CX2_AI5", "FX_X2CX2_AI9",
        ],
        "浮选充气量是用户要求的必保概念，按所有可观测实际气量点聚合。",
    )
    add(
        "agg_flotation_airflow_setpoint",
        "浮选_充气量设定值",
        "浮选",
        "充气量设定",
        "mean",
        [
            "FX_X2JX_AI11", "FX_X2JX_AI15", "FX_X2SX1_AI11", "FX_X2SX1_AI15",
            "FX_X1JX_AI15", "FX_X1SX1_AI11", "FX_X1SX1_AI15", "FX_X1SX3_AI6",
            "FX_X1CX1_AI6", "FX_X1CX2_AI6", "FX_X1CX3_AI6", "FX_X1JX_AI11",
        ],
        "设定值单列保留，以区分执行结果与控制意图。",
        required=False,
    )
    add(
        "agg_flotation_valve_opening",
        "浮选_阀门开度",
        "浮选",
        "阀门开度",
        "mean",
        [
            "FX_X2CX2_AI27", "FX_X2CX3_AI7", "FX_X2CX3_AI11", "FX_X2JX_AI3", "FX_X2CX1_AI7",
            "FX_X2SX1_AI3", "FX_X2SX1_AI7", "FX_X2SX1_AI38", "FX_X2SX2_AI4", "FX_X2CX1_AI12",
            "FX_X2CX2_AI3", "FX_X2CX2_AI12", "FX_X1SX1_AI7", "FX_X1SX2_AI3", "FX_X1SX3_AI4",
            "FX_X1SX3_AI7", "FX_X1SX3_AI12", "FX_X1CX3_AI3", "FX_X1JX_AI3",
        ],
        "用户要求保留浮选阀门开度，按可观测阀门开度/设定聚合。",
    )
    add(
        "agg_flotation_tailings_pool_level",
        "浮选_尾矿相关泵池液位",
        "浮选",
        "尾矿泵池液位",
        "mean",
        ["FX_LT_1603", "FX_LT_1604", "FX_LT_2602", "FX_LT_2603", "FX_LT_2604", "FX_LT_1605", "FX_LT_2605", "FX_LT_1701"],
        "尾矿/中矿/溢流泵池液位反映浮选网络缓冲状态，作为辅助聚合特征保留。",
        required=False,
    )
    add(
        "agg_flotation_tailings_froth_image",
        "浮选_尾矿泡沫图像",
        "浮选",
        "尾矿泡沫图像",
        "mean",
        [],
        "专家知识要求该概念，但当前 parquet 中没有相机/RGB图像列，保留占位 schema。",
    )

    return concepts


def build_metadata(
    variables_df: pd.DataFrame,
    abc_df: pd.DataFrame,
    parquet_cols: set[str],
) -> pd.DataFrame:
    name_to_comment, _, name_to_group = build_comment_maps(abc_df)
    var_map: dict[str, dict[str, str]] = {}

    for _, row in variables_df.iterrows():
        name = normalize_text(row.get("Non_Collinear_Representative", ""))
        if not name:
            continue
        var_map[name] = {
            "Description_CN": normalize_text(row.get("Description_CN", "")) or name_to_comment.get(name, ""),
            "Group": normalize_text(row.get("Group", "")) or name_to_group.get(name, ""),
            "Source": "representative_csv",
        }

    for name in parquet_cols:
        if name.startswith("y_"):
            continue
        if name not in var_map:
            var_map[name] = {
                "Description_CN": name_to_comment.get(name, ""),
                "Group": name_to_group.get(name, ""),
                "Source": "parquet_only",
            }

    rows = [{"Variable_Name": k, **v} for k, v in sorted(var_map.items())]
    return pd.DataFrame(rows)


def build_analysis(metadata_df: pd.DataFrame, concepts: list[ConceptSpec], parquet_cols: set[str]) -> pd.DataFrame:
    concept_rows = []
    variable_to_concept: dict[str, ConceptSpec] = {}
    for concept in concepts:
        for var in concept.source_vars:
            if var in variable_to_concept:
                raise ValueError(f"变量 {var} 被重复分配到多个聚合概念。")
            variable_to_concept[var] = concept
        available_count = sum(1 for v in concept.source_vars if v in parquet_cols)
        concept_rows.append(
            {
                "feature_name": concept.feature_name,
                "feature_cn": concept.feature_cn,
                "process": concept.process,
                "role": concept.role,
                "method": concept.method,
                "source_var_count": len(concept.source_vars),
                "available_var_count": available_count,
                "required": "yes" if concept.required else "no",
                "reason": concept.reason,
                "placeholder_only": "yes" if len(concept.source_vars) == 0 else "no",
            }
        )

    rows = []
    for _, row in metadata_df.sort_values("Variable_Name").iterrows():
        var = row["Variable_Name"]
        concept = variable_to_concept.get(var)
        if concept is None:
            action = "drop"
            target = ""
            method = ""
            reason = "未进入专家定义的核心概念聚合层，且当前版本以概念级聚合为主。"
            process = ""
            role = ""
        else:
            action = "aggregate" if len(concept.source_vars) > 1 else "keep_single"
            target = concept.feature_name
            method = concept.method
            reason = concept.reason
            process = concept.process
            role = concept.role

        rows.append(
            {
                "Variable_Name": var,
                "Description_CN": row["Description_CN"],
                "Group": row["Group"],
                "Metadata_Source": row["Source"],
                "Present_In_Input_Parquet": "yes" if var in parquet_cols else "no",
                "Action": action,
                "Aggregate_Target": target,
                "Aggregation_Method": method,
                "Process": process,
                "Role": role,
                "Reason": reason,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(concept_rows)


def render_report(
    analysis_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    input_parquet: Path,
    output_parquet: Path,
    reduced_df: pd.DataFrame,
    expert_doc_excerpt: str,
) -> str:
    action_counts = analysis_df["Action"].value_counts().to_dict()

    lines = [
        "# 专家知识驱动的聚合优先降维报告",
        "",
        "## 本版策略",
        "",
        "- 本版以**聚合**为主，而不是以删除为主。",
        "- 每个变量都被逐一映射为：`keep_single` / `aggregate` / `drop`。",
        "- 用户明确要求保留的流程概念，即使当前数据源缺失，也保留为输出 schema 占位列。",
        "",
        "## 输入输出",
        "",
        f"- 输入 parquet：`{input_parquet}`",
        f"- 输出 parquet：`{output_parquet}`",
        f"- 输出 shape：**{reduced_df.shape}**",
        f"- 变量动作统计：`{action_counts}`",
        "",
        "## 用户要求必须保留的流程概念",
        "",
        "| 概念列 | 中文名 | 源变量数 | 当前可用数 |",
        "|---|---|---:|---:|",
    ]

    required_df = concept_df[concept_df["required"] == "yes"]
    for _, row in required_df.iterrows():
        lines.append(f"| `{row['feature_name']}` | {row['feature_cn']} | {row['source_var_count']} | {row['available_var_count']} |")

    lines.extend(
        [
            "",
            "## 聚合概念表",
            "",
            "| 特征名 | 中文名 | 工艺段 | 角色 | 方法 | 源变量数 | 可用数 |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for _, row in concept_df.iterrows():
        lines.append(
            f"| `{row['feature_name']}` | {row['feature_cn']} | {row['process']} | {row['role']} | "
            f"{row['method']} | {row['source_var_count']} | {row['available_var_count']} |"
        )

    if expert_doc_excerpt:
        lines.extend(["", "## 专家知识摘要", "", expert_doc_excerpt, ""])

    lines.extend(
        [
            "",
            "## 逐变量分析",
            "",
            "| 变量名 | 中文描述 | 动作 | 聚合目标 | 工艺段 | 角色 | 说明 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for _, row in analysis_df.iterrows():
        reason = str(row["Reason"]).replace("|", "｜").replace("\n", " ")
        desc = str(row["Description_CN"]).replace("|", "｜")
        lines.append(
            f"| `{row['Variable_Name']}` | {desc} | {row['Action']} | "
            f"`{row['Aggregate_Target']}` | {row['Process']} | {row['Role']} | {reason} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    variables_csv = Path(args.variables_csv)
    abc_csv = Path(args.abc_csv)
    input_parquet = Path(args.input_parquet)
    output_parquet = Path(args.output_parquet) if args.output_parquet else input_parquet.with_name(
        f"{input_parquet.stem}_expert_reduced.parquet"
    )
    analysis_csv = Path(args.analysis_csv)
    report_md = Path(args.report_md)
    concept_csv = Path(args.concept_csv)
    expert_doc = Path(args.expert_doc)

    if not variables_csv.exists():
        raise FileNotFoundError(f"找不到变量清单：{variables_csv}")
    if not abc_csv.exists():
        raise FileNotFoundError(f"找不到 ABC 元数据：{abc_csv}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"找不到输入 parquet：{input_parquet}")

    variables_df = pd.read_csv(variables_csv, encoding="utf-8-sig")
    abc_df = pd.read_csv(abc_csv, encoding="utf-8-sig")
    df = pd.read_parquet(input_parquet)
    parquet_cols = set(df.columns)

    _, comment_to_names, _ = build_comment_maps(abc_df)
    concepts = build_concepts(parquet_cols, comment_to_names)
    metadata_df = build_metadata(variables_df, abc_df, parquet_cols)
    analysis_df, concept_df = build_analysis(metadata_df, concepts, parquet_cols)

    reduced = pd.DataFrame(index=df.index)
    for concept in concepts:
        available_vars = [v for v in concept.source_vars if v in df.columns]
        reduced[concept.feature_name] = aggregate_series(df, available_vars, concept.method)

    y_cols = [c for c in df.columns if c.lower().startswith("y_")]
    for col in y_cols:
        reduced[col] = df[col]

    expert_excerpt = ""
    if expert_doc.exists():
        text = expert_doc.read_text(encoding="utf-8", errors="replace").strip()
        expert_excerpt = text[:1200] + ("..." if len(text) > 1200 else "")

    analysis_csv.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    concept_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    analysis_df.to_csv(analysis_csv, index=False, encoding="utf-8-sig")
    concept_df.to_csv(concept_csv, index=False, encoding="utf-8-sig")
    reduced.to_parquet(output_parquet)
    report_md.write_text(
        render_report(
            analysis_df=analysis_df,
            concept_df=concept_df,
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            reduced_df=reduced,
            expert_doc_excerpt=expert_excerpt,
        ),
        encoding="utf-8-sig",
    )

    print(f"analysis_csv={analysis_csv}")
    print(f"concept_csv={concept_csv}")
    print(f"report_md={report_md}")
    print(f"output_parquet={output_parquet}")
    print(f"output_shape={reduced.shape}")


if __name__ == "__main__":
    main()
