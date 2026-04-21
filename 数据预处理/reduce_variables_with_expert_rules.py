"""
专家知识驱动的"聚合优先"降维脚本（v3）。

核心原则：
1. 以 non_collinear_representative_vars_operability.csv 中的 470 个代表变量为基础，
   全部默认直通保留（keep_individual）。
2. 仅对真正并联的同类设备变量做显式聚合（aggregate）：
   - 所有聚合组均使用精确变量名列表，不做任何正则或文字匹配。
   - 主要聚合对象：20 台强磁机（MC2_QC501-QC610，115 个代表变量 → 12 个概念特征）、
     6 台塔磨主电机及温度（MC1_TM*）、12 台三旋给矿泵（MC1_GKB*）等。
3. 不设置占位列——确实没有源变量的概念直接跳过。
4. 对每个代表变量均有明确的处置决定（aggregate / keep_individual）记录在分析 CSV 中。

旋流器开关状态说明（2026-04）：
  经穷举检索，当前 modeling_dataset_final.parquet 中不存在任何"旋流器开关状态"类列。
  ABC 元数据中 204 个 _ZT 变量均为浮选药剂阀、鼓风机阀、分配器阀等，无一与旋流器开关相关。
  MC1 塔磨旋流器相关变量不在 parquet 中的是：泵频给定（_AO）、PID 参数（_P/_I）、阀位设定（_SP）、浓度计（_DE）。
  若实际 DCS 中存在此信号，需从未接入当前数据集的子系统补充。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
class AggGroup:
    """一组并联设备同类变量 → 聚合为单一概念特征。"""
    feature_name: str          # 输出列名
    feature_cn: str            # 中文名
    process: str               # 所属工艺段
    role: str                  # 工艺角色
    method: str                # 聚合方法 (mean / sum / max / any / first)
    source_vars: tuple[str, ...]  # 精确变量名列表（来自代表变量集）
    reason: str                # 聚合理由


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="专家知识驱动的聚合优先降维脚本 v3")
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


def build_comment_maps(abc_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    name_to_comment: dict[str, str] = {}
    name_to_group: dict[str, str] = {}
    for _, row in abc_df.iterrows():
        name = normalize_text(row.get("NAME", ""))
        if not name:
            continue
        comment = normalize_text(row.get("COMMENT", ""))
        group = normalize_text(row.get("Group", ""))
        if comment:
            name_to_comment[name] = comment
        if group:
            name_to_group[name] = group
    return name_to_comment, name_to_group


def aggregate_series(df: pd.DataFrame, vars_in_df: list[str], method: str) -> pd.Series:
    if not vars_in_df:
        return pd.Series(np.nan, index=df.index, dtype="float64")
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
    raise ValueError(f"未知聚合方法: {method}")


def define_parallel_groups() -> list[AggGroup]:
    """
    定义所有并联设备的聚合组。
    每个组的 source_vars 为来自 non_collinear_representative_vars_operability.csv
    的精确变量名，逐一核对确认，不依赖正则或文字匹配。
    """
    groups: list[AggGroup] = []

    def g(
        feature_name: str,
        feature_cn: str,
        process: str,
        role: str,
        method: str,
        source_vars: list[str],
        reason: str,
    ) -> None:
        groups.append(AggGroup(
            feature_name=feature_name,
            feature_cn=feature_cn,
            process=process,
            role=role,
            method=method,
            source_vars=tuple(source_vars),
            reason=reason,
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # 磁选（强磁分选机 MC2_QC501-MC2_QC610）
    # 代表变量集中共 115 个，分 12 种测量类型，每种聚合为一个概念特征。
    # 各机同类测量之间高度重复（并联运行），专家知识要求按概念层保留。
    # ──────────────────────────────────────────────────────────────────────────

    # 1. 励磁电压（13 台有代表变量）
    g("agg_mag_excit_voltage", "磁选_励磁电压", "磁选", "励磁电压", "mean",
      ["MC2_QC501_LCDY_AI", "MC2_QC502_LCDY_AI", "MC2_QC506_LCDY_AI",
       "MC2_QC507_LCDY_AI", "MC2_QC508_LCDY_AI", "MC2_QC510_LCDY_AI",
       "MC2_QC601_LCDY_AI", "MC2_QC602_LCDY_AI", "MC2_QC606_LCDY_AI",
       "MC2_QC607_LCDY_AI", "MC2_QC608_LCDY_AI", "MC2_QC609_LCDY_AI",
       "MC2_QC610_LCDY_AI"],
      "强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。")

    # 2. 励磁电流（12 台有代表变量）
    g("agg_mag_excit_current", "磁选_励磁电流", "磁选", "励磁电流", "mean",
      ["MC2_QC501_LCDL_AI", "MC2_QC502_LCDL_AI", "MC2_QC503_LCDL_AI",
       "MC2_QC504_LCDL_AI", "MC2_QC505_LCDL_AI", "MC2_QC507_LCDL_AI",
       "MC2_QC508_LCDL_AI", "MC2_QC510_LCDL_AI", "MC2_QC601_LCDL_AI",
       "MC2_QC607_LCDL_AI", "MC2_QC608_LCDL_AI", "MC2_QC609_LCDL_AI"],
      "强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。")

    # 3. 线圈温度（10 台有代表变量）
    g("agg_mag_coil_temp", "磁选_线圈温度", "磁选", "线圈温度", "mean",
      ["MC2_QC501_XQWD_AI", "MC2_QC503_XQWD_AI", "MC2_QC504_XQWD_AI",
       "MC2_QC506_XQWD_AI", "MC2_QC507_XQWD_AI", "MC2_QC508_XQWD_AI",
       "MC2_QC601_XQWD_AI", "MC2_QC606_XQWD_AI", "MC2_QC607_XQWD_AI",
       "MC2_QC609_XQWD_AI"],
      "线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。")

    # 4. 1# 尾矿阀门开度（10 台）
    g("agg_mag_tailings_valve1", "磁选_1号尾矿阀开度", "磁选", "尾矿阀1开度", "mean",
      ["MC2_QC501_WKF1_AI", "MC2_QC509_WKF1_AI", "MC2_QC510_WKF1_AI",
       "MC2_QC601_WKF1_AI", "MC2_QC603_WKF1_AI", "MC2_QC606_WKF1_AI",
       "MC2_QC607_WKF1_AI", "MC2_QC608_WKF1_AI", "MC2_QC609_WKF1_AI",
       "MC2_QC610_WKF1_AI"],
      "1# 尾矿阀门开度控制排尾量，10 台并联取均值。")

    # 5. 2# 尾矿阀门开度（14 台）
    g("agg_mag_tailings_valve2", "磁选_2号尾矿阀开度", "磁选", "尾矿阀2开度", "mean",
      ["MC2_QC501_WKF2_AI", "MC2_QC502_WKF2_AI", "MC2_QC503_WKF2_AI",
       "MC2_QC504_WKF2_AI", "MC2_QC505_WKF2_AI", "MC2_QC506_WKF2_AI",
       "MC2_QC507_WKF2_AI", "MC2_QC508_WKF2_AI", "MC2_QC510_WKF2_AI",
       "MC2_QC602_WKF2_AI", "MC2_QC606_WKF2_AI", "MC2_QC607_WKF2_AI",
       "MC2_QC609_WKF2_AI", "MC2_QC610_WKF2_AI"],
      "2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。")

    # 6. 排污阀门开度（10 台）
    g("agg_mag_blowdown_valve", "磁选_排污阀开度", "磁选", "排污阀开度", "mean",
      ["MC2_QC501_PWF_AI", "MC2_QC502_PWF_AI", "MC2_QC503_PWF_AI",
       "MC2_QC507_PWF_AI", "MC2_QC510_PWF_AI", "MC2_QC606_PWF_AI",
       "MC2_QC607_PWF_AI", "MC2_QC608_PWF_AI", "MC2_QC609_PWF_AI",
       "MC2_QC610_PWF_AI"],
      "排污阀开度反映磁选机排矿状态，10 台并联取均值。")

    # 7. 脉动电机频率（19 台，全区最多）
    g("agg_mag_pulsation_freq", "磁选_脉动电机频率", "磁选", "脉动频率", "mean",
      ["MC2_QC501_MDPL_AI", "MC2_QC502_MDPL_AI", "MC2_QC503_MDPL_AI",
       "MC2_QC504_MDPL_AI", "MC2_QC505_MDPL_AI", "MC2_QC506_MDPL_AI",
       "MC2_QC507_MDPL_AI", "MC2_QC508_MDPL_AI", "MC2_QC509_MDPL_AI",
       "MC2_QC510_MDPL_AI", "MC2_QC601_MDPL_AI", "MC2_QC602_MDPL_AI",
       "MC2_QC603_MDPL_AI", "MC2_QC604_MDPL_AI", "MC2_QC606_MDPL_AI",
       "MC2_QC607_MDPL_AI", "MC2_QC608_MDPL_AI", "MC2_QC609_MDPL_AI",
       "MC2_QC610_MDPL_AI"],
      "脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。")

    # 8. 转环电机频率（13 台）
    g("agg_mag_ring_freq", "磁选_转环频率", "磁选", "转环频率", "mean",
      ["MC2_QC501_ZHPL_AI", "MC2_QC502_ZHPL_AI", "MC2_QC507_ZHPL_AI",
       "MC2_QC508_ZHPL_AI", "MC2_QC509_ZHPL_AI", "MC2_QC510_ZHPL_AI",
       "MC2_QC601_ZHPL_AI", "MC2_QC602_ZHPL_AI", "MC2_QC604_ZHPL_AI",
       "MC2_QC607_ZHPL_AI", "MC2_QC608_ZHPL_AI", "MC2_QC609_ZHPL_AI",
       "MC2_QC610_ZHPL_AI"],
      "转环频率控制磁选机矿浆处理速率，13 台并联取均值。")

    # 9. 选矿液位（9 台）
    g("agg_mag_level", "磁选_选矿液位", "磁选", "选矿液位", "mean",
      ["MC2_QC501_XKYW_AI", "MC2_QC504_XKYW_AI", "MC2_QC505_XKYW_AI",
       "MC2_QC507_XKYW_AI", "MC2_QC508_XKYW_AI", "MC2_QC601_XKYW_AI",
       "MC2_QC606_XKYW_AI", "MC2_QC609_XKYW_AI", "MC2_QC610_XKYW_AI"],
      "磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。")

    # 10. 冲矿水压力（含入口、压力、出口三类测点，共 5 个代表变量）
    #     CKSYL（总）: QC508, QC510, QC610；CKSRKYL（入口）: QC502；CKSCKYL（出口）: QC609
    g("agg_mag_flush_water_pressure", "磁选_冲矿水压力", "磁选", "冲矿水压力", "mean",
      ["MC2_QC502_CKSRKYL_AI",
       "MC2_QC508_CKSYL_AI", "MC2_QC510_CKSYL_AI", "MC2_QC610_CKSYL_AI",
       "MC2_QC609_CKSCKYL_AI"],
      "冲矿水压力（入口/总/出口）反映磁选区冲洗条件，5 个代表测点取均值。")

    # 11. 转环电机 A 相电流代理（MC2_RC 磁选机电流，5 个代表变量：DL*2 + DY*3）
    #     注：MC2_RC* 是磁选机主电路电流/电压测量，DL=电流，DY=电压
    g("agg_mag_motor_current_rc", "磁选_主电机A相电流", "磁选", "电机电流", "mean",
      ["MC2_RC101_DL_AI", "MC2_RC102_DL_AI"],
      "MC2_RC101/102 的 A 相电流是磁选机主电机电流的代表变量，2 台取均值。")

    g("agg_mag_motor_voltage_rc", "磁选_主电机BC线电压", "磁选", "电机电压", "mean",
      ["MC2_RC101_DY_AI", "MC2_RC102_DY_AI", "MC2_RC106_DY_AI"],
      "MC2_RC101/102/106 的 BC 线电压是磁选机主电机电压代表，3 台取均值。")

    # ──────────────────────────────────────────────────────────────────────────
    # 塔磨（MC1_TM201-MC1_TM206，6 台）
    # 代表变量集中共 15 个，分为 3 类可聚合测量 + 4 个单独代表
    # ──────────────────────────────────────────────────────────────────────────

    # 12. 塔磨主电机电流（5 台有代表变量）
    g("agg_tm_motor_current", "塔磨_主电机电流", "塔磨", "主电机电流", "mean",
      ["MC1_TM201_ZDJ_DL_AI", "MC1_TM202_ZDJ_DL_AI",
       "MC1_TM204_ZDJ_DL_AI", "MC1_TM205_ZDJ_DL_AI", "MC1_TM206_ZDJ_DL_AI"],
      "6 台塔磨并联运行，主电机电流反映各机磨矿载荷，5 台有代表变量，取均值。")

    # 13. 塔磨减速机油池温度（2 台有代表变量）
    g("agg_tm_reducer_oil_temp", "塔磨_减速机油池温度", "塔磨", "减速机油温", "mean",
      ["MC1_TM201_JSJ_YC_WD_AI", "MC1_TM204_JSJ_YC_WD_AI"],
      "减速机油池温度反映润滑状态，2 台有代表变量，取均值。")

    # 14. 塔磨减速机出油口温度（4 台有代表变量）
    g("agg_tm_reducer_outlet_temp", "塔磨_减速机出油口温度", "塔磨", "减速机出油温", "mean",
      ["MC1_TM201_JSJ_CYK_WD_AI", "MC1_TM204_JSJ_CYK_WD_AI",
       "MC1_TM205_JSJ_CYK_WD_AI", "MC1_TM206_JSJ_CYK_WD_AI"],
      "减速机出油口温度是润滑系统实时热状态，4 台有代表变量，取均值。")

    # TM204_HDZC_1_WD_AI（滑动轴承1#温度）→ keep_individual（唯一代表）
    # TM204_ZDJ_DZ_A_WD_AI（主电机定子A温度）→ keep_individual
    # TM206_HDZC_2_WD_AI（滑动轴承2#温度）→ keep_individual
    # TM206_ZDJ_DZ_B_WD_AI（主电机定子B温度）→ keep_individual

    # ──────────────────────────────────────────────────────────────────────────
    # 塔磨区三次旋流器给矿泵（MC1_GKB701-MC1_GKB712，12 台）
    # 代表变量集中共 10 个：8 个 DL（电流）+ 2 个 HZ（频率反馈）
    # ──────────────────────────────────────────────────────────────────────────

    # 15. 三旋给矿泵电流（8 台）
    g("agg_tm_cyclone_pump_current", "塔磨_三旋给矿泵电流", "塔磨", "给矿泵电流", "mean",
      ["MC1_GKB702_DL", "MC1_GKB703_DL", "MC1_GKB704_DL", "MC1_GKB706_DL",
       "MC1_GKB707_DL", "MC1_GKB708_DL", "MC1_GKB710_DL", "MC1_GKB711_DL"],
      "12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。")

    # 16. 三旋给矿泵频率反馈（2 台）
    g("agg_tm_cyclone_pump_freq", "塔磨_三旋给矿泵频率", "塔磨", "给矿泵频率", "mean",
      ["MC1_GKB709_HZ", "MC1_GKB711_HZ"],
      "12 台三旋给矿泵中 2 台有频率反馈代表变量，取均值。")

    # ──────────────────────────────────────────────────────────────────────────
    # 塔磨区三次分级旋流器水量/阀位/液位（MC1_FET / MC1_FV / MC1_LV / MC1_LET）
    # 共 6 组旋流器，代表变量集中有 10+8+2+3=23 个
    # ──────────────────────────────────────────────────────────────────────────

    # 17. 旋流器沉砂加水管道流量（3 台有代表变量）
    g("agg_tm_cyclone_sand_water_flow", "塔磨_旋流器沉砂加水流量", "塔磨", "沉砂加水量", "mean",
      ["MC1_FET101_AI", "MC1_FET301_AI", "MC1_FET601_AI"],
      "三次分级旋流器沉砂补水，6 组中 3 组有代表变量，取均值。")

    # 18. 旋流器给矿管道流量（6 台有代表变量）
    g("agg_tm_cyclone_feed_flow", "塔磨_旋流器给矿管道流量", "塔磨", "给矿流量", "mean",
      ["MC1_FET102_AI", "MC1_FET202_AI", "MC1_FET302_AI",
       "MC1_FET402_AI", "MC1_FET502_AI", "MC1_FET602_AI"],
      "三次分级旋流器给矿流量，6 组均有代表变量，取均值反映整体给矿水平。")

    # 19. 旋流器给矿泵池加水管道流量（FET503，仅 1 台）→ keep_individual

    # 20. 旋流器沉砂加水阀位给定（5 台）
    g("agg_tm_cyclone_sand_valve_setpoint", "塔磨_旋流器沉砂水阀位给定", "塔磨", "沉砂水阀给定", "mean",
      ["MC1_FV101_AO", "MC1_FV201_AO", "MC1_FV301_AO", "MC1_FV401_AO", "MC1_FV501_AO"],
      "旋流器沉砂加水阀位给定（AO），5 组并联取均值为控制意图代表。")

    # 21. 旋流器沉砂加水阀位反馈（3 台）
    g("agg_tm_cyclone_sand_valve_feedback", "塔磨_旋流器沉砂水阀位反馈", "塔磨", "沉砂水阀反馈", "mean",
      ["MC1_FV101_AI", "MC1_FV201_AI", "MC1_FV401_AI"],
      "旋流器沉砂加水阀位反馈（AI），3 组并联取均值，与给定一起反映阀门控制状态。")

    # 22. 旋流器给矿泵池加水阀位给定（2 台）
    g("agg_tm_cyclone_pool_valve_setpoint", "塔磨_旋流器泵池水阀位给定", "塔磨", "泵池水阀给定", "mean",
      ["MC1_LV101_AO", "MC1_LV301_AO"],
      "旋流器给矿泵池加水阀位给定，2 组并联取均值。")

    # 23. 旋流器给矿泵池液位（3 台）
    g("agg_tm_cyclone_pool_level", "塔磨_旋流器给矿泵池液位", "塔磨", "泵池液位", "mean",
      ["MC1_LET101_AI", "MC1_LET301_AI", "MC1_LET501_AI"],
      "三次分级旋流器给矿泵池液位，3 组并联取均值反映整体液位水平。")

    # ──────────────────────────────────────────────────────────────────────────
    # 塔磨后段：旋流器溢流泵池液位（MC2_LET）
    # ──────────────────────────────────────────────────────────────────────────

    # 24. 三次分级旋流器溢流泵池液位（3 台）
    g("agg_tm_cyclone_overflow_pool_level", "塔磨_旋流器溢流泵池液位", "塔磨", "溢流泵池液位", "mean",
      ["MC2_LET_102_AI", "MC2_LET_302_AI", "MC2_LET_502_AI"],
      "三次分级旋流器溢流泵池液位，3 个并联泵池取均值，反映后段矿浆缓冲状态。")

    # 25. 旋流器溢流泵电流（MC2_YLB，4 台有代表变量）
    g("agg_tm_overflow_pump_current", "塔磨_旋流器溢流泵电流", "塔磨", "溢流泵电流", "mean",
      ["MC2_YLB802_DL", "MC2_YLB805_DL", "MC2_YLB806_DL", "MC2_YLB809_DL"],
      "旋流器溢流泵电流反映泵组负荷，4 台有代表变量，取均值。")

    # MC2_YLB811_HZ（溢流泵频率反馈，仅 1 台）→ keep_individual

    # ──────────────────────────────────────────────────────────────────────────
    # 事故泵（MC2_SGB，2 台有代表变量）
    # ──────────────────────────────────────────────────────────────────────────

    # 26. 事故泵频率
    g("agg_accident_pump_freq", "尾矿_事故泵频率", "尾矿", "事故泵频率", "mean",
      ["MC2_SGB1002_HZ", "MC2_SGB1003_HZ"],
      "事故泵 2# 和 3# 为同类设备，频率取均值。")

    # 27. 事故泵电流
    g("agg_accident_pump_current", "尾矿_事故泵电流", "尾矿", "事故泵电流", "mean",
      ["MC2_SGB1002_DL", "MC2_SGB1003_DL"],
      "事故泵 2# 和 3# 为同类设备，电流取均值。")

    # ──────────────────────────────────────────────────────────────────────────
    # 底流泵站（MC2_ZJB，4 个代表变量）
    # ──────────────────────────────────────────────────────────────────────────

    # 28. 底流泵站频率给定
    g("agg_bottom_pump_freq_setpoint", "底流_渣浆泵频率给定", "底流", "频率给定", "mean",
      ["MC2_ZJB02_AO", "MC2_ZJB03_AO"],
      "底流泵站 2# 和 3# 渣浆泵为同类设备，频率给定取均值。")

    # 29. 底流泵站电流
    g("agg_bottom_pump_current", "底流_渣浆泵电流", "底流", "电流", "mean",
      ["MC2_ZJB01_DL", "MC2_ZJB04_DL"],
      "底流泵站 1# 和 4# 渣浆泵为同类设备，电流取均值。")

    return groups


def build_analysis_df(
    rep_vars: list[str],
    agg_groups: list[AggGroup],
    rep_var_meta: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """为 470 个代表变量生成逐一处置分析表。"""
    var_to_group: dict[str, AggGroup] = {}
    for grp in agg_groups:
        for v in grp.source_vars:
            if v in var_to_group:
                raise ValueError(f"变量 {v} 被分配到多个聚合组：{var_to_group[v].feature_name} 和 {grp.feature_name}")
            var_to_group[v] = grp

    rows = []
    for v in rep_vars:
        meta = rep_var_meta.get(v, {})
        grp = var_to_group.get(v)
        if grp is not None:
            action = "aggregate"
            target = grp.feature_name
            method = grp.method
            process = grp.process
            role = grp.role
            reason = grp.reason
        else:
            action = "keep_individual"
            target = v
            method = "passthrough"
            process = meta.get("process", "")
            role = meta.get("Description_CN", "")
            reason = "该变量没有并联同类信号，按专家知识保留为独立特征。"
        rows.append({
            "Variable_Name": v,
            "Description_CN": meta.get("Description_CN", ""),
            "Group_ABC": meta.get("Group_ABC", ""),
            "Operability": meta.get("Operability", ""),
            "Action": action,
            "Output_Feature": target,
            "Aggregation_Method": method,
            "Process": process,
            "Role": role,
            "Reason": reason,
        })
    return pd.DataFrame(rows)


def build_concept_df(agg_groups: list[AggGroup], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp in agg_groups:
        available = sum(1 for v in grp.source_vars if v in df.columns)
        rows.append({
            "feature_name": grp.feature_name,
            "feature_cn": grp.feature_cn,
            "process": grp.process,
            "role": grp.role,
            "method": grp.method,
            "source_var_count": len(grp.source_vars),
            "available_var_count": available,
            "source_vars": "; ".join(grp.source_vars),
            "reason": grp.reason,
        })
    return pd.DataFrame(rows)


def render_report(
    analysis_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    input_parquet: Path,
    output_parquet: Path,
    reduced_df: pd.DataFrame,
) -> str:
    action_counts = analysis_df["Action"].value_counts().to_dict()
    n_agg_groups = len(concept_df)
    n_agg_vars = int(action_counts.get("aggregate", 0))
    n_keep = int(action_counts.get("keep_individual", 0))

    lines = [
        "# 专家知识驱动的聚合优先降维报告（v3）",
        "",
        "## 策略",
        "",
        "- 以 470 个代表变量为基础，全部默认直通。",
        "- 仅对确认为并联同类设备的变量做聚合，聚合组由精确变量名列表定义，不依赖正则或文字匹配。",
        "- 不设占位列：没有源数据的概念直接忽略。",
        "",
        "## 输入输出",
        "",
        f"- 输入 parquet：`{input_parquet}`",
        f"- 输出 parquet：`{output_parquet}`",
        f"- 输出 shape：**{reduced_df.shape}**",
        f"- 代表变量总数：470",
        f"- 参与聚合的变量数：{n_agg_vars}（合并为 {n_agg_groups} 个概念特征）",
        f"- 直通保留的变量数：{n_keep}",
        "",
        "## 关于旋流器开关状态",
        "",
        "经穷举检索（2026-04），当前 `modeling_dataset_final.parquet` 中不存在任何旋流器开关状态列。",
        "ABC 元数据中 204 个 `_ZT` 变量均为浮选药剂阀/鼓风机阀/分配器阀，无旋流器开关。",
        "若实际 DCS 存在该信号，需从未接入当前数据集的子系统补充数据后重新处理。",
        "",
        "## 聚合概念表",
        "",
        f"共 **{n_agg_groups}** 个聚合概念，{n_agg_vars} 个代表变量参与聚合。",
        "",
        "| 特征名 | 中文名 | 工艺段 | 角色 | 方法 | 源变量数 | 可用数 |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for _, row in concept_df.iterrows():
        lines.append(
            f"| `{row['feature_name']}` | {row['feature_cn']} | {row['process']} | {row['role']} | "
            f"{row['method']} | {row['source_var_count']} | {row['available_var_count']} |"
        )

    lines.extend([
        "",
        "## 逐变量处置明细",
        "",
        "| 变量名 | 中文描述 | 动作 | 输出特征 | 理由 |",
        "|---|---|---|---|---|",
    ])
    for _, row in analysis_df.sort_values("Variable_Name").iterrows():
        reason = str(row["Reason"]).replace("|", "｜").replace("\n", " ")
        desc = str(row["Description_CN"]).replace("|", "｜")
        lines.append(
            f"| `{row['Variable_Name']}` | {desc} | {row['Action']} | "
            f"`{row['Output_Feature']}` | {reason} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    variables_csv = Path(args.variables_csv)
    abc_csv = Path(args.abc_csv)
    input_parquet = Path(args.input_parquet)
    output_parquet = (
        Path(args.output_parquet)
        if args.output_parquet
        else input_parquet.with_name(f"{input_parquet.stem}_expert_reduced.parquet")
    )
    analysis_csv = Path(args.analysis_csv)
    report_md = Path(args.report_md)
    concept_csv = Path(args.concept_csv)

    for p in [variables_csv, abc_csv, input_parquet]:
        if not p.exists():
            raise FileNotFoundError(f"找不到文件：{p}")

    variables_df = pd.read_csv(variables_csv, encoding="utf-8-sig")
    abc_df = pd.read_csv(abc_csv, encoding="utf-8-sig")
    df = pd.read_parquet(input_parquet)

    name_to_comment, name_to_group = build_comment_maps(abc_df)

    # 建立 470 个代表变量的元数据字典
    rep_vars: list[str] = []
    rep_var_meta: dict[str, dict[str, str]] = {}
    for _, row in variables_df.iterrows():
        name = normalize_text(row.get("Non_Collinear_Representative", ""))
        if not name:
            continue
        rep_vars.append(name)
        rep_var_meta[name] = {
            "Description_CN": normalize_text(row.get("Description_CN", "")) or name_to_comment.get(name, ""),
            "Group_ABC": normalize_text(row.get("Group", "")) or name_to_group.get(name, ""),
            "Operability": normalize_text(row.get("Operability", "")),
            "process": "",
        }

    # 定义聚合组（精确变量名，无字符匹配）
    agg_groups = define_parallel_groups()

    # 检查聚合组中每个变量是否在代表变量集中
    rep_set = set(rep_vars)
    for grp in agg_groups:
        for v in grp.source_vars:
            if v not in rep_set:
                raise ValueError(
                    f"聚合组 {grp.feature_name} 中的变量 {v} 不在代表变量集（non_collinear_representative_vars_operability.csv）中。"
                    "请核实变量名后重新运行。"
                )

    # 确定哪些变量参与聚合
    vars_in_agg: set[str] = {v for grp in agg_groups for v in grp.source_vars}

    # 构建输出 DataFrame（用 pd.concat 避免逐列插入的碎片化 PerformanceWarning）
    parts: list[pd.DataFrame] = []

    # 1. 直通保留：未参与聚合的代表变量，原样保留
    passthrough_cols = [v for v in rep_vars if v not in vars_in_agg]
    # 区分在 parquet 中存在的 vs 缺失的
    existing = [v for v in passthrough_cols if v in df.columns]
    missing  = [v for v in passthrough_cols if v not in df.columns]
    if existing:
        parts.append(df[existing])
    if missing:
        parts.append(pd.DataFrame(np.nan, index=df.index, columns=missing))

    # 2. 聚合：并联设备组 → 概念特征
    agg_series: dict[str, pd.Series] = {}
    for grp in agg_groups:
        available_vars = [v for v in grp.source_vars if v in df.columns]
        agg_series[grp.feature_name] = aggregate_series(df, available_vars, grp.method)
    if agg_series:
        parts.append(pd.DataFrame(agg_series, index=df.index))

    # 3. 保留 y_ 标签列
    y_cols = [col for col in df.columns if col.lower().startswith("y_")]
    if y_cols:
        parts.append(df[y_cols])

    reduced = pd.concat(parts, axis=1)

    # 生成分析和报告
    analysis_df = build_analysis_df(rep_vars, agg_groups, rep_var_meta)
    concept_df = build_concept_df(agg_groups, df)
    report_text = render_report(analysis_df, concept_df, input_parquet, output_parquet, reduced)

    # 写出文件
    for p in [analysis_csv.parent, report_md.parent, concept_csv.parent, output_parquet.parent]:
        p.mkdir(parents=True, exist_ok=True)

    analysis_df.to_csv(analysis_csv, index=False, encoding="utf-8-sig")
    concept_df.to_csv(concept_csv, index=False, encoding="utf-8-sig")
    reduced.to_parquet(output_parquet)
    report_md.write_text(report_text, encoding="utf-8-sig")

    print(f"analysis_csv={analysis_csv}")
    print(f"concept_csv={concept_csv}")
    print(f"report_md={report_md}")
    print(f"output_parquet={output_parquet}")
    print(f"output_shape={reduced.shape}")


if __name__ == "__main__":
    main()
