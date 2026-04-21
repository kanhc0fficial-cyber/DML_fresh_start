# 专家知识驱动的聚合优先降维报告（v3）

## 策略

- 以 470 个代表变量为基础，全部默认直通。
- 仅对确认为并联同类设备的变量做聚合，聚合组由精确变量名列表定义，不依赖正则或文字匹配。
- 不设占位列：没有源数据的概念直接忽略。

## 输入输出

- 输入 parquet：`/home/runner/work/DML_fresh_start/DML_fresh_start/data/modeling_dataset_final.parquet`
- 输出 parquet：`/home/runner/work/DML_fresh_start/DML_fresh_start/data/modeling_dataset_final_expert_reduced.parquet`
- 输出 shape：**(1953, 323)**
- 代表变量总数：470
- 参与聚合的变量数：178（合并为 29 个概念特征）
- 直通保留的变量数：292

## 关于旋流器开关状态

经穷举检索（2026-04），当前 `modeling_dataset_final.parquet` 中不存在任何旋流器开关状态列。
ABC 元数据中 204 个 `_ZT` 变量均为浮选药剂阀/鼓风机阀/分配器阀，无旋流器开关。
若实际 DCS 存在该信号，需从未接入当前数据集的子系统补充数据后重新处理。

## 聚合概念表

共 **29** 个聚合概念，178 个代表变量参与聚合。

| 特征名 | 中文名 | 工艺段 | 角色 | 方法 | 源变量数 | 可用数 |
|---|---|---|---|---|---:|---:|
| `agg_mag_excit_voltage` | 磁选_励磁电压 | 磁选 | 励磁电压 | mean | 13 | 13 |
| `agg_mag_excit_current` | 磁选_励磁电流 | 磁选 | 励磁电流 | mean | 12 | 12 |
| `agg_mag_coil_temp` | 磁选_线圈温度 | 磁选 | 线圈温度 | mean | 10 | 10 |
| `agg_mag_tailings_valve1` | 磁选_1号尾矿阀开度 | 磁选 | 尾矿阀1开度 | mean | 10 | 10 |
| `agg_mag_tailings_valve2` | 磁选_2号尾矿阀开度 | 磁选 | 尾矿阀2开度 | mean | 14 | 14 |
| `agg_mag_blowdown_valve` | 磁选_排污阀开度 | 磁选 | 排污阀开度 | mean | 10 | 10 |
| `agg_mag_pulsation_freq` | 磁选_脉动电机频率 | 磁选 | 脉动频率 | mean | 19 | 19 |
| `agg_mag_ring_freq` | 磁选_转环频率 | 磁选 | 转环频率 | mean | 13 | 13 |
| `agg_mag_level` | 磁选_选矿液位 | 磁选 | 选矿液位 | mean | 9 | 9 |
| `agg_mag_flush_water_pressure` | 磁选_冲矿水压力 | 磁选 | 冲矿水压力 | mean | 5 | 5 |
| `agg_mag_motor_current_rc` | 磁选_主电机A相电流 | 磁选 | 电机电流 | mean | 2 | 2 |
| `agg_mag_motor_voltage_rc` | 磁选_主电机BC线电压 | 磁选 | 电机电压 | mean | 3 | 3 |
| `agg_tm_motor_current` | 塔磨_主电机电流 | 塔磨 | 主电机电流 | mean | 5 | 5 |
| `agg_tm_reducer_oil_temp` | 塔磨_减速机油池温度 | 塔磨 | 减速机油温 | mean | 2 | 2 |
| `agg_tm_reducer_outlet_temp` | 塔磨_减速机出油口温度 | 塔磨 | 减速机出油温 | mean | 4 | 4 |
| `agg_tm_cyclone_pump_current` | 塔磨_三旋给矿泵电流 | 塔磨 | 给矿泵电流 | mean | 8 | 8 |
| `agg_tm_cyclone_pump_freq` | 塔磨_三旋给矿泵频率 | 塔磨 | 给矿泵频率 | mean | 2 | 2 |
| `agg_tm_cyclone_sand_water_flow` | 塔磨_旋流器沉砂加水流量 | 塔磨 | 沉砂加水量 | mean | 3 | 3 |
| `agg_tm_cyclone_feed_flow` | 塔磨_旋流器给矿管道流量 | 塔磨 | 给矿流量 | mean | 6 | 6 |
| `agg_tm_cyclone_sand_valve_setpoint` | 塔磨_旋流器沉砂水阀位给定 | 塔磨 | 沉砂水阀给定 | mean | 5 | 5 |
| `agg_tm_cyclone_sand_valve_feedback` | 塔磨_旋流器沉砂水阀位反馈 | 塔磨 | 沉砂水阀反馈 | mean | 3 | 3 |
| `agg_tm_cyclone_pool_valve_setpoint` | 塔磨_旋流器泵池水阀位给定 | 塔磨 | 泵池水阀给定 | mean | 2 | 2 |
| `agg_tm_cyclone_pool_level` | 塔磨_旋流器给矿泵池液位 | 塔磨 | 泵池液位 | mean | 3 | 3 |
| `agg_tm_cyclone_overflow_pool_level` | 塔磨_旋流器溢流泵池液位 | 塔磨 | 溢流泵池液位 | mean | 3 | 3 |
| `agg_tm_overflow_pump_current` | 塔磨_旋流器溢流泵电流 | 塔磨 | 溢流泵电流 | mean | 4 | 4 |
| `agg_accident_pump_freq` | 尾矿_事故泵频率 | 尾矿 | 事故泵频率 | mean | 2 | 2 |
| `agg_accident_pump_current` | 尾矿_事故泵电流 | 尾矿 | 事故泵电流 | mean | 2 | 2 |
| `agg_bottom_pump_freq_setpoint` | 底流_渣浆泵频率给定 | 底流 | 频率给定 | mean | 2 | 2 |
| `agg_bottom_pump_current` | 底流_渣浆泵电流 | 底流 | 电流 | mean | 2 | 2 |

## 逐变量处置明细

| 变量名 | 中文描述 | 动作 | 输出特征 | 理由 |
|---|---|---|---|---|
| `FX_1FP2_ZT` | 电动阀(分配器)1#1FP2状态 | keep_individual | `FX_1FP2_ZT` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_3QF05_ZT` | 5#阀状态 | keep_individual | `FX_3QF05_ZT` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_3QF11_ZT` | 11#阀状态 | keep_individual | `FX_3QF11_ZT` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_8F16_ZT` | 电动阀(输送二次K6-1粗选)8F16状态 | keep_individual | `FX_8F16_ZT` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH10_AI6` | 浮选3#低压室2#变压器总瞬时有功功率 | keep_individual | `FX_AH10_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH12_AI6` | 浮选4#鼓风机变频总瞬时有功功率 | keep_individual | `FX_AH12_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH14_AI7` | 浮选5#鼓风机变频总瞬时无功功率 | keep_individual | `FX_AH14_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH16_AI5` | 浮选6#鼓风机变频总功率因素 | keep_individual | `FX_AH16_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH16_AI6` | 浮选6#鼓风机变频总瞬时有功功率 | keep_individual | `FX_AH16_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH5_AI6` | 浮选1#低压室1#变压器总瞬时有功功率 | keep_individual | `FX_AH5_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH6_AI6` | 浮选1#低压室2#变压器总瞬时有功功率 | keep_individual | `FX_AH6_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH7_AI6` | 浮选2#低压室1#变压器总瞬时有功功率 | keep_individual | `FX_AH7_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AH9_AI6` | 浮选3#低压室1#变压器总瞬时有功功率 | keep_individual | `FX_AH9_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AT_1102` | 一系列3#-1高效搅拌槽PH值 | keep_individual | `FX_AT_1102` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_AT_2104` | 二系列3#-4高效搅拌槽PH值 | keep_individual | `FX_AT_2104` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_DT_2602` | 粗选给矿泵1104出口管道浓度 | keep_individual | `FX_DT_2602` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_1601` | 一系列粗选给矿泵1101出口管道流量 | keep_individual | `FX_FT_1601` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_1602` | 一系列粗选给矿泵1102出口管道流量 | keep_individual | `FX_FT_1602` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_1701` | 一系列1#-1高效高效搅拌槽入矿管道流量 | keep_individual | `FX_FT_1701` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_1702` | 一系列1#-2高效高效搅拌槽入矿管道流量 | keep_individual | `FX_FT_1702` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_2601` | 二系列粗选给矿泵1101出口管道流量 | keep_individual | `FX_FT_2601` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_2602` | 二系列粗选给矿泵1102出口管道流量 | keep_individual | `FX_FT_2602` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_2602_BJ` | 二系列粗选给矿泵1102出口管道流量报警 | keep_individual | `FX_FT_2602_BJ` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_2701` | 二系列1#-3高效高效搅拌槽入矿管道流量 | keep_individual | `FX_FT_2701` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_2702` | 二系列1#-4高效高效搅拌槽入矿管道流量 | keep_individual | `FX_FT_2702` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_612C` |  | keep_individual | `FX_FT_612C` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_612D` |  | keep_individual | `FX_FT_612D` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_612E` |  | keep_individual | `FX_FT_612E` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_612F` |  | keep_individual | `FX_FT_612F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_612G` |  | keep_individual | `FX_FT_612G` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_612H` |  | keep_individual | `FX_FT_612H` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622A` |  | keep_individual | `FX_FT_622A` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622A_LJ` |  | keep_individual | `FX_FT_622A_LJ` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622B` |  | keep_individual | `FX_FT_622B` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622C` |  | keep_individual | `FX_FT_622C` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622D` |  | keep_individual | `FX_FT_622D` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622E` |  | keep_individual | `FX_FT_622E` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622E_SP` |  | keep_individual | `FX_FT_622E_SP` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622F` |  | keep_individual | `FX_FT_622F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622G` |  | keep_individual | `FX_FT_622G` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_622H` |  | keep_individual | `FX_FT_622H` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_642A` |  | keep_individual | `FX_FT_642A` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_642B` |  | keep_individual | `FX_FT_642B` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_642C` |  | keep_individual | `FX_FT_642C` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FT_642D` |  | keep_individual | `FX_FT_642D` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ401_I` | 浮选机(粗选)401电流 | keep_individual | `FX_FXJ401_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ401_U` | 浮选机(粗选)401电压 | keep_individual | `FX_FXJ401_U` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ402_I` | 浮选机(粗选)402电流 | keep_individual | `FX_FXJ402_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ407_I` | 浮选机(粗选)407电流 | keep_individual | `FX_FXJ407_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ410_I` | 浮选机(粗选)410电流 | keep_individual | `FX_FXJ410_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ504_I` | 浮选机(精选)504电流 | keep_individual | `FX_FXJ504_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ601_I` | 浮选机(扫选一)601电流 | keep_individual | `FX_FXJ601_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ602_I` | 浮选机(扫选一)602电流 | keep_individual | `FX_FXJ602_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ603_I` | 浮选机(扫选一)603电流 | keep_individual | `FX_FXJ603_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ604_I` | 浮选机(扫选一)604电流 | keep_individual | `FX_FXJ604_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ605_I` | 浮选机(扫选一)605电流 | keep_individual | `FX_FXJ605_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ701_I` | 浮选机(扫选二)701电流 | keep_individual | `FX_FXJ701_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ801_I` | 浮选机(扫选三)801电流 | keep_individual | `FX_FXJ801_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ802_I` | 浮选机(扫选三)802电流 | keep_individual | `FX_FXJ802_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_FXJ803_I` | 浮选机(扫选三)803电流 | keep_individual | `FX_FXJ803_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_GFJ901_I` | 离心鼓风机901电流反馈 | keep_individual | `FX_GFJ901_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_GFJ902_F` | 离心鼓风机902频率反馈 | keep_individual | `FX_GFJ902_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_GFJ903_I` | 离心鼓风机903电流反馈 | keep_individual | `FX_GFJ903_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_GFJ904_I` | 离心鼓风机904电流反馈 | keep_individual | `FX_GFJ904_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_GFJ905_I` | 离心鼓风机905电流反馈 | keep_individual | `FX_GFJ905_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_GFJ906_F` | 离心鼓风机906频率反馈 | keep_individual | `FX_GFJ906_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HGB2201_F` | 化工泵(输送一次CaO)2201频率反馈 | keep_individual | `FX_HGB2201_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HGB2202_I` | 化工泵(输送一次CaO)2202电流反馈 | keep_individual | `FX_HGB2202_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HGB2203_I` | 化工泵(输送一次CaO)2203电流反馈 | keep_individual | `FX_HGB2203_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HGB2204_F` | 化工泵(输送一次CaO)2204频率反馈 | keep_individual | `FX_HGB2204_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HGB2204_I` | 化工泵(输送一次CaO)2204电流反馈 | keep_individual | `FX_HGB2204_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HV_621A_AI` |  | keep_individual | `FX_HV_621A_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HV_621B_AI` |  | keep_individual | `FX_HV_621B_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HV_621C_AI` |  | keep_individual | `FX_HV_621C_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HV_621D_AI` |  | keep_individual | `FX_HV_621D_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HV_631_AI` | 热水桶加蒸汽电动调节蝶阀HV-631阀位反馈信号 | keep_individual | `FX_HV_631_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_HV_632_AI` | 二次TD-Ⅱ搅拌槽加入热水主管道电动调节球阀HV-632阀位反馈信号 | keep_individual | `FX_HV_632_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_JBC102_I` | 1#高效搅拌槽102电流 | keep_individual | `FX_JBC102_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_JBC103_I` | 1#高效搅拌槽103电流 | keep_individual | `FX_JBC103_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_JBC1801_I` | 搅拌槽(一次CaO)1801电流 | keep_individual | `FX_JBC1801_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_JBC1901_I` | 搅拌槽(二次CaO)1901电流 | keep_individual | `FX_JBC1901_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_JBC1902_I` | 搅拌槽(二次CaO)1902电流 | keep_individual | `FX_JBC1902_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_JBC1903_I` | 搅拌槽(二次CaO)1903电流 | keep_individual | `FX_JBC1903_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_JBC203_I` | 2#高效搅拌槽203电流 | keep_individual | `FX_JBC203_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_JBC303_I` | 3#高效搅拌槽303电流 | keep_individual | `FX_JBC303_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2302_F` | 螺杆泵(输送二次TD-II粗选)2302频率反馈 | keep_individual | `FX_LGB2302_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2302_I` | 螺杆泵(输送二次TD-II粗选)2302电流反馈 | keep_individual | `FX_LGB2302_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2303_F_W` | 螺杆泵(输送二次TD-II粗选)2303频率给定 | keep_individual | `FX_LGB2303_F_W` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2303_I` | 螺杆泵(输送二次TD-II粗选)2303电流反馈 | keep_individual | `FX_LGB2303_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2401_I` | 螺杆泵(输送二次TD-II精选)2401电流反馈 | keep_individual | `FX_LGB2401_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2403_F_W` | 螺杆泵(输送二次TD-II精选)2403频率给定 | keep_individual | `FX_LGB2403_F_W` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2403_I` | 螺杆泵(输送二次TD-II精选)2403电流反馈 | keep_individual | `FX_LGB2403_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2404_F_W` | 螺杆泵(输送二次TD-II精选)2404频率给定 | keep_individual | `FX_LGB2404_F_W` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2502_F` | 螺杆泵(输送二次K6-1粗选)2502频率反馈 | keep_individual | `FX_LGB2502_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2602_F` | 螺杆泵(输送二次K6-1一扫)2602频率反馈 | keep_individual | `FX_LGB2602_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2602_I` | 螺杆泵(输送二次K6-1一扫)2602电流反馈 | keep_individual | `FX_LGB2602_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2701_F` | 螺杆泵(输送二次NaOH)2701频率反馈 | keep_individual | `FX_LGB2701_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2702_F` | 螺杆泵(输送二次NaOH)2702频率反馈 | keep_individual | `FX_LGB2702_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2801_F` | 螺杆泵((皂化泵TD-II粗选))2801频率反馈 | keep_individual | `FX_LGB2801_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LGB2801_I` | 螺杆泵((皂化泵TD-II粗选))2801电流反馈 | keep_individual | `FX_LGB2801_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1601` | 一系列精选泡沫及一扫底流泵池(给粗选)液位 | keep_individual | `FX_LT_1601` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1601_BJ` | 一系列精选泡沫及一扫底流泵池(给粗选)液位报警 | keep_individual | `FX_LT_1601_BJ` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1601_HSV` | 一系列粗选给矿泵池液位液位控制上限 | keep_individual | `FX_LT_1601_HSV` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1602` | 一系列粗选泡沫及二扫底流泵池(给一扫)液位 | keep_individual | `FX_LT_1602` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1602_BJ` | 一系列粗选泡沫及二扫底流泵池(给一扫)液位报警 | keep_individual | `FX_LT_1602_BJ` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1603` | 一系列三扫底流泵池(给二扫)液位 | keep_individual | `FX_LT_1603` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1604` | 一系列尾矿泵池(给尾矿浓缩机)液位 | keep_individual | `FX_LT_1604` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1604_BJ` | 一系列尾矿泵池(给尾矿浓缩机)液位报警 | keep_individual | `FX_LT_1604_BJ` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1605` | 一系列精矿泵(给精矿浓缩机)液位 | keep_individual | `FX_LT_1605` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1606` | 一系列事故池液位 | keep_individual | `FX_LT_1606` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_1701` | 浮选溢流泵池液位 | keep_individual | `FX_LT_1701` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_2602` | 二系列粗选泡沫及二扫底流泵池(给一扫)液位 | keep_individual | `FX_LT_2602` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_2603` | 二系列三扫底流泵池(给二扫)液位 | keep_individual | `FX_LT_2603` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_2604` | 二系列尾矿泵池(给尾矿浓缩机)液位 | keep_individual | `FX_LT_2604` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_2605` | 二系列精矿泵(给精矿浓缩机)液位 | keep_individual | `FX_LT_2605` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_611` | 二次K6-1贮药箱(贮罐)液位 | keep_individual | `FX_LT_611` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_621B` |  | keep_individual | `FX_LT_621B` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_621C` |  | keep_individual | `FX_LT_621C` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_621D` |  | keep_individual | `FX_LT_621D` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_LT_641A` |  | keep_individual | `FX_LT_641A` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_P1_N2_I` | 4#Φ30m浮选前浓缩机N2电流 | keep_individual | `FX_P1_N2_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_P2_N1_I` | 6#Φ30m浮选前浓缩机N1电流 | keep_individual | `FX_P2_N1_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_P3_N2_I` | 4#Φ53m浮选前浓缩机N2电流 | keep_individual | `FX_P3_N2_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_PT_901_CK` | 离心鼓风机901出口压力 | keep_individual | `FX_PT_901_CK` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_PT_902_CK` | 离心鼓风机902出口压力 | keep_individual | `FX_PT_902_CK` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_1101` | 一系列1#-1高效搅拌槽温度 | keep_individual | `FX_TT_1101` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_1105` | 一系列3#-1高效搅拌槽温度 | keep_individual | `FX_TT_1105` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_2102` | 二系列2#-3高效搅拌槽温度 | keep_individual | `FX_TT_2102` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_2104` | 二系列2#-4高效搅拌槽温度 | keep_individual | `FX_TT_2104` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_2105` | 二系列3#-3高效搅拌槽温度 | keep_individual | `FX_TT_2105` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_2106` | 二系列3#-4高效搅拌槽温度 | keep_individual | `FX_TT_2106` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_621B2` |  | keep_individual | `FX_TT_621B2` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_621C1` |  | keep_individual | `FX_TT_621C1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_631B` |  | keep_individual | `FX_TT_631B` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_901_JK` | 离心鼓风机901进口温度 | keep_individual | `FX_TT_901_JK` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_902_JK` | 离心鼓风机902进口温度 | keep_individual | `FX_TT_902_JK` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TT_906_JK` | 离心鼓风机906进口温度 | keep_individual | `FX_TT_906_JK` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_1101_AI` | 一系列1#-1高效搅拌槽加蒸汽电动调节蝶阀TV-1101阀位反馈信号 | keep_individual | `FX_TV_1101_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_1103_AO` | 一系列1#-2高效搅拌槽加蒸汽电动调节蝶阀TV-1103阀位给定信号 | keep_individual | `FX_TV_1103_AO` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_1104_AI` | 一系列2#-2高效搅拌槽加蒸汽电动调节蝶阀TV-1104阀位反馈信号 | keep_individual | `FX_TV_1104_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_2101_AO` | 二系列1#-1高效搅拌槽加蒸汽电动调节蝶阀TV-2101阀位给定信号 | keep_individual | `FX_TV_2101_AO` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_2102_AI` | 二系列2#-1高效搅拌槽加蒸汽电动调节蝶阀TV-2102阀位反馈信号 | keep_individual | `FX_TV_2102_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_2103_AI` | 二系列1#-2高效搅拌槽加蒸汽电动调节蝶阀TV-2103阀位反馈信号 | keep_individual | `FX_TV_2103_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_2104_AI` | 二系列2#-2高效搅拌槽加蒸汽电动调节蝶阀TV-2104阀位反馈信号 | keep_individual | `FX_TV_2104_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_611_AI` | 二次K6-1贮药箱(贮罐)加蒸汽电动调节蝶阀TV-611阀位反馈信号 | keep_individual | `FX_TV_611_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_621A_AI` |  | keep_individual | `FX_TV_621A_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_621B_AI` |  | keep_individual | `FX_TV_621B_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_621C_AI` |  | keep_individual | `FX_TV_621C_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_TV_621D_AI` |  | keep_individual | `FX_TV_621D_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX1_AI1` | Ⅰ系列粗选1作业泡沫层厚度实际值 | keep_individual | `FX_X1CX1_AI1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX1_AI5` | Ⅰ系列粗选1作业气量1实际值 | keep_individual | `FX_X1CX1_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX1_AI9` | Ⅰ系列粗选1作业气量2实际值 | keep_individual | `FX_X1CX1_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX2_AI1` | Ⅰ系列粗选2作业泡沫层厚度实际值 | keep_individual | `FX_X1CX2_AI1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX2_AI12` | Ⅰ系列粗选2作业蝶阀2开度设定值显示 | keep_individual | `FX_X1CX2_AI12` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX2_AI3` | Ⅰ系列粗选2作业液位阀1开度实际值 | keep_individual | `FX_X1CX2_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX2_AI5` | Ⅰ系列粗选2作业气量1实际值 | keep_individual | `FX_X1CX2_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX2_AI6` | Ⅰ系列粗选2作业气量1设定值显示 | keep_individual | `FX_X1CX2_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX2_AI7` | Ⅰ系列粗选2作业蝶阀1开度实际值 | keep_individual | `FX_X1CX2_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX2_AI9` | Ⅰ系列粗选2作业气量2实际值 | keep_individual | `FX_X1CX2_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX3_AI1` | Ⅰ系列粗选3作业泡沫层厚度实际值 | keep_individual | `FX_X1CX3_AI1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX3_AI3` | Ⅰ系列粗选3作业液位阀1开度实际值 | keep_individual | `FX_X1CX3_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX3_AI5` | Ⅰ系列粗选3作业气量1实际值 | keep_individual | `FX_X1CX3_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX3_AI6` | Ⅰ系列粗选3作业气量1设定值显示 | keep_individual | `FX_X1CX3_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1CX3_AI9` | Ⅰ系列粗选3作业气量2实际值 | keep_individual | `FX_X1CX3_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1JX_AI1` | Ⅰ系列精选作业泡沫层厚度实际值 | keep_individual | `FX_X1JX_AI1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1JX_AI11` | Ⅰ系列精选作业气量2设定值显示 | keep_individual | `FX_X1JX_AI11` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1JX_AI13` | Ⅰ系列精选作业气量3实际值 | keep_individual | `FX_X1JX_AI13` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1JX_AI15` | Ⅰ系列精选作业气量3设定值显示 | keep_individual | `FX_X1JX_AI15` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1JX_AI3` | Ⅰ系列精选作业液位阀1开度实际值 | keep_individual | `FX_X1JX_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1JX_AI5` | Ⅰ系列精选作业气量1实际值 | keep_individual | `FX_X1JX_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1JX_AI9` | Ⅰ系列精选作业气量2实际值 | keep_individual | `FX_X1JX_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX1_AI11` | Ⅰ系列扫选1作业气量2设定值显示 | keep_individual | `FX_X1SX1_AI11` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX1_AI13` | Ⅰ系列扫选1作业气量3实际值 | keep_individual | `FX_X1SX1_AI13` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX1_AI15` | Ⅰ系列扫选1作业气量3设定值显示 | keep_individual | `FX_X1SX1_AI15` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX1_AI16` | Ⅰ系列扫选1作业蝶阀3开度实际值 | keep_individual | `FX_X1SX1_AI16` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX1_AI3` | Ⅰ系列扫选1作业液位阀1开度实际值 | keep_individual | `FX_X1SX1_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX1_AI5` | Ⅰ系列扫选1作业气量1实际值 | keep_individual | `FX_X1SX1_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX1_AI7` | Ⅰ系列扫选1作业蝶阀1开度实际值 | keep_individual | `FX_X1SX1_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX2_AI12` | Ⅰ系列扫选2作业蝶阀2开度设定值显示 | keep_individual | `FX_X1SX2_AI12` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX2_AI3` | Ⅰ系列扫选2作业液位阀1开度实际值 | keep_individual | `FX_X1SX2_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX2_AI5` | Ⅰ系列扫选2作业气量1实际值 | keep_individual | `FX_X1SX2_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX2_AI7` | Ⅰ系列扫选2作业蝶阀1开度实际值 | keep_individual | `FX_X1SX2_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX2_AI9` | Ⅰ系列扫选2作业气量2实际值 | keep_individual | `FX_X1SX2_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX3_AI1` | Ⅰ系列扫选3作业泡沫层厚度实际值 | keep_individual | `FX_X1SX3_AI1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX3_AI12` | Ⅰ系列扫选3作业蝶阀2开度设定值显示 | keep_individual | `FX_X1SX3_AI12` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX3_AI4` | Ⅰ系列扫选3作业液位阀1开度设定显示 | keep_individual | `FX_X1SX3_AI4` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX3_AI5` | Ⅰ系列扫选3作业气量1实际值 | keep_individual | `FX_X1SX3_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX3_AI7` | Ⅰ系列扫选3作业蝶阀1开度实际值 | keep_individual | `FX_X1SX3_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X1SX3_AI9` | Ⅰ系列扫选3作业气量2实际值 | keep_individual | `FX_X1SX3_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX1_AI1` | Ⅱ系列粗选1作业泡沫层厚度实际值 | keep_individual | `FX_X2CX1_AI1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX1_AI12` | Ⅱ系列粗选1作业蝶阀2开度设定值显示 | keep_individual | `FX_X2CX1_AI12` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX1_AI5` | Ⅱ系列粗选1作业气量1实际值 | keep_individual | `FX_X2CX1_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX1_AI7` | Ⅱ系列粗选1作业蝶阀1开度实际值 | keep_individual | `FX_X2CX1_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX1_AI9` | Ⅱ系列粗选1作业气量2实际值 | keep_individual | `FX_X2CX1_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX2_AI12` | Ⅱ系列粗选2作业蝶阀2开度设定值显示 | keep_individual | `FX_X2CX2_AI12` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX2_AI27` | Ⅱ系列粗选2作业电动阀开度 | keep_individual | `FX_X2CX2_AI27` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX2_AI3` | Ⅱ系列粗选2作业液位阀1开度实际值 | keep_individual | `FX_X2CX2_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX2_AI4` | Ⅱ系列粗选2作业液位阀1开度设定显示 | keep_individual | `FX_X2CX2_AI4` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX2_AI5` | Ⅱ系列粗选2作业气量1实际值 | keep_individual | `FX_X2CX2_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX2_AI7` | Ⅱ系列粗选2作业蝶阀1开度实际值 | keep_individual | `FX_X2CX2_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX2_AI9` | Ⅱ系列粗选2作业气量2实际值 | keep_individual | `FX_X2CX2_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX3_AI11` | Ⅱ系列粗选3作业蝶阀2开度实际值 | keep_individual | `FX_X2CX3_AI11` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX3_AI13` | Ⅱ系列粗选3作业1#减速机低速端上轴温 | keep_individual | `FX_X2CX3_AI13` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX3_AI27` | Ⅱ系列粗选3作业电动阀开度 | keep_individual | `FX_X2CX3_AI27` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX3_AI3` | Ⅱ系列粗选3作业液位阀1开度实际值 | keep_individual | `FX_X2CX3_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX3_AI5` | Ⅱ系列粗选3作业气量1实际值 | keep_individual | `FX_X2CX3_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX3_AI7` | Ⅱ系列粗选3作业蝶阀1开度实际值 | keep_individual | `FX_X2CX3_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2CX3_AI9` | Ⅱ系列粗选3作业气量2实际值 | keep_individual | `FX_X2CX3_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2JX_AI1` | Ⅱ系列精选作业泡沫层厚度实际值 | keep_individual | `FX_X2JX_AI1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2JX_AI11` | Ⅱ系列精选作业气量2设定值显示 | keep_individual | `FX_X2JX_AI11` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2JX_AI13` | Ⅱ系列精选作业气量3实际值 | keep_individual | `FX_X2JX_AI13` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2JX_AI17` | Ⅱ系列精选作业1#减速机低速端上轴温 | keep_individual | `FX_X2JX_AI17` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2JX_AI3` | Ⅱ系列精选作业液位阀1开度实际值 | keep_individual | `FX_X2JX_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2JX_AI5` | Ⅱ系列精选作业气量1实际值 | keep_individual | `FX_X2JX_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2JX_AI9` | Ⅱ系列精选作业气量2实际值 | keep_individual | `FX_X2JX_AI9` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX1_AI1` | Ⅱ系列扫选1作业泡沫层厚度实际值 | keep_individual | `FX_X2SX1_AI1` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX1_AI13` | Ⅱ系列扫选1作业气量3实际值 | keep_individual | `FX_X2SX1_AI13` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX1_AI3` | Ⅱ系列扫选1作业液位阀1开度实际值 | keep_individual | `FX_X2SX1_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX1_AI38` | Ⅱ系列扫选1作业电动阀开度 | keep_individual | `FX_X2SX1_AI38` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX1_AI4` | Ⅱ系列扫选1作业液位阀1开度设定显示 | keep_individual | `FX_X2SX1_AI4` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX1_AI5` | Ⅱ系列扫选1作业气量1实际值 | keep_individual | `FX_X2SX1_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX1_AI7` | Ⅱ系列扫选1作业蝶阀1开度实际值 | keep_individual | `FX_X2SX1_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX2_AI3` | Ⅱ系列扫选2作业液位阀1开度实际值 | keep_individual | `FX_X2SX2_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX2_AI4` | Ⅱ系列扫选2作业液位阀1开度设定显示 | keep_individual | `FX_X2SX2_AI4` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX3_AI3` | Ⅱ系列扫选3作业液位阀1开度实际值 | keep_individual | `FX_X2SX3_AI3` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX3_AI4` | Ⅱ系列扫选3作业液位阀1开度设定显示 | keep_individual | `FX_X2SX3_AI4` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_X2SX3_AI5` | Ⅱ系列扫选3作业气量1实际值 | keep_individual | `FX_X2SX3_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1101_F` | 渣浆泵(粗选给矿泵)1101频率反馈 | keep_individual | `FX_ZJB1101_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1101_I` | 渣浆泵(粗选给矿泵)1101电流反馈 | keep_individual | `FX_ZJB1101_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1104_F` | 渣浆泵(粗选给矿泵)1104频率反馈 | keep_individual | `FX_ZJB1104_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1104_I` | 渣浆泵(粗选给矿泵)1104电流反馈 | keep_individual | `FX_ZJB1104_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1201_F` | 渣浆泵(一扫给矿泵)1201频率反馈 | keep_individual | `FX_ZJB1201_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1201_I` | 渣浆泵(一扫给矿泵)1201电流反馈 | keep_individual | `FX_ZJB1201_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1202_I` | 渣浆泵(一扫给矿泵)1202电流反馈 | keep_individual | `FX_ZJB1202_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1203_I` | 渣浆泵(一扫给矿泵)1203电流反馈 | keep_individual | `FX_ZJB1203_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1204_F` | 渣浆泵(一扫给矿泵)1204频率反馈 | keep_individual | `FX_ZJB1204_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1204_I` | 渣浆泵(一扫给矿泵)1204电流反馈 | keep_individual | `FX_ZJB1204_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1301_I` | 渣浆泵(二扫给矿泵)1301电流反馈 | keep_individual | `FX_ZJB1301_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1302_I` | 渣浆泵(二扫给矿泵)1302电流反馈 | keep_individual | `FX_ZJB1302_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1303_F` | 渣浆泵(二扫给矿泵)1303频率反馈 | keep_individual | `FX_ZJB1303_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1303_I` | 渣浆泵(二扫给矿泵)1303电流反馈 | keep_individual | `FX_ZJB1303_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1304_F` | 渣浆泵(二扫给矿泵)1304频率反馈 | keep_individual | `FX_ZJB1304_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1304_I` | 渣浆泵(二扫给矿泵)1304电流反馈 | keep_individual | `FX_ZJB1304_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB13_F` | 渣浆泵13频率反馈 | keep_individual | `FX_ZJB13_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1401_F` | 渣浆泵(精矿泵)1401频率反馈 | keep_individual | `FX_ZJB1401_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1401_I` | 渣浆泵(精矿泵)1401电流反馈 | keep_individual | `FX_ZJB1401_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1402_I` | 渣浆泵(精矿泵)1402电流反馈 | keep_individual | `FX_ZJB1402_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1404_F` | 渣浆泵(精矿泵)1404频率反馈 | keep_individual | `FX_ZJB1404_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1404_I` | 渣浆泵(精矿泵)1404电流反馈 | keep_individual | `FX_ZJB1404_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB14_I` | 渣浆泵14电流反馈 | keep_individual | `FX_ZJB14_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1502_F` | 渣浆泵(事故泵)1502频率反馈 | keep_individual | `FX_ZJB1502_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1502_I` | 渣浆泵(事故泵)1502电流反馈 | keep_individual | `FX_ZJB1502_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1504_F` | 渣浆泵(事故泵)1504频率反馈 | keep_individual | `FX_ZJB1504_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1504_I` | 渣浆泵(事故泵)1504电流反馈 | keep_individual | `FX_ZJB1504_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1601_F` | 渣浆泵(尾矿泵)1601频率反馈 | keep_individual | `FX_ZJB1601_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1601_I` | 渣浆泵(尾矿泵)1601电流反馈 | keep_individual | `FX_ZJB1601_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1602_I` | 渣浆泵(尾矿泵)1602电流反馈 | keep_individual | `FX_ZJB1602_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1603_I` | 渣浆泵(尾矿泵)1603电流反馈 | keep_individual | `FX_ZJB1603_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB1604_I` | 渣浆泵(尾矿泵)1604电流反馈 | keep_individual | `FX_ZJB1604_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB3002_I` | 渣浆泵(溢流泵)3002电流反馈 | keep_individual | `FX_ZJB3002_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB3004_I` | 渣浆泵(溢流泵)3004电流反馈 | keep_individual | `FX_ZJB3004_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB33_F` | 渣浆泵33频率反馈 | keep_individual | `FX_ZJB33_F` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB33_I` | 渣浆泵33电流反馈 | keep_individual | `FX_ZJB33_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB34_I` | 渣浆泵34电流反馈 | keep_individual | `FX_ZJB34_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB41_I` | 渣浆泵41电流反馈 | keep_individual | `FX_ZJB41_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB43_I` | 渣浆泵43电流反馈 | keep_individual | `FX_ZJB43_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `FX_ZJB62_I` | 渣浆泵62电流反馈 | keep_individual | `FX_ZJB62_I` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_AH10_AI5` | 磨磁1#配电室2#变压器总功率因素 | keep_individual | `MC1_AH10_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_AH12_AI5` | 磨磁6#塔磨机总功率因素 | keep_individual | `MC1_AH12_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_AH12_AI7` | 磨磁6#塔磨机总瞬时无功功率 | keep_individual | `MC1_AH12_AI7` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_AH6_AI6` | 磨磁3#配电室2#变压器总瞬时有功功率 | keep_individual | `MC1_AH6_AI6` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_AH8_AI5` | 磨磁2#配电室2#变压器总功率因素 | keep_individual | `MC1_AH8_AI5` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_AH9_AI4` | 磨磁3#配电室1#变压器电网频率 | keep_individual | `MC1_AH9_AI4` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_FET101_AI` | 1#三次分级旋流器沉砂加水管道流量 | aggregate | `agg_tm_cyclone_sand_water_flow` | 三次分级旋流器沉砂补水，6 组中 3 组有代表变量，取均值。 |
| `MC1_FET102_AI` | 1#三次分级旋流器给矿管道流量 | aggregate | `agg_tm_cyclone_feed_flow` | 三次分级旋流器给矿流量，6 组均有代表变量，取均值反映整体给矿水平。 |
| `MC1_FET202_AI` | 2#三次分级旋流器给矿管道流量 | aggregate | `agg_tm_cyclone_feed_flow` | 三次分级旋流器给矿流量，6 组均有代表变量，取均值反映整体给矿水平。 |
| `MC1_FET301_AI` | 3#三次分级旋流器沉砂加水管道流量 | aggregate | `agg_tm_cyclone_sand_water_flow` | 三次分级旋流器沉砂补水，6 组中 3 组有代表变量，取均值。 |
| `MC1_FET302_AI` | 3#三次分级旋流器给矿管道流量 | aggregate | `agg_tm_cyclone_feed_flow` | 三次分级旋流器给矿流量，6 组均有代表变量，取均值反映整体给矿水平。 |
| `MC1_FET402_AI` | 4#三次分级旋流器给矿管道流量 | aggregate | `agg_tm_cyclone_feed_flow` | 三次分级旋流器给矿流量，6 组均有代表变量，取均值反映整体给矿水平。 |
| `MC1_FET502_AI` | 5#三次分级旋流器给矿管道流量 | aggregate | `agg_tm_cyclone_feed_flow` | 三次分级旋流器给矿流量，6 组均有代表变量，取均值反映整体给矿水平。 |
| `MC1_FET503_AI` | 3#三次分级旋流器给矿泵池加水管道流量 | keep_individual | `MC1_FET503_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_FET601_AI` | 6#三次分级旋流器沉砂加水管道流量 | aggregate | `agg_tm_cyclone_sand_water_flow` | 三次分级旋流器沉砂补水，6 组中 3 组有代表变量，取均值。 |
| `MC1_FET602_AI` | 6#三次分级旋流器给矿管道流量 | aggregate | `agg_tm_cyclone_feed_flow` | 三次分级旋流器给矿流量，6 组均有代表变量，取均值反映整体给矿水平。 |
| `MC1_FV101_AI` | 1#三次分级旋流器沉砂加水管道阀位反馈 | aggregate | `agg_tm_cyclone_sand_valve_feedback` | 旋流器沉砂加水阀位反馈（AI），3 组并联取均值，与给定一起反映阀门控制状态。 |
| `MC1_FV101_AO` | 1#三次分级旋流器沉砂加水管道阀位给定 | aggregate | `agg_tm_cyclone_sand_valve_setpoint` | 旋流器沉砂加水阀位给定（AO），5 组并联取均值为控制意图代表。 |
| `MC1_FV201_AI` | 2#三次分级旋流器沉砂加水管道阀位反馈 | aggregate | `agg_tm_cyclone_sand_valve_feedback` | 旋流器沉砂加水阀位反馈（AI），3 组并联取均值，与给定一起反映阀门控制状态。 |
| `MC1_FV201_AO` | 2#三次分级旋流器沉砂加水管道阀位给定 | aggregate | `agg_tm_cyclone_sand_valve_setpoint` | 旋流器沉砂加水阀位给定（AO），5 组并联取均值为控制意图代表。 |
| `MC1_FV301_AO` | 3#三次分级旋流器沉砂加水管道阀位给定 | aggregate | `agg_tm_cyclone_sand_valve_setpoint` | 旋流器沉砂加水阀位给定（AO），5 组并联取均值为控制意图代表。 |
| `MC1_FV401_AI` | 4#三次分级旋流器沉砂加水管道阀位反馈 | aggregate | `agg_tm_cyclone_sand_valve_feedback` | 旋流器沉砂加水阀位反馈（AI），3 组并联取均值，与给定一起反映阀门控制状态。 |
| `MC1_FV401_AO` | 4#三次分级旋流器沉砂加水管道阀位给定 | aggregate | `agg_tm_cyclone_sand_valve_setpoint` | 旋流器沉砂加水阀位给定（AO），5 组并联取均值为控制意图代表。 |
| `MC1_FV501_AO` | 5#三次分级旋流器沉砂加水管道阀位给定 | aggregate | `agg_tm_cyclone_sand_valve_setpoint` | 旋流器沉砂加水阀位给定（AO），5 组并联取均值为控制意图代表。 |
| `MC1_GKB702_DL` | 2#三旋给矿泵电流 | aggregate | `agg_tm_cyclone_pump_current` | 12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。 |
| `MC1_GKB703_DL` | 3#三旋给矿泵电流 | aggregate | `agg_tm_cyclone_pump_current` | 12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。 |
| `MC1_GKB704_DL` | 4#三旋给矿泵电流 | aggregate | `agg_tm_cyclone_pump_current` | 12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。 |
| `MC1_GKB706_DL` | 6#三旋给矿泵电流 | aggregate | `agg_tm_cyclone_pump_current` | 12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。 |
| `MC1_GKB707_DL` | 7#三旋给矿泵电流 | aggregate | `agg_tm_cyclone_pump_current` | 12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。 |
| `MC1_GKB708_DL` | 8#三旋给矿泵电流 | aggregate | `agg_tm_cyclone_pump_current` | 12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。 |
| `MC1_GKB709_HZ` | 9#三旋给矿泵频率反馈 | aggregate | `agg_tm_cyclone_pump_freq` | 12 台三旋给矿泵中 2 台有频率反馈代表变量，取均值。 |
| `MC1_GKB710_DL` | 10#三旋给矿泵电流 | aggregate | `agg_tm_cyclone_pump_current` | 12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。 |
| `MC1_GKB711_DL` | 11#三旋给矿泵电流 | aggregate | `agg_tm_cyclone_pump_current` | 12 台三旋给矿泵并联，8 台有电流代表变量，取均值反映泵组整体负荷。 |
| `MC1_GKB711_HZ` | 11#三旋给矿泵频率反馈 | aggregate | `agg_tm_cyclone_pump_freq` | 12 台三旋给矿泵中 2 台有频率反馈代表变量，取均值。 |
| `MC1_LET101_AI` | 1#三次分级旋流器给矿泵池液位 | aggregate | `agg_tm_cyclone_pool_level` | 三次分级旋流器给矿泵池液位，3 组并联取均值反映整体液位水平。 |
| `MC1_LET301_AI` | 2#三次分级旋流器给矿泵池液位 | aggregate | `agg_tm_cyclone_pool_level` | 三次分级旋流器给矿泵池液位，3 组并联取均值反映整体液位水平。 |
| `MC1_LET501_AI` | 3#三次分级旋流器给矿泵池液位 | aggregate | `agg_tm_cyclone_pool_level` | 三次分级旋流器给矿泵池液位，3 组并联取均值反映整体液位水平。 |
| `MC1_LV101_AO` | 1#三次分级旋流器给矿泵池加水阀位给定 | aggregate | `agg_tm_cyclone_pool_valve_setpoint` | 旋流器给矿泵池加水阀位给定，2 组并联取均值。 |
| `MC1_LV301_AO` | 2#三次分级旋流器给矿泵池加水阀位给定 | aggregate | `agg_tm_cyclone_pool_valve_setpoint` | 旋流器给矿泵池加水阀位给定，2 组并联取均值。 |
| `MC1_TM201_JSJ_CYK_WD_AI` | 减速机出油口温度 | aggregate | `agg_tm_reducer_outlet_temp` | 减速机出油口温度是润滑系统实时热状态，4 台有代表变量，取均值。 |
| `MC1_TM201_JSJ_YC_WD_AI` | 减速机油池温度 | aggregate | `agg_tm_reducer_oil_temp` | 减速机油池温度反映润滑状态，2 台有代表变量，取均值。 |
| `MC1_TM201_ZDJ_DL_AI` | 主电机电流 | aggregate | `agg_tm_motor_current` | 6 台塔磨并联运行，主电机电流反映各机磨矿载荷，5 台有代表变量，取均值。 |
| `MC1_TM202_ZDJ_DL_AI` | 主电机电流 | aggregate | `agg_tm_motor_current` | 6 台塔磨并联运行，主电机电流反映各机磨矿载荷，5 台有代表变量，取均值。 |
| `MC1_TM204_HDZC_1_WD_AI` | 滑动轴承1#温度 | keep_individual | `MC1_TM204_HDZC_1_WD_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_TM204_JSJ_CYK_WD_AI` | 减速机出油口温度 | aggregate | `agg_tm_reducer_outlet_temp` | 减速机出油口温度是润滑系统实时热状态，4 台有代表变量，取均值。 |
| `MC1_TM204_JSJ_YC_WD_AI` | 减速机油池温度 | aggregate | `agg_tm_reducer_oil_temp` | 减速机油池温度反映润滑状态，2 台有代表变量，取均值。 |
| `MC1_TM204_ZDJ_DL_AI` | 主电机电流 | aggregate | `agg_tm_motor_current` | 6 台塔磨并联运行，主电机电流反映各机磨矿载荷，5 台有代表变量，取均值。 |
| `MC1_TM204_ZDJ_DZ_A_WD_AI` | 主电机定子A温度 | keep_individual | `MC1_TM204_ZDJ_DZ_A_WD_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_TM205_JSJ_CYK_WD_AI` | 减速机出油口温度 | aggregate | `agg_tm_reducer_outlet_temp` | 减速机出油口温度是润滑系统实时热状态，4 台有代表变量，取均值。 |
| `MC1_TM205_ZDJ_DL_AI` | 主电机电流 | aggregate | `agg_tm_motor_current` | 6 台塔磨并联运行，主电机电流反映各机磨矿载荷，5 台有代表变量，取均值。 |
| `MC1_TM206_HDZC_2_WD_AI` | 滑动轴承2#温度 | keep_individual | `MC1_TM206_HDZC_2_WD_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC1_TM206_JSJ_CYK_WD_AI` | 减速机出油口温度 | aggregate | `agg_tm_reducer_outlet_temp` | 减速机出油口温度是润滑系统实时热状态，4 台有代表变量，取均值。 |
| `MC1_TM206_ZDJ_DL_AI` | 主电机电流 | aggregate | `agg_tm_motor_current` | 6 台塔磨并联运行，主电机电流反映各机磨矿载荷，5 台有代表变量，取均值。 |
| `MC1_TM206_ZDJ_DZ_B_WD_AI` | 主电机定子B温度 | keep_individual | `MC1_TM206_ZDJ_DZ_B_WD_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_CQC_GNJNJ_AI` | 挂泥电机扭矩 | keep_individual | `MC2_CQC_GNJNJ_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_FET102_DL_AI` | 2#新建强磁前浓缩机底流泵出口管道流量 | keep_individual | `MC2_FET102_DL_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_FET201_DL_AI` | 1#新建机械加速澄清池底流泵出口管道流量 | keep_individual | `MC2_FET201_DL_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_FET202_DL_AI` | 2#新建机械加速澄清池底流泵出口管道流量 | keep_individual | `MC2_FET202_DL_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_JYB2_DL_AI` | A相电流 | keep_individual | `MC2_JYB2_DL_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_LET_102_AI` | 1#三次分级旋流器溢流泵池液位 | aggregate | `agg_tm_cyclone_overflow_pool_level` | 三次分级旋流器溢流泵池液位，3 个并联泵池取均值，反映后段矿浆缓冲状态。 |
| `MC2_LET_1101_AI` | 1#事故池液位 | keep_individual | `MC2_LET_1101_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_LET_1103_AI` | 3#事故池液位 | keep_individual | `MC2_LET_1103_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_LET_302_AI` | 2#三次分级旋流器溢流泵池液位 | aggregate | `agg_tm_cyclone_overflow_pool_level` | 三次分级旋流器溢流泵池液位，3 个并联泵池取均值，反映后段矿浆缓冲状态。 |
| `MC2_LET_502_AI` | 3#三次分级旋流器溢流泵池液位 | aggregate | `agg_tm_cyclone_overflow_pool_level` | 三次分级旋流器溢流泵池液位，3 个并联泵池取均值，反映后段矿浆缓冲状态。 |
| `MC2_NSJ_LPYL_AI` | 落耙压力 | keep_individual | `MC2_NSJ_LPYL_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_NSJ_TPYL_AI` | 提耙压力 | keep_individual | `MC2_NSJ_TPYL_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_PET101_AI` | 1#新建强磁前浓缩机底流泵出口管道压力 | keep_individual | `MC2_PET101_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_PET201_AI` | 1#机械澄清池底流泵出口管道压力 | keep_individual | `MC2_PET201_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_PET202_AI` | 2#机械澄清池底流泵出口管道压力 | keep_individual | `MC2_PET202_AI` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_QC501_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC501_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC501_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC501_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC501_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC501_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC501_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC501_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC501_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC502_CKSRKYL_AI` | 冲矿水入口压力值 | aggregate | `agg_mag_flush_water_pressure` | 冲矿水压力（入口/总/出口）反映磁选区冲洗条件，5 个代表测点取均值。 |
| `MC2_QC502_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC502_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC502_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC502_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC502_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC502_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC503_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC503_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC503_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC503_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC503_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC504_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC504_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC504_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC504_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC504_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC505_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC505_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC505_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC505_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC506_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC506_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC506_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC506_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC507_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC507_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC507_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC507_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC507_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC507_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC507_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC507_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC508_CKSYL_AI` | 冲矿水压力值 | aggregate | `agg_mag_flush_water_pressure` | 冲矿水压力（入口/总/出口）反映磁选区冲洗条件，5 个代表测点取均值。 |
| `MC2_QC508_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC508_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC508_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC508_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC508_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC508_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC508_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC509_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC509_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC509_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC510_CKSYL_AI` | 冲矿水压力值 | aggregate | `agg_mag_flush_water_pressure` | 冲矿水压力（入口/总/出口）反映磁选区冲洗条件，5 个代表测点取均值。 |
| `MC2_QC510_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC510_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC510_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC510_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC510_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC510_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC510_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC601_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC601_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC601_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC601_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC601_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC601_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC601_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC602_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC602_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC602_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC602_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC603_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC603_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC604_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC604_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC606_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC606_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC606_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC606_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC606_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC606_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC606_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC607_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC607_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC607_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC607_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC607_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC607_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC607_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC607_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC608_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC608_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC608_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC608_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC608_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC608_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC609_CKSCKYL_AI` | 冲矿水出口压力值 | aggregate | `agg_mag_flush_water_pressure` | 冲矿水压力（入口/总/出口）反映磁选区冲洗条件，5 个代表测点取均值。 |
| `MC2_QC609_LCDL_AI` | 励磁电流值 | aggregate | `agg_mag_excit_current` | 强磁机励磁电流是磁场强度的实际测量值，12 台并联机器取均值。 |
| `MC2_QC609_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC609_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC609_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC609_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC609_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC609_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC609_XQWD_AI` | 线圈温度值 | aggregate | `agg_mag_coil_temp` | 线圈温度反映磁选机热负荷状态，取均值作为全区代表温度。 |
| `MC2_QC609_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_QC610_CKSYL_AI` | 冲矿水压力值 | aggregate | `agg_mag_flush_water_pressure` | 冲矿水压力（入口/总/出口）反映磁选区冲洗条件，5 个代表测点取均值。 |
| `MC2_QC610_LCDY_AI` | 励磁电压值 | aggregate | `agg_mag_excit_voltage` | 强磁机励磁电压是磁选强度的直接控制量，13 台并联机器取均值反映全区励磁水平。 |
| `MC2_QC610_MDPL_AI` | 脉动电机实际工作频率显示值 | aggregate | `agg_mag_pulsation_freq` | 脉动频率直接影响磁选分离效果，19 台并联取均值为全区脉动水平代表。 |
| `MC2_QC610_PWF_AI` | 排污阀门开度实际值 | aggregate | `agg_mag_blowdown_valve` | 排污阀开度反映磁选机排矿状态，10 台并联取均值。 |
| `MC2_QC610_WKF1_AI` | 1#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve1` | 1# 尾矿阀门开度控制排尾量，10 台并联取均值。 |
| `MC2_QC610_WKF2_AI` | 2#尾矿阀门实际开度反馈值 | aggregate | `agg_mag_tailings_valve2` | 2# 尾矿阀门开度与 1# 合并可看出阀门调节整体状态，14 台并联取均值。 |
| `MC2_QC610_XKYW_AI` | 选矿液位 | aggregate | `agg_mag_level` | 磁选槽内液位反映矿浆充填程度，9 台并联取均值。注：部分仪表精度较低。 |
| `MC2_QC610_ZHPL_AI` | 转环电机实际工作频率显示值 | aggregate | `agg_mag_ring_freq` | 转环频率控制磁选机矿浆处理速率，13 台并联取均值。 |
| `MC2_RC101_DL_AI` | A相电流 | aggregate | `agg_mag_motor_current_rc` | MC2_RC101/102 的 A 相电流是磁选机主电机电流的代表变量，2 台取均值。 |
| `MC2_RC101_DY_AI` | BC线电压 | aggregate | `agg_mag_motor_voltage_rc` | MC2_RC101/102/106 的 BC 线电压是磁选机主电机电压代表，3 台取均值。 |
| `MC2_RC102_DL_AI` | A相电流 | aggregate | `agg_mag_motor_current_rc` | MC2_RC101/102 的 A 相电流是磁选机主电机电流的代表变量，2 台取均值。 |
| `MC2_RC102_DY_AI` | BC线电压 | aggregate | `agg_mag_motor_voltage_rc` | MC2_RC101/102/106 的 BC 线电压是磁选机主电机电压代表，3 台取均值。 |
| `MC2_RC106_DY_AI` | BC线电压 | aggregate | `agg_mag_motor_voltage_rc` | MC2_RC101/102/106 的 BC 线电压是磁选机主电机电压代表，3 台取均值。 |
| `MC2_SGB1002_DL` | 2#事故泵电流 | aggregate | `agg_accident_pump_current` | 事故泵 2# 和 3# 为同类设备，电流取均值。 |
| `MC2_SGB1002_HZ` | 2#事故泵频率反馈 | aggregate | `agg_accident_pump_freq` | 事故泵 2# 和 3# 为同类设备，频率取均值。 |
| `MC2_SGB1003_DL` | 3#事故泵电流 | aggregate | `agg_accident_pump_current` | 事故泵 2# 和 3# 为同类设备，电流取均值。 |
| `MC2_SGB1003_HZ` | 3#事故泵频率反馈 | aggregate | `agg_accident_pump_freq` | 事故泵 2# 和 3# 为同类设备，频率取均值。 |
| `MC2_WKB903_DL` | 3#尾矿泵电流 | keep_individual | `MC2_WKB903_DL` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_YLB802_DL` | 2#旋流器溢流泵电流 | aggregate | `agg_tm_overflow_pump_current` | 旋流器溢流泵电流反映泵组负荷，4 台有代表变量，取均值。 |
| `MC2_YLB805_DL` | 5#旋流器溢流泵电流 | aggregate | `agg_tm_overflow_pump_current` | 旋流器溢流泵电流反映泵组负荷，4 台有代表变量，取均值。 |
| `MC2_YLB806_DL` | 6#旋流器溢流泵电流 | aggregate | `agg_tm_overflow_pump_current` | 旋流器溢流泵电流反映泵组负荷，4 台有代表变量，取均值。 |
| `MC2_YLB809_DL` | 9#旋流器溢流泵电流 | aggregate | `agg_tm_overflow_pump_current` | 旋流器溢流泵电流反映泵组负荷，4 台有代表变量，取均值。 |
| `MC2_YLB811_HZ` | 11#旋流器溢流泵频率反馈 | keep_individual | `MC2_YLB811_HZ` | 该变量没有并联同类信号，按专家知识保留为独立特征。 |
| `MC2_ZJB01_DL` | 底流泵站1#渣浆泵电流 | aggregate | `agg_bottom_pump_current` | 底流泵站 1# 和 4# 渣浆泵为同类设备，电流取均值。 |
| `MC2_ZJB02_AO` | 底流泵站2#渣浆泵频率给定 | aggregate | `agg_bottom_pump_freq_setpoint` | 底流泵站 2# 和 3# 渣浆泵为同类设备，频率给定取均值。 |
| `MC2_ZJB03_AO` | 底流泵站3#渣浆泵频率给定 | aggregate | `agg_bottom_pump_freq_setpoint` | 底流泵站 2# 和 3# 渣浆泵为同类设备，频率给定取均值。 |
| `MC2_ZJB04_DL` | 底流泵站4#渣浆泵电流 | aggregate | `agg_bottom_pump_current` | 底流泵站 1# 和 4# 渣浆泵为同类设备，电流取均值。 |
