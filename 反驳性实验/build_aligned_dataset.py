"""
build_aligned_dataset.py
以 Y 测量时间点为锚，对 X/D 做窗口聚合，
生成高信息密度的对齐数据集。

核心思路：
  原来：每行 = 一个时间步（Y 大量缺失）
  改后：每行 = 一次 Y 测量（D/X 用窗口统计量填充）

每个原始特征扩展为 4 个统计量：mean / std / last / trend。
"""

import numpy as np
import pandas as pd

# ── 窗口配置 ──────────────────────────────────────────
WINDOW_MINUTES = 30      # Y 测量前 N 分钟的 X/D 数据纳入聚合
MIN_WINDOW_POINTS = 3    # 窗口内至少有几个有效点才保留该行
Y_FFILL_LIMIT = 2        # Y 少量前向填充（仅补1~2步，缩小缺失率）
                         # 这是唯一需要对 Y 动手的地方，limit 控制风险


def build_aligned_dataset(df_raw: pd.DataFrame,
                          operable_cols: list,
                          observable_cols: list,
                          window_minutes: int = WINDOW_MINUTES,
                          min_points: int = MIN_WINDOW_POINTS,
                          y_ffill_limit: int = Y_FFILL_LIMIT):
    """
    参数
    ----
    df_raw         : 原始宽表，index 为 DatetimeIndex，含 Y_grade 列
    operable_cols  : 操作变量列名列表（将在窗口内聚合）
    observable_cols: 状态变量列名列表（将在窗口内聚合）

    返回
    ----
    df_aligned      : 以 Y 观测时间点为行、窗口聚合特征为列的 DataFrame
    new_operable    : 聚合后的操作变量列名列表（*_mean/*_std/*_last/*_trend）
    new_observable  : 聚合后的状态变量列名列表
    """
    df = df_raw.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    all_feature_cols = list(operable_cols) + list(observable_cols)
    # 去重，保持顺序
    seen = set()
    all_feature_cols = [c for c in all_feature_cols if not (c in seen or seen.add(c))]

    # ── Step 1：对 X/D 做前向填充（物理合理）─────────────────────
    df[all_feature_cols] = df[all_feature_cols].ffill()

    # ── Step 2：对 Y 做极少量前向填充（limit 控制污染范围）───────
    # limit=2 意味着最多把上一次质量读数往后延 2 步
    # 对毕设而言可接受，能显著增加有效行数
    df["Y_grade"] = df["Y_grade"].ffill(limit=y_ffill_limit)

    # ── Step 3：以 Y 非空时间点为锚，构造窗口聚合特征 ────────────
    y_times = df.index[df["Y_grade"].notna()]
    window = pd.Timedelta(minutes=window_minutes)

    rows = []
    for t in y_times:
        window_df = df.loc[
            (df.index > t - window) & (df.index <= t),
            all_feature_cols
        ]
        if len(window_df) < min_points:
            continue

        row = {"time": t, "Y_grade": df.loc[t, "Y_grade"]}

        for col in all_feature_cols:
            vals = window_df[col].dropna()
            if len(vals) == 0:
                row[col + "_mean"]  = np.nan
                row[col + "_std"]   = 0.0
                row[col + "_last"]  = np.nan
                row[col + "_trend"] = 0.0
            else:
                row[col + "_mean"]  = vals.mean()
                row[col + "_std"]   = vals.std(ddof=0)
                row[col + "_last"]  = vals.iloc[-1]
                # 简单线性趋势（斜率）：反映窗口内的变化方向
                if len(vals) >= 3:
                    x_idx = np.arange(len(vals))
                    slope = np.polyfit(x_idx, vals.values, 1)[0]
                    row[col + "_trend"] = slope
                else:
                    row[col + "_trend"] = 0.0

        rows.append(row)

    df_aligned = pd.DataFrame(rows).set_index("time")
    df_aligned = df_aligned.dropna(subset=["Y_grade"])

    print(f"[对齐数据集] 原始行数: {len(df_raw)}  →  有效观测点: {len(df_aligned)}")
    print(f"[对齐数据集] 特征数: {len(df_aligned.columns) - 1}  "
          f"（每个原始特征 × 4 个统计量: mean/std/last/trend）")
    print(f"[对齐数据集] Y 覆盖率: {df_aligned['Y_grade'].notna().mean():.1%}")

    # ── Step 4：更新列名映射，返回新的 operable/observable 列名 ──
    new_operable = [c + sfx for c in operable_cols
                    for sfx in ("_mean", "_std", "_last", "_trend")]
    new_observable = [c + sfx for c in observable_cols
                      for sfx in ("_mean", "_std", "_last", "_trend")]

    # 只保留存在于 df_aligned 中的列
    new_operable   = [c for c in new_operable   if c in df_aligned.columns]
    new_observable = [c for c in new_observable if c in df_aligned.columns]

    return df_aligned, new_operable, new_observable
