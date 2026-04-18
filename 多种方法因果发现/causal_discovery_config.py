"""
causal_discovery_config.py
===========================
多产线因果发现公共配置与数据加载工具。

两条产线逻辑：
  - LINE='xin1': 使用 Group A（新1专线）+ Group C（公用）变量, Y = y_fx_xin1
  - LINE='xin2': 使用 Group B（新2专线）+ Group C（公用）变量, Y = y_fx_xin2

来自专家知识:
  - Stage 6 浮选区: FX_X1* -> 新1产线; FX_X2* -> 新2产线; FX_FXJ* -> 公用
  - Stage 7/8 收尾区: 一系列 -> 新1; 二系列 -> 新2
  - Stage 2/3/4/5: 部分有产线归属（MC1/MC2命名）
  - Stage 0/1: 全厂公用

物理拓扑约束（can_cause规则）:
  - 全面支持前馈因果流: stage 低 -> stage 高
  - 浮选(6) -> 尾矿(7) / 脱水(8): 允许
  - 公共/辅助(0,1) -> 主流程(2+): 允许
  - F 禁止跨产线: Group A 变量不能影响 xin2 的 Y，反之亦然
"""
import pandas as pd
import numpy as np
import os

# ─── 全局路径 ──────────────────────────────────────────────────────────────
BASE_DIR = r"C:\DML_fresh_start\数据存储"
X_PARQUET = os.path.join(BASE_DIR, "X_features_new.parquet")
Y_CSV = os.path.join(BASE_DIR, "y_target_new.csv")
VAR_CSV = (r"C:\DML_fresh_start\数据预处理\数据与处理结果-分阶段-去共线性后"
           r"\non_collinear_representative_vars_annotated.csv")

# 产线 -> Y列名映射
LINE_TO_Y_COL = {
    "xin1": "y_fx_xin1",
    "xin2": "y_fx_xin2",
}

# 产线 -> 允许的 Group 集合
LINE_TO_GROUPS = {
    "xin1": {"A", "C"},
    "xin2": {"B", "C"},
}


def load_vars_and_stages(line: str):
    """
    读取变量表，按产线过滤，返回:
      - var_to_stage: {变量名: Stage_ID}
      - var_to_group: {变量名: Group}
      - valid_vars: 过滤后有序变量列表
    
    过滤规则:
      1. Keep_Remove == 'keep'
      2. Group 属于该产线允许的集合
      3. change_status 为 'Active'（排除 Dead/LowChange 死变量）
         注意: 化验指标变量 change_status='Unknown'，一律保留（这些是关键领域知识变量）
    """
    df = pd.read_csv(VAR_CSV)
    
    allowed_groups = LINE_TO_GROUPS[line]
    
    mask = (
        (df["Keep_Remove"] == "keep") &
        (df["Group"].isin(allowed_groups)) &
        (
            (df["change_status"] == "Active") |
            (df["change_status"] == "Unknown")  # 化验/质检类变量没有 SCADA change_status
        )
    )
    df_filtered = df[mask].copy()
    
    var_to_stage = dict(zip(df_filtered["Variable_Name"], df_filtered["Stage_ID"]))
    var_to_group = dict(zip(df_filtered["Variable_Name"], df_filtered["Group"]))
    valid_vars = df_filtered["Variable_Name"].tolist()
    
    print(f"  [产线={line}] 过滤后变量数: {len(valid_vars)} "
          f"(A:{sum(1 for v in valid_vars if var_to_group[v]=='A')}, "
          f"B:{sum(1 for v in valid_vars if var_to_group[v]=='B')}, "
          f"C:{sum(1 for v in valid_vars if var_to_group[v]=='C')})")
    
    return var_to_stage, var_to_group, valid_vars


def can_cause(stage_src, stage_dst, group_src=None, group_dst=None, line=None):
    """
    物理拓扑因果可行性判断。
    
    参数:
      stage_src/dst: Stage_ID 或 'Y'
      group_src/dst: Group (A/B/C) 或 None
      line: 'xin1'/'xin2'，用于产线隔离
    
    返回 True 表示 src -> dst 因果方向物理上合理。
    """
    # ── 跨产线硬隔离 ──
    if line and group_src and group_dst:
        opposite = "B" if line == "xin1" else "A"
        if group_src == opposite or group_dst == opposite:
            return False  # 对立产线设备不参与本产线因果
    
    # ── Y 节点规则 ──
    if stage_dst == 'Y':
        if stage_src == 'Y':
            return False  # Y 不因果 Y
        # 尾矿(7)/脱水(8)不能指向精矿品位（这两个是结果，不是原因）
        try:
            if int(stage_src) in [7, 8]:
                return False
        except (ValueError, TypeError):
            pass
        return True
    
    if stage_src == 'Y':
        return False  # Y 不影响任何前端变量（反向因果禁止）
    
    # ── 同 Stage 内部允许 ──
    if stage_src == stage_dst:
        return True
    
    try:
        s = int(stage_src)
        d = int(stage_dst)
    except (ValueError, TypeError):
        return False
    
    # 公共辅助(0,1) -> 主流程(2+): 允许
    if s in [0, 1]:
        return d >= 2
    
    # 主流程前馈: 低 Stage -> 高 Stage
    if s >= 2 and d >= 2:
        if s == 6 and d in [7, 8]:
            return True  # 浮选 -> 尾矿/脱水: 允许
        if s in [7, 8] and d in [7, 8] and s != d:
            return False  # 尾矿 <-> 脱水: 不互相影响
        if s in [7, 8] and d < s:
            return False  # 末端不能反向影响前端
        return s < d
    
    return False


def prepare_data(line: str, resample_freq: str = "10min"):
    """
    载入并对齐 X 特征和 Y 目标，按产线过滤变量。
    
    返回:
      df: 包含所有特征列和 'y_grade' 列的 DataFrame
      X_cols: 特征列名列表（不包含 y_grade）
      var_to_stage: {变量名: Stage_ID}
      var_to_group: {变量名: Group}
    """
    var_to_stage, var_to_group, valid_vars = load_vars_and_stages(line)
    y_col = LINE_TO_Y_COL[line]
    
    X = pd.read_parquet(X_PARQUET)
    X.index = pd.to_datetime(X.index).tz_localize(None)
    
    y_df = pd.read_csv(Y_CSV, parse_dates=["time"])
    y_df["time"] = pd.to_datetime(y_df["time"]).dt.tz_localize(None)
    y_df = y_df.dropna(subset=[y_col])
    
    # 只保留在 valid_vars 中且实际存在于 X 的列
    X_cols = [c for c in valid_vars if c in X.columns]
    missing = [c for c in valid_vars if c not in X.columns]
    if missing:
        print(f"  [警告] {len(missing)} 个变量不在 X 特征文件中，将跳过: {missing[:5]}...")
    
    X_re = X[X_cols].resample(resample_freq).mean().ffill().bfill()
    y_re = y_df.set_index("time")[y_col].resample(resample_freq).mean().interpolate()
    
    common_idx = X_re.index.intersection(y_re.index)
    df = pd.concat([X_re.loc[common_idx], y_re.loc[common_idx].rename("y_grade")],
                   axis=1).dropna()
    
    # 更新 X_cols 只含实际存在的列
    X_cols = [c for c in X_cols if c in df.columns]
    
    print(f"  [产线={line}] 数据集: {len(df)} 行 x {len(df.columns)} 列, "
          f"Y={y_col}, 时间范围: {df.index.min()} ~ {df.index.max()}")
    
    return df, X_cols, var_to_stage, var_to_group
