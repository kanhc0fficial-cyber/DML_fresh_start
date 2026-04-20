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
import re
import pandas as pd
import numpy as np
import os

# ─── 全局路径（相对于仓库根目录） ──────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录：data/ 下按产线拆分的建模宽表（X 特征 + Y 品位已合并）
DATA_DIR = os.path.join(_REPO_ROOT, "data")

# 变量注释 Markdown（包含 Stage_ID 信息）
VAR_MD = os.path.join(
    _REPO_ROOT,
    "数据预处理",
    "数据与处理结果-分阶段-去共线性后",
    "non_collinear_representative_vars_annotated.md",
)

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

# 非特征列（Y 目标列），从 X 特征中排除
_Y_COLS = {"y_fx_xin1", "y_fx_xin2"}


def _infer_group(var_name: str) -> str:
    """
    根据变量命名规则推断产线归属（Group）。

    规则（来自专家知识文档）：
      - 名称中含 'X1'（新1系列浮选机）→ Group A
      - 名称中含 'X2'（新2系列浮选机）→ Group B
      - 其余                           → Group C（全厂公用）

    该规则覆盖浮选区 Stage 6 的主要产线专线设备，
    磨磁（MC1/MC2）和公用辅助（Stage 0/1）默认归为 Group C。
    """
    upper = var_name.upper()
    if 'FX_X1' in upper:
        return "A"
    if 'FX_X2' in upper:
        return "B"
    return "C"


def _parse_var_stage_from_md(md_path: str) -> dict:
    """
    解析变量注释 Markdown 文件，提取 {变量名: Stage_ID} 映射。

    文件格式示例：
      ## Stage 6 - 浮选网络
      | # | 变量名 | ... |
      | 1 | FX_X1CX1_AI7 | ... |

    返回：{变量名: stage_str}，stage_str 为字符串形式（如 '0', '6'）。
    """
    if not os.path.exists(md_path):
        print(f"  [警告] 变量注释 Markdown 文件不存在: {md_path}")
        return {}

    with open(md_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    var_to_stage = {}
    # 按 "## Stage N" 章节切分
    sections = re.split(r'## Stage (\S+)', content)
    i = 1
    while i + 1 < len(sections):
        stage = sections[i]  # captured group: the stage identifier (e.g., '0', '6')
        body = sections[i + 1]
        # 匹配表格行中的变量名（第2列，可能带反引号）
        for m in re.finditer(r'\|\s*\d+\s*\|\s*`?(\S+?)`?\s*\|', body):
            var_to_stage[m.group(1)] = stage
        i += 2

    return var_to_stage


def load_vars_and_stages(line: str):
    """
    读取变量表，按产线过滤，返回:
      - var_to_stage: {变量名: Stage_ID}
      - var_to_group: {变量名: Group}
      - valid_vars: 过滤后有序变量列表

    数据来源（新格式）：
      1. 从 non_collinear_representative_vars_annotated.md 解析 Stage 信息
      2. 根据变量命名规则推断 Group（A/B/C）
      3. 按产线允许的 Group 集合过滤变量
    """
    var_to_stage_all = _parse_var_stage_from_md(VAR_MD)

    allowed_groups = LINE_TO_GROUPS[line]

    var_to_stage = {}
    var_to_group = {}
    valid_vars = []

    # 保持 Markdown 中的原始顺序
    for var, stage in var_to_stage_all.items():
        group = _infer_group(var)
        if group in allowed_groups:
            var_to_stage[var] = stage
            var_to_group[var] = group
            valid_vars.append(var)

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

    数据源：时序宽表 data/timeseries_dataset_{line}_final.parquet
      - 1min 频率连续时序，~128K 行
      - Y 列稀疏（仅化验时间点有值，其余为 NaN）→ 前向/后向填充补全
      - X 列少量缺失 → 前向/后向填充（处理传感器短暂停采）
      - 全列 NaN 的 X 特征列自动剔除（整段停产传感器）
    该文件通常超过 100 MB，不随代码库分发，需先运行
    data_processing/merge_final.py --line {line} 生成。

    返回:
      df: 包含所有特征列和 'y_grade' 列的 DataFrame（无 NaN）
      X_cols: 特征列名列表（不包含 y_grade）
      var_to_stage: {变量名: Stage_ID}
      var_to_group: {变量名: Group}
    """
    var_to_stage, var_to_group, valid_vars = load_vars_and_stages(line)
    y_col = LINE_TO_Y_COL[line]

    # 加载时序宽表（连续 1min 时序，供时序因果发现算法使用）
    dataset_path = os.path.join(DATA_DIR, f"timeseries_dataset_{line}_final.parquet")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"找不到时序宽表: {dataset_path}\n"
            f"请先运行 data_processing/merge_final.py --line {line}"
        )

    df_raw = pd.read_parquet(dataset_path)
    df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)

    # 确认 Y 列存在
    if y_col not in df_raw.columns:
        raise KeyError(
            f"时序宽表中没有 Y 列 '{y_col}'，请检查 data_processing/merge_final.py 的输出。"
        )

    # 只保留在 valid_vars 中且实际存在于数据集的特征列
    X_cols = [c for c in valid_vars if c in df_raw.columns]
    missing = [c for c in valid_vars if c not in df_raw.columns]
    if missing:
        print(f"  [警告] {len(missing)} 个变量不在时序宽表中，将跳过: {missing[:5]}...")

    # 构建输出 DataFrame：选取 X 特征列 + Y 列（重命名为 y_grade）
    df = df_raw[X_cols].copy()
    df["y_grade"] = df_raw[y_col]

    # Y 列稀疏 → 前向填充后向填充（品位为缓变量，最近测量值是合理估计）
    df["y_grade"] = df["y_grade"].ffill().bfill()

    # X 列前向/后向填充（处理传感器短暂停采缺口，保持时序连续性）
    df[X_cols] = df[X_cols].ffill().bfill()

    # 剔除填充后仍为全 NaN 的 X 特征列（整段停产的传感器通道）
    all_nan_cols = [c for c in X_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"  [提示] 剔除 {len(all_nan_cols)} 个全 NaN 的 X 特征列")
        df = df.drop(columns=all_nan_cols)
        X_cols = [c for c in X_cols if c not in all_nan_cols]

    # 丢弃 y_grade 仍为 NaN 的行（ffill+bfill 后极少，说明整段无任何化验记录）
    df = df.dropna(subset=["y_grade"])

    # 更新 var_to_stage / var_to_group，只保留实际存在的特征
    X_cols_set = set(X_cols)
    var_to_stage = {v: s for v, s in var_to_stage.items() if v in X_cols_set}
    var_to_group = {v: g for v, g in var_to_group.items() if v in X_cols_set}

    print(f"  [产线={line}] 时序宽表: {len(df)} 行 x {len(df.columns)} 列, "
          f"Y={y_col}, 时间范围: {df.index.min()} ~ {df.index.max()}")

    return df, X_cols, var_to_stage, var_to_group
