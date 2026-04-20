"""
preprocess_X.py — X 特征主处理脚本
=====================================
选矿厂 DCS 数据批量预处理，专为双重机器学习 / 因果发现场景设计。

【设计原则】
  1. 物理硬阈值裁剪（per-tag 上下界）：超范围值直接置 NaN，不做插值平滑。
  2. Exception-based ffill + 系统级活跃掩码：
       - PLC/DCS exception-based recording（变化才记录）下，稳定段无记录 ≠ 缺失，
         应在系统活跃区间内无限制 ffill（"最后已知值"语义）。
       - 若全系统超过 MAX_GAP_MINUTES 分钟均无任何新记录，视为真实停产空白，保留 NaN。
  3. 整段大缺失直接保留 NaN：不用全局中位数、不加噪声、不做 EWMA 平滑，
     保护因果时滞结构，下游建模自行处理。
  4. 不做孤立森林等 ML 异常检测：防止把因果信号当作噪声清除掉。
  5. Hash 刷新机制：白名单文件 md5 改变后自动失效所有批次缓存，避免用旧缓存。

【数据放置要求】
  data/
  ├── 大量长时间数据/           ← DCS 批次目录，每个子目录含 .xlsx
  │   ├── 10.13/
  │   ├── 10.22/
  │   └── ...
  ├── 操作变量和混杂变量/
  │   ├── output_MC_可用版.csv                       ← 白名单来源1（NAME 列）
  │   └── output_最终可用变量合集_无digital版.csv    ← 白名单来源2（NAME 列）
  └── .cache/                   ← 自动生成，用于断点续传

【输出】
  data/X_features_final.parquet

【用法】
  python data_processing/preprocess_X.py
  或在脚本底部修改配置常量后运行。

【依赖】
  pip install python-calamine pandas pyarrow numpy
"""

import gc
import hashlib
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from python_calamine import CalamineWorkbook
    _CALAMINE_AVAILABLE = True
except ImportError:
    _CALAMINE_AVAILABLE = False
    print("[警告] python_calamine 未安装，将回退到 openpyxl（速度慢）。")
    print("       建议安装：pip install python-calamine")


# ─── 全局配置 ──────────────────────────────────────────────────────────────────
# 脚本所在目录的上级即为仓库根，data/ 相对于仓库根。
# 如需改变数据路径，修改 BASE_DATA_DIR 即可。
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 白名单来源目录（含两个 CSV，NAME 列合并）
WHITELIST_DIR = BASE_DATA_DIR / "操作变量和混杂变量"
WHITELIST_CSVS = [
    WHITELIST_DIR / "output_whitelist_active.csv",
]

# 多批次数据目录（按时间顺序排列；格式：每目录下含 .xlsx，sheet=变量名，列含 time/value/quality）
# 在此追加新批次路径即可
DATA_BATCH_DIRS: list[Path] = [
    BASE_DATA_DIR / "大量长时间数据" / "10.13",
    BASE_DATA_DIR / "大量长时间数据" / "10.22",
    BASE_DATA_DIR / "大量长时间数据" / "11.4~10.22",
    BASE_DATA_DIR / "大量长时间数据" / "12.12",
    BASE_DATA_DIR / "大量长时间数据" / "12.25数据",
    BASE_DATA_DIR / "大量长时间数据" / "1.9-1.29数据",
    BASE_DATA_DIR / "大量长时间数据" / "3.12plc数据",
]

# 断点缓存目录（会在 data/ 下自动创建）
CACHE_DIR = BASE_DATA_DIR / ".cache"

# 输出文件
OUTPUT_PARQUET = BASE_DATA_DIR / "X_features_final.parquet"

# 日志文件
LOG_FILE = BASE_DATA_DIR / "preprocess_X_log.txt"

# 重采样频率（1 分钟宽表）
RESAMPLE_FREQ = "1min"

# 系统级活跃掩码阈值（分钟）
# PLC/DCS exception-based recording：变化才记录，稳定段无记录不等于缺失。
# 若全系统超过此时长均无任何新记录，视为真实停产空白，停产区间内保留 NaN（不 ffill）。
# 工厂连续生产场景下 24 小时通常足够覆盖所有正常稳定段；检修停车超过此值则保留 NaN 符合预期。
MAX_GAP_MINUTES = 24 * 60  # 单位：分钟

# 物理阈值裁剪字典（可选，格式：{TAG_UPPER: (lo, hi)}）
# 如果某个 tag 没有配置，则不裁剪，但仍会去掉明显的传感器饱和值（±1e6 硬上界）
# 示例：
# PHYSICAL_BOUNDS = {
#     "FX_X1_CONC_GRADE": (0.0, 100.0),
#     "FX_X2_CONC_GRADE": (0.0, 100.0),
# }
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {}

# 全局硬安全上下界（防止传感器饱和/断线时的极端值污染数据）
GLOBAL_LO = -1e6
GLOBAL_HI = 1e6


# ─── 日志 ──────────────────────────────────────────────────────────────────────
def _log(msg: str):
    """打印并追加写入日志文件。"""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ─── 白名单加载（带 md5 hash 缓存机制）─────────────────────────────────────────
def _whitelist_hash() -> str:
    """计算所有白名单 CSV 文件内容的联合 md5，用于检测文件是否变更。"""
    h = hashlib.md5()
    for csv_path in sorted(WHITELIST_CSVS):
        if csv_path.exists():
            h.update(csv_path.read_bytes())
    return h.hexdigest()


def load_whitelist() -> set[str]:
    """
    读取两个白名单 CSV 的 NAME 列，合并去重后返回大写 tag 集合。

    两个 CSV 的来源：
      - output_MC_可用版.csv：磨矿/磁选操作变量（NAME 列）
      - output_最终可用变量合集_无digital版.csv：浮选变量无数字型（NAME 列）
    """
    tags: set[str] = set()
    for csv_path in WHITELIST_CSVS:
        if not csv_path.exists():
            _log(f"[警告] 白名单文件不存在，跳过: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["NAME"])
            batch = set(df["NAME"].dropna().astype(str).str.strip().str.upper())
            batch.discard("")
            tags.update(batch)
            _log(f"[白名单] 从 {csv_path.name} 读取 {len(batch)} 个变量")
        except Exception as e:
            _log(f"[错误] 读取白名单 {csv_path.name} 失败: {e}")
    _log(f"[白名单] 合并后共 {len(tags)} 个唯一变量")
    return tags


# ─── 物理硬阈值裁剪 ────────────────────────────────────────────────────────────
def apply_physical_clip(series: pd.Series, tag: str) -> pd.Series:
    """
    对单变量时间序列做物理范围裁剪：
      - 超出全局安全界（±1e6）的值直接设 NaN（传感器饱和/断线保护）
      - 如果在 PHYSICAL_BOUNDS 中配置了该 tag 的上下界，进一步裁剪

    注意：不做插值，只做置 NaN，保留原始信号完整性。
    """
    # 全局安全界
    s = series.where((series >= GLOBAL_LO) & (series <= GLOBAL_HI))

    # per-tag 物理界
    if tag in PHYSICAL_BOUNDS:
        lo, hi = PHYSICAL_BOUNDS[tag]
        s = s.where((s >= lo) & (s <= hi))

    return s


# ─── 缺失值处理（exception-based ffill + 系统活跃掩码）────────────────────────
def handle_missing_exception_based(
    series: pd.Series,
    fillable_mask: pd.Series,
) -> pd.Series:
    """
    针对 exception-based recording（变化才记录）的缺失处理：
      1. 先无限制 ffill（把所有"值没变所以没记录"的 NaN 都填上）。
      2. 再用 fillable_mask 把真实停产空白重新置回 NaN。

    fillable_mask: 布尔 Series（与 series 同 index），
                   True  = 此分钟距上次系统有任何记录 ≤ MAX_GAP_MINUTES，可填充；
                   False = 全系统超过 MAX_GAP_MINUTES 无任何新记录，停产空白，保留 NaN。
    """
    filled = series.ffill()               # 无限制，先把所有稳定段填满
    return filled.where(fillable_mask)    # 真实停产区间重新清空


# ─── 单个 Excel 文件解析（calamine 高性能 / openpyxl 回退）────────────────────
def _parse_sheet_calamine(wb, sheet_name: str, tag_upper: str) -> pd.Series | None:
    """用 calamine 解析一个 sheet，返回 pd.Series 或 None。"""
    try:
        raw = wb.get_sheet_by_name(sheet_name).to_python()
    except Exception as e:
        _log(f"  [跳过-Sheet] [{sheet_name}]: {e}")
        return None

    if not raw or len(raw) < 2:
        return None

    # 解析表头
    header = [str(h).lower().strip() if h is not None else "" for h in raw[0]]
    if "time" not in header or "value" not in header:
        return None

    idx_time = header.index("time")
    idx_value = header.index("value")
    idx_qual = header.index("quality") if "quality" in header else None

    times, values = [], []
    for row in raw[1:]:
        if row is None or len(row) <= max(idx_time, idx_value):
            continue

        # quality 过滤：只保留 "good"
        if idx_qual is not None and idx_qual < len(row):
            q = str(row[idx_qual]).lower().strip() if row[idx_qual] is not None else ""
            if q not in ("good", ""):
                continue

        t = row[idx_time]
        v = row[idx_value]
        if t is None or v is None:
            continue

        try:
            t = pd.Timestamp(t)
        except Exception:
            continue

        if isinstance(v, bool):
            v = int(v)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue

        times.append(t)
        values.append(np.float32(v))

    if not times:
        return None

    s = pd.Series(
        values,
        index=pd.DatetimeIndex(times),
        name=tag_upper,
        dtype=np.float32,
    )
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def _parse_sheet_openpyxl(xlsx_path: Path, sheet_name: str, tag_upper: str) -> pd.Series | None:
    """用 openpyxl 解析一个 sheet（回退方案），返回 pd.Series 或 None。"""
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, usecols=["time", "value", "quality"],
                           engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
            df.columns = [str(c).lower().strip() for c in df.columns]
            if "time" not in df.columns or "value" not in df.columns:
                return None
        except Exception as e:
            _log(f"  [跳过-Sheet openpyxl] [{sheet_name}]: {e}")
            return None

    if "quality" in df.columns:
        df = df[df["quality"].astype(str).str.lower().str.strip().isin(["good", "nan", ""])]

    df = df.dropna(subset=["time", "value"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    if df.empty:
        return None

    s = pd.Series(
        df["value"].values.astype(np.float32),
        index=pd.DatetimeIndex(df["time"].values),
        name=tag_upper,
    )
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def process_single_excel(xlsx_path: Path, whitelist: set[str]) -> dict[str, pd.Series]:
    """
    解析单个 xlsx 文件，返回 {tag_upper: pd.Series}。

    步骤：
      1. 用白名单做 sheet 预过滤，只读取匹配的 sheet（大写比较）。
      2. quality != 'good' 的行丢弃。
      3. 每个 tag 返回带时间索引的 float32 Series。
    """
    result: dict[str, pd.Series] = {}

    if _CALAMINE_AVAILABLE:
        try:
            wb = CalamineWorkbook.from_path(str(xlsx_path))
            sheet_names = wb.sheet_names
        except Exception as e:
            _log(f"  [跳过] 无法打开 {xlsx_path.name}: {e}")
            return result

        matching = [(s, s.strip().upper()) for s in sheet_names if s.strip().upper() in whitelist]
        for sheet_name, tag_upper in matching:
            s = _parse_sheet_calamine(wb, sheet_name, tag_upper)
            if s is not None:
                result[tag_upper] = s
    else:
        # openpyxl 回退：先获取 sheet 名再逐一解析
        try:
            import openpyxl
            wb_meta = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
            sheet_names = wb_meta.sheetnames
            wb_meta.close()
        except Exception as e:
            _log(f"  [跳过] 无法打开 {xlsx_path.name}: {e}")
            return result

        matching = [(s, s.strip().upper()) for s in sheet_names if s.strip().upper() in whitelist]
        for sheet_name, tag_upper in matching:
            s = _parse_sheet_openpyxl(xlsx_path, sheet_name, tag_upper)
            if s is not None:
                result[tag_upper] = s

    return result


# ─── 批次处理（含断点续传 + hash 校验）────────────────────────────────────────
def _cache_hash_file(whitelist_hash: str) -> Path:
    """返回存放 whitelist hash 的标记文件路径。"""
    return CACHE_DIR / "_whitelist_hash.txt"


def _check_and_invalidate_cache(whitelist_hash: str) -> bool:
    """
    检查缓存目录中的 hash 标记文件：
      - 若 hash 与当前一致，缓存有效，返回 True。
      - 若 hash 不一致或文件不存在，删除所有批次缓存，返回 False。
    """
    hash_file = _cache_hash_file(whitelist_hash)
    if hash_file.exists():
        stored = hash_file.read_text(encoding="utf-8").strip()
        if stored == whitelist_hash:
            return True  # 缓存有效
        _log(f"[缓存] 白名单已变更（旧 hash={stored[:8]}… 新 hash={whitelist_hash[:8]}…）")
    else:
        _log(f"[缓存] 未找到 hash 标记文件，缓存视为无效")

    # 删除所有旧批次缓存
    for old_cache in CACHE_DIR.glob("batch_*.parquet"):
        old_cache.unlink()
        _log(f"[缓存] 已删除旧缓存: {old_cache.name}")

    return False  # 缓存无效


def _save_whitelist_hash(whitelist_hash: str):
    """将当前 whitelist hash 写入标记文件。"""
    CACHE_DIR.mkdir(exist_ok=True)
    (_cache_hash_file(whitelist_hash)).write_text(whitelist_hash, encoding="utf-8")


def process_batch_dir(
    data_dir: Path,
    whitelist: set[str],
    accumulated: dict[str, list],
    batch_idx: int,
    total_batches: int,
):
    """
    扫描 data_dir 下所有 xlsx，提取白名单变量，追加到 accumulated。
    accumulated: {tag: [Series, ...]} 跨批次共享。
    """
    all_files = sorted(data_dir.rglob("*.xlsx"))
    total = len(all_files)
    _log(f"[批次 {batch_idx}/{total_batches}] 目录: {data_dir}  共 {total} 个 xlsx 文件")

    t_start = time.time()
    for i, fpath in enumerate(all_files):
        rel = str(fpath.relative_to(data_dir))
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            _log(f"  [{i+1}/{total}] {rel}  "
                 f"(已用 {elapsed/60:.1f}min, 预计剩余 {eta/60:.1f}min)")

        parts = process_single_excel(fpath, whitelist)
        for tag, s in parts.items():
            # 提前重采样到目标频率，避免将原始高频数据全部堆入内存
            accumulated[tag].append(s.resample(RESAMPLE_FREQ).mean().astype(np.float32))

        if (i + 1) % 50 == 0:
            gc.collect()

    _log(f"[批次 {batch_idx}/{total_batches}] 完成，耗时 {(time.time()-t_start)/60:.1f} 分钟")


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def build_features(
    batch_dirs: list[Path],
    whitelist: set[str],
    output_path: Path,
):
    """
    处理所有批次数据目录，合并后按时间顺序输出宽表 Parquet。

    流程：
      1. 检查白名单 hash，决定是否复用批次缓存。
      2. 按批次读取 xlsx，每批次解析完立即写入缓存（断点保护）。
      3. 合并所有批次 → 按 1min 重采样 → 物理裁剪 → exception-based ffill（系统活跃掩码）→ 停产空白保留 NaN。
      4. 拼宽表输出 Parquet。
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # hash 校验
    wl_hash = _whitelist_hash()
    cache_valid = _check_and_invalidate_cache(wl_hash)

    # 初始化每个白名单变量的 Series 列表
    accumulated: dict[str, list] = defaultdict(list)

    t_all_start = time.time()

    for idx, data_dir in enumerate(batch_dirs, start=1):
        if not data_dir.exists():
            _log(f"[警告] 批次目录不存在，已跳过: {data_dir}")
            continue

        cache_file = CACHE_DIR / f"batch_{idx}_{data_dir.name}.parquet"

        # 断点续传：如果缓存有效且文件存在，直接读缓存
        if cache_valid and cache_file.exists():
            _log(f"[断点续传] 读取批次缓存 ({idx}/{len(batch_dirs)}): {cache_file.name}")
            try:
                df_cache = pd.read_parquet(cache_file)
                for col in df_cache.columns:
                    col_upper = col.upper()
                    if col_upper in whitelist:
                        s = df_cache[col].dropna().astype(np.float32)
                        if not s.empty:
                            accumulated[col_upper].append(s)
                continue
            except Exception as e:
                _log(f"[警告] 缓存文件损坏 ({e})，退回原始处理流程...")

        # 从原始 Excel 读取
        len_before = {tag: len(accumulated[tag]) for tag in whitelist}
        process_batch_dir(data_dir, whitelist, accumulated, idx, len(batch_dirs))

        # 写批次缓存（断点保护）
        _log(f"  [断点保护] 正在写入批次缓存...")
        batch_parts: dict[str, pd.Series] = {}
        for tag in whitelist:
            added = accumulated[tag][len_before[tag]:]
            if added:
                s = pd.concat(added)
                s = s[~s.index.duplicated(keep="first")]
                batch_parts[tag] = s

        if batch_parts:
            pd.DataFrame(batch_parts).to_parquet(cache_file, compression="snappy")
            size_mb = cache_file.stat().st_size / 1024 / 1024
            _log(f"  [断点保护] 已保存批次快照: {cache_file.name} ({size_mb:.1f} MB)")

    # hash 标记（只有全部批次处理完才写，中断重跑时不会误判为有效）
    _save_whitelist_hash(wl_hash)

    # ── 合并 + 重采样 + 物理裁剪 + 缺失处理 ─────────────────────────────────
    _log("\n[合并] 开始合并、重采样、物理裁剪...")

    # ── 第一循环：resample + 物理裁剪，暂不 ffill，存原始稀疏序列 ──────────
    all_resampled_raw: dict[str, pd.Series] = {}

    for tag in whitelist:
        parts = accumulated[tag]
        if not parts:
            continue

        # 多批次 concat + 去重 + 时间排序
        combined = pd.concat(parts).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # 1min 重采样（均值）
        resampled = combined.resample(RESAMPLE_FREQ).mean()

        # 物理硬阈值裁剪（超范围置 NaN，不插值）
        resampled = apply_physical_clip(resampled, tag)

        # 暂不 ffill，保留稀疏版本
        all_resampled_raw[tag] = resampled.astype(np.float32)

    found = len(all_resampled_raw)
    _log(f"[合并] 在所有批次中共找到 {found} / {len(whitelist)} 个白名单变量")

    if not all_resampled_raw:
        _log("[错误] 没有任何变量有数据！请检查白名单与文件格式。")
        return

    # ── 计算系统级活跃掩码 ────────────────────────────────────────────────
    _log("[活跃掩码] 基于全变量并集计算系统数据活跃区间...")

    # 任意分钟只要有任何一个变量有真实记录，就标记为活跃
    _sample_series = next(iter(all_resampled_raw.values()))
    any_data = pd.Series(False, index=_sample_series.index)
    for s in all_resampled_raw.values():
        any_data |= s.notna()

    # 向前传播活跃标记：从最后一次有数据的时刻起，向后延伸 MAX_GAP_MINUTES 分钟。
    # 若超过 MAX_GAP_MINUTES 仍无任何数据 → fillable=False → 停产空白，保留 NaN。
    fillable = any_data.astype(float).where(any_data, other=np.nan)
    fillable = fillable.ffill(limit=MAX_GAP_MINUTES).notna()

    n_active = int(fillable.sum())
    n_total = len(fillable)
    _log(f"[活跃掩码] 活跃分钟数: {n_active:,} / {n_total:,} ({n_active/n_total:.1%})")
    _log(f"[活跃掩码] 真实空白（停产）分钟数: {n_total - n_active:,}")

    # ── 第二循环：在活跃区间内 ffill，停产区间保留 NaN ──────────────────
    _log("[ffill] 对各变量应用 exception-based ffill 策略...")
    all_resampled: dict[str, pd.Series] = {}
    for tag, s in all_resampled_raw.items():
        all_resampled[tag] = handle_missing_exception_based(s, fillable).astype(np.float32)

    # ── 拼宽表 ──────────────────────────────────────────────────────────────
    _log("\n[拼接] 合并宽表...")
    X_df = pd.concat(list(all_resampled.values()), axis=1, sort=True)
    X_df.columns = list(all_resampled.keys())
    X_df.index.name = "time"
    X_df = X_df.sort_index()

    _log(f"[完成] 宽表形状: {X_df.shape}")
    _log(f"       时间范围: {X_df.index.min()} ~ {X_df.index.max()}")
    _log(f"       整体缺失率: {X_df.isnull().mean().mean():.2%}")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"\n[保存] 写入 {output_path} ...")
    X_df.to_parquet(output_path, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    _log(f"[完成] 保存成功！文件大小: {size_mb:.1f} MB")
    _log(f"[总耗时] {(time.time()-t_all_start)/60:.1f} 分钟")

    # 未找到的变量报告
    missing = sorted(whitelist - set(all_resampled.keys()))
    if missing:
        _log(f"\n[提示] {len(missing)} 个白名单变量在所有批次中均未找到")
        for v in missing[:30]:
            _log(f"  {v}")
        if len(missing) > 30:
            _log(f"  ...及其余 {len(missing)-30} 个")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 初始化日志
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    _log("=" * 60)
    _log("preprocess_X.py — X 特征主处理脚本 启动")
    _log(f"数据目录: {BASE_DATA_DIR}")
    _log(f"输出文件: {OUTPUT_PARQUET}")
    _log(f"批次目录数: {len(DATA_BATCH_DIRS)}")
    for d in DATA_BATCH_DIRS:
        _log(f"  {d}")
    _log(f"停产判断阈值: MAX_GAP_MINUTES = {MAX_GAP_MINUTES} 分钟（{MAX_GAP_MINUTES/60:.0f} 小时）")
    _log("=" * 60)

    whitelist = load_whitelist()
    if not whitelist:
        _log("[错误] 白名单为空，退出。请检查白名单 CSV 文件路径和 NAME 列。")
    else:
        build_features(DATA_BATCH_DIRS, whitelist, OUTPUT_PARQUET)
        _log("\n全部完成！")
