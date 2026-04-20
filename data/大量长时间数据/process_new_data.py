"""
process_new_data.py  (v5 - 多批次支持 + 新白名单版)
====================================================
主要变更（v5）：
  1. 白名单来源改为 操作变量和混杂变量/ 下两个CSV的 NAME 列合并
     - output_MC_可用版.csv          （磨矿/磁选操作变量）
     - output_最终可用变量合集_无digital版.csv  （浮选变量无数字型）
  2. 支持多批次数据目录（DATA_BATCH_DIRS），按时间顺序统一处理
     - 所有批次格式与 大量长时间数据/ 相同（xlsx, sheet=变量名, 列=time/value/quality）
     - 处理完所有批次后，时间轴自动排序合并
  3. 保持 calamine 高性能读取（v4 优化成果）
  4. 输出文件保存时间范围、批次来源到 process_log.txt

安装依赖：
    pip install python-calamine

运行：
    python 大量长时间数据/process_new_data.py
"""

import gc, time
import numpy as np
import pandas as pd
from pathlib import Path
from python_calamine import CalamineWorkbook
from sklearn.ensemble import IsolationForest


# ─── 配置 ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"c:\backup\doubleml")
OUTPUT_PARQUET = BASE_DIR / "X_features_new.parquet"

# ── 白名单来源（操作变量和混杂变量）──────────────
WHITELIST_DIR      = BASE_DIR / "操作变量和混杂变量"
WHITELIST_CSV      = WHITELIST_DIR / "output_ABC_分类合集_带变化标签.csv"

# ── 多批次数据目录（按先后顺序填写；格式要求：xlsx, sheet=变量名, 列含 time/value/quality）
# 说明：将来新增批次时，在此列表末尾追加对应 Path 即可。
DATA_BATCH_DIRS: list[Path] = [
    BASE_DIR / "大量长时间数据" / "10.13",
    BASE_DIR / "大量长时间数据" / "10.22",
    BASE_DIR / "大量长时间数据" / "11.4~10.22",
    BASE_DIR / "大量长时间数据" / "12.12",
    BASE_DIR / "大量长时间数据" / "12.25数据",
    BASE_DIR / "大量长时间数据" / "1.9-1.29数据",
    BASE_DIR / "大量长时间数据" / "3.12plc数据",
]

LOG_FILE      = BASE_DIR / "大量长时间数据" / "process_log.txt"

RESAMPLE_FREQ = "1min"
FFILL_LIMIT   = 60


# ─── 日志 ──────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ─── 白名单加载（从分类合集中读取，过滤出变化状态为 Active 的变量）─────────────────
def load_whitelist() -> set:
    """
    读取 output_ABC_分类合集_带变化标签.csv
    仅保留 "change_status" (或用户口中的 change_count) 状态为 "Active" 的行对应的 NAME 列。
    这是在之前排查中验证过有实际物理波动的“活变量”。
    """
    try:
        df = pd.read_csv(WHITELIST_CSV, encoding='utf-8')
        
        # 很多用户的 csv 是由不同流程写出的，这里做一个兼容保护，检查列是否存在
        status_col = 'change_status' if 'change_status' in df.columns else 'change_count'
        
        # 过滤出值为 Active 的行 (大小写全兼容处理)
        df_active = df[df[status_col].astype(str).str.strip().str.lower() == 'active']
        
        tags = set(df_active["NAME"].dropna().str.strip().str.upper())
        tags.discard("")
        log(f"[白名单] 从 {WHITELIST_CSV.name} 过滤出 Active 变量，共获取到 {len(tags)} 个")
        return tags
    except Exception as e:
        log(f"[错误] 白名单加载失败: {e}")
        return set()


# ─── 处理单个 Excel 文件（calamine 高性能版）────────────────────────────────────
def process_single_excel(xlsx_path: Path, whitelist: set) -> dict[str, pd.Series]:
    """
    用 calamine 读取 xlsx（约比 openpyxl 快 25-200 倍）：
    1. 打开 workbook，获取所有 sheet 名
    2. 用白名单做预过滤，只读取匹配的 sheet
    3. 解析 time / value / quality 列，quality 非 'good' 的行丢弃
    4. 返回 {tag_upper: pd.Series(index=DatetimeIndex, dtype=float32)}
    """
    result: dict[str, pd.Series] = {}
    try:
        wb = CalamineWorkbook.from_path(str(xlsx_path))
        sheet_names = wb.sheet_names
    except Exception as e:
        log(f"  [跳过] 无法打开 {xlsx_path.name}: {e}")
        return result

    matching = [(s, s.upper()) for s in sheet_names if s.upper() in whitelist]
    if not matching:
        return result

    for sheet_name, tag_upper in matching:
        try:
            raw = wb.get_sheet_by_name(sheet_name).to_python()
        except Exception as e:
            log(f"  [跳过-Sheet] {xlsx_path.name}[{sheet_name}]: {e}")
            continue

        if not raw or len(raw) < 2:
            continue

        # 解析 header
        header = [str(h).lower().strip() if h is not None else "" for h in raw[0]]
        if "time" not in header or "value" not in header:
            continue

        idx_time  = header.index("time")
        idx_value = header.index("value")
        idx_qual  = header.index("quality") if "quality" in header else None

        times, values = [], []
        for row in raw[1:]:
            if row is None or len(row) <= max(idx_time, idx_value):
                continue

            # quality 过滤
            if idx_qual is not None and idx_qual < len(row):
                q = str(row[idx_qual]).lower().strip() if row[idx_qual] is not None else ""
                if q != "good":
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
            continue

        series = pd.Series(
            values,
            index=pd.DatetimeIndex(times),
            name=tag_upper,
            dtype=np.float32,
        )
        series = series[~series.index.duplicated(keep="first")].sort_index()
        result[tag_upper] = series

    return result


# ─── 处理单个批次目录 ─────────────────────────────────────────────────────────
def process_batch_dir(
    data_dir: Path,
    whitelist: set,
    accumulated: dict[str, list],
    batch_idx: int,
    total_batches: int,
):
    """
    扫描 data_dir 下所有 xlsx，提取白名单变量，追加到 accumulated。
    accumulated: {tag: [Series, ...]} 跨批次共享，在本函数内追加。
    """
    all_files = sorted(data_dir.rglob("*.xlsx"))
    total = len(all_files)
    log(f"[批次 {batch_idx}/{total_batches}] 目录: {data_dir}  共 {total} 个 xlsx 文件")

    t_start = time.time()
    for i, fpath in enumerate(all_files):
        rel = str(fpath.relative_to(data_dir))

        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            log(f"  [{i+1}/{total}] {rel}  "
                f"(已用{elapsed/60:.1f}min, 预计剩余{eta/60:.1f}min)")

        parts = process_single_excel(fpath, whitelist)
        for tag, s in parts.items():
            accumulated[tag].append(s)

        if (i + 1) % 50 == 0:
            gc.collect()

    log(f"[批次 {batch_idx}/{total_batches}] 完成，耗时 {(time.time()-t_start)/60:.1f} 分钟\n")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def build_features(
    batch_dirs: list[Path],
    whitelist: set,
    output_path: Path,
):
    """
    处理所有批次数据目录，合并后按时间顺序输出宽表 Parquet。
    已增持【批次级防灾/断点续传】：每个批次解析完毕立刻缓存，中断后秒恢复。
    """
    # 初始化每个白名单变量的 Series 列表（所有批次共享）
    accumulated: dict[str, list] = {tag: [] for tag in whitelist}

    CACHE_DIR = BASE_DIR / "大量长时间数据" / ".cache"
    CACHE_DIR.mkdir(exist_ok=True)

    t_all_start = time.time()
    for idx, data_dir in enumerate(batch_dirs, start=1):
        if not data_dir.exists():
            log(f"[警告] 批次目录不存在，已跳过: {data_dir}")
            continue
            
        cache_file = CACHE_DIR / f"batch_{idx}_{data_dir.name}.parquet"
        
        # 1. 触发断点续传尝试
        if cache_file.exists():
            log(f"[断点续传] 读取已缓存批次 ({idx}/{len(batch_dirs)}): {cache_file.name}")
            try:
                df_cache = pd.read_parquet(cache_file)
                for col in df_cache.columns:
                    if col in whitelist:
                        s = df_cache[col].dropna()
                        if not s.empty:
                            accumulated[col].append(s)
                continue
            except Exception as e:
                log(f"[警告] 缓存文件损坏({e})，退回原始处理流程...")
        
        # 2. 从原始 Excel 读取（普通流程）
        # 记录处理本批次前积累的数组长度，以便后续剥离该批次结果
        len_before = {tag: len(accumulated[tag]) for tag in whitelist}
        process_batch_dir(data_dir, whitelist, accumulated, idx, len(batch_dirs))
        
        # 3. 稳妥起见，每个批次结束后立即释放并落盘一个 Checkpoint
        log(f"  [断点防灾] 正在将本批次抽取的数据写入硬盘缓存区...")
        batch_parts = {}
        for tag in whitelist:
            added = accumulated[tag][len_before[tag]:]
            if added:
                s = pd.concat(added)
                s = s[~s.index.duplicated(keep="first")]
                batch_parts[tag] = s
                
        if batch_parts:
            # 外连接构建当前批次的 Dataframe 生成缓存
            pd.DataFrame(batch_parts).to_parquet(cache_file, compression="snappy")
            size_mb = cache_file.stat().st_size / 1024 / 1024
            log(f"  [断点防灾] 已保存批次快照: {cache_file.name} ({size_mb:.1f} MB)\n")

    # ── 合并 + 重采样（按时间排序） ─────────────────────────────────────────
    log("\n[合并] 开始按变量合并、重采样（时间顺序），并执行机器学习异常过滤...")
    all_resampled: dict[str, pd.Series] = {}
    total_anomalies_removed = 0

    for tag in whitelist:
        parts = accumulated[tag]
        if not parts:
            continue

        # 多批次 Series 直接 concat，sort_index 保证时间顺序
        combined = pd.concat(parts).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        resampled = combined.resample(RESAMPLE_FREQ).mean()
        
        # --- 孤立森林异常点屏蔽 (Masking) ---
        # 工业场景下，用机器学习捕捉单变量非线性跳变，比传统的 3-sigma 更稳健
        notna_idx = resampled.dropna().index
        if len(notna_idx) > 50:
            # contamination=0.005 意味着我们只重点打击约0.5%的极端离群值
            clf = IsolationForest(n_estimators=50, contamination=0.005, random_state=42, n_jobs=-1)
            X_arr = resampled.loc[notna_idx].values.reshape(-1, 1)
            preds = clf.fit_predict(X_arr)
            
            # -1 代表被孤立森林标记出来的极端异常点
            anomaly_mask = (preds == -1)
            anomalies_count = anomaly_mask.sum()
            
            if anomalies_count > 0:
                # 核心逻辑：不要整行删掉破坏时间轴，而是设为空值 (NaN)，交给后续特征工程去"缝合"它
                resampled.loc[notna_idx[anomaly_mask]] = np.nan
                total_anomalies_removed += anomalies_count
        # --------------------------------

        # ====== 1. 缺失值处理: 绝对避开论文的“拉格朗日插值”和“稳态大段删除” ======
        # 对于短期的偶尔丢包(如2小时内)：采用 Time-aware 时间感知线性插值，规避拉格朗日多项式的龙格震荡现象
        resampled = resampled.interpolate(method='time', limit=120)
        
        # 对于重大背景：“大量工作人员未及时拷贝导致的大片缺失”：
        # 坚决不使用论文中按稳态判定去切断并删除时间轴的方法！（删除会直接破坏下游 PCMCI 连续的时间因果结构）。
        # 这里改用 全局中位数结合微小高斯底噪 进行恒态平稳填充，底噪用于防止填补成绝对直线导致下游双重机器学习提取引发“零方差”异常。
        missing_mask = resampled.isna()
        missing_count = missing_mask.sum()
        if missing_count > 0:
            median_val = resampled.median()
            std_val = resampled.std()
            if pd.isna(median_val):
                median_val = 0.0
            if pd.isna(std_val) or std_val == 0.0:
                std_val = 0.01
            
            # 使用高斯随机数生成极弱的底噪向量
            noise = np.random.normal(0, std_val * 0.01, size=missing_count)
            resampled.loc[missing_mask] = median_val + noise

        # ====== 2. 数据降噪处理: 绝对避开论文的“Savitzky-Golay 多项式平滑去噪” ======
        # 采用工业控制中更通用的 Exponential Weighted Moving Average (EWMA) 一阶滞后低通滤波。
        # 它完美模拟了现场仪表的 RC 模拟滤波电路响应，不存在 Savitzky-Golay 多项式拟合特有的局部过度平滑或边缘跳变问题。
        resampled = resampled.ewm(span=15, adjust=False).mean()

        all_resampled[tag] = resampled.astype(np.float32)

    found = len(all_resampled)
    log(f"[合并] 在所有批次中共找到 {found} / {len(whitelist)} 个白名单变量")
    log(f"[异常清洗] 孤立森林共屏蔽剔除了 {total_anomalies_removed} 个极其恶劣的传感器噪声点！")

    if not all_resampled:
        log("[错误] 没有任何变量有数据！请检查白名单与文件格式。")
        return

    # ── 拼宽表 ──────────────────────────────────────────────────────────────
    log("\n[拼接] 合并宽表...")
    X_df = pd.concat(list(all_resampled.values()), axis=1, sort=True)
    X_df.columns = list(all_resampled.keys())
    X_df.index.name = "time"
    X_df = X_df.sort_index()   # 保证整体时间顺序

    log(f"[完成] 宽表形状: {X_df.shape}")
    log(f"       时间范围: {X_df.index.min()} ~ {X_df.index.max()}")
    log(f"       整体缺失率: {X_df.isnull().mean().mean():.2%}")

    log(f"\n[保存] 写入 {output_path} ...")
    X_df.to_parquet(output_path, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    log(f"[完成] 保存成功！文件大小: {size_mb:.1f} MB")
    log(f"[总耗时] {(time.time()-t_all_start)/60:.1f} 分钟")

    # 未找到的变量
    missing = sorted(whitelist - set(all_resampled.keys()))
    if missing:
        log(f"\n[提示] {len(missing)} 个白名单变量在所有批次中均未找到")
        for v in missing[:30]:
            log(f"  {v}")
        if len(missing) > 30:
            log(f"  ...及其余 {len(missing)-30} 个")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 60)
    log("新数据预处理脚本 v5（多批次 + 新白名单版）启动")
    log(f"批次目录列表：")
    for d in DATA_BATCH_DIRS:
        log(f"  {d}")
    log("=" * 60)

    whitelist = load_whitelist()
    build_features(DATA_BATCH_DIRS, whitelist, OUTPUT_PARQUET)

    log("\n全部完成！")
