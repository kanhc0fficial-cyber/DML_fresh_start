"""
get_time_range.py  (v2 - 多批次支持版)
========================================
扫描多个批次数据目录的所有 xlsx 文件，输出整体时间范围。
各批次目录格式要求：xlsx, sheet 含 time 列。

新增功能：
  - 支持多批次目录（DATA_BATCH_DIRS），与 process_new_data.py 配置保持一致
  - 对每个批次单独汇报时间范围，最后给出所有批次合并后的时间跨度
  - 使用 calamine 高性能读取（比 openpyxl 快 25-200 倍）
  - 利用每个 sheet 首/末行做快速估算，不全量遍历（大幅提速）
"""
import time
import pandas as pd
from pathlib import Path
from python_calamine import CalamineWorkbook


# ─── 配置（与 process_new_data.py 保持一致）─────────────────────────────────
BASE_DIR = Path(r"c:\backup\doubleml")

DATA_BATCH_DIRS: list[Path] = [
    BASE_DIR / "大量长时间数据",
    # BASE_DIR / "第二批数据",
    # BASE_DIR / "第三批数据",
]


# ─── 从一个 sheet 的首/末行快速提取时间边界 ───────────────────────────────────
def _quick_time_range(raw: list) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    不全量扫描，取首行（raw[1]）和末行（raw[-1]）的 time 列做快速边界估计。
    对已按时间排序的文件完全准确；对乱序文件可能低估范围（可接受）。
    """
    if not raw or len(raw) < 2:
        return None, None

    header = [str(h).lower().strip() if h is not None else "" for h in raw[0]]
    if "time" not in header:
        return None, None

    idx_time = header.index("time")

    candidates = []
    # 首行、末行、以及中间抽样几行（应对极少数乱序情况）
    sample_rows = [raw[1], raw[-1]]
    if len(raw) > 3:
        mid = len(raw) // 2
        sample_rows.append(raw[mid])

    for row in sample_rows:
        if row is None or len(row) <= idx_time:
            continue
        t = row[idx_time]
        if t is not None:
            try:
                candidates.append(pd.Timestamp(t))
            except Exception:
                pass

    if not candidates:
        return None, None
    return min(candidates), max(candidates)


# ─── 扫描单个批次目录 ─────────────────────────────────────────────────────────
def scan_batch(data_dir: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """返回该批次目录内所有文件的（最早时间, 最晚时间）"""
    all_files = sorted(data_dir.rglob("*.xlsx"))
    total = len(all_files)
    print(f"  目录: {data_dir}  共 {total} 个 xlsx 文件")

    global_min = pd.Timestamp.max
    global_max = pd.Timestamp.min
    t_start = time.time()

    for i, fpath in enumerate(all_files):
        try:
            wb = CalamineWorkbook.from_path(str(fpath))
            sheet_names = wb.sheet_names
        except Exception:
            continue

        file_min = pd.Timestamp.max
        file_max = pd.Timestamp.min

        for sheet_name in sheet_names:
            try:
                raw = wb.get_sheet_by_name(sheet_name).to_python()
            except Exception:
                continue

            t_min, t_max = _quick_time_range(raw)
            if t_min is not None and t_min < file_min:
                file_min = t_min
            if t_max is not None and t_max > file_max:
                file_max = t_max

        if file_min != pd.Timestamp.max and file_max != pd.Timestamp.min:
            if file_min < global_min:
                global_min = file_min
            if file_max > global_max:
                global_max = file_max

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"    [{i+1}/{total}] 当前范围: {global_min} ~ {global_max}")

    elapsed = time.time() - t_start
    result_min = global_min if global_min != pd.Timestamp.max  else None
    result_max = global_max if global_max != pd.Timestamp.min else None
    print(f"  耗时: {elapsed:.1f}s  范围: {result_min} ~ {result_max}\n")
    return result_min, result_max


# ─── 主函数 ───────────────────────────────────────────────────────────────────
def get_time_range():
    print("=" * 55)
    print("多批次时间范围扫描（快速首/末行模式）")
    print("=" * 55)

    overall_min = pd.Timestamp.max
    overall_max = pd.Timestamp.min
    t_total = time.time()

    for idx, data_dir in enumerate(DATA_BATCH_DIRS, start=1):
        print(f"\n[批次 {idx}/{len(DATA_BATCH_DIRS)}]")
        if not data_dir.exists():
            print(f"  [警告] 目录不存在，已跳过: {data_dir}")
            continue

        b_min, b_max = scan_batch(data_dir)
        if b_min is not None and b_min < overall_min:
            overall_min = b_min
        if b_max is not None and b_max > overall_max:
            overall_max = b_max

    print("\n" + "=" * 55)
    print("所有批次扫描完成！")
    if overall_min != pd.Timestamp.max:
        print(f"数据起始时间点 (Earliest): {overall_min}")
        print(f"数据结束时间点 (Latest)  : {overall_max}")
        span = overall_max - overall_min
        print(f"总时间跨度              : {span}")
    else:
        print("未找到任何有效时间数据！")
    print(f"总耗时: {time.time()-t_total:.1f} 秒")
    print("=" * 55)


if __name__ == "__main__":
    get_time_range()
