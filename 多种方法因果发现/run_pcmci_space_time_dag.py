"""
run_pcmci_space_time_dag.py
===========================
基于时间和空间拓扑限制，运行 PCMCI 并构建因果 DAG。
【双产线版】：支持 line='xin1' / line='xin2' 分别运行，
             Y 分别对应新1精矿品位 和 新2精矿品位。
【完全修复版】：引入递归锁，确保进度条稳定运行。
"""

import os
import sys
import time
import re
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# ─── 导入公共配置 ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config import prepare_data, can_cause

# ─── 全局变量 ─────────────────────────────────────────────────────────────
VAR_NAMES_GLOBAL = []

class StdoutProxy:
    """拦截 stdout 以驱动 tqdm 进度条，使用递归锁防止死循环"""
    def __init__(self, total, original_stdout):
        self.terminal = original_stdout
        self.pbar = tqdm(total=total, desc="[准备中]", unit="var", dynamic_ncols=True, file=self.terminal)
        self.last_j = -1
        self.current_step = "PC"
        self._locked = False

    def write(self, message):
        if self._locked:
            self.terminal.write(message)
            return
        self._locked = True
        try:
            if "Step 1:" in message:
                self.current_step = "PC"
                self.pbar.set_description(f"[{self.current_step}] 启动中...")
            elif "Step 2:" in message:
                self.current_step = "MCI"
                self.pbar.reset()
                self.last_j = -1
                self.pbar.set_description(f"[{self.current_step}] 启动中...")

            match = re.search(r'Variable j\s*=\s*(\d+)', message)
            if match:
                curr_j = int(match.group(1))
                if curr_j > self.last_j:
                    self.pbar.update(curr_j - self.last_j)
                    self.last_j = curr_j
                    if curr_j < len(VAR_NAMES_GLOBAL):
                        name = VAR_NAMES_GLOBAL[curr_j]
                        self.pbar.set_description(f"[{self.current_step}] {name[:15]}")

            if "Step" in message or "analysis finished" in message:
                self.pbar.write(message.strip())
        finally:
            self._locked = False

    def flush(self):
        self.terminal.flush()


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果")
os.makedirs(OUT_DIR, exist_ok=True)


def run_pcmci(line: str = "xin1"):
    global VAR_NAMES_GLOBAL
    print("=" * 70)
    print(f"PCMCI 因果分析任务启动 [产线 = {line}]")
    print("=" * 70)

    df, X_cols, var_to_stage, var_to_group = prepare_data(line)

    VAR_NAMES_GLOBAL = list(df.columns)  # 含 y_grade
    dataframe = pp.DataFrame(df.values, datatime=np.arange(len(df)), var_names=VAR_NAMES_GLOBAL)
    tau_max = 12

    print(f"[{time.strftime('%H:%M:%S')}] 变量: {len(VAR_NAMES_GLOBAL)}, 样本: {len(df)}")

    # ── 构建时空约束链接假设矩阵 ──
    link_assumptions = {j: {} for j in range(len(VAR_NAMES_GLOBAL))}
    for i, src_name in enumerate(VAR_NAMES_GLOBAL):
        src_stage = 'Y' if src_name == 'y_grade' else var_to_stage.get(src_name, -1)
        src_group = None if src_name == 'y_grade' else var_to_group.get(src_name)
        for j, dst_name in enumerate(VAR_NAMES_GLOBAL):
            dst_stage = 'Y' if dst_name == 'y_grade' else var_to_stage.get(dst_name, -1)
            dst_group = None if dst_name == 'y_grade' else var_to_group.get(dst_name)
            if can_cause(src_stage, dst_stage, src_group, dst_group, line):
                for lag in range(1, tau_max + 1):
                    link_assumptions[j][(i, -lag)] = '-?>'

    cond_ind_test = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=1)

    old_stdout = sys.stdout
    proxy = StdoutProxy(len(VAR_NAMES_GLOBAL), old_stdout)
    sys.stdout = proxy

    try:
        t0 = time.time()
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.01, link_assumptions=link_assumptions)
        proxy.pbar.close()
    finally:
        sys.stdout = old_stdout

    print(f"\n[{time.strftime('%H:%M:%S')}] 分析耗时: {time.time()-t0:.1f} 秒")

    # ── 提取对 y_grade 的显著因果链接 ──
    y_idx = VAR_NAMES_GLOBAL.index("y_grade")
    causal_links = []
    for i, src_name in enumerate(VAR_NAMES_GLOBAL):
        p_mat = results['p_matrix'][i, y_idx, :]
        val_mat = results['val_matrix'][i, y_idx, :]
        for lag in range(1, tau_max + 1):
            if p_mat[lag] <= 0.01:
                stage = 'Y' if src_name == 'y_grade' else var_to_stage.get(src_name, "Unknown")
                group = 'Y' if src_name == 'y_grade' else var_to_group.get(src_name, "?")
                causal_links.append({
                    "Stage": stage, "Group": group, "Variable": src_name,
                    "Lag_Step": lag, "Effect_Size": val_mat[lag], "P_Value": p_mat[lag]
                })

    causal_df = pd.DataFrame(causal_links)
    if not causal_df.empty:
        causal_df = causal_df.sort_values(by="P_Value")
        out_csv = os.path.join(OUT_DIR, f"pcmci_space_time_effects_on_y_{line}.csv")
        causal_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {out_csv}")
        print(causal_df.head(10).to_string(index=False))
    else:
        print("\n未发现显著路径。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCMCI 双产线因果发现")
    parser.add_argument("--line", choices=["xin1", "xin2", "both"], default="both",
                        help="运行哪条产线: xin1/xin2/both (默认 both)")
    args = parser.parse_args()

    lines = ["xin1", "xin2"] if args.line == "both" else [args.line]
    for ln in lines:
        run_pcmci(ln)
