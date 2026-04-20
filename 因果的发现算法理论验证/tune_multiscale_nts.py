"""
tune_multiscale_nts.py
======================
MultiScale-NTS 超参数调优脚本（针对模拟数据场景）

使用网格搜索对 MultiScale-NTS 的关键超参数进行调优，
评估指标为多次蒙特卡洛实验的平均 F1 分数。

可调超参数:
  - window_size    : 时间窗口大小
  - kernel_sizes   : 多尺度卷积核大小组合
  - hidden_mult    : 隐藏通道倍率 (d * hidden_mult)
  - lr             : 学习率
  - epochs         : 训练轮数
  - batch_size     : 批大小
  - l1_lambda      : L1 稀疏正则化权重
  - h_lambda       : NOTEARS 无环惩罚权重
  - threshold      : 邻接矩阵二值化阈值

用法:
  python tune_multiscale_nts.py                        # 使用默认配置
  python tune_multiscale_nts.py --n_trials 5           # 每组超参数跑5次取平均
  python tune_multiscale_nts.py --n_trials 3 --quick   # 快速模式（缩小搜索空间）
"""

import gc
import os
import sys
import time
import json
import itertools
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 导入合成数据生成器
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_dag_generator import (
    SyntheticDAGGenerator,
    compute_dag_metrics,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "因果发现结果", "hyperparameter_tuning")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 模拟数据生成固定配置（与 benchmark 保持一致） ─────────────────────────
N_NODES = 20
N_SAMPLES = 1000
NOISE_SCALE = 0.1
GRAPH_TYPE = 'layered'
NOISE_TYPE = 'heteroscedastic'
USE_INDUSTRIAL_FUNCTIONS = True
ADD_TIME_LAG = True


# ─── 工具函数 ────────────────────────────────────────────────────────────

def build_windows(X, window_size):
    """构建时间窗口"""
    T, d = X.shape
    xs, ys = [], []
    for start in range(0, T - window_size):
        xs.append(X[start:start + window_size, :])
        ys.append(X[start + window_size, :])
    return np.array(xs), np.array(ys)


# ─── 参数化的 MultiScaleNTSNet ────────────────────────────────────────────

class MultiScaleNTSNet(nn.Module):
    """
    参数化版本的多尺度并行卷积 + 可学习融合权重 + NOTEARS 约束

    支持自定义 kernel_sizes、hidden_mult 和 window_size。
    """
    def __init__(self, d, kernel_sizes, hidden_mult=8, window_size=10):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.01, 0.01))

        # 确保所有 kernel_size <= window_size
        valid_kernels = [ks for ks in kernel_sizes if ks <= window_size]
        if not valid_kernels:
            valid_kernels = [min(kernel_sizes[0], window_size)]
        self.kernel_sizes = valid_kernels

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d, d * hidden_mult, kernel_size=ks, groups=d),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(d * hidden_mult, d, kernel_size=1, groups=d)
            )
            for ks in self.kernel_sizes
        ])

        self.alpha = nn.Parameter(torch.zeros(len(self.kernel_sizes)))

    def forward(self, x):
        x_agg = torch.matmul(x, self.W)
        x_t   = x_agg.transpose(1, 2)

        alpha_norm = torch.softmax(self.alpha, dim=0)
        pred = sum(
            alpha_norm[i] * conv(x_t).squeeze(2)
            for i, conv in enumerate(self.convs)
        )
        return pred

    def notears_penalty(self):
        M = self.W * self.W
        E = torch.matrix_exp(M)
        return torch.trace(E) - self.d


def train_and_evaluate(X, adj_true, hparams):
    """
    使用给定超参数训练 MultiScale-NTS 并返回评估指标。

    参数:
        X         : 模拟数据 (n_samples, d)
        adj_true  : 真实邻接矩阵 (d, d)
        hparams   : 超参数字典

    返回:
        metrics   : 评估指标字典
    """
    d = X.shape[1]
    window_size = hparams['window_size']
    kernel_sizes = hparams['kernel_sizes']
    hidden_mult = hparams['hidden_mult']
    lr = hparams['lr']
    epochs = hparams['epochs']
    batch_size = hparams['batch_size']
    l1_lambda = hparams['l1_lambda']
    h_lambda = hparams['h_lambda']
    threshold = hparams['threshold']

    # 数据预处理
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    wx, wy = build_windows(X_norm, window_size)

    if len(wx) == 0:
        return {'SHD': float('inf'), 'F1': 0.0, 'Precision': 0.0,
                'Recall': 0.0, 'TPR': 0.0, 'FDR': 1.0}

    model = MultiScaleNTSNet(d, kernel_sizes, hidden_mult, window_size).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)

    xb = torch.tensor(wx, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(wy, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb),
        batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        for b_x, b_y in loader:
            opt.zero_grad()
            pred  = model(b_x)
            mse   = F.mse_loss(pred, b_y)
            h_val = model.notears_penalty()
            loss  = mse + l1_lambda * torch.sum(torch.abs(model.W)) + h_lambda * h_val * h_val
            loss.backward()
            opt.step()

    adj_pred = model.W.detach().cpu().numpy()
    metrics = compute_dag_metrics(adj_true, adj_pred, threshold=threshold)

    # 清理
    del model, opt, xb, yb, loader
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return metrics


def evaluate_hparams(hparams, n_trials=5, seed_offset=0):
    """
    对一组超参数运行 n_trials 次蒙特卡洛实验，返回聚合指标。

    参数:
        hparams    : 超参数字典
        n_trials   : 蒙特卡洛实验次数
        seed_offset: 随机种子偏移（避免与 benchmark 重叠）

    返回:
        agg_metrics: 聚合指标 (mean ± std)
    """
    all_metrics = []

    for trial in range(n_trials):
        seed = seed_offset + trial
        gen = SyntheticDAGGenerator(n_nodes=N_NODES, seed=seed)
        X, adj_true, metadata = gen.generate_complete_synthetic_dataset(
            graph_type=GRAPH_TYPE,
            n_samples=N_SAMPLES,
            noise_scale=NOISE_SCALE,
            noise_type=NOISE_TYPE,
            add_time_lag=ADD_TIME_LAG,
            use_industrial_functions=USE_INDUSTRIAL_FUNCTIONS,
            n_layers=5
        )

        try:
            metrics = train_and_evaluate(X, adj_true, hparams)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  [警告] Trial {trial} 失败: {e}")

    if not all_metrics:
        return None

    agg = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        agg[f'{key}_mean'] = np.mean(vals)
        agg[f'{key}_std']  = np.std(vals)

    return agg


# ─── 超参数搜索空间 ──────────────────────────────────────────────────────

# 完整搜索空间
FULL_SEARCH_SPACE = {
    'window_size':   [8, 10, 15],
    'kernel_sizes':  [[3, 5], [3, 5, 10], [3, 7], [2, 4, 8]],
    'hidden_mult':   [4, 8, 16],
    'lr':            [0.001, 0.005, 0.01],
    'epochs':        [20, 30, 50],
    'batch_size':    [32, 64],
    'l1_lambda':     [0.0005, 0.001, 0.005],
    'h_lambda':      [0.25, 0.5, 1.0],
    'threshold':     [0.03, 0.05, 0.1],
}

# 快速搜索空间（缩小网格）
QUICK_SEARCH_SPACE = {
    'window_size':   [10],
    'kernel_sizes':  [[3, 5], [3, 5, 10], [2, 4, 8]],
    'hidden_mult':   [8],
    'lr':            [0.005, 0.01],
    'epochs':        [30],
    'batch_size':    [64],
    'l1_lambda':     [0.0005, 0.001, 0.005],
    'h_lambda':      [0.5],
    'threshold':     [0.03, 0.05, 0.1],
}


def generate_param_grid(search_space):
    """从搜索空间生成所有超参数组合"""
    keys = list(search_space.keys())
    values = list(search_space.values())
    grid = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # 处理 kernel_sizes 中包含 window_size 的组合
        # 追加 window_size 本身作为第三路（如果未包含且合理）
        ks = list(params['kernel_sizes'])
        ws = params['window_size']
        if ws not in ks:
            ks.append(ws)
        params['kernel_sizes'] = ks
        grid.append(params)
    return grid


def run_tuning(n_trials=5, quick=False, seed_offset=1000):
    """
    运行超参数调优。

    参数:
        n_trials   : 每组超参数的蒙特卡洛实验次数
        quick      : 是否使用缩小的搜索空间
        seed_offset: 随机种子偏移

    返回:
        results_df : 结果 DataFrame
        best_params: 最佳超参数
    """
    search_space = QUICK_SEARCH_SPACE if quick else FULL_SEARCH_SPACE
    param_grid = generate_param_grid(search_space)

    print("=" * 70)
    print("MultiScale-NTS 超参数调优")
    print("=" * 70)
    print(f"搜索模式:     {'快速' if quick else '完整'}")
    print(f"超参数组合数: {len(param_grid)}")
    print(f"每组实验次数: {n_trials}")
    print(f"总实验数:     {len(param_grid) * n_trials}")
    print(f"设备:         {DEVICE}")
    print("=" * 70)

    results = []
    best_f1 = -1
    best_params = None

    for idx, hparams in enumerate(tqdm(param_grid, desc="超参数搜索")):
        # 显示当前超参数
        ks_str = str(hparams['kernel_sizes'])
        tqdm.write(f"\n[{idx+1}/{len(param_grid)}] "
                   f"ws={hparams['window_size']} ks={ks_str} "
                   f"hm={hparams['hidden_mult']} lr={hparams['lr']} "
                   f"ep={hparams['epochs']} bs={hparams['batch_size']} "
                   f"l1={hparams['l1_lambda']} h={hparams['h_lambda']} "
                   f"thr={hparams['threshold']}")

        agg = evaluate_hparams(hparams, n_trials=n_trials, seed_offset=seed_offset)

        if agg is None:
            tqdm.write("  → 所有试验失败，跳过")
            continue

        row = {
            'window_size':  hparams['window_size'],
            'kernel_sizes': str(hparams['kernel_sizes']),
            'hidden_mult':  hparams['hidden_mult'],
            'lr':           hparams['lr'],
            'epochs':       hparams['epochs'],
            'batch_size':   hparams['batch_size'],
            'l1_lambda':    hparams['l1_lambda'],
            'h_lambda':     hparams['h_lambda'],
            'threshold':    hparams['threshold'],
            **agg,
        }
        results.append(row)

        f1_mean = agg.get('F1_mean', 0)
        shd_mean = agg.get('SHD_mean', float('inf'))
        tqdm.write(f"  → F1={f1_mean:.4f}±{agg.get('F1_std', 0):.4f}  "
                   f"SHD={shd_mean:.1f}±{agg.get('SHD_std', 0):.1f}  "
                   f"Prec={agg.get('Precision_mean', 0):.4f}  "
                   f"Rec={agg.get('Recall_mean', 0):.4f}")

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_params = hparams.copy()
            tqdm.write(f"  ★ 新最佳! F1={best_f1:.4f}")

    results_df = pd.DataFrame(results)

    # 按 F1 降序排序
    if not results_df.empty:
        results_df = results_df.sort_values('F1_mean', ascending=False).reset_index(drop=True)

    return results_df, best_params


def save_results(results_df, best_params):
    """保存调优结果"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # 保存完整结果 CSV
    csv_path = os.path.join(OUT_DIR, f"tuning_results_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n完整结果已保存: {csv_path}")

    # 保存最佳参数 JSON
    if best_params is not None:
        # 将 kernel_sizes list 转为可序列化格式
        best_params_save = best_params.copy()
        best_params_save['kernel_sizes'] = list(best_params_save['kernel_sizes'])

        json_path = os.path.join(OUT_DIR, f"best_params_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(best_params_save, f, indent=2, ensure_ascii=False)
        print(f"最佳参数已保存: {json_path}")

    # 生成报告
    report_path = os.path.join(OUT_DIR, f"tuning_report_{timestamp}.md")
    generate_tuning_report(results_df, best_params, report_path)
    print(f"调优报告已保存: {report_path}")


def generate_tuning_report(results_df, best_params, report_path):
    """生成 Markdown 格式的调优报告"""
    report = f"""# MultiScale-NTS 超参数调优报告

## 模拟数据场景配置

| 参数 | 值 |
|------|-----|
| 节点数 | {N_NODES} |
| 样本数 | {N_SAMPLES} |
| 图类型 | {GRAPH_TYPE} |
| 噪声类型 | {NOISE_TYPE} |
| 噪声水平 | {NOISE_SCALE} |
| 工业函数 | {USE_INDUSTRIAL_FUNCTIONS} |
| 时序依赖 | {ADD_TIME_LAG} |

## 最佳超参数

"""
    if best_params is not None:
        for k, v in best_params.items():
            report += f"- **{k}**: `{v}`\n"

        # 找到最佳结果行
        if not results_df.empty:
            best_row = results_df.iloc[0]
            report += f"""
## 最佳性能（均值 ± 标准差）

| 指标 | 值 |
|------|-----|
| F1 | {best_row.get('F1_mean', 0):.4f} ± {best_row.get('F1_std', 0):.4f} |
| Precision | {best_row.get('Precision_mean', 0):.4f} ± {best_row.get('Precision_std', 0):.4f} |
| Recall | {best_row.get('Recall_mean', 0):.4f} ± {best_row.get('Recall_std', 0):.4f} |
| SHD | {best_row.get('SHD_mean', 0):.1f} ± {best_row.get('SHD_std', 0):.1f} |
| TPR | {best_row.get('TPR_mean', 0):.4f} ± {best_row.get('TPR_std', 0):.4f} |
| FDR | {best_row.get('FDR_mean', 0):.4f} ± {best_row.get('FDR_std', 0):.4f} |
"""

    report += f"""
## Top 10 超参数组合

"""
    if not results_df.empty:
        top_n = min(10, len(results_df))
        report += "| 排名 | window_size | kernel_sizes | hidden_mult | lr | epochs | l1_lambda | h_lambda | threshold | F1 | SHD |\n"
        report += "|" + "------|" * 11 + "\n"
        for i in range(top_n):
            row = results_df.iloc[i]
            report += (f"| {i+1} | {row.get('window_size', '')} | "
                       f"{row.get('kernel_sizes', '')} | {row.get('hidden_mult', '')} | "
                       f"{row.get('lr', '')} | {row.get('epochs', '')} | "
                       f"{row.get('l1_lambda', '')} | {row.get('h_lambda', '')} | "
                       f"{row.get('threshold', '')} | "
                       f"{row.get('F1_mean', 0):.4f} | {row.get('SHD_mean', 0):.1f} |\n")

    report += f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiScale-NTS 超参数调优（针对模拟数据场景）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--n_trials", type=int, default=5,
        help="每组超参数的蒙特卡洛实验次数（默认: 5）"
    )
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="快速模式：缩小搜索空间"
    )
    parser.add_argument(
        "--seed_offset", type=int, default=1000,
        help="随机种子偏移，避免与 benchmark 实验重叠（默认: 1000）"
    )

    args = parser.parse_args()

    print(f"使用设备: {DEVICE}")
    print(f"模拟数据配置: {N_NODES}节点 × {N_SAMPLES}样本 × {GRAPH_TYPE}图 × {NOISE_TYPE}噪声")

    results_df, best_params = run_tuning(
        n_trials=args.n_trials,
        quick=args.quick,
        seed_offset=args.seed_offset,
    )

    if best_params is not None:
        print("\n" + "=" * 70)
        print("最佳超参数:")
        print("=" * 70)
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print("=" * 70)

    save_results(results_df, best_params)

    print("\n调优完成！")
