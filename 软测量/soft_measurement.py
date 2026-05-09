#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
软测量/soft_measurement.py
===========================
基于多种机器学习方法的浮选精矿品位软测量
支持两个出矿口（新一线 xin1、新二线 xin2）的品位预测

方法：
  1. ElasticNet  — 结合 L1+L2 正则化的线性回归模型
  2. SVR         — 支持向量回归（RBF 核）
  3. GPR         — 高斯过程回归（为控制计算量对训练集子采样）
  4. XGBoost     — 极端梯度提升（引入二阶导数和正则化项）
  5. LightGBM    — 基于直方图和叶子生长策略的高效梯度提升树
  6. Bi-LSTM     — 双向长短期记忆网络（结合历史与未来信息）

数据来源：
  data/modeling_dataset_xin1_final.parquet  — 新一线特征 + 品位目标
  data/modeling_dataset_xin2_final.parquet  — 新二线特征 + 品位目标

输出：
  软测量/结果/metrics_summary.csv         — 所有模型在两个出矿口的评估指标
  软测量/结果/prediction_<outlet>.png     — 预测值 vs 真实值折线图
  软测量/结果/scatter_<outlet>.png        — 散点图（每个模型）

运行：
  python 软测量/soft_measurement.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════════════════════════════════════
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果")
os.makedirs(RESULT_DIR, exist_ok=True)

DS_XIN1 = os.path.join(DATA_DIR, "modeling_dataset_xin1_final.parquet")
DS_XIN2 = os.path.join(DATA_DIR, "modeling_dataset_xin2_final.parquet")

# ═══════════════════════════════════════════════════════════════════════════
#  全局超参数
# ═══════════════════════════════════════════════════════════════════════════
K_FEATURES       = 50    # SelectKBest 按互信息保留的特征数（供所有模型使用）
TEST_RATIO       = 0.20  # 测试集比例（时序分割，不打乱顺序）
RANDOM_SEED      = 42

# GPR
GPR_MAX_SAMPLES  = 300   # GPR 训练样本上限（控制 O(n³) 计算量）

# Bi-LSTM
LSTM_SEQ_LEN     = 8     # 输入窗口长度（个时间步）
LSTM_HIDDEN      = 64    # 双向 LSTM 隐藏维度（单向）
LSTM_LAYERS      = 2     # LSTM 堆叠层数
LSTM_DROPOUT     = 0.2   # Dropout 比例
LSTM_EPOCHS      = 150   # 最大训练轮次
LSTM_LR          = 1e-3  # Adam 学习率
LSTM_BATCH       = 64    # 批量大小
LSTM_PATIENCE    = 20    # 早停轮次

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTLET_CFG = {
    "xin1": {"dataset": DS_XIN1, "y_col": "y_fx_xin1", "label": "新一线精矿品位"},
    "xin2": {"dataset": DS_XIN2, "y_col": "y_fx_xin2", "label": "新二线精矿品位"},
}

MODEL_NAMES = ["ElasticNet", "SVR", "GPR", "XGBoost", "LightGBM", "Bi-LSTM"]


# ═══════════════════════════════════════════════════════════════════════════
#  Bi-LSTM 模型定义
# ═══════════════════════════════════════════════════════════════════════════
class BiLSTMRegressor(nn.Module):
    """双向 LSTM 回归网络：seq_len × n_features → 1 标量"""

    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)   # ×2 因为双向

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        out, _ = self.lstm(x)          # out: (B, seq_len, 2*hidden)
        last    = out[:, -1, :]        # 取最后一个时间步
        return self.head(self.drop(last)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 R², RMSE, MAE"""
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


def make_sequences(X: np.ndarray, y: np.ndarray,
                   seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """将 2D 时序数组切分为滑动窗口序列。

    Args:
        X:       (n_samples, n_features)
        y:       (n_samples,)
        seq_len: 窗口长度

    Returns:
        X_seq:   (n_samples - seq_len + 1, seq_len, n_features)
        y_seq:   (n_samples - seq_len + 1,)  — 对应每个窗口末尾的目标值
    """
    n = len(X)
    xs, ys = [], []
    for i in range(seq_len - 1, n):
        xs.append(X[i - seq_len + 1: i + 1])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def load_and_prepare(outlet_key: str) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray,
                                                pd.DatetimeIndex]:
    """加载数据、特征选择、时序分割、标准化。

    Returns:
        X_train_s, X_test_s: 标准化后特征（2D）
        y_train, y_test: 目标值
        X_raw_train, X_raw_test: 未标准化特征（供 Bi-LSTM 序列构建）
        test_index: 测试集时间戳索引
    """
    cfg   = OUTLET_CFG[outlet_key]
    y_col = cfg["y_col"]

    print(f"\n{'='*60}")
    print(f"  出矿口：{cfg['label']} ({outlet_key})")
    print(f"{'='*60}")

    # ── 1. 读取数据 ──────────────────────────────────────────────
    df = pd.read_parquet(cfg["dataset"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "time"
    df = df.sort_index()

    # 目标列
    other_y = "y_fx_xin2" if outlet_key == "xin1" else "y_fx_xin1"
    if other_y in df.columns:
        df = df.drop(columns=[other_y])

    target = df[y_col].values.astype(np.float32)
    feat_cols = [c for c in df.columns if c != y_col]
    X_raw = df[feat_cols].values.astype(np.float32)

    print(f"  原始特征维度：{X_raw.shape[1]}，样本数：{len(X_raw)}")

    # ── 2. 时序分割（不打乱） ────────────────────────────────────
    n_total = len(X_raw)
    n_train = int(n_total * (1 - TEST_RATIO))
    X_train_raw, X_test_raw = X_raw[:n_train],   X_raw[n_train:]
    y_train,     y_test     = target[:n_train],  target[n_train:]
    test_index              = df.index[n_train:]

    print(f"  训练集：{n_train} 条  |  测试集：{n_total - n_train} 条")

    # ── 3. 特征选择（仅在训练集上拟合） ─────────────────────────
    # 3a. 去方差极小特征
    vt = VarianceThreshold(threshold=1e-4)
    X_tr_vt = vt.fit_transform(X_train_raw)
    X_te_vt = vt.transform(X_test_raw)
    print(f"  方差阈值后：{X_tr_vt.shape[1]} 个特征")

    # 3b. 处理训练集中的 NaN（ffill + 列均值填充）
    X_tr_df = pd.DataFrame(X_tr_vt).ffill().bfill()
    X_te_df = pd.DataFrame(X_te_vt).ffill().bfill()
    # 如果还有 NaN，用训练集均值填充
    col_means = X_tr_df.mean()
    X_tr_df   = X_tr_df.fillna(col_means)
    X_te_df   = X_te_df.fillna(col_means)
    X_tr_vt   = X_tr_df.values
    X_te_vt   = X_te_df.values

    # 3c. 互信息选 Top-K
    k = min(K_FEATURES, X_tr_vt.shape[1])
    selector = SelectKBest(mutual_info_regression, k=k)
    X_tr_sel = selector.fit_transform(X_tr_vt, y_train)
    X_te_sel = selector.transform(X_te_vt)
    print(f"  互信息选特征后：{X_tr_sel.shape[1]} 个特征")

    # ── 4. 标准化 ────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_tr_sel)
    X_test_s  = scaler.transform(X_te_sel)

    return X_train_s, X_test_s, y_train, y_test, X_tr_sel, X_te_sel, test_index


# ═══════════════════════════════════════════════════════════════════════════
#  模型训练与预测
# ═══════════════════════════════════════════════════════════════════════════

def run_elasticnet(X_train, X_test, y_train, y_test):
    t0 = time.time()
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=5, max_iter=5000, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    print(f"  [ElasticNet] α={model.alpha_:.4f} l1={model.l1_ratio_:.2f}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_svr(X_train, X_test, y_train, y_test):
    t0 = time.time()
    model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    print(f"  [SVR]        支持向量数={model.n_support_.sum()}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_gpr(X_train, X_test, y_train, y_test):
    """GPR 对训练集子采样以控制 O(n³) 复杂度。"""
    t0 = time.time()
    rng = np.random.RandomState(RANDOM_SEED)
    if len(X_train) > GPR_MAX_SAMPLES:
        idx = rng.choice(len(X_train), GPR_MAX_SAMPLES, replace=False)
        idx.sort()
        X_tr_gpr, y_tr_gpr = X_train[idx], y_train[idx]
    else:
        X_tr_gpr, y_tr_gpr = X_train, y_train

    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-5, 1e1))
    model  = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, random_state=RANDOM_SEED,
        normalize_y=True
    )
    model.fit(X_tr_gpr, y_tr_gpr)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    n_sub = len(X_tr_gpr)
    print(f"  [GPR]        子采样={n_sub}/{len(X_train)}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_xgboost(X_train, X_test, y_train, y_test):
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric="rmse",
        random_state=RANDOM_SEED, verbosity=0, n_jobs=-1,
        device="cpu",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    print(f"  [XGBoost]    最佳轮次={model.best_iteration}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_lightgbm(X_train, X_test, y_train, y_test):
    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    print(f"  [LightGBM]   最佳轮次={model.best_iteration_}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_bilstm(X_train_raw, X_test_raw, y_train, y_test, seq_len: int = LSTM_SEQ_LEN):
    """使用未标准化特征构建滑动窗口序列，内部做序列级 StandardScaler。"""
    t0 = time.time()

    # ── 序列级标准化（逐特征） ──────────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_te_s = scaler.transform(X_test_raw)

    # ── 构建滑动窗口 ─────────────────────────────────────────────
    X_tr_seq, y_tr_seq = make_sequences(X_tr_s, y_train, seq_len)
    X_te_seq, y_te_seq = make_sequences(X_te_s, y_test,  seq_len)

    n_features = X_tr_seq.shape[2]

    # ── 目标值归一化（方便训练稳定） ─────────────────────────────
    y_mean, y_std = y_tr_seq.mean(), y_tr_seq.std() + 1e-8
    y_tr_norm = (y_tr_seq - y_mean) / y_std
    y_te_norm = (y_te_seq - y_mean) / y_std   # 用训练集统计量

    # ── DataLoader ───────────────────────────────────────────────
    ds_train = TensorDataset(
        torch.from_numpy(X_tr_seq),
        torch.from_numpy(y_tr_norm),
    )
    ds_test = TensorDataset(
        torch.from_numpy(X_te_seq),
        torch.from_numpy(y_te_norm),
    )
    dl_train = DataLoader(ds_train, batch_size=LSTM_BATCH, shuffle=True,  drop_last=False)
    dl_test  = DataLoader(ds_test,  batch_size=LSTM_BATCH, shuffle=False, drop_last=False)

    # ── 模型 ─────────────────────────────────────────────────────
    model = BiLSTMRegressor(
        n_features=n_features,
        hidden=LSTM_HIDDEN,
        n_layers=LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, LSTM_EPOCHS + 1):
        # Train
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_test:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= LSTM_PATIENCE:
                print(f"    早停于 epoch {epoch}（patience={LSTM_PATIENCE}）")
                break

    # ── 预测 ─────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def _predict_loader(dl):
        preds = []
        with torch.no_grad():
            for xb, _ in dl:
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
        return np.concatenate(preds)

    pred_tr_norm = _predict_loader(dl_train)
    pred_te_norm = _predict_loader(dl_test)

    # 反标准化
    pred_train_full = pred_tr_norm * y_std + y_mean
    pred_test_full  = pred_te_norm * y_std + y_mean

    elapsed = time.time() - t0
    print(f"  [Bi-LSTM]    最终验证损失={best_val_loss:.5f}  耗时={elapsed:.1f}s")

    # 对齐回完整序列长度（前 seq_len-1 个样本无预测，用 NaN 填充）
    pad = seq_len - 1
    pred_train_aligned = np.concatenate([np.full(pad, np.nan), pred_train_full])
    pred_test_aligned  = np.concatenate([np.full(pad, np.nan), pred_test_full])

    return pred_train_aligned, pred_test_aligned, y_tr_seq, y_te_seq


# ═══════════════════════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════════════════════

def plot_predictions(outlet_key: str, test_index: pd.DatetimeIndex,
                     y_test: np.ndarray, preds: dict, label: str):
    """测试集上各模型的预测值 vs 真实值折线图（时间轴）。"""
    n_models = len(preds)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 3 * n_models), sharex=True)
    fig.suptitle(f"{label} — 测试集预测值 vs 真实值", fontsize=13, y=1.01)

    for ax, (name, pred) in zip(axes, preds.items()):
        valid_mask = ~np.isnan(pred)
        ax.plot(test_index[valid_mask], y_test[valid_mask],
                label="真实值", color="steelblue", linewidth=0.8, alpha=0.9)
        ax.plot(test_index[valid_mask], pred[valid_mask],
                label=f"{name} 预测", color="orangered", linewidth=0.8, alpha=0.85,
                linestyle="--")
        r2 = r2_score(y_test[valid_mask], pred[valid_mask])
        ax.set_title(f"{name}  R²={r2:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylabel("品位 (%)", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间")
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f"prediction_{outlet_key}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  折线图已保存：{save_path}")


def plot_scatter(outlet_key: str, y_test: np.ndarray, preds: dict, label: str):
    """散点图：每个模型一个子图（预测值 vs 真实值）。"""
    n_models = len(preds)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else axes.flatten()
    fig.suptitle(f"{label} — 散点图（预测 vs 真实）", fontsize=13)

    for ax, (name, pred) in zip(axes_flat, preds.items()):
        valid_mask = ~np.isnan(pred)
        yt = y_test[valid_mask]
        yp = pred[valid_mask]
        ax.scatter(yt, yp, s=10, alpha=0.5, color="royalblue", edgecolors="none")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        r2   = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        ax.set_title(f"{name}  R²={r2:.3f}  RMSE={rmse:.3f}", fontsize=9)
        ax.set_xlabel("真实值 (%)", fontsize=8)
        ax.set_ylabel("预测值 (%)", fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f"scatter_{outlet_key}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  散点图已保存：{save_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════

def run_outlet(outlet_key: str) -> pd.DataFrame:
    """对单个出矿口运行全部 6 个模型，返回评估指标 DataFrame。"""
    cfg = OUTLET_CFG[outlet_key]

    (X_train_s, X_test_s, y_train, y_test,
     X_train_raw, X_test_raw, test_index) = load_and_prepare(outlet_key)

    records   = []
    test_preds = {}   # 供可视化

    # ── 1. ElasticNet ──────────────────────────────────────────────
    ptr, pte = run_elasticnet(X_train_s, X_test_s, y_train, y_test)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "ElasticNet",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["ElasticNet"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 2. SVR ─────────────────────────────────────────────────────
    ptr, pte = run_svr(X_train_s, X_test_s, y_train, y_test)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "SVR",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["SVR"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 3. GPR ─────────────────────────────────────────────────────
    ptr, pte = run_gpr(X_train_s, X_test_s, y_train, y_test)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "GPR",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["GPR"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 4. XGBoost ─────────────────────────────────────────────────
    ptr, pte = run_xgboost(X_train_s, X_test_s, y_train, y_test)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "XGBoost",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["XGBoost"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 5. LightGBM ────────────────────────────────────────────────
    ptr, pte = run_lightgbm(X_train_s, X_test_s, y_train, y_test)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "LightGBM",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["LightGBM"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 6. Bi-LSTM ─────────────────────────────────────────────────
    ptr, pte, y_tr_seq, y_te_seq = run_bilstm(
        X_train_raw, X_test_raw, y_train, y_test, seq_len=LSTM_SEQ_LEN
    )
    # 对齐：前 seq_len-1 个位置无预测，跳过
    pad = LSTM_SEQ_LEN - 1
    m_tr = compute_metrics(y_train[pad:], ptr[pad:][~np.isnan(ptr[pad:])])
    # test: y_test 与 pte 长度一致
    valid = ~np.isnan(pte)
    m_te  = compute_metrics(y_test[valid], pte[valid])
    records.append({"模型": "Bi-LSTM",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["Bi-LSTM"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 可视化 ─────────────────────────────────────────────────────
    plot_predictions(outlet_key, test_index, y_test, test_preds, cfg["label"])
    plot_scatter(outlet_key, y_test, test_preds, cfg["label"])

    metrics_df = pd.DataFrame(records)
    metrics_df.insert(0, "出矿口", cfg["label"])
    return metrics_df


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    all_results = []
    for outlet in ["xin1", "xin2"]:
        df_metrics = run_outlet(outlet)
        all_results.append(df_metrics)

    summary = pd.concat(all_results, ignore_index=True)

    # 打印汇总表
    print("\n" + "=" * 80)
    print("  软测量结果汇总（测试集）")
    print("=" * 80)
    disp_cols = ["出矿口", "模型", "测试R2", "测试RMSE", "测试MAE"]
    print(summary[disp_cols].to_string(index=False))

    # 保存完整结果（含训练集）
    csv_path = os.path.join(RESULT_DIR, "metrics_summary.csv")
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  完整指标已保存：{csv_path}")


if __name__ == "__main__":
    main()
