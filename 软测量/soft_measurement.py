#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
软测量/soft_measurement.py
===========================
基于多种机器学习方法的浮选精矿品位软测量
支持两个出矿口（新一线 xin1、新二线 xin2）的品位预测

方法：
  1. ElasticNet  — 结合 L1+L2 正则化的线性回归模型（内置 CV 选参）
  2. SVR         — 支持向量回归（RBF 核）
  3. GPR         — 高斯过程回归（为控制计算量对训练集子采样）
  4. XGBoost     — 极端梯度提升（引入二阶导数和正则化项）
  5. LightGBM    — 基于直方图和叶子生长策略的高效梯度提升树
  6. LSTM        — 单向长短期记忆网络（因果时序，适用于在线软测量）

超参数优化：
  通过 Optuna（TPE 采样 + 中值剪枝）在验证集上搜索各模型的最优超参数。
  测试集在调优全过程中保持严格隔离，仅用于最终指标评估。
  可通过 ENABLE_OPTUNA = False 快速跳过调优，直接使用默认超参数。

数据来源：
  data/modeling_dataset_xin1_final.parquet  — 新一线特征 + 品位目标
  data/modeling_dataset_xin2_final.parquet  — 新二线特征 + 品位目标

输出：
  软测量/结果/metrics_summary.csv              — 所有模型在两个出矿口的评估指标
  软测量/结果/best_hyperparams_<outlet>.json   — Optuna 搜索到的各模型最优超参数
  软测量/结果/prediction_<outlet>.png          — 预测值 vs 真实值折线图
  软测量/结果/scatter_<outlet>.png             — 散点图（每个模型）

运行：
  python 软测量/soft_measurement.py
"""

import json
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
VAL_RATIO        = 0.10  # 验证集比例（从训练集末尾切出，用于 early stopping）
TEST_RATIO       = 0.20  # 测试集比例（时序分割，不打乱顺序，最终评估专用）
RANDOM_SEED      = 42

# GPR
GPR_MAX_SAMPLES  = 300   # GPR 训练样本上限（控制 O(n³) 计算量）

# LSTM
LSTM_SEQ_LEN     = 8     # 输入窗口长度（个时间步）
LSTM_HIDDEN      = 64    # LSTM 隐藏维度
LSTM_LAYERS      = 2     # LSTM 堆叠层数
LSTM_DROPOUT     = 0.2   # Dropout 比例
LSTM_EPOCHS      = 150   # 最大训练轮次
LSTM_LR          = 1e-3  # Adam 学习率
LSTM_BATCH       = 64    # 批量大小
LSTM_PATIENCE    = 20    # 早停轮次

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════
#  Optuna 超参数优化配置
# ═══════════════════════════════════════════════════════════════════════════
ENABLE_OPTUNA        = True   # 设为 False 可跳过调优，直接使用默认超参数
OPTUNA_N_TRIALS      = 30     # 常规模型（SVR / XGBoost / LightGBM）的搜索次数
OPTUNA_N_TRIALS_GPR  = 10     # GPR 搜索次数（因 O(n³) 计算量高，适当减少）
OPTUNA_N_TRIALS_LSTM = 15     # LSTM 搜索次数（每次 trial 需跑若干 epoch）
OPTUNA_TUNE_EPOCHS   = 60     # Optuna 搜索阶段 LSTM 最大轮次（加速 trial 评估）
OPTUNA_PATIENCE_LSTM = 10     # Optuna 搜索阶段 LSTM 早停轮次

OUTLET_CFG = {
    "xin1": {"dataset": DS_XIN1, "y_col": "y_fx_xin1", "label": "新一线精矿品位"},
    "xin2": {"dataset": DS_XIN2, "y_col": "y_fx_xin2", "label": "新二线精矿品位"},
}

MODEL_NAMES = ["ElasticNet", "SVR", "GPR", "XGBoost", "LightGBM", "LSTM"]


# ═══════════════════════════════════════════════════════════════════════════
#  LSTM 模型定义（单向，适用于因果时序软测量）
# ═══════════════════════════════════════════════════════════════════════════
class LSTMRegressor(nn.Module):
    """单向 LSTM 回归网络：seq_len × n_features → 1 标量

    单向设计确保 t 时刻预测仅依赖 t 及之前的信息，符合在线软测量的因果性要求。
    """

    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        out, _ = self.lstm(x)          # out: (B, seq_len, hidden)
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


def load_and_prepare(outlet_key: str):
    """加载数据、特征选择、时序三段分割（train/val/test）、标准化。

    数据切分顺序（按时间）：
      train : [0,  n_train)        — 占总量约 70%，用于拟合模型
      val   : [n_train, n_val_end) — 占总量约 10%，用于 early stopping / 超参选择
      test  : [n_val_end, end)     — 占总量约 20%，仅用于最终评估，不参与任何训练决策

    Returns:
        X_train_s, X_val_s, X_test_s : 标准化后特征（2D）
        y_train, y_val, y_test       : 目标值
        X_train_raw, X_val_raw, X_test_raw : 未标准化特征（供 LSTM 序列构建）
        val_index, test_index        : 时间戳索引
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

    # ── 2. 时序三段分割（train / val / test，不打乱） ────────────
    n_total   = len(X_raw)
    n_test    = int(n_total * TEST_RATIO)
    n_val     = int(n_total * VAL_RATIO)
    n_train   = n_total - n_val - n_test
    n_val_end = n_train + n_val

    X_train_raw = X_raw[:n_train]
    X_val_raw   = X_raw[n_train:n_val_end]
    X_test_raw  = X_raw[n_val_end:]
    y_train     = target[:n_train]
    y_val       = target[n_train:n_val_end]
    y_test      = target[n_val_end:]
    val_index   = df.index[n_train:n_val_end]
    test_index  = df.index[n_val_end:]

    print(f"  训练集：{n_train} 条  |  验证集：{n_val} 条  |  测试集：{n_test} 条")

    # ── 3. 特征选择（仅在训练集上拟合） ─────────────────────────
    # 3a. 去方差极小特征
    vt = VarianceThreshold(threshold=1e-4)
    X_tr_vt = vt.fit_transform(X_train_raw)
    X_va_vt = vt.transform(X_val_raw)
    X_te_vt = vt.transform(X_test_raw)
    print(f"  方差阈值后：{X_tr_vt.shape[1]} 个特征")

    # 3b. 处理 NaN（仅 ffill，再用训练集列均值补全剩余缺失）
    # 训练集：ffill 后用自身均值填充；验证/测试集：ffill 后用训练集均值填充，避免 bfill 引入未来信息
    X_tr_df   = pd.DataFrame(X_tr_vt).ffill()
    col_means = X_tr_df.mean()
    X_tr_vt   = X_tr_df.fillna(col_means).values
    X_va_vt   = pd.DataFrame(X_va_vt).ffill().fillna(col_means).values
    X_te_vt   = pd.DataFrame(X_te_vt).ffill().fillna(col_means).values

    # 3c. 互信息选 Top-K（仅在训练集上拟合）
    k = min(K_FEATURES, X_tr_vt.shape[1])
    selector  = SelectKBest(mutual_info_regression, k=k)
    X_tr_sel  = selector.fit_transform(X_tr_vt, y_train)
    X_va_sel  = selector.transform(X_va_vt)
    X_te_sel  = selector.transform(X_te_vt)
    print(f"  互信息选特征后：{X_tr_sel.shape[1]} 个特征")

    # ── 4. 标准化（仅在训练集上拟合） ───────────────────────────
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_tr_sel)
    X_val_s   = scaler.transform(X_va_sel)
    X_test_s  = scaler.transform(X_te_sel)

    return (X_train_s, X_val_s, X_test_s,
            y_train, y_val, y_test,
            X_tr_sel, X_va_sel, X_te_sel,
            val_index, test_index)


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


def run_svr(X_train, X_test, y_train, y_test,
            C: float = 10.0, epsilon: float = 0.1, gamma: str = "scale"):
    t0 = time.time()
    model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    print(f"  [SVR]        支持向量数={model.n_support_.sum()}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_gpr(X_train, X_test, y_train, y_test,
            n_restarts: int = 3, noise_level: float = 0.1,
            max_samples: int = GPR_MAX_SAMPLES):
    """GPR 对训练集取最近连续 N 条以控制 O(n³) 复杂度。

    使用末尾连续窗口（而非随机抽样）以保留时序局部结构。
    """
    t0 = time.time()
    if len(X_train) > max_samples:
        X_tr_gpr = X_train[-max_samples:]
        y_tr_gpr = y_train[-max_samples:]
    else:
        X_tr_gpr, y_tr_gpr = X_train, y_train

    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level, (1e-5, 1e1))
    model  = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, random_state=RANDOM_SEED,
        normalize_y=True
    )
    model.fit(X_tr_gpr, y_tr_gpr)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    n_sub = len(X_tr_gpr)
    print(f"  [GPR]        子采样={n_sub}/{len(X_train)}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_xgboost(X_train, X_val, X_test, y_train, y_val, y_test,
                learning_rate: float = 0.05, max_depth: int = 5,
                subsample: float = 0.8, colsample_bytree: float = 0.8,
                reg_alpha: float = 0.1, reg_lambda: float = 1.0,
                n_estimators: int = 500):
    """XGBoost：使用独立验证集做 early stopping，测试集仅用于最终评估。"""
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=subsample, colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        early_stopping_rounds=30,
        eval_metric="rmse",
        random_state=RANDOM_SEED, verbosity=0, n_jobs=-1,
        device="cpu",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   # 验证集做 early stopping，不使用测试集
        verbose=False,
    )
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    print(f"  [XGBoost]    最佳轮次={model.best_iteration}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test,
                 learning_rate: float = 0.05, max_depth: int = 6,
                 num_leaves: int = 63, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0, n_estimators: int = 500):
    """LightGBM：使用独立验证集做 early stopping，测试集仅用于最终评估。"""
    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        num_leaves=num_leaves, subsample=subsample, colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   # 验证集做 early stopping，不使用测试集
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    elapsed = time.time() - t0
    print(f"  [LightGBM]   最佳轮次={model.best_iteration_}  耗时={elapsed:.1f}s")
    return pred_train, pred_test


def run_lstm(X_train_raw, X_val_raw, X_test_raw,
             y_train, y_val, y_test,
             seq_len: int = LSTM_SEQ_LEN,
             hidden: int = LSTM_HIDDEN,
             n_layers: int = LSTM_LAYERS,
             dropout: float = LSTM_DROPOUT,
             lr: float = LSTM_LR,
             batch_size: int = LSTM_BATCH,
             max_epochs: int = LSTM_EPOCHS,
             patience: int = LSTM_PATIENCE):
    """单向 LSTM：使用独立验证集做 early stopping 和 LR 调度，测试集仅用于最终评估。

    标准化、窗口构建均仅在训练集上拟合，然后分别 transform 验证集与测试集。
    """
    t0 = time.time()

    # ── 序列级标准化（逐特征，仅训练集拟合） ────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_va_s = scaler.transform(X_val_raw)
    X_te_s = scaler.transform(X_test_raw)

    # ── 构建滑动窗口 ─────────────────────────────────────────────
    X_tr_seq, y_tr_seq = make_sequences(X_tr_s, y_train, seq_len)
    X_va_seq, y_va_seq = make_sequences(X_va_s, y_val,   seq_len)
    X_te_seq, y_te_seq = make_sequences(X_te_s, y_test,  seq_len)

    n_features = X_tr_seq.shape[2]

    # ── 目标值归一化（仅训练集统计量） ──────────────────────────
    y_mean, y_std = y_tr_seq.mean(), y_tr_seq.std() + 1e-8
    y_tr_norm = (y_tr_seq - y_mean) / y_std
    y_va_norm = (y_va_seq - y_mean) / y_std
    y_te_norm = (y_te_seq - y_mean) / y_std

    # ── DataLoader ───────────────────────────────────────────────
    dl_train = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_norm)),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    dl_val = DataLoader(
        TensorDataset(torch.from_numpy(X_va_seq), torch.from_numpy(y_va_norm)),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )
    dl_test = DataLoader(
        TensorDataset(torch.from_numpy(X_te_seq), torch.from_numpy(y_te_norm)),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )

    # ── 模型 ─────────────────────────────────────────────────────
    model = LSTMRegressor(
        n_features=n_features,
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # LR 调度依赖验证集损失
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )
    # Huber 损失对品位异常值更稳健（delta=1.0 在均方误差和平均绝对误差之间平衡）
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate on val set only (test set not touched)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)   # LR 调度依赖验证集损失

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    早停于 epoch {epoch}（patience={patience}）")
                break

    # ── 预测（加载最佳检查点） ───────────────────────────────────
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
    print(f"  [LSTM]       最终验证损失={best_val_loss:.5f}  耗时={elapsed:.1f}s")

    # 对齐回完整序列长度（前 seq_len-1 个样本无预测，用 NaN 填充）
    pad = seq_len - 1
    pred_train_aligned = np.concatenate([np.full(pad, np.nan), pred_train_full])
    pred_test_aligned  = np.concatenate([np.full(pad, np.nan), pred_test_full])

    return pred_train_aligned, pred_test_aligned


# ═══════════════════════════════════════════════════════════════════════════
#  Optuna 超参数搜索（目标函数 + 调优入口）
# ═══════════════════════════════════════════════════════════════════════════

def _objective_svr(trial, X_train, X_val, y_train, y_val):
    C       = trial.suggest_float("C",       0.1,  200.0,  log=True)
    epsilon = trial.suggest_float("epsilon", 0.005, 1.0,   log=True)
    gamma   = trial.suggest_categorical("gamma", ["scale", "auto"])
    model   = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train, y_train)
    return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))


def tune_svr(X_train, X_val, y_train, y_val) -> dict:
    """用 Optuna TPE 搜索 SVR 超参数，返回验证集 RMSE 最优的参数字典。"""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda t: _objective_svr(t, X_train, X_val, y_train, y_val),
        n_trials=OPTUNA_N_TRIALS, show_progress_bar=False,
    )
    print(f"    [SVR 调优]   最优验证RMSE={study.best_value:.5f}  参数={study.best_params}")
    return study.best_params


def _objective_gpr(trial, X_train, X_val, y_train, y_val):
    n_restarts  = trial.suggest_int("n_restarts", 1, 5)
    noise_level = trial.suggest_float("noise_level", 1e-5, 1.0, log=True)
    max_samples = trial.suggest_int("max_samples", 100, GPR_MAX_SAMPLES, step=50)
    n_fit = min(len(X_train), max_samples)
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level, (1e-5, 1e1))
    model  = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts,
        random_state=RANDOM_SEED, normalize_y=True,
    )
    model.fit(X_train[-n_fit:], y_train[-n_fit:])
    return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))


def tune_gpr(X_train, X_val, y_train, y_val) -> dict:
    """用 Optuna TPE 搜索 GPR 超参数，返回验证集 RMSE 最优的参数字典。"""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda t: _objective_gpr(t, X_train, X_val, y_train, y_val),
        n_trials=OPTUNA_N_TRIALS_GPR, show_progress_bar=False,
    )
    print(f"    [GPR 调优]   最优验证RMSE={study.best_value:.5f}  参数={study.best_params}")
    return study.best_params


def _objective_xgboost(trial, X_train, X_val, y_train, y_val):
    params = dict(
        learning_rate    = trial.suggest_float("learning_rate",   0.005, 0.3,  log=True),
        max_depth        = trial.suggest_int("max_depth",         2,     8),
        subsample        = trial.suggest_float("subsample",       0.5,   1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree",0.5,   1.0),
        reg_alpha        = trial.suggest_float("reg_alpha",       1e-4,  10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda",      1e-4,  10.0, log=True),
        n_estimators     = trial.suggest_int("n_estimators",      100,   800,  step=50),
    )
    model = xgb.XGBRegressor(
        **params, early_stopping_rounds=20, eval_metric="rmse",
        random_state=RANDOM_SEED, verbosity=0, n_jobs=-1, device="cpu",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))


def tune_xgboost(X_train, X_val, y_train, y_val) -> dict:
    """用 Optuna TPE 搜索 XGBoost 超参数，返回验证集 RMSE 最优的参数字典。"""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda t: _objective_xgboost(t, X_train, X_val, y_train, y_val),
        n_trials=OPTUNA_N_TRIALS, show_progress_bar=False,
    )
    print(f"    [XGBoost 调优] 最优验证RMSE={study.best_value:.5f}  参数={study.best_params}")
    return study.best_params


def _objective_lightgbm(trial, X_train, X_val, y_train, y_val):
    num_leaves = trial.suggest_int("num_leaves", 15, 127)
    params = dict(
        learning_rate    = trial.suggest_float("learning_rate",    0.005, 0.3,  log=True),
        max_depth        = trial.suggest_int("max_depth",          2,     8),
        num_leaves       = num_leaves,
        subsample        = trial.suggest_float("subsample",        0.5,   1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5,   1.0),
        reg_alpha        = trial.suggest_float("reg_alpha",        1e-4,  10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda",       1e-4,  10.0, log=True),
        n_estimators     = trial.suggest_int("n_estimators",       100,   800,  step=50),
    )
    model = lgb.LGBMRegressor(
        **params, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
    )
    return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))


def tune_lightgbm(X_train, X_val, y_train, y_val) -> dict:
    """用 Optuna TPE 搜索 LightGBM 超参数，返回验证集 RMSE 最优的参数字典。"""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda t: _objective_lightgbm(t, X_train, X_val, y_train, y_val),
        n_trials=OPTUNA_N_TRIALS, show_progress_bar=False,
    )
    print(f"    [LightGBM 调优] 最优验证RMSE={study.best_value:.5f}  参数={study.best_params}")
    return study.best_params


def _objective_lstm(trial, X_train_raw, X_val_raw, y_train, y_val):
    seq_len    = trial.suggest_categorical("seq_len",    [4, 8, 12, 16])
    hidden     = trial.suggest_categorical("hidden",     [32, 64, 128])
    n_layers   = trial.suggest_int("n_layers",           1, 3)
    dropout    = trial.suggest_float("dropout",          0.05, 0.5)
    lr         = trial.suggest_float("lr",               1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_va_s = scaler.transform(X_val_raw)

    X_tr_seq, y_tr_seq = make_sequences(X_tr_s, y_train, seq_len)
    X_va_seq, y_va_seq = make_sequences(X_va_s, y_val,   seq_len)

    if len(X_tr_seq) == 0 or len(X_va_seq) == 0:
        return float("inf")

    n_features = X_tr_seq.shape[2]
    y_mean, y_std = y_tr_seq.mean(), y_tr_seq.std() + 1e-8
    y_tr_norm = (y_tr_seq - y_mean) / y_std
    y_va_norm = (y_va_seq - y_mean) / y_std

    dl_train = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_norm)),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    dl_val = DataLoader(
        TensorDataset(torch.from_numpy(X_va_seq), torch.from_numpy(y_va_norm)),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )

    model = LSTMRegressor(
        n_features=n_features, hidden=hidden, n_layers=n_layers, dropout=dropout,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    patience_cnt  = 0

    for epoch in range(1, OPTUNA_TUNE_EPOCHS + 1):
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= OPTUNA_PATIENCE_LSTM:
                break

        # Optuna 剪枝（中间步骤报告最优验证损失）
        trial.report(best_val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(best_val_loss)


def tune_lstm(X_train_raw, X_val_raw, y_train, y_val) -> dict:
    """用 Optuna TPE + 中值剪枝搜索 LSTM 超参数，返回验证集损失最优的参数字典。"""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda t: _objective_lstm(t, X_train_raw, X_val_raw, y_train, y_val),
        n_trials=OPTUNA_N_TRIALS_LSTM, show_progress_bar=False,
    )
    print(f"    [LSTM 调优]  最优验证损失={study.best_value:.5f}  参数={study.best_params}")
    return study.best_params


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
    axes_flat = axes.flatten()
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
    """对单个出矿口运行全部 6 个模型，返回评估指标 DataFrame。

    当 ENABLE_OPTUNA=True 时，在正式训练前使用 Optuna 在验证集上搜索最优超参数；
    测试集在整个调优过程中保持完全隔离，仅用于最终评估。
    """
    cfg = OUTLET_CFG[outlet_key]

    (X_train_s, X_val_s, X_test_s,
     y_train, y_val, y_test,
     X_train_raw, X_val_raw, X_test_raw,
     val_index, test_index) = load_and_prepare(outlet_key)

    records    = []
    test_preds = {}   # 供可视化
    best_params_all = {}  # 用于保存各模型最优超参数

    # ── 1. ElasticNet ──────────────────────────────────────────────
    # ElasticNetCV 内部已通过交叉验证自动选择 alpha 和 l1_ratio，无需额外调优
    ptr, pte = run_elasticnet(X_train_s, X_test_s, y_train, y_test)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "ElasticNet",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["ElasticNet"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 2. SVR ─────────────────────────────────────────────────────
    svr_params = {}
    if ENABLE_OPTUNA:
        print("  [SVR] 开始 Optuna 超参数搜索 ...")
        svr_params = tune_svr(X_train_s, X_val_s, y_train, y_val)
        best_params_all["SVR"] = svr_params
    ptr, pte = run_svr(X_train_s, X_test_s, y_train, y_test, **svr_params)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "SVR",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["SVR"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 3. GPR ─────────────────────────────────────────────────────
    gpr_params = {}
    if ENABLE_OPTUNA:
        print("  [GPR] 开始 Optuna 超参数搜索 ...")
        gpr_params = tune_gpr(X_train_s, X_val_s, y_train, y_val)
        best_params_all["GPR"] = gpr_params
    ptr, pte = run_gpr(X_train_s, X_test_s, y_train, y_test, **gpr_params)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "GPR",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["GPR"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 4. XGBoost ─────────────────────────────────────────────────
    xgb_params = {}
    if ENABLE_OPTUNA:
        print("  [XGBoost] 开始 Optuna 超参数搜索 ...")
        xgb_params = tune_xgboost(X_train_s, X_val_s, y_train, y_val)
        best_params_all["XGBoost"] = xgb_params
    ptr, pte = run_xgboost(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, **xgb_params)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "XGBoost",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["XGBoost"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 5. LightGBM ────────────────────────────────────────────────
    lgbm_params = {}
    if ENABLE_OPTUNA:
        print("  [LightGBM] 开始 Optuna 超参数搜索 ...")
        lgbm_params = tune_lightgbm(X_train_s, X_val_s, y_train, y_val)
        best_params_all["LightGBM"] = lgbm_params
    ptr, pte = run_lightgbm(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, **lgbm_params)
    m_tr = compute_metrics(y_train, ptr)
    m_te = compute_metrics(y_test,  pte)
    records.append({"模型": "LightGBM",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["LightGBM"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 6. LSTM ────────────────────────────────────────────────────
    lstm_params = {}
    if ENABLE_OPTUNA:
        print("  [LSTM] 开始 Optuna 超参数搜索 ...")
        tuned = tune_lstm(X_train_raw, X_val_raw, y_train, y_val)
        lstm_params = {k: tuned[k] for k in ("seq_len", "hidden", "n_layers", "dropout", "lr", "batch_size")}
        best_params_all["LSTM"] = lstm_params

    seq_len = lstm_params.pop("seq_len", LSTM_SEQ_LEN)
    ptr, pte = run_lstm(
        X_train_raw, X_val_raw, X_test_raw,
        y_train, y_val, y_test,
        seq_len=seq_len, **lstm_params,
    )
    # 对齐：前 seq_len-1 个位置无预测（NaN 填充），跳过这些位置
    pad = seq_len - 1
    valid_tr_full = np.concatenate([np.zeros(pad, dtype=bool), ~np.isnan(ptr[pad:])])
    m_tr = compute_metrics(y_train[valid_tr_full], ptr[valid_tr_full])
    valid_te = ~np.isnan(pte)
    m_te = compute_metrics(y_test[valid_te], pte[valid_te])
    records.append({"模型": "LSTM",
                    "训练R2": m_tr["R2"], "训练RMSE": m_tr["RMSE"], "训练MAE": m_tr["MAE"],
                    "测试R2":  m_te["R2"], "测试RMSE":  m_te["RMSE"], "测试MAE":  m_te["MAE"]})
    test_preds["LSTM"] = pte
    print(f"    测试 R²={m_te['R2']:.4f}  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}")

    # ── 保存最优超参数 ─────────────────────────────────────────────
    if ENABLE_OPTUNA and best_params_all:
        hp_path = os.path.join(RESULT_DIR, f"best_hyperparams_{outlet_key}.json")
        with open(hp_path, "w", encoding="utf-8") as f:
            json.dump(best_params_all, f, ensure_ascii=False, indent=2)
        print(f"  最优超参数已保存：{hp_path}")

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
    summary.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  完整指标已保存：{csv_path}")


if __name__ == "__main__":
    main()
