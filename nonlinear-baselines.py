#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Baselines for mixed-frequency MIDAS-NN prediction (MLP / RNN / LSTM / Random Forest).

Input: a 2-column, tab-separated text file with a header:
  - Column 0: X (monthly), always present
  - Column 1: Y (quarterly), blank on non-quarter months

Pipeline:
  1) Load raw series (Y has NaNs on non-quarter months)
  2) Build true MIDAS design: quarterly Y lags (N), monthly X lags (M)
  3) Time split + standardise using TRAIN stats only
  4) Fit:
       - MLP baseline (dense net on concatenated lags)
       - RNN baseline (sequence model on X lags + linear on Y lags)
       - LSTM baseline (same)
       - RandomForest baseline (on concatenated lags)
  5) Print metrics + plot predicted vs actual (train/test, original units)
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


# -----------------------------------------------------------------------------
# Data: load txt + build mixed-frequency MIDAS design
# -----------------------------------------------------------------------------

def load_txt_two_series(txt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Expects tab-separated file with header.
    Column 0: X (monthly, float)
    Column 1: Y (quarterly, float or blank)

    Returns:
      x_monthly: np.ndarray (T_months,)
      y_monthly: np.ndarray (T_months,) with np.nan where missing
    """
    x_monthly: list[float] = []
    y_monthly: list[float] = []

    with open(txt_path, "r") as f:
        f.readline()  # header
        for line in f:
            if not line.strip():
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue

            x_str = parts[0].strip()
            y_str = parts[1].strip()

            if x_str == "":
                continue

            x_monthly.append(float(x_str))
            y_monthly.append(np.nan if y_str == "" else float(y_str))

    return np.array(x_monthly, dtype=np.float32), np.array(y_monthly, dtype=np.float32)


def build_midas_design(
    x_monthly: np.ndarray,
    y_monthly: np.ndarray,
    n_quarter_lags: int,
    n_month_lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds samples for true mixed-frequency MIDAS.

    For each quarterly observation (month index t where Y is not NaN):
      target: y_quarter[q] = y_monthly[t]
      y_hist: [y_quarter[q-1], ..., y_quarter[q-n_quarter_lags]]  (quarterly lags)
      x_hist: [x_monthly[t-1], ..., x_monthly[t-n_month_lags]]    (monthly lags)

    Returns:
      y_hist: (K, n_quarter_lags)
      x_hist: (K, n_month_lags)
      y_tgt : (K,)
      q_month_index: (K,) month indices for each quarterly observation
    """
    q_month_idx_all = [t for t, y in enumerate(y_monthly) if not np.isnan(y)]
    y_quarter = [y_monthly[t] for t in q_month_idx_all]

    y_hist_list: list[np.ndarray] = []
    x_hist_list: list[np.ndarray] = []
    y_tgt_list: list[np.float32] = []
    q_month_index: list[int] = []

    for q, t in enumerate(q_month_idx_all):
        if q < n_quarter_lags or t < n_month_lags:
            continue

        y_hist = np.array([y_quarter[q - j] for j in range(1, n_quarter_lags + 1)], dtype=np.float32)
        x_hist = x_monthly[t - np.arange(1, n_month_lags + 1)].astype(np.float32)
        y_tgt = np.float32(y_quarter[q])

        y_hist_list.append(y_hist)
        x_hist_list.append(x_hist)
        y_tgt_list.append(y_tgt)
        q_month_index.append(t)

    if not y_tgt_list:
        raise ValueError("No samples created. Reduce N/M or check that Y has enough non-empty entries.")

    return (
        np.stack(y_hist_list, axis=0),
        np.stack(x_hist_list, axis=0),
        np.array(y_tgt_list, dtype=np.float32),
        np.array(q_month_index, dtype=int),
    )


@dataclass(frozen=True)
class StandardisationStats:
    y_mean: np.ndarray
    y_std: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    yt_mean: float
    yt_std: float


def split_and_standardise(
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    y_tgt: np.ndarray,
    q_month_index: np.ndarray,
    train_frac: float = 0.8,
    eps: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    """
    Time split (first train_frac for training) + standardise using TRAIN stats only.

    Note: test set is taken as [split-1:] to keep continuity in plots.
    """
    K = len(y_tgt)
    split = int(train_frac * K)

    y_hist_tr, y_hist_te = y_hist[:split], y_hist[split - 1 :]
    x_hist_tr, x_hist_te = x_hist[:split], x_hist[split - 1 :]
    y_tgt_tr, y_tgt_te = y_tgt[:split], y_tgt[split - 1 :]
    q_idx_tr, q_idx_te = q_month_index[:split], q_month_index[split - 1 :]

    y_mean = y_hist_tr.mean(axis=0, keepdims=True)
    y_std = y_hist_tr.std(axis=0, keepdims=True) + eps
    x_mean = x_hist_tr.mean(axis=0, keepdims=True)
    x_std = x_hist_tr.std(axis=0, keepdims=True) + eps

    y_hist_tr_s = (y_hist_tr - y_mean) / y_std
    y_hist_te_s = (y_hist_te - y_mean) / y_std
    x_hist_tr_s = (x_hist_tr - x_mean) / x_std
    x_hist_te_s = (x_hist_te - x_mean) / x_std

    yt_mean = float(y_tgt_tr.mean())
    yt_std = float(y_tgt_tr.std() + eps)

    y_tgt_tr_s = (y_tgt_tr - yt_mean) / yt_std
    y_tgt_te_s = (y_tgt_te - yt_mean) / yt_std

    stats = StandardisationStats(
        y_mean=y_mean, y_std=y_std,
        x_mean=x_mean, x_std=x_std,
        yt_mean=yt_mean, yt_std=yt_std,
    )

    return (
        y_hist_tr_s, x_hist_tr_s, y_tgt_tr_s, q_idx_tr,
        y_hist_te_s, x_hist_te_s, y_tgt_te_s, q_idx_te,
        stats,
    )


# -----------------------------------------------------------------------------
# Utilities: predict + metrics + plotting
# -----------------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def predict(model: Any, y_hist: Any, x_hist: Any) -> Any:
    """
    Unified prediction wrapper:
      - Torch modules: calls forward(y_hist, x_hist)
      - sklearn models: concatenates [y_hist, x_hist] and calls .predict(X)
    """
    if hasattr(model, "forward"):
        with torch.no_grad():
            return model(y_hist, x_hist)

    if hasattr(model, "predict"):
        X = np.concatenate([_to_numpy(y_hist), _to_numpy(x_hist)], axis=1)
        return model.predict(X)

    raise ValueError("Unknown model type")


def regression_metrics(y_true: Any, y_pred: Any) -> tuple[float, float, float, float]:
    y_true_np = _to_numpy(y_true).astype(np.float64)
    y_pred_np = _to_numpy(y_pred).astype(np.float64)

    mse = float(np.mean((y_pred_np - y_true_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred_np - y_true_np)))

    denom = float(np.sum((y_true_np - y_true_np.mean()) ** 2))
    r2 = float("nan") if denom == 0.0 else float(1.0 - np.sum((y_pred_np - y_true_np) ** 2) / denom)

    return mse, rmse, mae, r2


def plot_train_test(
    model: Any,
    model_name: str,
    y_hist_tr: Any,
    x_hist_tr: Any,
    y_tgt_tr: Any,
    q_idx_tr: np.ndarray,
    y_hist_te: Any,
    x_hist_te: Any,
    y_tgt_te: Any,
    q_idx_te: np.ndarray,
    stats: StandardisationStats,
    train_frac: float,
    ylabel: str = "GVA",
) -> None:
    yhat_tr_s = predict(model, y_hist_tr, x_hist_tr)
    yhat_te_s = predict(model, y_hist_te, x_hist_te)

    yhat_tr = _to_numpy(yhat_tr_s) * stats.yt_std + stats.yt_mean
    yhat_te = _to_numpy(yhat_te_s) * stats.yt_std + stats.yt_mean
    y_tr_u = _to_numpy(y_tgt_tr) * stats.yt_std + stats.yt_mean
    y_te_u = _to_numpy(y_tgt_te) * stats.yt_std + stats.yt_mean

    print(f"\n--- {model_name} | TRAIN performance (original units) ---")
    mse, rmse, mae, r2 = regression_metrics(y_tr_u, yhat_tr)
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"R2   = {r2:.3f}")

    print(f"\n--- {model_name} | TEST performance (original units) ---")
    mse, rmse, mae, r2 = regression_metrics(y_te_u, yhat_te)
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"R2   = {r2:.3f}")

    plt.figure()
    plt.plot(q_idx_tr, y_tr_u, label="Actual (train)")
    plt.plot(q_idx_tr, yhat_tr, label="Predicted (train)")
    plt.plot(q_idx_te, y_te_u, label="Actual (test)")
    plt.plot(q_idx_te, yhat_te, label="Predicted (test)")
    plt.xlabel("Month index of quarter observation")
    plt.ylabel(ylabel)
    plt.title(f"Baselines ({model_name}): first {int(100*train_frac)}% train, last {100-int(100*train_frac)}% test")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Baseline models
# -----------------------------------------------------------------------------

class MIDAS_RNN(nn.Module):
    def __init__(self, n_quarter_lags: int, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(hidden + n_quarter_lags, 1)

    def forward(self, y_hist: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        seq = x_hist.unsqueeze(-1)  # (batch, M) -> (batch, M, 1)
        _, h = self.rnn(seq)
        h_last = h[-1]              # (batch, hidden)
        z = torch.cat([h_last, y_hist], dim=1)
        return self.fc(z).squeeze()


class MIDAS_LSTM(nn.Module):
    def __init__(self, n_month_lags: int, n_quarter_lags: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden + n_quarter_lags, 1)

    def forward(self, y_hist: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        seq = x_hist.unsqueeze(-1)  # (batch, M, 1)
        _, (h, _) = self.lstm(seq)
        h_last = h[-1]              # (batch, hidden)
        z = torch.cat([h_last, y_hist], dim=1)
        return self.fc(z).squeeze()


class MIDAS_MLP(nn.Module):
    def __init__(self, n_quarter_lags: int, n_month_lags: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_quarter_lags + n_month_lags, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, y_hist: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        z = torch.cat([y_hist, x_hist], dim=1)
        return self.net(z).squeeze()


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    y_hist: torch.Tensor,
    x_hist: torch.Tensor,
    y_tgt: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.003,
    verbose_every: int = 50,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        pred = model(y_hist, x_hist)
        loss = loss_fn(pred, y_tgt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose_every and ep % verbose_every == 0:
            print(f"epoch {ep} loss {loss.item():.6f}")

    return model


def train_random_forest(
    y_hist_tr: Any,
    x_hist_tr: Any,
    y_tgt_tr: Any,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 0,
) -> RandomForestRegressor:
    X_tr = np.concatenate([_to_numpy(y_hist_tr), _to_numpy(x_hist_tr)], axis=1)
    y_tr = _to_numpy(y_tgt_tr)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    return rf


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main_pipeline(
    txt_path: str,
    n_quarter_lags: int,
    n_month_lags: int,
    train_frac: float = 0.8,
    device: str | None = None,
) -> tuple[Any, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_monthly, y_monthly = load_txt_two_series(txt_path)
    y_hist, x_hist, y_tgt, q_month_index = build_midas_design(x_monthly, y_monthly, n_quarter_lags, n_month_lags)

    (
        y_hist_tr, x_hist_tr, y_tgt_tr, q_idx_tr,
        y_hist_te, x_hist_te, y_tgt_te, q_idx_te,
        stats,
    ) = split_and_standardise(y_hist, x_hist, y_tgt, q_month_index, train_frac=train_frac)

    # Torch tensors
    y_hist_tr_t = torch.tensor(y_hist_tr, dtype=torch.float32, device=device)
    x_hist_tr_t = torch.tensor(x_hist_tr, dtype=torch.float32, device=device)
    y_tgt_tr_t = torch.tensor(y_tgt_tr, dtype=torch.float32, device=device)

    y_hist_te_t = torch.tensor(y_hist_te, dtype=torch.float32, device=device)
    x_hist_te_t = torch.tensor(x_hist_te, dtype=torch.float32, device=device)
    y_tgt_te_t = torch.tensor(y_tgt_te, dtype=torch.float32, device=device)

    # Models
    mlp = MIDAS_MLP(n_quarter_lags=y_hist.shape[1], n_month_lags=x_hist.shape[1]).to(device)
    rnn = MIDAS_RNN(n_quarter_lags=y_hist.shape[1]).to(device)
    lstm = MIDAS_LSTM(n_month_lags=x_hist.shape[1], n_quarter_lags=y_hist.shape[1]).to(device)

    print("\nTraining MLP")
    train_model(mlp, y_hist_tr_t, x_hist_tr_t, y_tgt_tr_t)

    print("\nTraining RNN")
    train_model(rnn, y_hist_tr_t, x_hist_tr_t, y_tgt_tr_t)

    print("\nTraining LSTM")
    train_model(lstm, y_hist_tr_t, x_hist_tr_t, y_tgt_tr_t)

    print("\nTraining RandomForest")
    rf = train_random_forest(y_hist_tr_t, x_hist_tr_t, y_tgt_tr_t, n_estimators=400, max_depth=None, random_state=42)

    # Metrics (standardised target units)
    pred_mlp = predict(mlp, y_hist_te_t, x_hist_te_t)
    pred_rnn = predict(rnn, y_hist_te_t, x_hist_te_t)
    pred_lstm = predict(lstm, y_hist_te_t, x_hist_te_t)
    pred_rf = predict(rf, y_hist_te_t, x_hist_te_t)

    print("\nMLP:", regression_metrics(y_tgt_te_t, pred_mlp))
    print("RNN:", regression_metrics(y_tgt_te_t, pred_rnn))
    print("LSTM:", regression_metrics(y_tgt_te_t, pred_lstm))
    print("RandomForest:", regression_metrics(y_tgt_te_t, pred_rf))

    # Plots (original units via stats)
    plot_train_test(mlp, "MLP", y_hist_tr_t, x_hist_tr_t, y_tgt_tr_t, q_idx_tr,
                    y_hist_te_t, x_hist_te_t, y_tgt_te_t, q_idx_te, stats, train_frac=train_frac)

    plot_train_test(rnn, "RNN", y_hist_tr_t, x_hist_tr_t, y_tgt_tr_t, q_idx_tr,
                    y_hist_te_t, x_hist_te_t, y_tgt_te_t, q_idx_te, stats, train_frac=train_frac)

    plot_train_test(lstm, "LSTM", y_hist_tr_t, x_hist_tr_t, y_tgt_tr_t, q_idx_tr,
                    y_hist_te_t, x_hist_te_t, y_tgt_te_t, q_idx_te, stats, train_frac=train_frac)

    plot_train_test(rf, "RandomForest", y_hist_tr_t, x_hist_tr_t, y_tgt_tr_t, q_idx_tr,
                    y_hist_te_t, x_hist_te_t, y_tgt_te_t, q_idx_te, stats, train_frac=train_frac)

    return pred_mlp, pred_lstm

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MIDAS baselines (MLP / RNN / LSTM / RandomForest)."
    )

    p.add_argument(
        "--data",
        required=True,
        help="Path to tab-separated data file (X monthly, Y quarterly/blank).",
    )
    p.add_argument("--N", type=int, default=4, help="Number of quarterly lags.")
    p.add_argument("--M", type=int, default=12, help="Number of monthly lags.")
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.9,
        help="Fraction of samples used for training (time split).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu' or 'cuda'. Default: auto-detect.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main_pipeline(
        txt_path=args.data,
        n_quarter_lags=args.N,
        n_month_lags=args.M,
        train_frac=args.train_frac,
        device=args.device,
    )

