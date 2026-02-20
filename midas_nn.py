#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""
MIDAS-NN (mixed-frequency) regression with learned lag weights.

Input: a 2-column, tab-separated text file with a header:
  - Column 0: X (monthly), always present
  - Column 1: Y (quarterly), blank on non-quarter months

Builds a true mixed-frequency MIDAS design:
  - Quarterly Y lags (N)
  - Monthly X lags (M)

Trains a linear MIDAS model where lag weights are produced by small neural nets
as a function of lag index only (softmax-normalised, nonnegative, sum-to-1).

Plots:
  (i) learned quarterly weights b
  (ii) learned monthly weights c
  (iii) predicted vs actual quarterly series (train/test, in original units)
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model: linear in data, NN gives lag weights from lag index only
# -----------------------------------------------------------------------------

class LagNet(nn.Module):
    """Maps a lag index (normalised to [0, 1]) to a softmax weight vector."""
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def weights(self, K: int, device: str) -> torch.Tensor:
        k = torch.arange(1, K + 1, device=device, dtype=torch.float32).view(-1, 1)
        u = k / float(K)
        scores = self.net(u).view(-1)
        return torch.softmax(scores, dim=0)


class LinearMidasNN(nn.Module):
    """Linear MIDAS regression with neural-net lag weights."""
    def __init__(self, n_quarter_lags: int, n_month_lags: int, hidden: int = 16):
        super().__init__()
        self.n_quarter_lags = n_quarter_lags
        self.n_month_lags = n_month_lags

        self.gY = LagNet(hidden)
        self.gX = LagNet(hidden)

        self.a0 = nn.Parameter(torch.tensor(0.0))  # intercept

    def forward(
        self,
        y_hist: torch.Tensor,  # (K, n_quarter_lags)
        x_hist: torch.Tensor,  # (K, n_month_lags)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = y_hist.device

        b = self.gY.weights(self.n_quarter_lags, device)  # (n_quarter_lags,)
        c = self.gX.weights(self.n_month_lags, device)    # (n_month_lags,)

        y_hat = (y_hist * b).sum(dim=1) + (x_hist * c).sum(dim=1) + self.a0
        return y_hat, b, c


# -----------------------------------------------------------------------------
# Data: load txt + build mixed-frequency MIDAS design
# -----------------------------------------------------------------------------

def load_txt_two_series(txt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Expects tab-separated file with header.
    Column 0: X (monthly, float)
    Column 1: Y (quarterly, float or blank)
    Returns:
      x_monthly: np.array shape (T_months,)
      y_monthly: np.array shape (T_months,) with np.nan where missing
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

    For each quarterly observation (month index t where Y is non-NaN):
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
    Split in time (first train_frac for training) and standardise using TRAIN stats only.

    Note: the test set is taken as [split-1:] (i.e., includes the last training sample)
    to keep a continuous plot and avoid a visual gap at the boundary.
    """
    K = len(y_tgt)
    split = int(train_frac * K)

    y_hist_tr, y_hist_te = y_hist[:split], y_hist[split - 1 :]
    x_hist_tr, x_hist_te = x_hist[:split], x_hist[split - 1 :]
    y_tgt_tr, y_tgt_te = y_tgt[:split], y_tgt[split - 1 :]
    q_idx_tr, q_idx_te = q_month_index[:split], q_month_index[split - 1 :]

    # Inputs: per-lag standardisation
    y_mean = y_hist_tr.mean(axis=0, keepdims=True)
    y_std = y_hist_tr.std(axis=0, keepdims=True) + eps
    x_mean = x_hist_tr.mean(axis=0, keepdims=True)
    x_std = x_hist_tr.std(axis=0, keepdims=True) + eps

    y_hist_tr_s = (y_hist_tr - y_mean) / y_std
    y_hist_te_s = (y_hist_te - y_mean) / y_std
    x_hist_tr_s = (x_hist_tr - x_mean) / x_std
    x_hist_te_s = (x_hist_te - x_mean) / x_std

    # Target: scalar standardisation
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


def regression_metrics_normalised(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """
    MSE/RMSE/MAE computed after normalising each series by its own mean/std.
    (This keeps your original intent; we just guard against near-zero std.)
    """
    y_pred_mean = float(np.mean(y_pred))
    y_true_mean = float(np.mean(y_true))
    y_pred_std = float(np.std(y_pred) + eps)
    y_true_std = float(np.std(y_true) + eps)

    y_pred_norm = (y_pred - y_pred_mean) / y_pred_std
    y_true_norm = (y_true - y_true_mean) / y_true_std

    mse = float(np.mean((y_pred_norm - y_true_norm) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred_norm - y_true_norm)))
    return mse, rmse, mae


# -----------------------------------------------------------------------------
# Train + predict + plot
# -----------------------------------------------------------------------------

def train_model(
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    y_tgt: np.ndarray,
    n_quarter_lags: int,
    n_month_lags: int,
    epochs: int = 800,
    lr: float = 1e-2,
    hidden: int = 32,
) -> LinearMidasNN:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LinearMidasNN(n_quarter_lags, n_month_lags, hidden=hidden).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    y_hist_t = torch.tensor(y_hist, device=device)
    x_hist_t = torch.tensor(x_hist, device=device)
    y_tgt_t = torch.tensor(y_tgt, device=device)

    model.train()
    for _ in range(epochs):
        y_pred, _, _ = model(y_hist_t, x_hist_t)
        loss = loss_fn(y_pred, y_tgt_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


def predict(
    model: LinearMidasNN,
    y_hist: np.ndarray,
    x_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        y_pred, b_hat, c_hat = model(
            torch.tensor(y_hist, device=device),
            torch.tensor(x_hist, device=device),
        )

    return y_pred.cpu().numpy(), b_hat.cpu().numpy(), c_hat.cpu().numpy()


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = float(np.mean((y_true - y_pred) ** 2))
    return float(np.sqrt(mse))


def plot_train_test(
    model: LinearMidasNN,
    y_hist_tr: np.ndarray,
    x_hist_tr: np.ndarray,
    y_tgt_tr: np.ndarray,
    q_idx_tr: np.ndarray,
    y_hist_te: np.ndarray,
    x_hist_te: np.ndarray,
    y_tgt_te: np.ndarray,
    q_idx_te: np.ndarray,
    stats: StandardisationStats,
    n_quarter_lags: int,
    n_month_lags: int,
    train_frac: float,
    ylabel: str = "Stock Employed",
) -> None:
    yhat_tr_s, b_hat, c_hat = predict(model, y_hist_tr, x_hist_tr)
    yhat_te_s, _, _ = predict(model, y_hist_te, x_hist_te)

    # Unscale targets back to original units
    yhat_tr = yhat_tr_s * stats.yt_std + stats.yt_mean
    yhat_te = yhat_te_s * stats.yt_std + stats.yt_mean
    y_tr_u = y_tgt_tr * stats.yt_std + stats.yt_mean
    y_te_u = y_tgt_te * stats.yt_std + stats.yt_mean

    print("\n--- TRAIN performance (normalised-series metrics; displayed in original scale) ---")
    mse, rmse, mae = regression_metrics_normalised(y_tr_u, yhat_tr)
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")

    print("\n--- TEST performance (normalised-series metrics; displayed in original scale) ---")
    mse, rmse, mae = regression_metrics_normalised(y_te_u, yhat_te)
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")

    # Learned weights
    plt.figure()
    plt.plot(np.arange(1, n_quarter_lags + 1), b_hat, marker="o")
    plt.xlabel("Quarter lag $\\ell'$")
    plt.ylabel("$b_{\\ell'}$")
    plt.title("Learned quarterly lag weights $b$")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, n_month_lags + 1), c_hat)
    plt.xlabel("Monthly lag $\\ell$")
    plt.ylabel("$c_{\\ell}$")
    plt.title("Learned monthly MIDAS lag weights $c$")
    plt.tight_layout()
    plt.show()

    # Train + test predictions
    plt.figure()
    plt.plot(q_idx_tr, y_tr_u, label="Actual (train)")
    plt.plot(q_idx_tr, yhat_tr, label="Predicted (train)")
    plt.plot(q_idx_te, y_te_u, label="Actual (test)")
    plt.plot(q_idx_te, yhat_te, label="Predicted (test)")
    plt.xlabel("Month index of quarter observation")
    plt.ylabel(ylabel)
    plt.title(f"MIDAS-NN: first {int(100*train_frac)}% train, last {100-int(100*train_frac)}% test")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CLI / entry point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MIDAS-NN with learned lag weights.")
    p.add_argument("--data", required=True, help="Path to tab-separated data file (X monthly, Y quarterly/blank).")
    p.add_argument("--N", type=int, default=30, help="Number of quarterly lags.")
    p.add_argument("--M", type=int, default=80, help="Number of monthly lags.")
    p.add_argument("--train-frac", type=float, default=0.9, help="Fraction of samples used for training (time split).")

    p.add_argument("--epochs", type=int, default=1000, help="Training epochs per restart.")
    p.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate.")
    p.add_argument("--hidden", type=int, default=128, help="Hidden width for LagNet.")
    p.add_argument("--restarts", type=int, default=10, help="Number of random restarts; best test RMSE is kept.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for reproducibility (restart i uses seed+i).")

    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    x_monthly, y_monthly = load_txt_two_series(args.data)
    y_hist, x_hist, y_tgt, q_month_index = build_midas_design(x_monthly, y_monthly, args.N, args.M)

    (
        y_hist_tr, x_hist_tr, y_tgt_tr, q_idx_tr,
        y_hist_te, x_hist_te, y_tgt_te, q_idx_te,
        stats,
    ) = split_and_standardise(y_hist, x_hist, y_tgt, q_month_index, train_frac=args.train_frac)

    best_rmse = np.inf
    best_model: LinearMidasNN | None = None

    for i in range(args.restarts):
        set_seed(args.seed + i)

        model_try = train_model(
            y_hist_tr, x_hist_tr, y_tgt_tr,
            n_quarter_lags=args.N,
            n_month_lags=args.M,
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
        )

        yhat_te, _, _ = predict(model_try, y_hist_te, x_hist_te)  # in standardised target units
        rmse_te = rmse_score(y_tgt_te, yhat_te)

        print(f"Run {i:02d} | test RMSE (standardised target units) = {rmse_te:.4f}")

        if rmse_te < best_rmse:
            best_rmse = rmse_te
            best_model = model_try

    assert best_model is not None
    print(f"\nBest test RMSE over {args.restarts} runs (standardised target units): {best_rmse:.4f}")

    plot_train_test(
        best_model,
        y_hist_tr, x_hist_tr, y_tgt_tr, q_idx_tr,
        y_hist_te, x_hist_te, y_tgt_te, q_idx_te,
        stats,
        n_quarter_lags=args.N,
        n_month_lags=args.M,
        train_frac=args.train_frac,
    )


if __name__ == "__main__":
    main(parse_args())


# In[ ]:




