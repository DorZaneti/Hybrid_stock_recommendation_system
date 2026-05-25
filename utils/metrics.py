"""
Evaluation metrics for stock-price predictions.

These operate on real (inverse-scaled) prices so the numbers are interpretable
by an analyst: RMSE is in dollars, MAPE is a percent, directional accuracy is
the success rate of predicting the right next-day direction.
"""
from typing import List
import math
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

EPSILON = 1e-10


def directional_accuracy(
    y_true_prev: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Percentage of samples where the predicted direction matches the actual direction.

    Direction is taken relative to the previous-day price: a prediction is
    "correct" when sign(y_pred - y_true_prev) == sign(y_true - y_true_prev).
    Flat (zero) movements are treated as a separate sign and only count as a hit
    when both prediction and reality are flat.

    Args:
        y_true_prev: previous-day actual prices, shape (n_samples, n_stocks)
        y_true: actual prices for the predicted day, same shape
        y_pred: predicted prices, same shape

    Returns:
        Accuracy in [0, 100] as a percentage.
    """
    if y_true.shape != y_pred.shape or y_true.shape != y_true_prev.shape:
        raise ValueError(
            f"Shape mismatch: prev={y_true_prev.shape}, true={y_true.shape}, pred={y_pred.shape}"
        )

    if y_true.size == 0:
        return 0.0

    true_dir = np.sign(y_true - y_true_prev)
    pred_dir = np.sign(y_pred - y_true_prev)
    hits = (true_dir == pred_dir).sum()
    return float(hits) / y_true.size * 100.0


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute percentage error, robust to zeros via EPSILON.

    Returns a value in percent (e.g. 3.2 means 3.2% average error).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true={y_true.shape}, pred={y_pred.shape}")

    if y_true.size == 0:
        return 0.0

    denom = np.where(np.abs(y_true) < EPSILON, EPSILON, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error in the same units as the input prices."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true={y_true.shape}, pred={y_pred.shape}")
    if y_true.size == 0:
        return 0.0
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def per_stock_metrics(
    y_true_prev: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tickers: List[str],
) -> pd.DataFrame:
    """
    Compute RMSE, MAPE and DirectionalAccuracy per ticker.

    All inputs share shape (n_samples, n_stocks) where columns correspond to
    `tickers` in order.

    Returns a DataFrame indexed by ticker with three columns.
    """
    if y_true.shape != y_pred.shape or y_true.shape != y_true_prev.shape:
        raise ValueError(
            f"Shape mismatch: prev={y_true_prev.shape}, true={y_true.shape}, pred={y_pred.shape}"
        )
    if y_true.shape[1] != len(tickers):
        raise ValueError(
            f"tickers length {len(tickers)} != number of columns {y_true.shape[1]}"
        )

    rows = []
    for i, ticker in enumerate(tickers):
        col_prev = y_true_prev[:, i:i + 1]
        col_true = y_true[:, i:i + 1]
        col_pred = y_pred[:, i:i + 1]
        rows.append(
            {
                "Ticker": ticker,
                "RMSE": rmse(col_true, col_pred),
                "MAPE": mape(col_true, col_pred),
                "DirectionalAccuracy": directional_accuracy(col_prev, col_true, col_pred),
            }
        )

    df = pd.DataFrame(rows).set_index("Ticker")
    logger.debug(f"per_stock_metrics computed for {len(df)} tickers")
    return df
