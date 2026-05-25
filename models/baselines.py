"""
Reference predictors that any real model has to beat.

Both operate in **unscaled log-return space** — same interface as the LSTM
predictor wrappers in `training.walk_forward`. They produce a return prediction
per (sample, stock); the walk-forward loop converts that to a price using the
day's anchor price, then runs it through the same metrics as the LSTMs. No
special-casing in plotting or in the dashboard.
"""
from typing import Optional
import numpy as np


class NaivePersistence:
    """
    Predict tomorrow's return = the most recent return in the input window.

    Directionally: "tomorrow's move matches today's." Empirically near 50% on
    daily US equities — daily returns have very weak autocorrelation.
    """

    name = "Naive"
    input_type = "returns"

    def predict(self, X_returns: np.ndarray) -> np.ndarray:
        # X_returns: (n_samples, seq_length, n_stocks)
        if X_returns.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {X_returns.shape}")
        return X_returns[:, -1, :].copy()


class MajorityClass:
    """
    Predict a constant equal to the training-set mean log-return per stock.

    On a bull market this predicts "up every day"; on a bear market, "down every
    day." Captures the simplest directional drift you can extract from the data.
    """

    name = "Majority"
    input_type = "returns"

    def __init__(self, train_mean_returns: np.ndarray):
        # Shape (n_stocks,)
        if train_mean_returns.ndim != 1:
            raise ValueError(
                f"train_mean_returns must be 1-D, got shape {train_mean_returns.shape}"
            )
        self.mean = train_mean_returns.astype(np.float64)

    def predict(self, X_returns: np.ndarray) -> np.ndarray:
        if X_returns.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {X_returns.shape}")
        n_samples, _, n_stocks = X_returns.shape
        if n_stocks != self.mean.shape[0]:
            raise ValueError(
                f"Input has {n_stocks} stocks, baseline was fit for {self.mean.shape[0]}"
            )
        return np.broadcast_to(self.mean, (n_samples, n_stocks)).copy()
