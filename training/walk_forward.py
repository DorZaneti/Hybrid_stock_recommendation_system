"""
Walk-forward evaluation across an out-of-sample period.

Slides a fixed-size window across the OOS slice; each window produces one row
of metrics. Aggregating across windows gives a much more stable success-rate
estimate than a single tail slice.

The interface is intentionally simple: any object with `.predict(X_returns)`
returning unscaled log-returns can be evaluated. The LSTM wrappers below adapt
the trained PyTorch models to that interface (handling the scale/unscale +
torch tensor conversion); the baseline classes in `models.baselines` already
implement it directly.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from utils.logger import get_logger
from utils.metrics import directional_accuracy, mape, per_stock_metrics, rmse

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# Predictor wrappers: bring everything to a common (unscaled-return) interface.
# ----------------------------------------------------------------------------


class LSTMPredictor:
    """
    Wraps a trained multi-feature StockLSTM into a `.predict(X_unscaled)` interface.

    Accepts separate ``input_scaler`` (for the feature columns) and
    ``output_scaler`` (for the return columns) so the inputs and outputs can
    have different dimensionality — required for Part 6 Phase 2. When only one
    scaler is provided, it is used for both (legacy behavior).
    """

    input_type = "features"

    def __init__(self, model, scaler, device: torch.device, name: str, output_scaler=None):
        self.model = model
        self.input_scaler = scaler
        self.output_scaler = output_scaler if output_scaler is not None else scaler
        self.device = device
        self.name = name
        self.val_da: Optional[float] = None  # set after training for ensemble weighting

    def predict(self, X_unscaled: np.ndarray) -> np.ndarray:
        if X_unscaled.ndim != 3:
            raise ValueError(f"Expected 3D input, got {X_unscaled.shape}")
        n, seq, k = X_unscaled.shape
        flat = X_unscaled.reshape(-1, k)
        scaled = self.input_scaler.transform(flat).reshape(n, seq, k)

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(scaled, dtype=torch.float32, device=self.device)
            out = self.model(x).cpu().numpy()

        if getattr(self, "direction_mode", False):
            # Logits → signed unit predictions (±1e-4) so the existing
            # directional-accuracy metric (which uses sign(pred)) works unchanged.
            probs = 1.0 / (1.0 + np.exp(-out))
            return np.where(probs >= 0.5, 1e-4, -1e-4)
        return self.output_scaler.inverse_transform(out)


class SingleFeaturePredictor:
    """
    Wraps a trained per-stock LSTM into the standard `.predict(X_unscaled)` interface.

    For each stock j the inputs are the contiguous block of feature columns
    ``[j*n_features_per_stock : (j+1)*n_features_per_stock]`` from the (n, seq,
    n_stocks*n_features) feature array. The model outputs one scalar per stock
    (its predicted log-return), which is unscaled via ``output_scaler``.

    When ``output_scaler is None`` the input scaler is reused for the output —
    this is the legacy n_features_per_stock=1 path where input and output
    share the same scaler.
    """

    input_type = "features"

    def __init__(
        self,
        model,
        scaler,
        device: torch.device,
        name: str,
        output_scaler=None,
        n_features_per_stock: int = 1,
        n_macro_features: int = 0,
    ):
        self.model = model
        self.input_scaler = scaler
        self.output_scaler = output_scaler if output_scaler is not None else scaler
        self.device = device
        self.name = name
        self.n_features_per_stock = int(n_features_per_stock)
        self.n_macro_features = int(n_macro_features)
        self.val_da: Optional[float] = None  # set after training for ensemble weighting

    def predict(self, X_unscaled: np.ndarray) -> np.ndarray:
        if X_unscaled.ndim != 3:
            raise ValueError(f"Expected 3D input, got {X_unscaled.shape}")
        n, seq, k = X_unscaled.shape

        # Derive n_stocks from the per-stock feature count.
        stock_cols = k - self.n_macro_features
        if stock_cols <= 0 or stock_cols % self.n_features_per_stock != 0:
            raise ValueError(
                f"Input has {k} columns; after removing {self.n_macro_features} "
                f"macro columns, {stock_cols} stock columns remain which is not "
                f"divisible by n_features_per_stock={self.n_features_per_stock}"
            )
        n_stocks = stock_cols // self.n_features_per_stock

        # Scale ALL columns using the fitted scaler — slicing only what we need later.
        in_means = self.input_scaler.mean_[:k]
        in_scales = self.input_scaler.scale_[:k]
        scaled = (X_unscaled - in_means) / in_scales   # (n, seq, k)

        self.model.eval()
        out_scaled = np.zeros((n, n_stocks), dtype=np.float32)
        with torch.no_grad():
            for j in range(n_stocks):
                start = j * self.n_features_per_stock
                end = start + self.n_features_per_stock
                x_stock = scaled[:, :, start:end]
                if self.n_macro_features > 0:
                    x_stock = np.concatenate(
                        [x_stock, scaled[:, :, stock_cols:stock_cols + self.n_macro_features]],
                        axis=-1,
                    )
                x_j = torch.tensor(x_stock, dtype=torch.float32, device=self.device)
                out_scaled[:, j:j + 1] = self.model(x_j).cpu().numpy()

        if getattr(self, "direction_mode", False):
            probs = 1.0 / (1.0 + np.exp(-out_scaled))
            return np.where(probs >= 0.5, 1e-4, -1e-4)
        out_means = self.output_scaler.mean_
        out_scales = self.output_scaler.scale_
        return out_scaled * out_scales + out_means


# ----------------------------------------------------------------------------
# Walk-forward loop.
# ----------------------------------------------------------------------------


def walk_forward_predict(
    predictor: Any,
    X_oos_features: np.ndarray,
    y_oos_unscaled: np.ndarray,
    anchor_prices: np.ndarray,
    dates: Sequence,
    tickers: Sequence[str],
    window_days: int,
    stride_days: int,
    X_oos_returns: np.ndarray = None,
) -> List[Dict[str, Any]]:
    """
    Slide a window across the OOS slice and evaluate at each step.

    The predictor's ``input_type`` class attribute decides which X is passed
    to it:
        - ``"features"`` (default for LSTMs / Ensemble): receives ``X_oos_features``
          of shape (n, seq, n_stocks * n_features_per_stock).
        - ``"returns"`` (Naive / Majority baselines): receives ``X_oos_returns``
          of shape (n, seq, n_stocks) — the raw log-returns of Close.

    For backward compatibility, if ``X_oos_returns`` is None the same array is
    passed to both kinds of predictor (the Phase 0-1 single-array pipeline).

    Targets and anchors are always in price/return space (unchanged from Phase 0).
    """
    if X_oos_returns is None:
        X_oos_returns = X_oos_features
    n = len(y_oos_unscaled)
    if n < window_days:
        raise ValueError(
            f"OOS slice has {n} samples but window_days={window_days}; need more data."
        )

    input_type = getattr(predictor, "input_type", "features")
    X_source = X_oos_returns if input_type == "returns" else X_oos_features

    results: List[Dict[str, Any]] = []
    dates_arr = pd.to_datetime(list(dates))

    for w_idx, start in enumerate(range(0, n - window_days + 1, stride_days)):
        end = start + window_days
        X_w = X_source[start:end]
        y_w_returns = y_oos_unscaled[start:end]
        anchors_w = anchor_prices[start:end]
        dates_w = dates_arr[start:end]

        pred_returns = predictor.predict(X_w)
        pred_prices = anchors_w * np.exp(pred_returns)
        true_prices = anchors_w * np.exp(y_w_returns)

        results.append(
            {
                "window_idx": w_idx,
                "window_start": dates_w[0],
                "window_end": dates_w[-1],
                "n_samples": int(window_days),
                "RMSE": rmse(true_prices, pred_prices),
                "MAPE": mape(true_prices, pred_prices),
                "DirectionalAccuracy": directional_accuracy(
                    anchors_w, true_prices, pred_prices
                ),
                "pred_prices": pred_prices,
                "true_prices": true_prices,
                "anchors": anchors_w,
                "dates": dates_w,
            }
        )

    logger.info(
        f"walk_forward_predict[{getattr(predictor, 'name', type(predictor).__name__)}]: "
        f"{len(results)} windows of {window_days} days (stride {stride_days})"
    )
    return results


def windows_to_frame(
    per_window_results: List[Dict[str, Any]], model_name: str
) -> pd.DataFrame:
    """Strip raw arrays and return a tidy DataFrame for CSV output."""
    rows = []
    for r in per_window_results:
        rows.append(
            {
                "model": model_name,
                "window_idx": r["window_idx"],
                "window_start": r["window_start"],
                "window_end": r["window_end"],
                "n_samples": r["n_samples"],
                "RMSE": r["RMSE"],
                "MAPE": r["MAPE"],
                "DirectionalAccuracy": r["DirectionalAccuracy"],
            }
        )
    return pd.DataFrame(rows)


def pool_windows(per_window_results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Concatenate all windows into one big array per quantity (for pooled per-stock metrics)."""
    if not per_window_results:
        return {}
    return {
        "pred_prices": np.concatenate([r["pred_prices"] for r in per_window_results]),
        "true_prices": np.concatenate([r["true_prices"] for r in per_window_results]),
        "anchors": np.concatenate([r["anchors"] for r in per_window_results]),
        "dates": np.concatenate([r["dates"] for r in per_window_results]),
    }


def pooled_per_stock(
    pooled: Dict[str, np.ndarray], tickers: Sequence[str]
) -> pd.DataFrame:
    """Run per_stock_metrics on the concatenated walk-forward results."""
    return per_stock_metrics(
        pooled["anchors"], pooled["true_prices"], pooled["pred_prices"], list(tickers)
    )
