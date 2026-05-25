"""Unit tests for training.walk_forward."""
import numpy as np
import pandas as pd
import pytest

from training.walk_forward import walk_forward_predict, windows_to_frame, pool_windows


class _ZeroPredictor:
    """Predicts a zero return for every sample — direction is 'flat'."""
    name = "Zero"

    def predict(self, X):
        return np.zeros((X.shape[0], X.shape[2]))


class _PerfectPredictor:
    """Predicts the actual return — direction always correct."""
    name = "Oracle"

    def __init__(self, y):
        self.y = y

    def predict(self, X):
        # In the walk-forward loop we only get the X for the current window,
        # so look up y by the X identity using the global registry.
        # Simpler: accept that this predictor sees the full y and returns the
        # first len(X) rows of remaining y.
        out = self.y[: X.shape[0]]
        self.y = self.y[X.shape[0]:]
        return out


def _synthetic(n_samples=10, seq_length=3, n_stocks=2):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.01, size=(n_samples, seq_length, n_stocks))
    y = rng.normal(0, 0.01, size=(n_samples, n_stocks))
    anchors = 100 + rng.normal(0, 5, size=(n_samples, n_stocks))
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="B")
    return X, y, anchors, dates


def test_windows_non_overlapping_and_ordered():
    X, y, anchors, dates = _synthetic(n_samples=10)
    results = walk_forward_predict(
        _ZeroPredictor(), X, y, anchors, dates, ["AAA", "BBB"],
        window_days=4, stride_days=4,
    )
    assert len(results) == 2
    assert results[0]["window_idx"] == 0
    assert results[1]["window_idx"] == 1
    assert results[1]["window_start"] > results[0]["window_end"]


def test_windows_to_frame_columns():
    X, y, anchors, dates = _synthetic()
    results = walk_forward_predict(
        _ZeroPredictor(), X, y, anchors, dates, ["AAA", "BBB"],
        window_days=5, stride_days=5,
    )
    df = windows_to_frame(results, "Zero")
    assert set(df.columns) == {
        "model", "window_idx", "window_start", "window_end",
        "n_samples", "RMSE", "MAPE", "DirectionalAccuracy",
    }


def test_pool_windows_concatenates():
    X, y, anchors, dates = _synthetic(n_samples=10)
    results = walk_forward_predict(
        _ZeroPredictor(), X, y, anchors, dates, ["AAA", "BBB"],
        window_days=4, stride_days=4,
    )
    pooled = pool_windows(results)
    assert pooled["pred_prices"].shape == (8, 2)
    assert pooled["true_prices"].shape == (8, 2)


def test_too_short_raises():
    X, y, anchors, dates = _synthetic(n_samples=3)
    with pytest.raises(ValueError):
        walk_forward_predict(
            _ZeroPredictor(), X, y, anchors, dates, ["AAA", "BBB"],
            window_days=10, stride_days=5,
        )
