"""Unit tests for utils.metrics."""
import numpy as np
import pandas as pd
import pytest

from utils.metrics import directional_accuracy, mape, per_stock_metrics, rmse


def test_directional_accuracy_perfect():
    prev = np.array([[100.0, 50.0], [101.0, 49.0]])
    true = np.array([[101.0, 49.0], [102.0, 48.0]])
    pred = np.array([[110.0, 30.0], [120.0, 10.0]])  # right direction, wrong magnitude
    assert directional_accuracy(prev, true, pred) == 100.0


def test_directional_accuracy_perfectly_wrong():
    prev = np.array([[100.0, 50.0]])
    true = np.array([[101.0, 49.0]])    # up, down
    pred = np.array([[90.0, 60.0]])     # down, up
    assert directional_accuracy(prev, true, pred) == 0.0


def test_directional_accuracy_mixed():
    prev = np.array([[100.0, 50.0]])
    true = np.array([[101.0, 49.0]])    # up, down
    pred = np.array([[110.0, 60.0]])    # up, up  -> half right
    assert directional_accuracy(prev, true, pred) == 50.0


def test_directional_accuracy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        directional_accuracy(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((3, 2)),
        )


def test_directional_accuracy_empty():
    empty = np.empty((0, 2))
    assert directional_accuracy(empty, empty, empty) == 0.0


def test_mape_basic():
    true = np.array([100.0, 200.0])
    pred = np.array([110.0, 180.0])  # 10% off, 10% off
    assert mape(true, pred) == pytest.approx(10.0)


def test_mape_zero_division_guarded():
    true = np.array([0.0, 100.0])
    pred = np.array([5.0, 110.0])
    # Should not raise and should produce a finite number.
    val = mape(true, pred)
    assert np.isfinite(val)


def test_rmse_zero_when_perfect():
    true = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    assert rmse(true, pred) == 0.0


def test_per_stock_metrics_shape_and_columns():
    prev = np.array([[100.0, 50.0], [101.0, 49.0]])
    true = np.array([[101.0, 49.0], [102.0, 48.0]])
    pred = np.array([[101.5, 48.5], [101.0, 48.2]])

    df = per_stock_metrics(prev, true, pred, tickers=["AAA", "BBB"])
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["AAA", "BBB"]
    assert set(df.columns) == {"RMSE", "MAPE", "DirectionalAccuracy"}


def test_per_stock_metrics_ticker_count_mismatch_raises():
    arr = np.zeros((2, 2))
    with pytest.raises(ValueError):
        per_stock_metrics(arr, arr, arr, tickers=["only_one"])
