"""Tests for prepare_multifeature_array + create_sequences_xy (Phase 2A)."""
import numpy as np
import pandas as pd
import pytest

from data.features import calculate_all_features
from data.preprocessing import (
    FEATURES_PER_STOCK,
    prepare_multifeature_array,
    create_sequences_xy,
)


def _toy_ticker_df(seed=0, n=120):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    log_p = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)
    close = np.exp(log_p)
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    df = pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return calculate_all_features(
        df, rsi_window=14, momentum_window=10, ma_window=20, bb_window=20, bb_std=2
    )


def test_prepare_multifeature_array_shape_and_columns():
    data = {"AAPL": _toy_ticker_df(0), "MSFT": _toy_ticker_df(1)}
    features, targets, names = prepare_multifeature_array(
        data, ["AAPL", "MSFT"], "2020-01-01", "2020-12-31"
    )
    # 2 stocks × 9 features = 18 feature columns
    assert features.shape[1] == 2 * len(FEATURES_PER_STOCK)
    assert list(targets.columns) == ["AAPL", "MSFT"]
    assert features.shape[0] == targets.shape[0]
    # Column ordering: all features of stock 0, then all features of stock 1.
    assert names[0] == "AAPL_log_return"
    assert names[len(FEATURES_PER_STOCK)] == "MSFT_log_return"


def test_prepare_multifeature_array_drops_warmup_rows():
    data = {"AAPL": _toy_ticker_df(0)}
    features, targets, _ = prepare_multifeature_array(
        data, ["AAPL"], "2020-01-01", "2020-12-31"
    )
    # No NaNs anywhere — warmup rows were dropped.
    assert not features.isna().any().any()
    assert not targets.isna().any().any()


def test_prepare_multifeature_array_target_equals_log_return_column():
    """Targets should match the log_return feature column exactly."""
    data = {"AAPL": _toy_ticker_df(0)}
    features, targets, _ = prepare_multifeature_array(
        data, ["AAPL"], "2020-01-01", "2020-12-31"
    )
    np.testing.assert_allclose(
        targets["AAPL"].values,
        features["AAPL_log_return"].values,
    )


def test_prepare_multifeature_array_no_valid_tickers_raises():
    with pytest.raises(ValueError):
        prepare_multifeature_array({}, ["AAPL"], "2020-01-01", "2020-12-31")


def test_peer_return_5d_with_two_tickers():
    """With 2 tickers, peer_return_5d[A] must equal the 5-day log-return of B."""
    data = {"AAPL": _toy_ticker_df(0), "MSFT": _toy_ticker_df(1)}
    features, _, _ = prepare_multifeature_array(
        data, ["AAPL", "MSFT"], "2020-01-01", "2020-12-31"
    )
    # Manually compute MSFT's 5-day return aligned to features.index
    msft_close = data["MSFT"]["Close"]
    expected_msft_5d = np.log(msft_close / msft_close.shift(5)).reindex(features.index)
    np.testing.assert_allclose(
        features["AAPL_peer_return_5d"].values,
        expected_msft_5d.values,
        rtol=1e-5,
        err_msg="peer_return_5d[AAPL] should equal 5d return of MSFT",
    )
    # Symmetrically, peer of MSFT should equal AAPL's 5d return
    aapl_close = data["AAPL"]["Close"]
    expected_aapl_5d = np.log(aapl_close / aapl_close.shift(5)).reindex(features.index)
    np.testing.assert_allclose(
        features["MSFT_peer_return_5d"].values,
        expected_aapl_5d.values,
        rtol=1e-5,
        err_msg="peer_return_5d[MSFT] should equal 5d return of AAPL",
    )


def test_peer_return_5d_single_ticker_is_zero():
    """Single-ticker case: peer feature defaults to 0 (no peers)."""
    data = {"AAPL": _toy_ticker_df(0)}
    features, _, _ = prepare_multifeature_array(
        data, ["AAPL"], "2020-01-01", "2020-12-31"
    )
    assert (features["AAPL_peer_return_5d"] == 0.0).all()


def test_create_sequences_xy_shapes_and_indexing():
    X = np.arange(60, dtype=float).reshape(20, 3)  # 20 rows, 3 features
    y = np.arange(20, dtype=float).reshape(20, 1)   # 20 rows, 1 target
    X_seq, y_seq = create_sequences_xy(X, y, seq_length=5)
    assert X_seq.shape == (15, 5, 3)
    assert y_seq.shape == (15, 1)
    # Sample 0: X is rows [0..4], target is row 5.
    np.testing.assert_array_equal(X_seq[0], X[0:5])
    np.testing.assert_array_equal(y_seq[0], y[5])


def test_create_sequences_xy_mismatched_rows_raises():
    with pytest.raises(ValueError):
        create_sequences_xy(np.zeros((10, 3)), np.zeros((9, 1)), seq_length=3)


def test_create_sequences_xy_invalid_seq_length_raises():
    with pytest.raises(ValueError):
        create_sequences_xy(np.zeros((10, 3)), np.zeros((10, 1)), seq_length=20)
