"""Unit tests for data.returns."""
import numpy as np
import pandas as pd
import pytest

from data.returns import (
    prices_to_log_returns,
    returns_to_prices,
    compute_forward_returns,
    compute_past_returns,
)


def _sample_prices():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = {
        "AAA": np.linspace(100, 120, 20),
        "BBB": np.linspace(200, 180, 20),
    }
    return pd.DataFrame(data, index=dates)


def test_roundtrip_returns_to_prices():
    prices = _sample_prices()
    returns = prices_to_log_returns(prices)
    # Reconstruct: anchors are the previous day's actual prices.
    anchors = prices.shift(1).loc[returns.index].values
    reconstructed = returns_to_prices(returns, anchors)
    np.testing.assert_allclose(reconstructed, prices.loc[returns.index].values, atol=1e-9)


def test_constant_prices_yield_zero_returns():
    prices = pd.DataFrame({"AAA": [100.0] * 10})
    returns = prices_to_log_returns(prices)
    assert (returns["AAA"].abs() < 1e-12).all()


def test_returns_to_prices_shape_mismatch_raises():
    with pytest.raises(ValueError):
        returns_to_prices(np.zeros((3, 2)), np.zeros((3, 3)))


def test_forward_returns_h1_matches_legacy():
    """At horizon=1, forward returns must match the standard 1-day log-return."""
    prices = _sample_prices()
    legacy = prices_to_log_returns(prices)
    forward1 = compute_forward_returns(prices, horizon=1)
    np.testing.assert_allclose(
        forward1.values, legacy.loc[forward1.index].values, atol=1e-12
    )


def test_forward_returns_h5_value_check():
    """y_data[t] should equal log(p[t-1+h] / p[t-1]) for the chosen t."""
    prices = _sample_prices()
    fw = compute_forward_returns(prices, horizon=5)
    # Pick a row in the middle of the frame.
    t_idx = 10
    t = fw.index[t_idx]
    pos_in_prices = prices.index.get_loc(t)
    expected = np.log(prices.iloc[pos_in_prices - 1 + 5] / prices.iloc[pos_in_prices - 1])
    np.testing.assert_allclose(fw.iloc[t_idx].values, expected.values, atol=1e-12)


def test_forward_returns_drops_horizon_rows():
    """h rows must be dropped: 1 from prev-day anchor, h-1 from forward shift."""
    prices = _sample_prices()  # 20 rows
    fw = compute_forward_returns(prices, horizon=5)
    assert len(fw) == len(prices) - 5


def test_past_returns_value_check():
    """past[t] should equal log(p[t-1] / p[t-1-h])."""
    prices = _sample_prices()
    pr = compute_past_returns(prices, horizon=5)
    t_idx = 3
    t = pr.index[t_idx]
    pos = prices.index.get_loc(t)
    expected = np.log(prices.iloc[pos - 1] / prices.iloc[pos - 1 - 5])
    np.testing.assert_allclose(pr.iloc[t_idx].values, expected.values, atol=1e-12)


def test_horizon_validation():
    with pytest.raises(ValueError):
        compute_forward_returns(_sample_prices(), horizon=0)
    with pytest.raises(ValueError):
        compute_past_returns(_sample_prices(), horizon=-1)
