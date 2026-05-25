"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from data.features import (
    calculate_rsi,
    calculate_momentum,
    calculate_moving_average,
    calculate_bollinger_bands,
    calculate_all_features,
    calculate_macd,
    calculate_log_volume_ratio,
    calculate_atr,
    calculate_hl_pct,
)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.random.uniform(100, 200, 100)
    df = pd.DataFrame({'Close': prices}, index=dates)
    return df


def test_calculate_rsi(sample_price_data):
    """Test RSI calculation."""
    rsi = calculate_rsi(sample_price_data, window=14)

    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(sample_price_data)
    # RSI should be between 0 and 100 (ignoring NaN values)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_calculate_momentum(sample_price_data):
    """Test momentum calculation."""
    momentum = calculate_momentum(sample_price_data, window=10)

    assert isinstance(momentum, pd.Series)
    assert len(momentum) == len(sample_price_data)


def test_calculate_moving_average(sample_price_data):
    """Test moving average calculation."""
    ma = calculate_moving_average(sample_price_data, window=20)

    assert isinstance(ma, pd.Series)
    assert len(ma) == len(sample_price_data)


def test_calculate_bollinger_bands(sample_price_data):
    """Test Bollinger Bands calculation."""
    upper, lower = calculate_bollinger_bands(sample_price_data, window=20)

    assert isinstance(upper, pd.Series)
    assert isinstance(lower, pd.Series)
    assert len(upper) == len(sample_price_data)
    assert len(lower) == len(sample_price_data)
    # Upper band should be higher than lower band
    valid_data = upper.dropna().index.intersection(lower.dropna().index)
    assert (upper.loc[valid_data] >= lower.loc[valid_data]).all()


def test_calculate_all_features(sample_price_data):
    """Test calculation of all features."""
    df_with_features = calculate_all_features(sample_price_data)

    required_columns = ['Close', 'RSI', 'Momentum', 'Moving_Average', 'Bollinger_Upper', 'Bollinger_Lower']
    for col in required_columns:
        assert col in df_with_features.columns

    assert len(df_with_features) == len(sample_price_data)


def test_calculate_all_features_missing_column():
    """Test that calculate_all_features raises error for missing Close column."""
    df = pd.DataFrame({'Price': [100, 101, 102]})

    with pytest.raises(ValueError):
        calculate_all_features(df)


# ---- OHLCV fixture --------------------------------------------------------

@pytest.fixture
def sample_ohlcv_data():
    """OHLCV fixture for testing new features."""
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    log_p = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)
    close = np.exp(log_p)
    return pd.DataFrame(
        {
            'Open':   close * (1 + rng.uniform(-0.003, 0.003, n)),
            'High':   close * (1 + rng.uniform(0.001, 0.01, n)),
            'Low':    close * (1 - rng.uniform(0.001, 0.01, n)),
            'Close':  close,
            'Volume': rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )


# ---- New feature tests ----------------------------------------------------

def test_calculate_macd_shape_and_no_nan_after_warmup(sample_ohlcv_data):
    """MACD histogram should fill from the start (EWM has no strict NaN warmup)."""
    macd = calculate_macd(sample_ohlcv_data, fast=12, slow=26, signal=9)
    assert isinstance(macd, pd.Series)
    assert len(macd) == len(sample_ohlcv_data)
    # After the slow EWM has enough data (row ~26+), values should be finite.
    assert macd.iloc[30:].notna().all()


def test_calculate_macd_normalised_by_price(sample_ohlcv_data):
    """MACD_norm should be much smaller in magnitude than raw MACD histogram."""
    raw_macd_scale = (
        sample_ohlcv_data['Close'].ewm(span=12, adjust=False).mean()
        - sample_ohlcv_data['Close'].ewm(span=26, adjust=False).mean()
    ).abs().mean()
    macd_norm = calculate_macd(sample_ohlcv_data).abs().mean()
    # Normalised values should be orders of magnitude smaller than raw prices.
    assert macd_norm < raw_macd_scale


def test_calculate_log_volume_ratio_shape(sample_ohlcv_data):
    lvr = calculate_log_volume_ratio(sample_ohlcv_data, window=20)
    assert isinstance(lvr, pd.Series)
    assert len(lvr) == len(sample_ohlcv_data)
    # After warmup, values should be finite; rough mean close to 0.
    valid = lvr.dropna()
    assert len(valid) > 0
    assert abs(valid.mean()) < 1.0  # log-ratio centred near 0


def test_calculate_atr_positive_and_normalised(sample_ohlcv_data):
    atr = calculate_atr(sample_ohlcv_data, window=14)
    assert isinstance(atr, pd.Series)
    valid = atr.dropna()
    assert (valid > 0).all(), "ATR_norm should always be positive"
    assert (valid < 1.0).all(), "ATR_norm / Close should be well below 1.0 for equity"


def test_calculate_hl_pct_matches_formula(sample_ohlcv_data):
    hl_pct = calculate_hl_pct(sample_ohlcv_data)
    expected = (
        (sample_ohlcv_data['High'] - sample_ohlcv_data['Low'])
        / sample_ohlcv_data['Close']
    )
    pd.testing.assert_series_equal(hl_pct, expected, check_names=False)
    assert (hl_pct.dropna() > 0).all(), "HL_pct should be positive"
