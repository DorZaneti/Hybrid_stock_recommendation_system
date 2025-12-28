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
    calculate_all_features
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
