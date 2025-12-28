"""
Technical indicator calculation for stock data.
"""
import pandas as pd
from typing import Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


def calculate_rsi(df: pd.DataFrame, window: int = 14, column: str = 'Close') -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Args:
        df: Stock data DataFrame
        window: Lookback period for RSI calculation
        column: Price column to use

    Returns:
        Series containing RSI values (0-100)

    Example:
        >>> df = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
        >>> rsi = calculate_rsi(df, window=14)
    """
    try:
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        # Avoid division by zero
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        logger.debug(f"Calculated RSI with window={window}")
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        raise


def calculate_momentum(df: pd.DataFrame, window: int = 10, column: str = 'Close') -> pd.Series:
    """
    Calculate price momentum.

    Momentum measures the rate of change in price over a specified period.

    Args:
        df: Stock data DataFrame
        window: Lookback period for momentum calculation
        column: Price column to use

    Returns:
        Series containing momentum values

    Example:
        >>> df = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
        >>> momentum = calculate_momentum(df, window=10)
    """
    try:
        momentum = df[column].diff(window)
        logger.debug(f"Calculated momentum with window={window}")
        return momentum
    except Exception as e:
        logger.error(f"Error calculating momentum: {str(e)}")
        raise


def calculate_moving_average(df: pd.DataFrame, window: int = 20, column: str = 'Close') -> pd.Series:
    """
    Calculate simple moving average.

    Args:
        df: Stock data DataFrame
        window: Lookback period for moving average
        column: Price column to use

    Returns:
        Series containing moving average values

    Example:
        >>> df = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
        >>> ma = calculate_moving_average(df, window=20)
    """
    try:
        ma = df[column].rolling(window=window).mean()
        logger.debug(f"Calculated moving average with window={window}")
        return ma
    except Exception as e:
        logger.error(f"Error calculating moving average: {str(e)}")
        raise


def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    std_multiplier: int = 2,
    column: str = 'Close'
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of a moving average and two standard deviation bands
    above and below it, used to measure price volatility.

    Args:
        df: Stock data DataFrame
        window: Lookback period for calculations
        std_multiplier: Number of standard deviations for bands
        column: Price column to use

    Returns:
        Tuple of (upper_band, lower_band) Series

    Example:
        >>> df = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
        >>> upper, lower = calculate_bollinger_bands(df, window=20)
    """
    try:
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std = df[column].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_multiplier)
        lower_band = rolling_mean - (rolling_std * std_multiplier)

        logger.debug(f"Calculated Bollinger Bands with window={window}, std={std_multiplier}")
        return upper_band, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        raise


def calculate_all_features(
    df: pd.DataFrame,
    rsi_window: int = 14,
    momentum_window: int = 10,
    ma_window: int = 20,
    bb_window: int = 20,
    bb_std: int = 2
) -> pd.DataFrame:
    """
    Calculate all technical indicators for a stock.

    Args:
        df: Stock data DataFrame
        rsi_window: Window for RSI calculation
        momentum_window: Window for momentum calculation
        ma_window: Window for moving average
        bb_window: Window for Bollinger Bands
        bb_std: Standard deviation multiplier for Bollinger Bands

    Returns:
        DataFrame with all features added

    Raises:
        ValueError: If required columns are missing
        Exception: If feature calculation fails

    Example:
        >>> df = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
        >>> df_with_features = calculate_all_features(df)
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    try:
        df_copy = df.copy()
        df_copy['RSI'] = calculate_rsi(df_copy, window=rsi_window)
        df_copy['Momentum'] = calculate_momentum(df_copy, window=momentum_window)
        df_copy['Moving_Average'] = calculate_moving_average(df_copy, window=ma_window)
        df_copy['Bollinger_Upper'], df_copy['Bollinger_Lower'] = calculate_bollinger_bands(
            df_copy, window=bb_window, std_multiplier=bb_std
        )

        logger.info(f"Calculated all features for {len(df_copy)} data points")
        return df_copy
    except Exception as e:
        logger.error(f"Error calculating features: {str(e)}")
        raise
