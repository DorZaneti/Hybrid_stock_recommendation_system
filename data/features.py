"""
Technical indicator calculation for stock data.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
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


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = 'Close'
) -> pd.Series:
    """
    MACD histogram normalised by price.

    Returns (EMA_fast - EMA_slow - EMA_signal(EMA_fast - EMA_slow)) / Close.
    Normalising by Close makes the value comparable across stocks at different
    price levels. No NaN warmup beyond what EWM produces (EWM starts immediately).
    """
    try:
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        result = histogram / df[column].replace(0, 1e-10)
        logger.debug(f"Calculated MACD ({fast}/{slow}/{signal}) normalised by {column}")
        return result
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        raise


def calculate_log_volume_ratio(
    df: pd.DataFrame,
    window: int = 20,
    column: str = 'Volume'
) -> pd.Series:
    """
    Log-volume ratio: log(Volume / rolling_mean(Volume, window)).

    Stationary (roughly zero-mean) representation of volume anomalies.
    High positive values = unusually high volume (potential breakout/reversal).
    """
    try:
        vol_mean = df[column].rolling(window=window, min_periods=window).mean()
        result = np.log(df[column] / vol_mean.replace(0, np.nan))
        logger.debug(f"Calculated log-volume ratio with window={window}")
        return result
    except Exception as e:
        logger.error(f"Error calculating log-volume ratio: {str(e)}")
        raise


def calculate_atr(
    df: pd.DataFrame,
    window: int = 14
) -> pd.Series:
    """
    Average True Range normalised by Close (ATR_norm = ATR / Close).

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|).
    Normalising removes price-level effects so values are comparable across stocks.
    Requires columns: 'High', 'Low', 'Close'.
    """
    try:
        high_low = df['High'] - df['Low']
        high_prev = (df['High'] - df['Close'].shift(1)).abs()
        low_prev = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=window).mean()
        result = atr / df['Close'].replace(0, np.nan)
        logger.debug(f"Calculated ATR_norm with window={window}")
        return result
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        raise


def calculate_hl_pct(df: pd.DataFrame) -> pd.Series:
    """
    High-Low percentage: (High - Low) / Close.

    Measures intraday range relative to price. A proxy for daily uncertainty /
    market indecision. Requires columns: 'High', 'Low', 'Close'.
    """
    try:
        result = (df['High'] - df['Low']) / df['Close'].replace(0, np.nan)
        logger.debug("Calculated HL_pct")
        return result
    except Exception as e:
        logger.error(f"Error calculating HL_pct: {str(e)}")
        raise


def calculate_all_features(
    df: pd.DataFrame,
    rsi_window: int = 14,
    momentum_window: int = 10,
    ma_window: int = 20,
    bb_window: int = 20,
    bb_std: int = 2,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_window: int = 14,
    volume_window: int = 20,
) -> pd.DataFrame:
    """
    Calculate all technical indicators for a stock.

    Close-only indicators (always computed):
        RSI, Momentum, Moving_Average, Bollinger_Upper/Lower, MACD_norm.

    OHLCV indicators (computed only when High, Low, Volume columns are present):
        ATR_norm, HL_pct, log_vol_ratio.

    Args:
        df: Stock data DataFrame (at minimum must have 'Close').
        rsi_window, momentum_window, ma_window, bb_window, bb_std: legacy params.
        macd_fast, macd_slow, macd_signal: MACD EMA periods.
        atr_window: ATR rolling window.
        volume_window: Rolling window for log-volume ratio.

    Returns:
        DataFrame with all computable features added.
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
        df_copy['MACD_norm'] = calculate_macd(
            df_copy, fast=macd_fast, slow=macd_slow, signal=macd_signal
        )

        # OHLCV-dependent features: skip gracefully if columns not present.
        ohlcv_cols = {'High', 'Low', 'Volume'}
        has_ohlcv = ohlcv_cols.issubset(df_copy.columns)
        if has_ohlcv:
            df_copy['ATR_norm'] = calculate_atr(df_copy, window=atr_window)
            df_copy['HL_pct'] = calculate_hl_pct(df_copy)
            df_copy['log_vol_ratio'] = calculate_log_volume_ratio(
                df_copy, window=volume_window
            )
        else:
            logger.debug(
                f"Skipping ATR_norm, HL_pct, log_vol_ratio — "
                f"missing columns: {ohlcv_cols - set(df_copy.columns)}"
            )

        logger.info(
            f"Calculated all features for {len(df_copy)} data points "
            f"(OHLCV features: {'yes' if has_ohlcv else 'no'})"
        )
        return df_copy
    except Exception as e:
        logger.error(f"Error calculating features: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# Market-regime (macro) feature computation
# ---------------------------------------------------------------------------

def calculate_macro_features(
    macro_data: Dict[str, pd.DataFrame],
    vix_key: str = "^VIX",
    sp500_key: str = "^GSPC",
    vix_change_window: int = 5,
) -> pd.DataFrame:
    """
    Build shared market-regime features from VIX and S&P 500 daily data.

    Three stationary features are produced:
    - ``VIX_norm``       VIX / 100 — absolute fear level (typical range 0.10–0.80)
    - ``VIX_5d_change``  log(VIX_t / VIX_{t-5}) — 5-day momentum in implied vol;
                          positive = fear is rising (regime shift signal)
    - ``SP500_return``   log(SP500_t / SP500_{t-1}) — daily S&P 500 log-return
                          (broad market direction context)

    All three columns are returned as a date-indexed DataFrame.  Rows with NaN
    (e.g. the first ``vix_change_window`` rows) are kept — callers should apply
    ``dropna()`` after aligning to the stock-feature index.

    Args:
        macro_data:        Dict mapping tickers to DataFrames with a 'Close' col.
        vix_key:           Key for the VIX series in ``macro_data``.
        sp500_key:         Key for the S&P 500 series in ``macro_data``.
        vix_change_window: Look-back window for VIX momentum (default 5 days).

    Returns:
        DataFrame with columns subset of [VIX_norm, VIX_5d_change, SP500_return]
        — only columns whose source data was present are included.

    Raises:
        ValueError: If neither VIX nor S&P 500 data is available.

    Example:
        >>> macro_data = download_macro_data(['^VIX', '^GSPC'], '2020-01-01', '2025-01-01')
        >>> macro_df = calculate_macro_features(macro_data)
        >>> macro_df.columns.tolist()
        ['VIX_norm', 'VIX_5d_change', 'SP500_return']
    """
    parts: Dict[str, pd.Series] = {}

    if vix_key in macro_data:
        vix = macro_data[vix_key]["Close"].copy()
        parts["VIX_norm"] = vix / 100.0
        parts["VIX_5d_change"] = np.log(vix / vix.shift(vix_change_window))
        logger.info(
            f"VIX features computed from {vix_key}: {len(vix)} rows, "
            f"range [{vix.min():.1f}, {vix.max():.1f}]"
        )
    else:
        logger.warning(f"VIX key '{vix_key}' not in macro_data — VIX features skipped")

    if sp500_key in macro_data:
        sp500 = macro_data[sp500_key]["Close"].copy()
        parts["SP500_return"] = np.log(sp500 / sp500.shift(1))
        logger.info(f"SP500_return computed from {sp500_key}: {len(sp500)} rows")
    else:
        logger.warning(f"S&P 500 key '{sp500_key}' not in macro_data — SP500_return skipped")

    # Any additional tickers (e.g. sector ETFs like XLK) → daily log-return.
    # Feature name: ticker with ^ and - stripped, suffixed with _return.
    known_keys = {vix_key, sp500_key}
    for ticker, df_t in macro_data.items():
        if ticker in known_keys:
            continue
        if "Close" not in df_t.columns:
            logger.warning(f"Macro ticker '{ticker}' has no Close column — skipped")
            continue
        close = df_t["Close"].copy()
        feat_name = ticker.lstrip("^").replace("-", "_") + "_return"
        parts[feat_name] = np.log(close / close.shift(1))
        logger.info(f"{feat_name} computed from {ticker}: {len(close)} rows")

    if not parts:
        raise ValueError(
            f"No macro features could be computed — "
            f"neither {vix_key} nor {sp500_key} was found in macro_data"
        )

    macro_df = pd.DataFrame(parts)
    logger.info(
        f"calculate_macro_features: shape {macro_df.shape}, "
        f"NaN rows = {macro_df.isna().any(axis=1).sum()}"
    )
    return macro_df
