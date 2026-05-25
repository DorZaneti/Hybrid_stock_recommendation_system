"""
Data preprocessing utilities for LSTM training.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.logger import get_logger

logger = get_logger(__name__)


def prepare_stock_dataframe(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Prepare a DataFrame with Close prices for selected stocks.

    Args:
        data: Dictionary mapping tickers to DataFrames
        tickers: List of ticker symbols to include
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        DataFrame with tickers as columns and dates as index

    Raises:
        ValueError: If no valid data available after filtering

    Example:
        >>> df = prepare_stock_dataframe(data, ['AAPL', 'MSFT'], '2020-01-01', '2023-01-01')
    """
    logger.info(f"Preparing DataFrame for {len(tickers)} stocks from {start_date} to {end_date}")

    processed_data = {}
    skipped = []

    for ticker in tickers:
        try:
            if ticker not in data:
                logger.warning(f"{ticker}: Not found in data dictionary")
                skipped.append(ticker)
                continue

            # Extract Close prices within date range
            ticker_df = data[ticker]

            if 'Close' not in ticker_df.columns:
                logger.warning(f"{ticker}: 'Close' column not found")
                skipped.append(ticker)
                continue

            close_prices = ticker_df['Close'].loc[start_date:end_date]

            if close_prices.empty:
                logger.warning(f"{ticker}: No data in date range")
                skipped.append(ticker)
                continue

            processed_data[ticker] = close_prices

        except Exception as e:
            logger.error(f"{ticker}: Error during preparation - {str(e)}")
            skipped.append(ticker)

    if not processed_data:
        raise ValueError("No valid stock data available after filtering")

    # Align all series to common dates
    common_index = None
    for series in processed_data.values():
        if common_index is None:
            common_index = series.index
        else:
            common_index = common_index.intersection(series.index)

    if common_index.empty:
        raise ValueError("No common dates found across selected stocks")

    aligned_data = {ticker: series.reindex(common_index) for ticker, series in processed_data.items()}
    df = pd.DataFrame(aligned_data)

    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    if skipped:
        logger.warning(f"Skipped tickers: {', '.join(skipped)}")

    return df


def scale_data(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scale data to [0, 1] range using MinMaxScaler.

    Args:
        df: DataFrame to scale

    Returns:
        Tuple of (scaled_data, scaler) for inverse transformation

    Example:
        >>> scaled, scaler = scale_data(df)
        >>> original = scaler.inverse_transform(scaled)
    """
    logger.info(f"Scaling data with shape {df.shape}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.dropna())

    logger.debug(f"Data scaled from range [{df.min().min():.2f}, {df.max().max():.2f}] to [0, 1]")
    return scaled_data, scaler


FEATURES_PER_STOCK = [
    "log_return",      # log(Close_t / Close_{t-1})            — 1-day price momentum
    "RSI_norm",        # RSI / 100                             — overbought/oversold signal
    "BB_pct",          # (Close - Lower) / (Upper - Lower)     — Bollinger band position
    "momentum_z",      # Momentum / rolling_std(Momentum)      — z-scored price momentum
    "MA_ratio",        # Close / MA - 1                        — price vs trend level
    "MACD_norm",       # MACD histogram / Close                — trend divergence signal
    "log_vol_ratio",   # log(Vol / rolling_mean(Vol))          — volume anomaly
    "ATR_norm",        # ATR(14) / Close                       — normalised intraday range
    "HL_pct",          # (High - Low) / Close                  — daily range %-of-price
    "peer_return_5d",  # mean log(Close_t/Close_{t-5}) of peer stocks  — sector momentum
]

# Shared macro (market-regime) features appended ONCE at the end of the feature
# matrix — same values for all stocks on a given day.  The multi-stock LSTM
# (input_size = n_stocks × len(FEATURES_PER_STOCK) + len(MACRO_FEATURES)) sees
# these as regime context; the single-stock predictor ignores them (it slices
# only the per-stock blocks).
MACRO_FEATURES = [
    "VIX_norm",       # VIX / 100 — absolute fear level
    "VIX_5d_change",  # log(VIX_t / VIX_{t-5}) — regime-shift signal
    "SP500_return",   # log(SP500_t / SP500_{t-1}) — broad market direction
]


def prepare_multifeature_array(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    start_date: str,
    end_date: str,
    momentum_z_window: int = 20,
    macro_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Build per-stock feature columns + per-stock target column for multi-feature training.

    For each ticker the following 5 stationary/bounded features are produced:
      - ``log_return``  log(Close_t / Close_{t-1})
      - ``RSI_norm``    RSI / 100   (~0-1)
      - ``BB_pct``      (Close - Lower) / (Upper - Lower)  (~0-1)
      - ``momentum_z``  Momentum / rolling_std(Momentum, ``momentum_z_window``)
      - ``MA_ratio``    Close / Moving_Average - 1

    The target column for each ticker is its log-return of Close (same as
    ``log_return`` above, but kept in its own DataFrame so callers can scale
    inputs and targets independently and read targets without slicing).

    Inputs must already have ``Close, RSI, Momentum, Moving_Average,
    Bollinger_Upper, Bollinger_Lower`` columns — i.e. the output of
    ``data.features.calculate_all_features``.

    Returns
    -------
    features_df : DataFrame, shape (T, n_stocks * n_features)
        Columns ordered ``[TICKER_feat for ticker in tickers for feat in FEATURES_PER_STOCK]``.
    targets_df  : DataFrame, shape (T, n_stocks)
        Columns are ``tickers`` in order; values are log-returns of Close.
    feature_names : list[str]
        Same as ``features_df.columns`` for convenience.

    Rows with NaN in any feature or target are dropped (the indicator warmups
    plus the first log-return are the usual sources).
    """
    n_macro = macro_df.shape[1] if macro_df is not None else 0
    logger.info(
        f"Preparing multi-feature array for {len(tickers)} tickers, "
        f"features per stock: {FEATURES_PER_STOCK}, macro features: {n_macro}"
    )

    required = [
        "Close", "RSI", "Momentum", "Moving_Average",
        "Bollinger_Upper", "Bollinger_Lower",
        "MACD_norm",                      # Close-only, always present
        "High", "Low", "Volume",          # needed for ATR_norm, HL_pct, log_vol_ratio
        "ATR_norm", "HL_pct", "log_vol_ratio",
    ]
    feature_blocks: Dict[str, pd.DataFrame] = {}
    target_series: Dict[str, pd.Series] = {}
    skipped: List[str] = []

    for ticker in tickers:
        if ticker not in data:
            logger.warning(f"{ticker}: not in data dict; skipping")
            skipped.append(ticker)
            continue
        df = data[ticker]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"{ticker}: missing columns {missing}; skipping")
            skipped.append(ticker)
            continue

        sub = df.loc[start_date:end_date, required].copy()
        if sub.empty:
            logger.warning(f"{ticker}: no rows in date range; skipping")
            skipped.append(ticker)
            continue

        log_return = np.log(sub["Close"] / sub["Close"].shift(1))
        rsi_norm = sub["RSI"] / 100.0
        band_width = (sub["Bollinger_Upper"] - sub["Bollinger_Lower"]).replace(0.0, np.nan)
        bb_pct = (sub["Close"] - sub["Bollinger_Lower"]) / band_width
        momentum = sub["Momentum"]
        mom_std = momentum.rolling(window=momentum_z_window, min_periods=momentum_z_window).std()
        momentum_z = momentum / mom_std.replace(0.0, np.nan)
        ma_ratio = sub["Close"] / sub["Moving_Average"].replace(0.0, np.nan) - 1.0

        # New OHLCV-derived features (Part 7).
        macd_norm = sub["MACD_norm"]
        log_vol_ratio = sub["log_vol_ratio"]
        atr_norm = sub["ATR_norm"]
        hl_pct = sub["HL_pct"]

        block = pd.DataFrame(
            {
                f"{ticker}_log_return": log_return,
                f"{ticker}_RSI_norm": rsi_norm,
                f"{ticker}_BB_pct": bb_pct,
                f"{ticker}_momentum_z": momentum_z,
                f"{ticker}_MA_ratio": ma_ratio,
                f"{ticker}_MACD_norm": macd_norm,
                f"{ticker}_log_vol_ratio": log_vol_ratio,
                f"{ticker}_ATR_norm": atr_norm,
                f"{ticker}_HL_pct": hl_pct,
            }
        )
        feature_blocks[ticker] = block
        target_series[ticker] = log_return.rename(ticker)

    if not feature_blocks:
        raise ValueError("No valid tickers after filtering")

    # Align on the intersection of indices so every (ticker, feature) covers the same dates.
    common_idx = None
    for block in feature_blocks.values():
        common_idx = block.index if common_idx is None else common_idx.intersection(block.index)
    if common_idx is None or common_idx.empty:
        raise ValueError("No common dates across tickers")

    # ── Cross-stock peer momentum (peer_return_5d) ─────────────────────────
    # For each stock j: mean log(Close_t / Close_{t-5}) of all OTHER stocks k ≠ j.
    # Captures sector/peer-group momentum visible at decision time t (no leakage).
    # With only 1 stock available the feature is set to 0 (no peers → no signal).
    if len(feature_blocks) > 1:
        closes_on_common = {
            t: data[t]["Close"].reindex(common_idx)
            for t in feature_blocks
        }
        returns_5d_df = pd.DataFrame(
            {t: np.log(c / c.shift(5)) for t, c in closes_on_common.items()},
            index=common_idx,
        )
        total_5d = returns_5d_df.sum(axis=1)
        n_f = len(feature_blocks)
        for t in feature_blocks:
            peer_mean = (total_5d - returns_5d_df[t]) / (n_f - 1)
            feature_blocks[t][f"{t}_peer_return_5d"] = peer_mean
    else:
        # Single-stock edge case: peer feature carries no information → fill with 0.
        for t in feature_blocks:
            feature_blocks[t][f"{t}_peer_return_5d"] = 0.0

    features_df = pd.concat(
        [feature_blocks[t].reindex(common_idx) for t in feature_blocks.keys()],
        axis=1,
    )
    targets_df = pd.concat(
        [target_series[t].reindex(common_idx) for t in target_series.keys()],
        axis=1,
    )

    # ── Append shared macro / regime features (appended ONCE at the right) ──
    if macro_df is not None and not macro_df.empty:
        macro_aligned = macro_df.reindex(common_idx)
        available_macro = [c for c in macro_df.columns if c in macro_aligned.columns]
        if available_macro:
            features_df = pd.concat(
                [features_df, macro_aligned[available_macro]], axis=1
            )
            logger.info(
                f"Appended {len(available_macro)} macro features: {available_macro}"
            )
        else:
            logger.warning("macro_df had no columns that aligned to the stock index")

    # Drop any row with a NaN in either array (warmup periods).
    valid = features_df.notna().all(axis=1) & targets_df.notna().all(axis=1)
    features_df = features_df.loc[valid]
    targets_df = targets_df.loc[valid]

    if features_df.empty:
        raise ValueError("All rows were dropped during NaN cleanup; check feature windows")

    logger.info(
        f"Multi-feature: features {features_df.shape}, targets {targets_df.shape}, "
        f"dropped {len(common_idx) - len(features_df)} warmup rows"
    )
    if skipped:
        logger.warning(f"Skipped tickers: {', '.join(skipped)}")

    return features_df, targets_df, list(features_df.columns)


def create_sequences_xy(
    X_data: np.ndarray,
    y_data: np.ndarray,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-feature sequence builder: X and y can have different column counts.

    ``X_data`` has shape (T, n_input_features) and ``y_data`` has shape
    (T, n_output_features); both must share the same number of rows.

    Returns ``(X_seq, y_seq)`` where ``X_seq[i] = X_data[i:i+seq_length]`` and
    ``y_seq[i] = y_data[i+seq_length]``.
    """
    if len(X_data) != len(y_data):
        raise ValueError(f"X rows {len(X_data)} != y rows {len(y_data)}")
    if seq_length <= 0 or seq_length >= len(X_data):
        raise ValueError(f"seq_length={seq_length} invalid for {len(X_data)} rows")

    xs, ys = [], []
    for i in range(len(X_data) - seq_length):
        xs.append(X_data[i:i + seq_length])
        ys.append(y_data[i + seq_length])

    X_seq = np.array(xs)
    y_seq = np.array(ys)
    logger.info(f"create_sequences_xy: X {X_seq.shape}, y {y_seq.shape}")
    return X_seq, y_seq


def fit_scaler_on_train(
    full_data: np.ndarray,
    train_rows: int,
    scaler_type: str = "standard",
) -> Tuple[np.ndarray, object]:
    """
    Fit a scaler on the first `train_rows` rows only, then transform the whole array.

    Use this instead of `scale_data` when working with a time-ordered series — it
    avoids the train/test leakage of fitting on the full dataset.

    Args:
        full_data: 2D array of shape (T, n_features) — includes train + val + oos.
        train_rows: number of leading rows to fit the scaler on.
        scaler_type: 'standard' (default, for log-returns) or 'minmax'.

    Returns:
        (scaled_full, scaler)
    """
    if train_rows <= 0 or train_rows > len(full_data):
        raise ValueError(
            f"train_rows={train_rows} must be in (0, {len(full_data)}]"
        )

    scaler_cls = {"standard": StandardScaler, "minmax": MinMaxScaler}[scaler_type]
    scaler = scaler_cls()
    scaler.fit(full_data[:train_rows])
    scaled_full = scaler.transform(full_data)
    logger.info(
        f"Fit {scaler_type} scaler on {train_rows} train rows; "
        f"transformed {len(full_data)} total rows"
    )
    return scaled_full, scaler


def split_train_val(
    train_rows: int,
    val_days: int,
) -> Tuple[int, int]:
    """
    Compute the row-index split point between train and val for a time-ordered series.

    Returns (n_train_rows, n_val_rows) where n_train_rows + n_val_rows == train_rows.
    The val slice is the LAST `val_days` rows of the train range.

    Raises ValueError if val_days >= train_rows.
    """
    if val_days <= 0:
        return train_rows, 0
    if val_days >= train_rows:
        raise ValueError(
            f"val_days={val_days} must be < train_rows={train_rows}"
        )
    n_train = train_rows - val_days
    return n_train, val_days


def create_sequences(
    data: np.ndarray,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.

    Transforms 2D data into 3D sequences where each sequence contains
    seq_length time steps, and the target is the next time step.

    Args:
        data: 2D array of shape (num_samples, num_features)
        seq_length: Number of time steps in each sequence

    Returns:
        Tuple of (X, y) where:
        - X: 3D array of shape (num_sequences, seq_length, num_features)
        - y: 2D array of shape (num_sequences, num_features)

    Example:
        >>> data = np.random.rand(100, 5)
        >>> X, y = create_sequences(data, seq_length=8)
        >>> print(X.shape, y.shape)
        (92, 8, 5) (92, 5)
    """
    xs, ys = [], []

    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    X = np.array(xs)
    y = np.array(ys)

    logger.info(f"Created {len(X)} sequences with shape {X.shape}")
    return X, y


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split sequences into train and test sets.

    Args:
        X: Input sequences
        y: Target values
        test_size: Number of samples to use for testing

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as PyTorch tensors

    Example:
        >>> X_train, y_train, X_test, y_test = split_train_test(X, y, test_size=5)
    """
    X_train = torch.tensor(X[:-test_size], dtype=torch.float32)
    y_train = torch.tensor(y[:-test_size], dtype=torch.float32)
    X_test = torch.tensor(X[-test_size:], dtype=torch.float32)
    y_test = torch.tensor(y[-test_size:], dtype=torch.float32)

    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    return X_train, y_train, X_test, y_test


def split_for_transfer_learning(
    X: np.ndarray,
    y: np.ndarray,
    test_size: int,
    finetune_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data for transfer learning: pretrain, finetune, and test sets.

    Args:
        X: Input sequences
        y: Target values
        test_size: Number of samples for testing
        finetune_size: Number of samples for fine-tuning

    Returns:
        Tuple of (X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test)

    Example:
        >>> X_pre, y_pre, X_ft, y_ft, X_test, y_test = split_for_transfer_learning(
        ...     X, y, test_size=5, finetune_size=15
        ... )
    """
    total_train = len(X) - test_size
    pretrain_size = total_train - finetune_size

    X_pretrain = torch.tensor(X[:pretrain_size], dtype=torch.float32)
    y_pretrain = torch.tensor(y[:pretrain_size], dtype=torch.float32)

    X_finetune = torch.tensor(X[pretrain_size:-test_size], dtype=torch.float32)
    y_finetune = torch.tensor(y[pretrain_size:-test_size], dtype=torch.float32)

    X_test = torch.tensor(X[-test_size:], dtype=torch.float32)
    y_test = torch.tensor(y[-test_size:], dtype=torch.float32)

    logger.info(
        f"Transfer learning split - "
        f"Pretrain: {X_pretrain.shape[0]}, "
        f"Finetune: {X_finetune.shape[0]}, "
        f"Test: {X_test.shape[0]}"
    )

    return X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test
