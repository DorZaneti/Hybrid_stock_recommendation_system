"""
Data preprocessing utilities for LSTM training.
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
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
