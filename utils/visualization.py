"""
Visualization utilities for stock predictions.
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def plot_predictions(
    df: pd.DataFrame,
    ticker: str,
    predictions_dict: Dict[str, np.ndarray],
    ticker_index: int,
    num_test_days: int = 5,
    figsize: Tuple[int, int] = (14, 7),
    y_limit_bottom: Optional[float] = None
) -> None:
    """
    Plot actual vs predicted prices for a single stock.

    Args:
        df: DataFrame with actual prices
        ticker: Ticker symbol to plot
        predictions_dict: Dictionary mapping model names to prediction arrays
        ticker_index: Index of ticker in prediction arrays
        num_test_days: Number of test days to plot
        figsize: Figure size (width, height)
        y_limit_bottom: Minimum y-axis value (optional)

    Example:
        >>> predictions = {
        ...     'Model 0': predicted_prices_0,
        ...     'Model 0f': predicted_prices_0f
        ... }
        >>> plot_predictions(df, 'AAPL', predictions, ticker_index=0)
    """
    if ticker not in df.columns:
        logger.warning(f"Ticker {ticker} not found in DataFrame")
        return

    plt.figure(figsize=figsize)

    # Plot actual prices
    actual_prices = df[ticker][-num_test_days:]
    plt.plot(df.index[-num_test_days:], actual_prices, label=f'Actual {ticker} Prices', linewidth=2, marker='o')

    # Plot predictions from each model
    for model_name, predictions in predictions_dict.items():
        try:
            predicted_values = predictions[:, ticker_index]
            plt.plot(
                df.index[-num_test_days:],
                predicted_values,
                label=f'Predicted {ticker} - {model_name}',
                linestyle='--',
                marker='x'
            )
        except IndexError:
            logger.warning(f"Could not plot predictions for {model_name} - index out of range")

    plt.title(f'{ticker} Stock Price Predictions', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    if y_limit_bottom is not None:
        plt.ylim(bottom=y_limit_bottom)

    plt.tight_layout()
    plt.show()

    logger.info(f"Plotted predictions for {ticker}")


def plot_multiple_stocks(
    df: pd.DataFrame,
    tickers: List[str],
    predictions_dict: Dict[str, np.ndarray],
    num_test_days: int = 5,
    figsize: Tuple[int, int] = (14, 7),
    y_limit_bottom: Optional[float] = None
) -> None:
    """
    Plot predictions for multiple stocks.

    Args:
        df: DataFrame with actual prices
        tickers: List of ticker symbols to plot
        predictions_dict: Dictionary mapping model names to prediction arrays
        num_test_days: Number of test days to plot
        figsize: Figure size
        y_limit_bottom: Minimum y-axis value

    Example:
        >>> plot_multiple_stocks(df, ['AAPL', 'MSFT'], predictions, num_test_days=5)
    """
    ticker_list = list(df.columns)

    for ticker in tickers:
        if ticker not in ticker_list:
            logger.warning(f"Ticker {ticker} not in DataFrame columns")
            continue

        ticker_index = ticker_list.index(ticker)
        plot_predictions(
            df,
            ticker,
            predictions_dict,
            ticker_index,
            num_test_days,
            figsize,
            y_limit_bottom
        )


def plot_training_history(
    losses: List[float],
    title: str = 'Training Loss History',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot training loss over epochs.

    Args:
        losses: List of loss values per epoch
        title: Plot title
        figsize: Figure size

    Example:
        >>> plot_training_history(losses, title='LSTM Training Loss')
    """
    plt.figure(figsize=figsize)
    plt.plot(losses, linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    logger.info(f"Plotted training history with {len(losses)} epochs")


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot comparison of multiple models' performance metrics.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
                     e.g., {'Model 0': {'MSE': 0.023, 'RMSE': 0.15}}
        figsize: Figure size

    Example:
        >>> metrics = {
        ...     'Model 0': {'MSE': 0.023, 'RMSE': 0.15},
        ...     'Model 1': {'MSE': 0.019, 'RMSE': 0.14}
        ... }
        >>> plot_model_comparison(metrics)
    """
    models = list(metrics_dict.keys())
    mse_values = [metrics_dict[m]['MSE'] for m in models]
    rmse_values = [metrics_dict[m]['RMSE'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # MSE comparison
    ax1.bar(models, mse_values, color='steelblue', alpha=0.7)
    ax1.set_title('MSE Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # RMSE comparison
    ax2.bar(models, rmse_values, color='coral', alpha=0.7)
    ax2.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    logger.info(f"Plotted comparison for {len(models)} models")


def save_plot(
    filename: str,
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save the current plot to a file.

    Args:
        filename: Output filename (include extension like .png, .pdf)
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box setting

    Example:
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> save_plot('my_plot.png')
    """
    try:
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Plot saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save plot: {str(e)}")
        raise
