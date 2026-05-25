"""
Stock data download utilities using yfinance.
"""
from typing import Dict, List
import yfinance as yf
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


def download_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    retry_count: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Download historical stock data from Yahoo Finance.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        retry_count: Number of retry attempts for failed downloads

    Returns:
        Dictionary mapping ticker symbols to their DataFrames

    Raises:
        ValueError: If no data could be downloaded for any ticker
        ConnectionError: If network connection fails after retries

    Example:
        >>> data = download_stock_data(['AAPL', 'MSFT'], '2020-01-01', '2023-01-01')
        >>> print(data['AAPL'].head())
    """
    logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")

    for attempt in range(retry_count):
        try:
            all_data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                group_by='ticker',
                progress=False
            )

            # Create dictionary mapping tickers to their data
            data_dict = {}
            failed_tickers = []

            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        # Single ticker download returns different structure
                        ticker_data = all_data
                    else:
                        ticker_data = all_data[ticker]

                    if ticker_data.empty or ticker_data.isna().all().all():
                        logger.warning(f"No data available for {ticker}")
                        failed_tickers.append(ticker)
                    else:
                        data_dict[ticker] = ticker_data
                        logger.debug(f"Successfully downloaded {len(ticker_data)} rows for {ticker}")
                except KeyError:
                    logger.warning(f"Ticker {ticker} not found in downloaded data")
                    failed_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
                    failed_tickers.append(ticker)

            if not data_dict:
                raise ValueError("No data could be downloaded for any ticker")

            success_rate = len(data_dict) / len(tickers) * 100
            logger.info(f"Successfully downloaded {len(data_dict)}/{len(tickers)} tickers ({success_rate:.1f}%)")

            if failed_tickers:
                logger.warning(f"Failed to download: {', '.join(failed_tickers)}")

            return data_dict

        except Exception as e:
            logger.error(f"Download attempt {attempt + 1}/{retry_count} failed: {str(e)}")
            if attempt == retry_count - 1:
                raise ConnectionError(f"Failed to download data after {retry_count} attempts: {str(e)}")

    raise ConnectionError("Unexpected error in download_stock_data")


def download_macro_data(
    macro_tickers: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Download macro / index data (e.g. VIX, S&P 500) from Yahoo Finance.

    Wrapper around ``download_stock_data`` with friendlier error handling for
    index tickers that may not appear in a bulk download.  Missing tickers are
    logged as warnings rather than raising an exception, so a partial result
    is acceptable.

    Args:
        macro_tickers: List of Yahoo Finance tickers (e.g. ['^VIX', '^GSPC']).
        start_date:    Start date in 'YYYY-MM-DD' format.
        end_date:      End date in 'YYYY-MM-DD' format.

    Returns:
        Dict mapping ticker symbol → DataFrame with at least a 'Close' column.
        May be empty if all downloads fail (caller should check).

    Example:
        >>> macro = download_macro_data(['^VIX', '^GSPC'], '2020-01-01', '2025-01-01')
        >>> vix_close = macro['^VIX']['Close']
    """
    logger.info(f"Downloading macro data for: {macro_tickers}")
    result: Dict[str, pd.DataFrame] = {}

    for ticker in macro_tickers:
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
            if raw is None or raw.empty:
                logger.warning(f"No macro data returned for {ticker}")
                continue
            # yfinance returns multi-level columns for single-ticker downloads
            # when called with auto_adjust — flatten if needed.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if "Close" not in raw.columns:
                logger.warning(f"{ticker}: 'Close' column missing after download")
                continue
            result[ticker] = raw[["Close"]].copy()
            logger.info(f"Downloaded {len(result[ticker])} rows for {ticker}")
        except Exception as exc:
            logger.warning(f"Failed to download macro ticker {ticker}: {exc}")

    if not result:
        logger.warning("No macro data could be downloaded; macro features will be disabled")
    return result


def validate_stock_data(data: pd.DataFrame, ticker: str) -> bool:
    """
    Validate that stock data meets minimum requirements.

    Args:
        data: Stock data DataFrame
        ticker: Ticker symbol for logging

    Returns:
        True if data is valid, False otherwise

    Example:
        >>> df = pd.DataFrame({'Close': [100, 101, 102]})
        >>> is_valid = validate_stock_data(df, 'AAPL')
    """
    if data is None or data.empty:
        logger.warning(f"{ticker}: Data is empty")
        return False

    required_columns = ['Close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.warning(f"{ticker}: Missing required columns: {missing_columns}")
        return False

    if data['Close'].isna().all():
        logger.warning(f"{ticker}: All Close prices are NaN")
        return False

    if len(data) < 10:
        logger.warning(f"{ticker}: Insufficient data points ({len(data)} < 10)")
        return False

    return True
