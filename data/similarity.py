"""
Stock similarity analysis using technical indicators.
"""
from typing import Dict, List
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from utils.logger import get_logger

logger = get_logger(__name__)


def find_similar_stocks(
    data: Dict[str, pd.DataFrame],
    base_ticker: str,
    num_similar: int = 10,
    features: List[str] = None
) -> List[str]:
    """
    Find stocks similar to a base ticker based on technical indicators.

    Uses Euclidean distance in feature space to measure similarity.
    Lower distance means more similar stocks.

    Args:
        data: Dictionary mapping ticker symbols to DataFrames with features
        base_ticker: Ticker symbol to find similar stocks for
        num_similar: Number of similar stocks to return
        features: List of feature column names to use for comparison.
                 If None, uses default technical indicators.

    Returns:
        List of similar ticker symbols (most similar first)

    Raises:
        ValueError: If base ticker not in data or has no valid features
        KeyError: If specified features not found in data

    Example:
        >>> data = {'AAPL': df1, 'MSFT': df2, 'GOOGL': df3}
        >>> similar = find_similar_stocks(data, 'AAPL', num_similar=5)
        >>> print(similar)
        ['MSFT', 'GOOGL', ...]
    """
    if features is None:
        features = ['RSI', 'Momentum', 'Moving_Average', 'Bollinger_Upper', 'Bollinger_Lower', 'Close']

    logger.info(f"Finding {num_similar} stocks similar to {base_ticker} using features: {features}")

    # Validate base ticker
    if base_ticker not in data:
        raise ValueError(f"Base ticker '{base_ticker}' not found in data")

    # Get base features and drop NaNs
    base_features = data[base_ticker][features].dropna()

    if base_features.empty:
        raise ValueError(
            f"Base ticker '{base_ticker}' has no valid features after dropping NaNs. "
            "Ensure features are calculated first."
        )

    logger.debug(f"Base ticker {base_ticker} has {len(base_features)} valid data points")

    # Calculate similarities
    similarities = {}
    skipped_tickers = []

    for ticker in data:
        if ticker == base_ticker:
            continue

        try:
            # Check if features exist
            missing_features = [f for f in features if f not in data[ticker].columns]
            if missing_features:
                logger.warning(f"{ticker}: Missing features {missing_features}")
                skipped_tickers.append(ticker)
                continue

            other_features = data[ticker][features].dropna()

            if other_features.empty:
                logger.warning(f"{ticker}: No valid features after dropping NaNs")
                skipped_tickers.append(ticker)
                continue

            # Calculate Euclidean distance
            distance = pairwise_distances(base_features, other_features, metric='euclidean').mean()
            similarities[ticker] = distance

            logger.debug(f"{ticker}: Distance = {distance:.4f}")

        except Exception as e:
            logger.error(f"Error calculating similarity for {ticker}: {str(e)}")
            skipped_tickers.append(ticker)

    if not similarities:
        raise ValueError(
            f"No valid stocks found for similarity comparison with {base_ticker}. "
            f"Skipped: {', '.join(skipped_tickers)}"
        )

    # Sort by distance (ascending) and return top N
    similar_stocks = sorted(similarities, key=similarities.get)[:num_similar]

    logger.info(f"Found {len(similar_stocks)} similar stocks. Skipped {len(skipped_tickers)} tickers.")
    if similar_stocks:
        logger.info(f"Most similar: {similar_stocks[0]} (distance: {similarities[similar_stocks[0]]:.4f})")

    return similar_stocks


def get_similarity_scores(
    data: Dict[str, pd.DataFrame],
    base_ticker: str,
    features: List[str] = None
) -> Dict[str, float]:
    """
    Get similarity scores for all stocks compared to a base ticker.

    Args:
        data: Dictionary mapping ticker symbols to DataFrames
        base_ticker: Ticker to compare against
        features: Features to use for comparison

    Returns:
        Dictionary mapping tickers to their similarity scores (distance)

    Example:
        >>> scores = get_similarity_scores(data, 'AAPL')
        >>> print(scores['MSFT'])
        0.523
    """
    if features is None:
        features = ['RSI', 'Momentum', 'Moving_Average', 'Bollinger_Upper', 'Bollinger_Lower', 'Close']

    if base_ticker not in data:
        raise ValueError(f"Base ticker '{base_ticker}' not found in data")

    base_features = data[base_ticker][features].dropna()
    if base_features.empty:
        raise ValueError(f"Base ticker '{base_ticker}' has no valid features")

    scores = {}
    for ticker in data:
        if ticker == base_ticker:
            continue

        try:
            other_features = data[ticker][features].dropna()
            if not other_features.empty:
                distance = pairwise_distances(base_features, other_features, metric='euclidean').mean()
                scores[ticker] = distance
        except Exception as e:
            logger.warning(f"Could not calculate score for {ticker}: {str(e)}")

    return scores
