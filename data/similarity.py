"""
Stock similarity analysis using return correlation.

Test A: Pearson correlation on daily log-returns during the training period only.
- Higher correlation → more similar (inverted from the old Euclidean distance).
- Restricting to the training period prevents OOS data leak.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


def _log_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily log-returns from the 'Close' column."""
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()


def find_similar_stocks(
    data: Dict[str, pd.DataFrame],
    base_ticker: str,
    num_similar: int = 10,
    features: Optional[List[str]] = None,        # kept for API compat, unused
    train_end_date: Optional[str] = None,
) -> List[str]:
    """
    Find stocks most correlated with *base_ticker* using Pearson correlation
    of daily log-returns, computed only over the training period.

    Args:
        data: Dict mapping ticker → DataFrame with at least a 'Close' column.
        base_ticker: Ticker to find peers for.
        num_similar: How many similar tickers to return.
        features: Ignored — kept so call-sites don't need to change.
        train_end_date: ISO date string (e.g. '2023-01-01').
                        Only dates *before* this cutoff are used.
                        If None, all available dates are used (no cutoff).

    Returns:
        List of ticker symbols sorted by descending correlation (most similar first).
    """
    logger.info(
        f"Finding {num_similar} stocks similar to {base_ticker} "
        f"(Pearson return-correlation, train_end={train_end_date or 'all'})"
    )

    if base_ticker not in data:
        raise ValueError(f"Base ticker '{base_ticker}' not found in data")

    # ── Slice to training period ──────────────────────────────────────────────
    cutoff = pd.Timestamp(train_end_date) if train_end_date else None

    def _train_slice(df: pd.DataFrame) -> pd.DataFrame:
        if cutoff is None:
            return df
        return df.loc[df.index < cutoff]

    # ── Base ticker returns ───────────────────────────────────────────────────
    base_ret = _log_returns(_train_slice(data[base_ticker]))

    if base_ret.empty or base_ret.std() == 0:
        raise ValueError(
            f"Base ticker '{base_ticker}' has no usable return data in the "
            f"training period (up to {train_end_date})."
        )

    logger.debug(f"{base_ticker}: {len(base_ret)} training-period return observations")

    # ── Compute correlations ──────────────────────────────────────────────────
    correlations: Dict[str, float] = {}
    skipped: List[str] = []

    for ticker, df in data.items():
        if ticker == base_ticker:
            continue

        try:
            ret = _log_returns(_train_slice(df))

            if ret.empty or ret.std() == 0:
                logger.warning(f"{ticker}: empty or constant returns — skipping")
                skipped.append(ticker)
                continue

            # Align on the overlapping date index
            common = base_ret.index.intersection(ret.index)
            if len(common) < 30:
                logger.warning(f"{ticker}: only {len(common)} overlapping days — skipping")
                skipped.append(ticker)
                continue

            corr = float(base_ret.loc[common].corr(ret.loc[common]))
            if np.isnan(corr):
                skipped.append(ticker)
                continue

            correlations[ticker] = corr
            logger.debug(f"{ticker}: corr = {corr:.4f}")

        except Exception as exc:
            logger.error(f"Error computing correlation for {ticker}: {exc}")
            skipped.append(ticker)

    if not correlations:
        raise ValueError(
            f"No valid stocks found for similarity comparison with {base_ticker}. "
            f"Skipped: {', '.join(skipped)}"
        )

    # Sort descending by correlation (most similar = highest positive correlation)
    similar = sorted(correlations, key=correlations.__getitem__, reverse=True)[:num_similar]

    logger.info(
        f"Top {len(similar)} peers of {base_ticker}: {similar}. "
        f"Skipped {len(skipped)} tickers."
    )
    if similar:
        logger.info(
            f"Highest corr: {similar[0]} ({correlations[similar[0]]:.4f}), "
            f"Lowest in top-{num_similar}: {similar[-1]} ({correlations[similar[-1]]:.4f})"
        )

    return similar


def get_similarity_scores(
    data: Dict[str, pd.DataFrame],
    base_ticker: str,
    features: Optional[List[str]] = None,        # kept for API compat
    train_end_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Return Pearson return-correlation scores for every ticker vs *base_ticker*.

    Returns:
        Dict mapping ticker → correlation (higher = more similar).
    """
    if base_ticker not in data:
        raise ValueError(f"Base ticker '{base_ticker}' not found in data")

    cutoff = pd.Timestamp(train_end_date) if train_end_date else None

    def _train_slice(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df.index < cutoff] if cutoff is not None else df

    base_ret = _log_returns(_train_slice(data[base_ticker]))
    scores: Dict[str, float] = {}

    for ticker, df in data.items():
        if ticker == base_ticker:
            continue
        try:
            ret = _log_returns(_train_slice(df))
            common = base_ret.index.intersection(ret.index)
            if len(common) >= 30:
                corr = float(base_ret.loc[common].corr(ret.loc[common]))
                if not np.isnan(corr):
                    scores[ticker] = corr
        except Exception as exc:
            logger.warning(f"Could not compute score for {ticker}: {exc}")

    return scores
