"""
Log-return conversions for stock-price data.

Why log-returns?
- Stationary (mean ~ 0, variance roughly constant over time).
- Direction is a first-class quantity: sign(return) == sign(price_t - price_{t-1}).
- The trivial "predict yesterday" cheat that dominates a price-target MSE objective
  does not exist here: a constant prediction is zero return, which is a real prior
  (no change), not a free lunch.
"""
from typing import Union
import numpy as np
import pandas as pd


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a price DataFrame to a log-return DataFrame.

    r_t = log(p_t / p_{t-1})

    The first row is dropped (no previous price to compute against).
    """
    if prices.empty:
        return prices.copy()
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


def compute_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Forward h-day log-returns aligned for the ``create_sequences_xy`` convention.

    For a sample whose X window ends on date ``t-1``, we want the target to be
    the h-day forward log-return from ``t-1`` to ``t-1+h``. Since
    ``create_sequences_xy`` builds ``y[i] = y_data[i + seq_length]`` (the row
    AFTER X ends), we index targets by the "target date" ``t`` such that:

        y_data[t] = log(p[t-1+h] / p[t-1])

    For h=1 this collapses to the standard 1-day return ``log(p_t / p_{t-1})``,
    so the existing pipeline behavior is preserved at horizon 1.

    Returns a DataFrame with ``h`` rows dropped (1 from the prev-day anchor,
    h-1 from the forward shift) so every remaining row has a well-defined target.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if prices.empty:
        return prices.copy()
    forward = np.log(prices.shift(-(horizon - 1)) / prices.shift(1))
    return forward.dropna(how="any")


def compute_past_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Past h-day log-return ending one day *before* each row.

        past[t] = log(p[t-1] / p[t-1-h])

    This is the natural "last observed h-day return" available at decision day
    ``t-1``, which is what a momentum-style naive baseline uses to forecast the
    next h-day forward return. For h=1 this collapses to ``log(p[t-1] / p[t-2])``,
    one day behind the standard 1-day return — same shape, same purpose, just
    horizon-aware.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if prices.empty:
        return prices.copy()
    past = np.log(prices.shift(1) / prices.shift(1 + horizon))
    return past.dropna(how="any")


def returns_to_prices(
    returns: Union[np.ndarray, pd.DataFrame],
    anchor_prices: Union[np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """
    Reconstruct prices from log-returns given the previous-day anchor prices.

    p_t = p_{t-1} * exp(r_t)

    Shapes must match. Returns a numpy array regardless of input type so callers
    can feed it directly to downstream metrics.
    """
    r = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    a = anchor_prices.values if isinstance(anchor_prices, pd.DataFrame) else np.asarray(anchor_prices)
    if r.shape != a.shape:
        raise ValueError(f"Shape mismatch: returns {r.shape} vs anchors {a.shape}")
    return a * np.exp(r)
