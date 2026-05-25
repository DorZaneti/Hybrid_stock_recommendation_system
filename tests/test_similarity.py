"""
Unit tests for the Pearson return-correlation similarity module (Test A).
"""
import numpy as np
import pandas as pd
import pytest

from data.similarity import find_similar_stocks, get_similarity_scores


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ticker(seed: int, n: int = 200, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    log_p = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)
    close = np.exp(log_p)
    return pd.DataFrame({"Close": close}, index=dates)


def _make_correlated(base_df: pd.DataFrame, noise: float = 0.01) -> pd.DataFrame:
    """Return a ticker whose returns are strongly correlated with *base_df*."""
    rng = np.random.default_rng(999)
    base_ret = np.log(base_df["Close"] / base_df["Close"].shift(1)).fillna(0)
    noisy_ret = base_ret + rng.normal(0, noise, len(base_df))
    close = 100 * np.exp(np.cumsum(noisy_ret))
    return pd.DataFrame({"Close": close}, index=base_df.index)


def _make_orthogonal(base_df: pd.DataFrame) -> pd.DataFrame:
    """Return a ticker whose returns are pure noise (uncorrelated with base)."""
    rng = np.random.default_rng(777)
    n = len(base_df)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Close": close}, index=base_df.index)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_correlated_stock_ranks_first():
    """A stock with near-identical returns should come before an orthogonal one."""
    base = _make_ticker(0)
    data = {
        "BASE": base,
        "CORR": _make_correlated(base, noise=0.002),
        "ORTH": _make_orthogonal(base),
    }
    result = find_similar_stocks(data, "BASE", num_similar=2)
    assert result[0] == "CORR", f"Expected CORR first, got {result}"


def test_train_end_date_excludes_oos():
    """Correlations computed only on training data must not use future rows."""
    base = _make_ticker(0, n=300)
    # Construct a ticker that is *anti*-correlated in OOS but correlated in train.
    train_cut = base.index[150]

    rng = np.random.default_rng(42)
    base_ret = np.log(base["Close"] / base["Close"].shift(1)).fillna(0).values
    mixed_ret = np.concatenate([
        base_ret[:150] + rng.normal(0, 0.001, 150),     # correlated in train
        -base_ret[150:] + rng.normal(0, 0.001, 150),    # anti-correlated in OOS
    ])
    close_mixed = 100 * np.exp(np.cumsum(mixed_ret))
    mixed_df = pd.DataFrame({"Close": close_mixed}, index=base.index)

    data = {"BASE": base, "MIXED": mixed_df, "OTHER": _make_orthogonal(base)}

    result_with_cutoff = find_similar_stocks(
        data, "BASE", num_similar=1, train_end_date=str(train_cut.date())
    )
    # When constrained to train period, MIXED should rank first (correlated in train)
    assert result_with_cutoff[0] == "MIXED"


def test_returns_num_similar_respected():
    data = {f"T{i}": _make_ticker(i) for i in range(10)}
    data["BASE"] = _make_ticker(99)
    result = find_similar_stocks(data, "BASE", num_similar=5)
    assert len(result) == 5


def test_missing_base_ticker_raises():
    data = {"AAPL": _make_ticker(0)}
    with pytest.raises(ValueError):
        find_similar_stocks(data, "MISSING", num_similar=1)


def test_no_valid_peers_raises():
    """Only the base ticker in data → should raise ValueError."""
    data = {"BASE": _make_ticker(0)}
    with pytest.raises(ValueError):
        find_similar_stocks(data, "BASE", num_similar=1)


def test_get_similarity_scores_returns_correlations():
    base = _make_ticker(0)
    corr_ticker = _make_correlated(base, noise=0.005)
    data = {"BASE": base, "CORR": corr_ticker, "ORTH": _make_orthogonal(base)}
    scores = get_similarity_scores(data, "BASE")
    assert "BASE" not in scores
    assert "CORR" in scores
    # Correlated ticker should have higher score than orthogonal
    assert scores["CORR"] > scores["ORTH"]


def test_features_param_ignored():
    """The old 'features' API param should be silently ignored (no error)."""
    base = _make_ticker(0)
    data = {"BASE": base, "OTHER": _make_ticker(1)}
    # Should not raise even though 'features' is passed
    result = find_similar_stocks(
        data, "BASE", num_similar=1,
        features=["RSI", "Close", "NonExistentCol"]
    )
    assert len(result) == 1
