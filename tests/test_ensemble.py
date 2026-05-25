"""Unit tests for models.ensemble.EnsembleAverage and EnsembleWeighted."""
import numpy as np
import pytest

from models.ensemble import EnsembleAverage, EnsembleWeighted


class _Const:
    """Predictor that always returns the same constant prediction."""
    def __init__(self, value, n_stocks=2, name="const", val_da=None):
        self.value = float(value)
        self.n_stocks = n_stocks
        self.name = name
        self.val_da = val_da

    def predict(self, X):
        return np.full((X.shape[0], self.n_stocks), self.value)


# ── EnsembleAverage ──────────────────────────────────────────────────────────

def test_ensemble_is_mean_of_constituents():
    ens = EnsembleAverage([_Const(0.0), _Const(2.0), _Const(4.0)])
    X = np.zeros((5, 3, 2))
    out = ens.predict(X)
    np.testing.assert_array_equal(out, np.full((5, 2), 2.0))


def test_ensemble_requires_predictors():
    with pytest.raises(ValueError):
        EnsembleAverage([])


def test_ensemble_shape_mismatch_raises():
    ens = EnsembleAverage([_Const(1.0, n_stocks=2), _Const(3.0, n_stocks=3)])
    X = np.zeros((4, 3, 2))
    with pytest.raises(ValueError):
        ens.predict(X)


# ── EnsembleWeighted ─────────────────────────────────────────────────────────

def test_weighted_equal_weights_when_val_da_missing():
    """If any predictor lacks val_da, fall back to equal weighting."""
    ens = EnsembleWeighted([_Const(0.0, val_da=None), _Const(4.0, val_da=0.6)])
    X = np.zeros((3, 2, 2))
    out = ens.predict(X)
    # Equal weights → mean of 0 and 4 = 2
    np.testing.assert_allclose(out, np.full((3, 2), 2.0))


def test_weighted_favours_high_val_da():
    """A predictor with much higher val_da should dominate the output."""
    # One predictor is very accurate (val_da=0.9), one is terrible (val_da=0.1)
    good = _Const(10.0, val_da=0.9)
    bad  = _Const(0.0,  val_da=0.1)
    ens = EnsembleWeighted([good, bad])
    X = np.zeros((5, 3, 2))
    out = ens.predict(X)
    # With temperature=20, a 0.8pp gap overwhelmingly weights 'good'
    # The output should be much closer to 10 than to 0
    assert out.mean() > 8.0, f"Expected output ~10, got {out.mean():.2f}"


def test_weighted_equal_val_da_gives_mean():
    """If all val_da values are identical, softmax gives equal weights."""
    ens = EnsembleWeighted([_Const(0.0, val_da=0.55), _Const(4.0, val_da=0.55)])
    X = np.zeros((4, 2, 2))
    out = ens.predict(X)
    np.testing.assert_allclose(out, np.full((4, 2), 2.0), atol=1e-6)


def test_weighted_requires_predictors():
    with pytest.raises(ValueError):
        EnsembleWeighted([])


def test_weighted_shape_mismatch_raises():
    ens = EnsembleWeighted([_Const(1.0, n_stocks=2), _Const(3.0, n_stocks=3)])
    X = np.zeros((4, 3, 2))
    with pytest.raises(ValueError):
        ens.predict(X)
