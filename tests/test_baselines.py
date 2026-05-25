"""Unit tests for models.baselines."""
import numpy as np
import pytest

from models.baselines import NaivePersistence, MajorityClass


def test_naive_returns_last_step():
    X = np.arange(24, dtype=float).reshape(2, 4, 3)  # 2 samples, seq=4, 3 stocks
    pred = NaivePersistence().predict(X)
    assert pred.shape == (2, 3)
    np.testing.assert_array_equal(pred, X[:, -1, :])


def test_naive_rejects_non_3d():
    with pytest.raises(ValueError):
        NaivePersistence().predict(np.zeros((4, 3)))


def test_majority_broadcasts_training_mean():
    mean = np.array([0.001, -0.002, 0.0005])
    X = np.zeros((5, 4, 3))
    pred = MajorityClass(mean).predict(X)
    assert pred.shape == (5, 3)
    for row in pred:
        np.testing.assert_array_equal(row, mean)


def test_majority_dim_check():
    with pytest.raises(ValueError):
        MajorityClass(np.zeros((2, 3)))


def test_majority_stock_count_mismatch():
    mean = np.array([0.001, -0.002])
    X = np.zeros((5, 4, 3))
    with pytest.raises(ValueError):
        MajorityClass(mean).predict(X)
