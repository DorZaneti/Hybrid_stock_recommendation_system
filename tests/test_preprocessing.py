"""
Unit tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
import torch
from data.preprocessing import (
    scale_data,
    create_sequences,
    split_train_test,
    split_for_transfer_learning
)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    data = {
        'AAPL': np.random.uniform(100, 200, 100),
        'MSFT': np.random.uniform(150, 250, 100),
        'GOOGL': np.random.uniform(200, 300, 100)
    }
    return pd.DataFrame(data)


def test_scale_data(sample_dataframe):
    """Test data scaling."""
    scaled, scaler = scale_data(sample_dataframe)

    assert isinstance(scaled, np.ndarray)
    assert scaled.shape[0] <= sample_dataframe.shape[0]
    assert scaled.shape[1] == sample_dataframe.shape[1]
    # Check that values are scaled to [0, 1]
    assert scaled.min() >= 0
    assert scaled.max() <= 1


def test_create_sequences():
    """Test sequence creation."""
    data = np.random.rand(100, 5)
    seq_length = 8

    X, y = create_sequences(data, seq_length)

    assert X.shape == (92, 8, 5)  # 100 - 8 = 92 sequences
    assert y.shape == (92, 5)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_split_train_test():
    """Test train/test split."""
    X = np.random.rand(100, 8, 5)
    y = np.random.rand(100, 5)
    test_size = 10

    X_train, y_train, X_test, y_test = split_train_test(X, y, test_size)

    assert isinstance(X_train, torch.Tensor)
    assert isinstance(y_train, torch.Tensor)
    assert isinstance(X_test, torch.Tensor)
    assert isinstance(y_test, torch.Tensor)

    assert X_train.shape[0] == 90
    assert X_test.shape[0] == 10
    assert y_train.shape[0] == 90
    assert y_test.shape[0] == 10


def test_split_for_transfer_learning():
    """Test transfer learning split."""
    X = np.random.rand(100, 8, 5)
    y = np.random.rand(100, 5)
    test_size = 5
    finetune_size = 15

    X_pre, y_pre, X_ft, y_ft, X_test, y_test = split_for_transfer_learning(
        X, y, test_size, finetune_size
    )

    assert X_pre.shape[0] == 80  # 100 - 5 - 15 = 80
    assert X_ft.shape[0] == 15
    assert X_test.shape[0] == 5

    assert isinstance(X_pre, torch.Tensor)
    assert isinstance(X_ft, torch.Tensor)
    assert isinstance(X_test, torch.Tensor)
