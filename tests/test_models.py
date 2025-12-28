"""
Unit tests for LSTM models.
"""
import pytest
import torch
from models.lstm import StockLSTM, SingleFeatureLSTM


def test_stock_lstm_initialization():
    """Test StockLSTM model initialization."""
    model = StockLSTM(
        input_size=10,
        hidden_size=100,
        num_layers=4,
        output_size=10,
        dropout=0.2
    )

    assert model.hidden_size == 100
    assert model.num_layers == 4
    assert isinstance(model, torch.nn.Module)


def test_stock_lstm_forward():
    """Test forward pass through StockLSTM."""
    model = StockLSTM(input_size=10, hidden_size=100, num_layers=4, output_size=10)
    x = torch.randn(32, 8, 10)  # batch_size=32, seq_length=8, input_size=10

    output = model(x)

    assert output.shape == (32, 10)  # batch_size, output_size
    assert not torch.isnan(output).any()


def test_stock_lstm_freeze_layers():
    """Test layer freezing for transfer learning."""
    model = StockLSTM(input_size=10, hidden_size=100, num_layers=4, output_size=10)

    # Count parameters before freezing
    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.freeze_layers(3)

    # Count parameters after freezing
    params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Some parameters should be frozen
    assert params_after < params_before


def test_single_feature_lstm():
    """Test SingleFeatureLSTM model."""
    model = SingleFeatureLSTM(hidden_size=100, num_layers=4)

    assert model.hidden_size == 100
    assert model.num_layers == 4

    # Test forward pass
    x = torch.randn(32, 8, 1)  # Single feature
    output = model(x)

    assert output.shape == (32, 1)
    assert not torch.isnan(output).any()


def test_model_parameter_count():
    """Test parameter counting."""
    model = StockLSTM(input_size=10, hidden_size=100, num_layers=4, output_size=10)

    param_count = model.count_parameters()

    assert param_count > 0
    assert isinstance(param_count, int)
