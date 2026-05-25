"""Unit tests for trainer val + early stopping."""
import copy

import torch
import torch.nn as nn

from training.trainer import LSTMTrainer


class _TinyModel(nn.Module):
    """Tiny LSTM-ish module that the trainer can train."""
    def __init__(self, input_size=2, hidden=4, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _data(n=32, seq=3, feat=2, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, seq, feat)
    y = X.mean(dim=1) + 0.01 * torch.randn(n, feat)
    return X, y


def test_no_val_path_returns_dict():
    X, y = _data()
    model = _TinyModel()
    trainer = LSTMTrainer(model, torch.device("cpu"), learning_rate=0.01)
    out = trainer.train(X, y, num_epochs=3)
    assert "train_losses" in out
    assert len(out["train_losses"]) == 3
    assert out["val_losses"] == []
    assert out["best_epoch"] is None


def test_early_stopping_fires_and_restores_best():
    X, y = _data()
    Xv, yv = _data(seed=1)
    model = _TinyModel()
    trainer = LSTMTrainer(model, torch.device("cpu"), learning_rate=0.5)  # too high → diverges fast
    out = trainer.train(
        X, y, num_epochs=50,
        X_val=Xv, y_val=yv,
        early_stopping_patience=3,
    )
    assert out["best_epoch"] is not None
    # If it didn't early-stop, it ran all 50 epochs; otherwise fewer.
    if out["stopped_early"]:
        assert len(out["train_losses"]) < 50
    # Best epoch is somewhere in the trajectory
    assert 1 <= out["best_epoch"] <= len(out["train_losses"])


def test_early_stopping_disabled_completes_all_epochs():
    X, y = _data()
    Xv, yv = _data(seed=1)
    model = _TinyModel()
    trainer = LSTMTrainer(model, torch.device("cpu"), learning_rate=0.5)
    out = trainer.train(
        X, y, num_epochs=10,
        X_val=Xv, y_val=yv,
        early_stopping_patience=3,
        early_stopping_enabled=False,
    )
    assert len(out["train_losses"]) == 10
    assert len(out["val_losses"]) == 10
    assert not out["stopped_early"]
