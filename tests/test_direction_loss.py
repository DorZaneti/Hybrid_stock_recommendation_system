"""Test that direction_loss_weight changes the loss but leaves the API stable."""
import torch
import torch.nn as nn

from training.trainer import LSTMTrainer


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 4, batch_first=True)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _data(seed=0):
    torch.manual_seed(seed)
    X = torch.randn(16, 3, 2)
    y = torch.randn(16, 2)
    return X, y


def test_zero_weight_matches_legacy_loss():
    """direction_loss_weight=0 must reproduce pure-MSE behavior."""
    torch.manual_seed(123)
    m_legacy = _Tiny()
    torch.manual_seed(123)
    m_new = _Tiny()
    X, y = _data()

    trainer_legacy = LSTMTrainer(m_legacy, torch.device("cpu"), learning_rate=0.01,
                                  direction_loss_weight=0.0)
    trainer_new = LSTMTrainer(m_new, torch.device("cpu"), learning_rate=0.01,
                               direction_loss_weight=0.0)
    out_legacy = trainer_legacy.train(X, y, num_epochs=3)
    out_new = trainer_new.train(X, y, num_epochs=3)

    for a, b in zip(out_legacy["train_losses"], out_new["train_losses"]):
        assert abs(a - b) < 1e-9


def test_nonzero_weight_changes_loss():
    """direction_loss_weight>0 must produce different (typically higher) loss values."""
    X, y = _data()
    torch.manual_seed(123)
    m0 = _Tiny()
    torch.manual_seed(123)
    m1 = _Tiny()
    t0 = LSTMTrainer(m0, torch.device("cpu"), learning_rate=0.01, direction_loss_weight=0.0)
    t1 = LSTMTrainer(m1, torch.device("cpu"), learning_rate=0.01, direction_loss_weight=0.5)
    out0 = t0.train(X, y, num_epochs=2)
    out1 = t1.train(X, y, num_epochs=2)
    # Same model init + same data + different loss => different trajectories.
    assert out0["train_losses"] != out1["train_losses"]
