"""
Unified training module for LSTM models.

This module replaces the duplicated training code from the original implementation
with a clean, reusable trainer class.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple
from utils.logger import get_logger
from models.persistence import save_model, save_checkpoint

logger = get_logger(__name__)


class LSTMTrainer:
    """
    Unified trainer for LSTM models with support for standard and transfer learning.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        scheduler_step_size: int = 50,
        scheduler_gamma: float = 0.1,
        direction_loss_weight: float = 0.0,
    ):
        """
        Initialize the trainer.

        Args:
            model: LSTM model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            scheduler_step_size: Step size for learning rate scheduler
            scheduler_gamma: Multiplicative factor for learning rate decay
            direction_loss_weight: weight on the BCE-on-direction auxiliary loss
                (Part 6 Phase 1C). 0.0 = pure MSE (legacy behavior).

        Example:
            >>> trainer = LSTMTrainer(model, device, learning_rate=0.001)
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.direction_loss_weight = float(direction_loss_weight)
        # When True the model is a binary classifier: y=(return>0), loss=BCEWithLogits.
        # The model still outputs shape (batch, n_stocks) but values are raw logits.
        self.direction_mode = False
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )

        logger.info(
            f"LSTMTrainer initialized with lr={learning_rate}, "
            f"direction_loss_weight={self.direction_loss_weight}"
        )

    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> float:
        """
        Train for one epoch.

        Args:
            X_train: Training input data
            y_train: Training target data

        Returns:
            Training loss for this epoch
        """
        self.model.train()

        # Forward pass
        outputs = self.model(X_train.to(self.device))
        targets = y_train.to(self.device)

        if self.direction_mode:
            # Pure direction classification: y is binary (return > 0).
            # targets should already be 0/1 floats; outputs are raw logits.
            loss = self.bce_logits(outputs, targets)
        else:
            # MSE on the (scaled) returns — magnitude term.
            loss = self.criterion(outputs, targets)

            # Optional direction-BCE auxiliary term (Part 6 Phase 1C).
            if self.direction_loss_weight > 0.0:
                pred_dir = torch.sigmoid(outputs * 50.0)
                true_dir = (targets > 0).float()
                dir_loss = self.bce(pred_dir, true_dir)
                loss = loss + self.direction_loss_weight * dir_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @staticmethod
    def _compute_val_da(val_preds: torch.Tensor, y_val: torch.Tensor) -> float:
        """
        Compute directional accuracy on validation logits vs binary targets.

        Used when ``direction_mode=True`` and ``monitor='val_da'``.

        Args:
            val_preds: raw logits from model, shape (T, n_stocks)
            y_val: binary targets (0/1), shape (T, n_stocks)

        Returns:
            Directional accuracy as a percentage (0–100).
        """
        import numpy as np
        pred_dir = (val_preds.cpu().numpy() > 0).astype(float)
        true_dir = (y_val.cpu().numpy() > 0.5).astype(float)
        return float((pred_dir == true_dir).mean()) * 100.0

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        num_epochs: int,
        log_interval: int = 10,
        save_dir: Optional[str] = None,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 10,
        early_stopping_enabled: bool = True,
        monitor: str = "val_mse",
    ) -> Dict[str, Any]:
        """
        Train the model for multiple epochs, with optional early stopping.

        ``monitor`` controls which metric drives early stopping and best-weight
        selection when a validation set is provided:

        - ``"val_mse"``  (default): minimise val MSE / BCE loss — legacy behaviour.
        - ``"val_da"``   (direction mode only): maximise directional accuracy on val.
                         Requires ``self.direction_mode = True``.

        If X_val/y_val are None, all monitoring is skipped (legacy behaviour).

        Returns:
            {
                "train_losses": [...],            # per epoch
                "val_losses":   [...] or [],      # per epoch when val provided
                "best_val":     float or None,    # best monitored metric value
                "best_epoch":   int  or None,     # 1-indexed
                "stopped_early": bool,
            }
        """
        use_da = monitor == "val_da" and self.direction_mode
        logger.info(
            f"Starting training for {num_epochs} epochs "
            f"(monitor={'val_da' if use_da else 'val_mse'})"
        )
        losses: List[float] = []
        val_losses: List[float] = []
        # For DA we maximise (initialise low); for loss we minimise (initialise high).
        best_val = 0.0 if use_da else float("inf")
        best_state = None
        best_epoch: Optional[int] = None
        epochs_since_improvement = 0
        stopped_early = False

        for epoch in range(num_epochs):
            loss = self.train_epoch(X_train, y_train)
            losses.append(loss)

            val_msg = ""
            if X_val is not None and y_val is not None:
                val_mse, _, val_preds = self.evaluate(X_val, y_val)
                val_losses.append(val_mse)

                if use_da:
                    monitor_val = self._compute_val_da(val_preds, y_val)
                    val_msg = f", Val DA: {monitor_val:.2f}%"
                    improved = monitor_val > best_val + 1e-4   # maximise
                else:
                    monitor_val = val_mse
                    val_msg = f", Val MSE: {val_mse:.4f}"
                    improved = monitor_val < best_val - 1e-9   # minimise

                if improved:
                    best_val = monitor_val
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch + 1
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if early_stopping_enabled and epochs_since_improvement >= early_stopping_patience:
                    metric_label = "Val DA" if use_da else "Val MSE"
                    fmt = ".2f" if use_da else ".6f"
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no val improvement for {early_stopping_patience} epochs "
                        f"(best {metric_label} {best_val:{fmt}} at epoch {best_epoch})"
                    )
                    stopped_early = True
                    break

            if (epoch + 1) % log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}{val_msg}")

            if save_dir and (epoch + 1) % 50 == 0:
                save_checkpoint(self.model, self.optimizer, epoch + 1, loss, save_dir)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            metric_label = "val DA" if use_da else "val MSE"
            fmt = ".2f" if use_da else ".6f"
            logger.info(
                f"Restored best weights from epoch {best_epoch} "
                f"({metric_label} {best_val:{fmt}})"
            )

        logger.info(
            f"Training completed. Final train loss: {losses[-1]:.4f}"
            + (f", best {('val DA' if use_da else 'val MSE')}: {best_val:.{2 if use_da else 6}f}" if val_losses else "")
        )

        return {
            "train_losses": losses,
            "val_losses": val_losses,
            "best_val": best_val if val_losses else None,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
        }

    def evaluate(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor
    ) -> Tuple[float, float, torch.Tensor]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test input data
            y_test: Test target data

        Returns:
            Tuple of (mse_loss, rmse_loss, predictions)

        Example:
            >>> mse, rmse, predictions = trainer.evaluate(X_test, y_test)
        """
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X_test.to(self.device))
            if self.direction_mode:
                # Return BCE loss as "mse" slot; pass raw logits as predictions
                # so the walk-forward predictor can convert to ±1 signs.
                test_loss = self.bce_logits(predictions, y_test.to(self.device))
                rmse_val = test_loss.item()
            else:
                test_loss = self.criterion(predictions, y_test.to(self.device))
                rmse_val = math.sqrt(test_loss.item())

        logger.info(f'Test Loss: {test_loss.item():.4f}')
        logger.info(f'Test RMSE / BCE: {rmse_val:.4f}')

        return test_loss.item(), rmse_val, predictions

    def freeze_layers_and_finetune(
        self,
        num_layers_to_freeze: int,
        X_finetune: torch.Tensor,
        y_finetune: torch.Tensor,
        num_epochs: int,
        log_interval: int = 10,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 10,
        early_stopping_enabled: bool = True,
        monitor: str = "val_mse",
    ) -> Dict[str, Any]:
        """Freeze first N layers and fine-tune; same return shape as `train`."""
        logger.info(f"Starting transfer learning: freezing {num_layers_to_freeze} layers")

        if hasattr(self.model, "freeze_layers"):
            self.model.freeze_layers(num_layers_to_freeze)
        else:
            logger.warning("Model doesn't have freeze_layers method. Skipping layer freezing.")

        # Reinitialize optimizer with only trainable parameters.
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.optimizer.param_groups[0]["lr"],
        )

        return self.train(
            X_finetune,
            y_finetune,
            num_epochs,
            log_interval=log_interval,
            X_val=X_val,
            y_val=y_val,
            early_stopping_patience=early_stopping_patience,
            early_stopping_enabled=early_stopping_enabled,
            monitor=monitor,
        )


class SingleFeatureTrainer(LSTMTrainer):
    """
    Trainer for single-feature (per-stock) LSTM models.

    This trainer handles training a model on individual stocks sequentially.
    """

    def train_per_stock(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        num_epochs: int,
        num_stocks: int,
        log_interval: int = 2,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 10,
        early_stopping_enabled: bool = True,
        n_features_per_stock: int = 1,
        n_macro: int = 0,
        monitor: str = "val_mse",
    ) -> Dict[str, Any]:
        """Train one stock at a time per epoch, with optional early stopping.

        ``monitor`` mirrors the same parameter in :meth:`LSTMTrainer.train`:
        ``"val_da"`` maximises directional accuracy (direction_mode only);
        ``"val_mse"`` (default) minimises val loss.

        When ``n_features_per_stock > 1`` (Part 6 Phase 2), each stock's input
        is a contiguous block of ``n_features_per_stock`` columns out of the
        ``num_stocks * n_features_per_stock`` total input columns.

        When ``n_macro > 0``, the last ``n_macro`` columns of the input tensor
        are shared macro/regime features (e.g. VIX, SP500) that are appended to
        every stock's per-stock slice before passing to the model.
        """
        use_da = monitor == "val_da" and getattr(self, "direction_mode", False)
        logger.info(
            f"Training on {num_stocks} stocks individually for {num_epochs} epochs "
            f"(features per stock: {n_features_per_stock}, macro: {n_macro}, "
            f"monitor={'val_da' if use_da else 'val_mse'})"
        )
        stock_cols = num_stocks * n_features_per_stock
        losses: List[float] = []
        val_losses: List[float] = []
        best_val = 0.0 if use_da else float("inf")
        best_state = None
        best_epoch: Optional[int] = None
        epochs_since_improvement = 0
        stopped_early = False

        for epoch in range(num_epochs):
            epoch_losses = []
            for i in range(num_stocks):
                start = i * n_features_per_stock
                end = start + n_features_per_stock
                X_stock = X_train[:, :, start:end]
                if n_macro > 0:
                    X_stock = torch.cat(
                        [X_stock, X_train[:, :, stock_cols:stock_cols + n_macro]], dim=-1
                    )
                y_stock = y_train[:, i:i + 1]
                epoch_losses.append(self.train_epoch(X_stock, y_stock))

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)

            val_msg = ""
            if X_val is not None and y_val is not None:
                val_mse, _, val_preds = self.evaluate_per_stock(
                    X_val, y_val, num_stocks,
                    n_features_per_stock=n_features_per_stock, n_macro=n_macro,
                )
                val_losses.append(val_mse)

                if use_da:
                    monitor_val = self._compute_val_da(val_preds, y_val)
                    val_msg = f", Val DA: {monitor_val:.2f}%"
                    improved = monitor_val > best_val + 1e-4
                else:
                    monitor_val = val_mse
                    val_msg = f", Val MSE: {val_mse:.4f}"
                    improved = monitor_val < best_val - 1e-9

                if improved:
                    best_val = monitor_val
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch + 1
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if early_stopping_enabled and epochs_since_improvement >= early_stopping_patience:
                    metric_label = "Val DA" if use_da else "Val MSE"
                    fmt = ".2f" if use_da else ".6f"
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"best {metric_label} {best_val:{fmt}} at epoch {best_epoch}"
                    )
                    stopped_early = True
                    break

            if (epoch + 1) % log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}{val_msg}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            metric_label = "val DA" if use_da else "val MSE"
            fmt = ".2f" if use_da else ".6f"
            logger.info(
                f"Restored best weights from epoch {best_epoch} "
                f"({metric_label} {best_val:{fmt}})"
            )

        logger.info(f"Per-stock training completed. Final avg loss: {losses[-1]:.4f}")
        return {
            "train_losses": losses,
            "val_losses": val_losses,
            "best_val": best_val if val_losses else None,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
        }

    def evaluate_per_stock(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        num_stocks: int,
        n_features_per_stock: int = 1,
        n_macro: int = 0,
    ) -> Tuple[float, float, torch.Tensor]:
        """
        Evaluate model on each stock individually.

        When ``n_features_per_stock > 1`` each stock's input is the
        corresponding contiguous block of feature columns.
        When ``n_macro > 0``, the last ``n_macro`` columns are appended.
        """
        self.model.eval()
        stock_cols = num_stocks * n_features_per_stock
        predictions = torch.zeros_like(y_test).to(self.device)

        with torch.no_grad():
            for i in range(num_stocks):
                start = i * n_features_per_stock
                end = start + n_features_per_stock
                X_stock = X_test[:, :, start:end]
                if n_macro > 0:
                    X_stock = torch.cat(
                        [X_stock, X_test[:, :, stock_cols:stock_cols + n_macro]], dim=-1
                    )
                pred = self.model(X_stock.to(self.device))
                predictions[:, i:i+1] = pred

        if getattr(self, "direction_mode", False):
            test_loss = self.bce_logits(predictions, y_test.to(self.device))
        else:
            test_loss = self.criterion(predictions, y_test.to(self.device))
        rmse = math.sqrt(test_loss.item())

        logger.info(f'Per-stock Test Loss: {test_loss.item():.4f}')
        logger.info(f'Per-stock Test RMSE/BCE: {rmse:.4f}')

        return test_loss.item(), rmse, predictions


def create_trainer(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    is_single_feature: bool = False,
    **kwargs
) -> LSTMTrainer:
    """
    Factory function to create appropriate trainer.

    Args:
        model: LSTM model
        device: Training device
        learning_rate: Learning rate
        is_single_feature: Whether this is a single-feature model
        **kwargs: Additional arguments for trainer

    Returns:
        Appropriate trainer instance

    Example:
        >>> trainer = create_trainer(model, device, lr=0.001, is_single_feature=False)
    """
    if is_single_feature:
        return SingleFeatureTrainer(model, device, learning_rate, **kwargs)
    else:
        return LSTMTrainer(model, device, learning_rate, **kwargs)
