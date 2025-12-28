"""
Unified training module for LSTM models.

This module replaces the duplicated training code from the original implementation
with a clean, reusable trainer class.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
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
        scheduler_gamma: float = 0.1
    ):
        """
        Initialize the trainer.

        Args:
            model: LSTM model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            scheduler_step_size: Step size for learning rate scheduler
            scheduler_gamma: Multiplicative factor for learning rate decay

        Example:
            >>> trainer = LSTMTrainer(model, device, learning_rate=0.001)
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )

        logger.info(f"LSTMTrainer initialized with lr={learning_rate}")

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

        # Calculate loss
        loss = self.criterion(outputs, y_train.to(self.device))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        num_epochs: int,
        log_interval: int = 10,
        save_dir: Optional[str] = None
    ) -> List[float]:
        """
        Train the model for multiple epochs.

        Args:
            X_train: Training input data
            y_train: Training target data
            num_epochs: Number of epochs to train
            log_interval: Log progress every N epochs
            save_dir: Optional directory to save checkpoints

        Returns:
            List of losses for each epoch

        Example:
            >>> losses = trainer.train(X_train, y_train, num_epochs=100)
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        losses = []

        for epoch in range(num_epochs):
            loss = self.train_epoch(X_train, y_train)
            losses.append(loss)

            if (epoch + 1) % log_interval == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

            # Save checkpoint if directory provided
            if save_dir and (epoch + 1) % 50 == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    loss,
                    save_dir
                )

        logger.info(f"Training completed. Final loss: {losses[-1]:.4f}")
        return losses

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
            test_loss = self.criterion(predictions, y_test.to(self.device))
            rmse = math.sqrt(test_loss.item())

        logger.info(f'Test Loss (MSE): {test_loss.item():.4f}')
        logger.info(f'Test RMSE: {rmse:.4f}')

        return test_loss.item(), rmse, predictions

    def freeze_layers_and_finetune(
        self,
        num_layers_to_freeze: int,
        X_finetune: torch.Tensor,
        y_finetune: torch.Tensor,
        num_epochs: int,
        log_interval: int = 10
    ) -> List[float]:
        """
        Freeze layers and fine-tune the model (transfer learning).

        Args:
            num_layers_to_freeze: Number of layers to freeze
            X_finetune: Fine-tuning input data
            y_finetune: Fine-tuning target data
            num_epochs: Number of epochs for fine-tuning
            log_interval: Log progress every N epochs

        Returns:
            List of losses during fine-tuning

        Example:
            >>> losses = trainer.freeze_layers_and_finetune(3, X_ft, y_ft, num_epochs=50)
        """
        logger.info(f"Starting transfer learning: freezing {num_layers_to_freeze} layers")

        # Freeze layers
        if hasattr(self.model, 'freeze_layers'):
            self.model.freeze_layers(num_layers_to_freeze)
        else:
            logger.warning("Model doesn't have freeze_layers method. Skipping layer freezing.")

        # Reinitialize optimizer with only trainable parameters
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.optimizer.param_groups[0]['lr']
        )

        # Fine-tune
        losses = self.train(X_finetune, y_finetune, num_epochs, log_interval)

        return losses


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
        log_interval: int = 2
    ) -> List[float]:
        """
        Train model on each stock individually.

        Args:
            X_train: Training input (shape: [samples, seq_len, num_stocks])
            y_train: Training target (shape: [samples, num_stocks])
            num_epochs: Number of epochs to train
            num_stocks: Number of stocks in the data
            log_interval: Log progress every N epochs

        Returns:
            List of losses

        Example:
            >>> losses = trainer.train_per_stock(X_train, y_train, num_epochs=10, num_stocks=10)
        """
        logger.info(f"Training on {num_stocks} stocks individually for {num_epochs} epochs")
        losses = []

        for epoch in range(num_epochs):
            epoch_losses = []

            for i in range(num_stocks):
                # Extract data for current stock
                X_stock = X_train[:, :, i:i+1]
                y_stock = y_train[:, i:i+1]

                # Train on this stock
                loss = self.train_epoch(X_stock, y_stock)
                epoch_losses.append(loss)

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)

            if (epoch + 1) % log_interval == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')

        logger.info(f"Per-stock training completed. Final avg loss: {losses[-1]:.4f}")
        return losses

    def evaluate_per_stock(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        num_stocks: int
    ) -> Tuple[float, float, torch.Tensor]:
        """
        Evaluate model on each stock individually.

        Args:
            X_test: Test input data
            y_test: Test target data
            num_stocks: Number of stocks

        Returns:
            Tuple of (mse_loss, rmse_loss, predictions)
        """
        self.model.eval()

        predictions = torch.zeros_like(y_test).to(self.device)

        with torch.no_grad():
            for i in range(num_stocks):
                X_stock = X_test[:, :, i:i+1]
                pred = self.model(X_stock.to(self.device))
                predictions[:, i:i+1] = pred

        test_loss = self.criterion(predictions, y_test.to(self.device))
        rmse = math.sqrt(test_loss.item())

        logger.info(f'Per-stock Test Loss (MSE): {test_loss.item():.4f}')
        logger.info(f'Per-stock Test RMSE: {rmse:.4f}')

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
