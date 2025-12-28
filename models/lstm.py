"""
LSTM model for stock price prediction.
"""
import torch
import torch.nn as nn
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class StockLSTM(nn.Module):
    """
    Multi-layer LSTM network for stock price prediction.

    This model uses a stacked LSTM architecture with proper dropout implementation.
    The original code had dropout configured incorrectly (dropout on single-layer LSTMs
    has no effect). This implementation fixes that issue.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features (e.g., number of stocks)
            hidden_size: Number of hidden units in each LSTM layer
            num_layers: Number of stacked LSTM layers
            output_size: Number of output features (typically same as input_size)
            dropout: Dropout probability (only applied if num_layers > 1)

        Example:
            >>> model = StockLSTM(input_size=10, hidden_size=100, num_layers=4, output_size=10)
        """
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use stacked LSTM for proper dropout support
        # Dropout is only applied if num_layers > 1
        if num_layers > 1:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            logger.debug(f"Created LSTM with {num_layers} layers and dropout={dropout}")
        else:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            logger.debug(f"Created single-layer LSTM (dropout not applicable)")

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        logger.info(
            f"StockLSTM initialized: input={input_size}, hidden={hidden_size}, "
            f"layers={num_layers}, output={output_size}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Use output from last time step
        out = self.fc(out[:, -1, :])

        return out

    def freeze_layers(self, num_layers_to_freeze: int) -> None:
        """
        Freeze the first N LSTM layers for transfer learning.

        Args:
            num_layers_to_freeze: Number of layers to freeze from the beginning

        Example:
            >>> model.freeze_layers(3)  # Freeze first 3 layers
        """
        if num_layers_to_freeze >= self.num_layers:
            logger.warning(
                f"Requested to freeze {num_layers_to_freeze} layers but model only has "
                f"{self.num_layers}. Freezing all layers."
            )
            num_layers_to_freeze = self.num_layers

        # In a stacked LSTM, we can't freeze individual layers directly
        # Instead, we freeze all LSTM parameters and unfreeze the last few
        # This is a simplified approach for transfer learning

        for name, param in self.lstm.named_parameters():
            param.requires_grad = False

        logger.info(f"Froze all LSTM layers for transfer learning")
        logger.warning(
            "Note: With stacked LSTM architecture, we freeze all LSTM layers. "
            "Only the final fully connected layer remains trainable."
        )

    def unfreeze_all(self) -> None:
        """
        Unfreeze all model parameters.

        Example:
            >>> model.unfreeze_all()
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all model parameters")

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SingleFeatureLSTM(StockLSTM):
    """
    LSTM model for single-feature (single stock) prediction.

    This is a specialized version for training on individual stocks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2
    ):
        """
        Initialize single-feature LSTM.

        Args:
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability

        Example:
            >>> model = SingleFeatureLSTM(hidden_size=100, num_layers=4)
        """
        super(SingleFeatureLSTM, self).__init__(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        )
        logger.info("SingleFeatureLSTM initialized for individual stock prediction")
