"""
Model persistence utilities for saving and loading trained models.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


def save_model(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None
) -> None:
    """
    Save a PyTorch model with optional metadata and training state.

    Args:
        model: PyTorch model to save
        save_path: Path to save the model
        metadata: Optional dictionary with model metadata (hyperparameters, metrics, etc.)
        optimizer: Optional optimizer state to save
        scaler: Optional MinMaxScaler to save for inverse transformations

    Example:
        >>> save_model(
        ...     model,
        ...     'saved_models/lstm_model.pt',
        ...     metadata={'loss': 0.0234, 'epochs': 100}
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
    }

    if metadata:
        checkpoint['metadata'] = metadata

    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scaler:
        checkpoint['scaler'] = scaler

    try:
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
        if metadata:
            logger.debug(f"Metadata: {metadata}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise


def load_model(
    model: nn.Module,
    load_path: str,
    device: torch.device,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load a saved PyTorch model.

    Args:
        model: Model instance to load weights into
        load_path: Path to saved model file
        device: Device to load model onto
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer instance to load state into (required if load_optimizer=True)

    Returns:
        Dictionary containing metadata and optional components

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model architecture doesn't match saved weights

    Example:
        >>> model = StockLSTM(input_size=10, hidden_size=100, num_layers=4, output_size=10)
        >>> checkpoint = load_model(model, 'saved_models/lstm_model.pt', device)
        >>> print(checkpoint['metadata'])
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    try:
        checkpoint = torch.load(load_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        logger.info(f"Model loaded from {load_path}")

        result = {
            'metadata': checkpoint.get('metadata', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'scaler': checkpoint.get('scaler')
        }

        if load_optimizer and optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            result['optimizer_loaded'] = True
            logger.info("Optimizer state loaded")

        return result

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Error loading model from {load_path}: {str(e)}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_dir: str,
    model_name: str = 'checkpoint'
) -> str:
    """
    Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        save_dir: Directory to save checkpoint
        model_name: Name prefix for checkpoint file

    Returns:
        Path to saved checkpoint

    Example:
        >>> path = save_checkpoint(model, optimizer, epoch=50, loss=0.023, save_dir='checkpoints')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"{model_name}_epoch_{epoch}.pt"

    metadata = {
        'epoch': epoch,
        'loss': loss,
    }

    save_model(model, str(checkpoint_path), metadata=metadata, optimizer=optimizer)
    return str(checkpoint_path)


def get_latest_checkpoint(checkpoint_dir: str, model_name: str = 'checkpoint') -> Optional[str]:
    """
    Find the most recent checkpoint file.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Name prefix of checkpoint files

    Returns:
        Path to latest checkpoint or None if no checkpoints found

    Example:
        >>> latest = get_latest_checkpoint('checkpoints')
        >>> if latest:
        ...     checkpoint = load_model(model, latest, device)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(f"{model_name}_epoch_*.pt"))

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    logger.info(f"Found latest checkpoint: {latest}")
    return str(latest)
