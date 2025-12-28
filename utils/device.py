"""
Device selection utilities for PyTorch.
"""
import torch
from typing import List
from .logger import get_logger

logger = get_logger(__name__)


def get_device(priority: List[str] = None) -> torch.device:
    """
    Get the best available PyTorch device based on priority.

    Args:
        priority: List of device types in order of preference.
                 Defaults to ['mps', 'cuda', 'cpu']

    Returns:
        torch.device: The best available device

    Example:
        >>> device = get_device()
        >>> print(device)
        device(type='mps')
    """
    if priority is None:
        priority = ['mps', 'cuda', 'cpu']

    for device_type in priority:
        if device_type == 'mps' and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS device")
            return torch.device('mps')
        elif device_type == 'cuda' and torch.cuda.is_available():
            logger.info(f"Using CUDA device (GPU: {torch.cuda.get_device_name(0)})")
            return torch.device('cuda')
        elif device_type == 'cpu':
            logger.info("Using CPU device")
            return torch.device('cpu')

    # Fallback to CPU if no priority matches
    logger.warning("No preferred device available, falling back to CPU")
    return torch.device('cpu')


def set_random_seed(seed: int, device: torch.device) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        device: PyTorch device being used

    Example:
        >>> device = get_device()
        >>> set_random_seed(42, device)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device.type == 'mps':
        torch.mps.manual_seed(seed)

    logger.info(f"Random seed set to {seed} for reproducibility")
