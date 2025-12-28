"""
Configuration loader for stock recommendation system.
"""
from typing import Dict, Any
import yaml
import os
from pathlib import Path


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default config.yaml

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file is invalid
    """
    if config_path is None:
        # Default to config/config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'config.yaml'

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")


def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve nested configuration value.

    Args:
        config: Configuration dictionary
        *keys: Nested keys to retrieve (e.g., 'model', 'hidden_size')
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> config = {'model': {'hidden_size': 100}}
        >>> get_config_value(config, 'model', 'hidden_size')
        100
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
