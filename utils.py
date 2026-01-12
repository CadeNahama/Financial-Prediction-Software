"""
Utility functions for path resolution and configuration loading.
"""
import os
from pathlib import Path
import yaml


def get_project_root():
    """Get project root directory."""
    return Path(__file__).parent


def get_config_path(config_name):
    """
    Get absolute path to a config file.
    
    Args:
        config_name: Name of config file (e.g., 'config.yaml', 'tickers.yaml')
    
    Returns:
        Absolute path to config file
    """
    return get_project_root() / 'configs' / config_name


def load_config(config_name):
    """
    Load a YAML config file.
    
    Args:
        config_name: Name of config file
    
    Returns:
        Dictionary with config data
    """
    config_path = get_config_path(config_name)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

