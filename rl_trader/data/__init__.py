"""
RL Trader — Data Module.
Multi-asset data loading + 42 technical indicators + caching.
"""

from .data_loader import DataLoader, get_feature_columns, load_data, split_data

__all__ = ["DataLoader", "load_data", "get_feature_columns", "split_data"]
