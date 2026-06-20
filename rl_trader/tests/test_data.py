"""Tests for data loader module."""

import pandas as pd
import pytest

from rl_trader.data import DataLoader, get_feature_columns, split_data


@pytest.fixture
def sample_df():
    """Create a small sample dataframe mimicking OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "open": range(100),
            "high": range(100, 200),
            "low": range(0, 100)[::-1],
            "close": range(50, 150),
            "volume": range(1000, 1100),
        },
        index=dates,
    )


def test_get_feature_columns(sample_df):
    features = get_feature_columns(sample_df)
    assert "open" not in features
    assert "high" not in features
    assert "low" not in features
    assert "close" not in features
    assert "volume" not in features
    assert len(features) == 0  # No feature columns in this sample


def test_split_data(sample_df):
    train, test = split_data(sample_df, train_ratio=0.7)
    assert len(train) == 70
    assert len(test) == 30
    assert train.index[-1] < test.index[0]  # Chronological split


def test_data_loader_init():
    loader = DataLoader(symbol="AAPL", cache_dir="data")
    assert loader.symbol == "AAPL"
    assert loader.start == "2015-01-01"
