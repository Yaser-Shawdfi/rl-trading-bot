"""Tests for trading environment module."""

import numpy as np
import pandas as pd
import pytest

from rl_trader.models.trading_env import TradingEnv


@pytest.fixture
def sample_env():
    """Create a small trading environment for testing."""
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.random.randint(1000, 10000, 200),
            "returns": np.concatenate([[0], np.diff(prices) / prices[:-1]]),
            "rsi": np.random.uniform(20, 80, 200),
            "macd": np.random.randn(200),
            "volatility": np.abs(np.random.randn(200)) * 0.02,
        },
        index=dates,
    )
    features = ["returns", "rsi", "macd", "volatility"]
    return TradingEnv(df, feature_cols=features, window_size=5, initial_balance=10000)


def test_env_reset(sample_env):
    obs, info = sample_env.reset()
    assert obs.shape == sample_env.observation_space.shape
    assert sample_env.net_worth == 10000
    assert sample_env.balance == 10000
    assert sample_env.shares_held == 0


def test_env_step(sample_env):
    sample_env.reset()
    action = np.array([0.5])  # Buy 50%
    obs, reward, terminated, truncated, info = sample_env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "net_worth" in info
    assert "action" in info
    assert "price" in info


def test_env_buy_action(sample_env):
    sample_env.reset()
    action = np.array([1.0])  # Buy everything
    _, _, _, _, info = sample_env.step(action)
    assert info["shares_held"] > 0
    assert info["balance"] < 10000  # Money spent


def test_env_sell_action(sample_env):
    sample_env.reset()
    # Buy first
    sample_env.step(np.array([1.0]))
    # Then sell
    _, _, _, _, info = sample_env.step(np.array([-1.0]))
    assert info["shares_held"] == 0  # All sold
    assert info["balance"] > 0  # Cash recovered


def test_env_hold_action(sample_env):
    sample_env.reset()
    obs1, _, _, _, _ = sample_env.step(np.array([0.0]))  # Hold
    assert sample_env.total_trades == 0  # No trades made


def test_env_episode_end(sample_env):
    sample_env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = sample_env.step(np.array([0.0]))
        steps += 1
    assert steps == sample_env.max_steps - sample_env.window_size


def test_env_action_space(sample_env):
    assert sample_env.action_space.shape == (1,)
    assert sample_env.action_space.low[0] == -1.0
    assert sample_env.action_space.high[0] == 1.0
