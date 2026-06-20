"""
Custom OpenAI Gymnasium trading environment.
The agent observes market data and can BUY, HOLD, or SELL.
Reward is based on portfolio value change (profit/loss).
"""
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from config import INITIAL_BALANCE, WINDOW_SIZE, FEE


class TradingEnv(Env):
    """
    Custom trading environment compatible with OpenAI Gymnasium.

    Observation: Window of past N days of market features + current position
    Action: 0=SELL, 1=HOLD, 2=BUY
    Reward: Change in portfolio value (profit/loss)
    """

    def __init__(self, df, feature_cols=None, initial_balance=INITIAL_BALANCE,
                 window_size=WINDOW_SIZE, fee=FEE):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols or self._auto_features()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.fee = fee

        # Normalize features for the observation
        self.feature_mean = self.df[self.feature_cols].mean()
        self.feature_std = self.df[self.feature_cols].std().replace(0, 1)
        self.normalized_features = (
            (self.df[self.feature_cols] - self.feature_mean) / self.feature_std
        ).fillna(0).values.astype(np.float32)

        # Spaces
        n_features = len(self.feature_cols) * window_size + 2  # +2 for position info
        self.observation_space = Box(
            low=-10, high=10, shape=(n_features,), dtype=np.float32
        )
        self.action_space = Discrete(3)  # 0=SELL, 1=HOLD, 2=BUY

        # Episode state
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.prev_net_worth = initial_balance
        self.total_trades = 0
        self.total_fees = 0.0

    def _auto_features(self):
        """Auto-detect feature columns (exclude OHLCV)."""
        exclude = {"open", "high", "low", "close", "volume", "adj close"}
        return [c for c in self.df.columns if c.lower() not in exclude]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.total_trades = 0
        self.total_fees = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        """Build observation: window of features + position info."""
        start = self.current_step - self.window_size
        end = self.current_step

        window = self.normalized_features[start:end].flatten()

        # Add position info: current net worth ratio and shares held ratio
        net_worth_ratio = np.array([self.net_worth / self.initial_balance], dtype=np.float32)
        shares_ratio = np.array([self.shares_held / 100.0], dtype=np.float32)

        obs = np.concatenate([window, net_worth_ratio, shares_ratio])
        return obs.astype(np.float32)

    def step(self, action):
        current_price = float(self.df.iloc[self.current_step]["close"])

        # Execute action
        if action == 2:  # BUY
            # Invest all available balance
            if self.balance > 0:
                shares_to_buy = self.balance / (current_price * (1 + self.fee))
                cost = shares_to_buy * current_price
                fee = shares_to_buy * current_price * self.fee
                self.balance -= (cost + fee)
                self.shares_held += shares_to_buy
                self.total_trades += 1
                self.total_fees += fee

        elif action == 0:  # SELL
            # Sell all shares
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                fee = revenue * self.fee
                self.balance += (revenue - fee)
                self.total_fees += fee
                self.shares_held = 0
                self.total_trades += 1

        # action == 1 (HOLD) → do nothing

        # Advance to next step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            # End of episode — liquidate position
            if self.shares_held > 0:
                current_price = float(self.df.iloc[self.current_step]["close"])
                revenue = self.shares_held * current_price
                fee = revenue * self.fee
                self.balance += (revenue - fee)
                self.total_fees += fee
                self.shares_held = 0

        # Calculate new net worth
        if self.current_step < self.max_steps:
            next_price = float(self.df.iloc[self.current_step]["close"])
        else:
            next_price = current_price

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * next_price

        # Reward = change in net worth (scaled)
        reward = (self.net_worth - self.prev_net_worth) / self.initial_balance

        # Small penalty for trading (encourages efficiency)
        if action in [0, 2]:
            reward -= self.fee * 0.1

        # Episode is done if we reach the end
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "price": next_price,
            "action": action,
            "total_trades": self.total_trades,
            "total_fees": self.total_fees,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """Print current state."""
        action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        price = float(self.df.iloc[self.current_step]["close"])
        print(f"Step {self.current_step:4d} | Price: ${price:>10.2f} | "
              f"Net Worth: ${self.net_worth:>10.2f} | "
              f"Balance: ${self.balance:>10.2f} | "
              f"Shares: {self.shares_held:>8.2f} | "
              f"Trades: {self.total_trades}")


if __name__ == "__main__":
    from data_loader import load_data, get_feature_columns, split_data

    df = load_data()
    features = get_feature_columns(df)
    train, test = split_data(df)

    env = TradingEnv(train, feature_cols=features)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Features: {features}")

    obs, _ = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial net worth: ${env.net_worth:,.2f}")

    # Run random actions
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if i % 20 == 0:
            env.render()
        if terminated:
            break

    print(f"\nAfter 100 steps — Net worth: ${env.net_worth:,.2f} | "
          f"Trades: {env.total_trades} | Total reward: {total_reward:.4f}")