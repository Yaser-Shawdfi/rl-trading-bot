"""
Improved Trading Environment with:
  1. Continuous action space (position sizing 0-100%)
  2. Risk-adjusted reward (Sharpe-like + drawdown penalty)
  3. Transaction cost optimization
  4. Peak tracking for drawdown calculation
"""
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box

from config import (
    INITIAL_BALANCE, WINDOW_SIZE, FEE, MAX_POSITION,
    REWARD_TYPE, DRAWDOWN_PENALTY, SHARPE_WINDOW,
    VOLATILITY_PENALTY,
)


class TradingEnv(Env):
    """
    Improved trading environment with continuous action space.

    Observation: Window of past N days of market features + position info
    Action: [-1, 1] → -1=sell everything, 0=hold, +1=invest everything
    Reward: Risk-adjusted return (Sharpe-like) with drawdown penalty
    """

    def __init__(self, df, feature_cols=None, initial_balance=INITIAL_BALANCE,
                 window_size=WINDOW_SIZE, fee=FEE):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols or self._auto_features()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.fee = fee

        # Normalize features
        self.feature_mean = self.df[self.feature_cols].mean()
        self.feature_std = self.df[self.feature_cols].std().replace(0, 1)
        self.normalized_features = (
            (self.df[self.feature_cols] - self.feature_mean) / self.feature_std
        ).fillna(0).values.astype(np.float32)

        # Spaces — CONTINUOUS action space
        n_features = len(self.feature_cols) * window_size + 2
        self.observation_space = Box(low=-10, high=10, shape=(n_features,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Episode state
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.prev_net_worth = initial_balance
        self.peak_net_worth = initial_balance
        self.total_trades = 0
        self.total_fees = 0.0
        self.returns_history = []
        self.max_drawdown = 0.0

    def _auto_features(self):
        exclude = {"open", "high", "low", "close", "volume", "adj close"}
        return [c for c in self.df.columns if c.lower() not in exclude]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        self.total_trades = 0
        self.total_fees = 0.0
        self.returns_history = []
        self.max_drawdown = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.normalized_features[start:end].flatten()

        net_worth_ratio = np.array([self.net_worth / self.initial_balance], dtype=np.float32)
        current_position = np.array([
            (self.shares_held * self.df.iloc[self.current_step]["close"]) / max(self.net_worth, 1)
        ], dtype=np.float32)

        obs = np.concatenate([window, net_worth_ratio, current_position])
        return obs.astype(np.float32)

    def step(self, action):
        current_price = float(self.df.iloc[self.current_step]["close"])

        # ─── CONTINUOUS ACTION: [-1, 1] → target position [0, 1] ──────────
        # action = -1 → sell everything (target position = 0)
        # action =  0 → hold (no change)
        # action = +1 → buy everything (target position = 1)
        action_val = float(np.clip(action[0] if hasattr(action, '__len__') else action, -1, 1))

        # Current position value as fraction of net worth
        current_position_value = self.shares_held * current_price
        current_position_frac = current_position_value / max(self.net_worth, 1)

        # Target position fraction (0 to 1)
        if action_val >= 0:
            target_position_frac = action_val * MAX_POSITION
        else:
            target_position_frac = current_position_frac * (1 + action_val)  # Reduce position

        target_position_value = target_position_frac * self.net_worth

        # Calculate trade needed
        trade_value = target_position_value - current_position_value

        if abs(trade_value) > 1:  # Only trade if meaningful change
            if trade_value > 0:  # BUY
                shares_to_buy = trade_value / (current_price * (1 + self.fee))
                cost = shares_to_buy * current_price
                fee = cost * self.fee
                self.balance -= (cost + fee)
                self.shares_held += shares_to_buy
            else:  # SELL
                shares_to_sell = min(abs(trade_value) / current_price, self.shares_held)
                revenue = shares_to_sell * current_price
                fee = revenue * self.fee
                self.balance += (revenue - fee)
                self.shares_held -= shares_to_sell

            self.total_fees += fee
            self.total_trades += 1

        # Advance
        self.current_step += 1
        if self.current_step >= self.max_steps:
            if self.shares_held > 0:  # Liquidate at end
                next_price = float(self.df.iloc[self.current_step]["close"])
                revenue = self.shares_held * next_price
                fee = revenue * self.fee
                self.balance += (revenue - fee)
                self.total_fees += fee
                self.shares_held = 0

        # Update net worth
        if self.current_step < self.max_steps:
            next_price = float(self.df.iloc[self.current_step]["close"])
        else:
            next_price = current_price

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * next_price

        # Track peak and drawdown
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # ─── IMPROVED REWARD: Risk-Adjusted ────────────────────────────────
        daily_return = (self.net_worth - self.prev_net_worth) / max(self.prev_net_worth, 1)
        self.returns_history.append(daily_return)

        if REWARD_TYPE == "risk_adjusted" and len(self.returns_history) > 1:
            # Rolling Sharpe-like ratio: reward / volatility
            recent_returns = self.returns_history[-SHARPE_WINDOW:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns) + 1e-8

            # Sharpe-like reward (higher = better risk-adjusted return)
            sharpe_reward = mean_return / std_return * np.sqrt(252) / 50  # Annualized, scaled

            # Drawdown penalty
            dd_penalty = DRAWDOWN_PENALTY * drawdown

            # Volatility penalty
            vol_penalty = VOLATILITY_PENALTY * std_return

            # Transaction cost penalty
            trade_penalty = self.fee * 0.5 if abs(trade_value) > 1 else 0

            reward = sharpe_reward - dd_penalty - vol_penalty - trade_penalty
        else:
            # Simple reward (fallback)
            reward = daily_return - (self.fee * 0.5 if abs(trade_value) > 1 else 0)

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "price": next_price,
            "action": action_val,
            "position_frac": float(self.shares_held * next_price / max(self.net_worth, 1)),
            "daily_return": daily_return,
            "drawdown": drawdown,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "total_fees": self.total_fees,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        price = float(self.df.iloc[self.current_step]["close"])
        pos = float(self.shares_held * price / max(self.net_worth, 1))
        print(f"Step {self.current_step:4d} | Price: ${price:>10.2f} | "
              f"Net Worth: ${self.net_worth:>10.2f} | "
              f"Position: {pos:>5.1%} | "
              f"DD: {self.max_drawdown:>5.1%} | "
              f"Trades: {self.total_trades}")


if __name__ == "__main__":
    from data_loader import load_data, get_feature_columns, split_data

    df = load_data()
    features = get_feature_columns(df)
    train, test = split_data(df)

    env = TradingEnv(train, feature_cols=features)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Features: {len(features)}")

    obs, _ = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")

    # Run random actions
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 20 == 0:
            env.render()
        if terminated:
            break

    print(f"\nAfter 100 steps — Net worth: ${env.net_worth:,.2f} | "
          f"Max DD: {env.max_drawdown:.2%} | Trades: {env.total_trades}")