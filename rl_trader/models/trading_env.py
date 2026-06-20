"""
Enterprise trading environment — continuous action space + risk-adjusted reward.
"""

import logging

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box

from rl_trader.config import RewardConfig

logger = logging.getLogger("rl_trader.env")


class TradingEnv(Env):
    """
    Custom trading environment with continuous action space.

    Action: [-1, 1] → target position fraction [0, max_position]
    Reward: Risk-adjusted (Sharpe-like + drawdown + volatility penalties)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        initial_balance: float = 10000,
        window_size: int = 10,
        fee: float = 0.001,
        max_position: float = 1.0,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols or self._auto_features()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.fee = fee
        self.max_position = max_position
        self.reward_config = reward_config or RewardConfig()

        # Normalize features
        self._feature_mean = self.df[self.feature_cols].mean()
        self._feature_std = self.df[self.feature_cols].std().replace(0, 1)
        self._normed = (
            ((self.df[self.feature_cols] - self._feature_mean) / self._feature_std)
            .fillna(0)
            .values.astype(np.float32)
        )

        # Spaces
        n_obs = len(self.feature_cols) * window_size + 2
        self.observation_space = Box(low=-10, high=10, shape=(n_obs,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._reset_state()

    def _auto_features(self) -> list[str]:
        exclude = {"open", "high", "low", "close", "volume", "adj close"}
        return [c for c in self.df.columns if c.lower() not in exclude]

    def _reset_state(self):
        self.current_step = self.window_size
        self.max_steps = len(self.df) - 1
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        self.total_trades = 0
        self.total_fees = 0.0
        self.returns_history: list = []
        self.max_drawdown = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        window = self._normed[start:end].flatten()

        nw_ratio = np.array([self.net_worth / self.initial_balance], dtype=np.float32)
        price = float(self.df.iloc[self.current_step]["close"])
        pos_frac = np.array([(self.shares_held * price) / max(self.net_worth, 1)], dtype=np.float32)

        return np.concatenate([window, nw_ratio, pos_frac]).astype(np.float32)

    def step(self, action):
        price = float(self.df.iloc[self.current_step]["close"])
        action_val = float(np.clip(action[0] if hasattr(action, "__len__") else action, -1, 1))

        # Compute target position
        current_pos_val = self.shares_held * price
        current_pos_frac = current_pos_val / max(self.net_worth, 1)
        if action_val >= 0:
            target_frac = action_val * self.max_position
        else:
            target_frac = current_pos_frac * (1 + action_val)

        target_val = target_frac * self.net_worth
        trade_value = target_val - current_pos_val

        if abs(trade_value) > 1:
            if trade_value > 0:  # BUY
                shares = trade_value / (price * (1 + self.fee))
                cost = shares * price
                fee = cost * self.fee
                self.balance -= cost + fee
                self.shares_held += shares
            else:  # SELL
                shares = min(abs(trade_value) / price, self.shares_held)
                revenue = shares * price
                fee = revenue * self.fee
                self.balance += revenue - fee
                self.shares_held -= shares
            self.total_fees += fee
            self.total_trades += 1

        self.current_step += 1
        if self.current_step >= self.max_steps and self.shares_held > 0:
            next_price = float(self.df.iloc[self.current_step]["close"])
            revenue = self.shares_held * next_price
            fee = revenue * self.fee
            self.balance += revenue - fee
            self.total_fees += fee
            self.shares_held = 0

        next_price = float(self.df.iloc[min(self.current_step, self.max_steps)]["close"])
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * next_price

        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        drawdown = (self.peak_net_worth - self.net_worth) / max(self.peak_net_worth, 1)
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        daily_return = (self.net_worth - self.prev_net_worth) / max(self.prev_net_worth, 1)
        self.returns_history.append(daily_return)

        reward = self._compute_reward(daily_return, drawdown, trade_value)

        terminated = self.current_step >= self.max_steps
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
        return self._get_obs(), float(reward), terminated, False, info

    def _compute_reward(self, daily_return: float, drawdown: float, trade_value: float) -> float:
        """Risk-adjusted reward with Sharpe-like component + penalties."""
        rc = self.reward_config

        if rc.type == "simple":
            return daily_return - (self.fee * 0.5 if abs(trade_value) > 1 else 0)

        # Risk-adjusted
        recent = self.returns_history[-rc.sharpe_window :]
        mean_r = np.mean(recent)
        std_r = np.std(recent) + 1e-8

        sharpe = mean_r / std_r * np.sqrt(252) / 50
        dd_penalty = rc.drawdown_penalty * drawdown
        vol_penalty = rc.volatility_penalty * std_r
        trade_penalty = self.fee * 0.5 if abs(trade_value) > 1 else 0

        return sharpe - dd_penalty - vol_penalty - trade_penalty
