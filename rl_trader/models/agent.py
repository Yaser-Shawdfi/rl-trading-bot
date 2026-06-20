"""
Enterprise RL Agent + Ensemble + Backtesting Engine.
PPO ensemble with seed averaging, full backtest metrics, model persistence.
"""
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from rl_trader.config import AppConfig, AgentConfig
from rl_trader.models.trading_env import TradingEnv

logger = logging.getLogger("rl_trader.agent")


class _TrainingCallback(BaseCallback):
    def __init__(self, check_freq: int = 10000):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self.episode_nw: List[float] = []
        self.episode_dd: List[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "net_worth" in info:
                self.episode_nw.append(info["net_worth"])
                self.episode_dd.append(info.get("max_drawdown", 0))

        if self.n_calls % self.check_freq == 0 and self.n_calls > 0 and self.episode_nw:
            avg_nw = np.mean(self.episode_nw[-50:])
            avg_dd = np.mean(self.episode_dd[-50:]) if self.episode_dd else 0
            logger.info(
                f"Step {self.n_calls:>7d} | Avg NW: ${avg_nw:>10.2f} | "
                f"Avg DD: {avg_dd:.2%} | Episodes: {len(self.episode_nw)}"
            )
        return True


class TradingAgent:
    """Single PPO agent with train/evaluate/save/load."""

    def __init__(self, config: AgentConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.model: Optional[PPO] = None

    def _build_model(self, env) -> PPO:
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            verbose=0,
            seed=self.seed,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=dict(net_arch=dict(pi=[128, 64], vf=[128, 64])),
        )

    def train(self, df_train: pd.DataFrame, features: List[str],
              initial_balance: float = 10000, window_size: int = 10,
              fee: float = 0.001, max_position: float = 1.0,
              reward_config=None, save_dir: Optional[Path] = None) -> PPO:
        """Train the agent."""
        logger.info(f"Training PPO (seed={self.seed}, steps={self.config.total_timesteps:,})")

        def make_env():
            return TradingEnv(df_train, feature_cols=features,
                              initial_balance=initial_balance,
                              window_size=window_size, fee=fee,
                              max_position=max_position, reward_config=reward_config)
        env = DummyVecEnv([make_env])

        self.model = self._build_model(env)
        callback = _TrainingCallback()
        self.model.learn(total_timesteps=self.config.total_timesteps, callback=callback)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"ppo_agent_seed_{self.seed}"
            self.model.save(path)
            logger.info(f"Saved model to {path}")

        return self.model

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: Path):
        if self.model:
            self.model.save(path)

    def load(self, path: Path):
        self.model = PPO.load(path)


class EnsembleAgent:
    """Ensemble of PPO agents with averaged predictions."""

    def __init__(self, config: AgentConfig, seeds: Optional[List[int]] = None):
        self.config = config
        self.seeds = seeds or config.ensemble_seeds
        self.agents: List[TradingAgent] = []

    def train(self, df_train, features, **kwargs) -> List[PPO]:
        save_dir = kwargs.pop("save_dir", None)
        for seed in self.seeds:
            agent = TradingAgent(self.config, seed=seed)
            agent.train(df_train, features, save_dir=save_dir, **kwargs)
            self.agents.append(agent)
        logger.info(f"Ensemble of {len(self.agents)} agents trained (seeds: {self.seeds})")
        return [a.model for a in self.agents]

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Average predictions across all agents."""
        actions = []
        for agent in self.agents:
            action = agent.predict(obs, deterministic)
            val = float(action[0]) if hasattr(action, "__len__") else float(action)
            actions.append(val)
        avg = np.mean(actions)
        return np.array([avg], dtype=np.float32)

    def load(self, model_dir: Path):
        """Load all ensemble models from a directory."""
        for seed in self.seeds:
            path = model_dir / f"ppo_agent_seed_{seed}.zip"
            if path.exists():
                agent = TradingAgent(self.config, seed=seed)
                agent.load(path.with_suffix(""))
                self.agents.append(agent)
                logger.info(f"Loaded agent (seed={seed})")
        if not self.agents:
            raise FileNotFoundError(f"No models found in {model_dir}")


class BacktestEngine:
    """Run backtests and compute comprehensive metrics."""

    @staticmethod
    def run(agent, df_test: pd.DataFrame, features: List[str],
            initial_balance: float = 10000, window_size: int = 10,
            fee: float = 0.001, max_position: float = 1.0,
            reward_config=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run backtest and return results DataFrame + summary metrics."""
        logger.info("Running backtest on test data...")

        env = TradingEnv(df_test, feature_cols=features,
                         initial_balance=initial_balance,
                         window_size=window_size, fee=fee,
                         max_position=max_position, reward_config=reward_config)
        obs, _ = env.reset()

        results = []
        done = False
        while not done:
            if hasattr(agent, "agents"):  # EnsembleAgent
                action = agent.predict(obs, deterministic=True)
            else:  # Single model
                action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            results.append({
                "step": info["step"],
                "net_worth": info["net_worth"],
                "balance": info["balance"],
                "shares_held": info["shares_held"],
                "price": info["price"],
                "action": info["action"],
                "position_frac": info.get("position_frac", 0),
                "daily_return": info.get("daily_return", 0),
                "drawdown": info.get("drawdown", 0),
                "max_drawdown": info.get("max_drawdown", 0),
            })
            done = terminated or truncated

        results_df = pd.DataFrame(results)
        summary = BacktestEngine._compute_metrics(results_df, env, df_test)
        return results_df, summary

    @staticmethod
    def _compute_metrics(results: pd.DataFrame, env, df_test) -> Dict[str, Any]:
        final_nw = results["net_worth"].iloc[-1]
        total_return = (final_nw - env.initial_balance) / env.initial_balance
        daily_returns = results["daily_return"].values

        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        max_dd = results["max_drawdown"].iloc[-1]
        win_rate = np.mean(daily_returns > 0)

        gains = daily_returns[daily_returns > 0].sum()
        losses = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = gains / (losses + 1e-8)

        first_price = float(df_test.iloc[env.window_size]["close"])
        last_price = float(df_test.iloc[-1]["close"])
        bh_return = (last_price - first_price) / first_price

        summary = {
            "initial_balance": env.initial_balance,
            "final_net_worth": final_nw,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": env.total_trades,
            "total_fees": env.total_fees,
            "bh_return": bh_return,
            "bh_final": env.initial_balance * (1 + bh_return),
            "outperformed": total_return > bh_return,
        }
        logger.info(f"Backtest: Return={total_return:.2%} Sharpe={sharpe:.2f} "
                     f"MaxDD={max_dd:.2%} WinRate={win_rate:.1%} B&H={bh_return:.2%}")
        return summary