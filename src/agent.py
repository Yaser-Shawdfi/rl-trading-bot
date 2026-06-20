"""
Improved RL Agent:
  1. Ensemble training (3 seeds, averaged predictions)
  2. PPO with tuned hyperparameters for continuous action space
  3. Detailed backtesting with Sharpe ratio, max drawdown, win rate
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    MODELS_DIR, TOTAL_TIMESTEPS, LEARNING_RATE, BATCH_SIZE,
    GAMMA, GAE_LAMBDA, CLIP_RANGE, N_STEPS, RANDOM_SEED,
    ENSEMBLE_SEEDS,
)
from data_loader import load_data, get_feature_columns, split_data
from trading_env import TradingEnv


class TrainingCallback(BaseCallback):
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_net_worths = []
        self.episode_drawdowns = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "net_worth" in info:
                self.episode_net_worths.append(info["net_worth"])
                self.episode_drawdowns.append(info.get("max_drawdown", 0))

        if self.n_calls % self.check_freq == 0 and self.n_calls > 0:
            if self.episode_net_worths:
                avg_nw = np.mean(self.episode_net_worths[-50:])
                avg_dd = np.mean(self.episode_drawdowns[-50:]) if self.episode_drawdowns else 0
                print(f"  Step {self.n_calls:>7d} | "
                      f"Avg Net Worth: ${avg_nw:>10.2f} | "
                      f"Avg Max DD: {avg_dd:.2%} | "
                      f"Episodes: {len(self.episode_net_worths)}")
        return True


def create_env(df, features):
    def make_env():
        return TradingEnv(df, feature_cols=features)
    return DummyVecEnv([make_env])


def train_single_agent(df_train, features, seed, total_timesteps, save=True):
    """Train a single PPO agent with a given random seed."""
    print(f"\n{'='*60}")
    print(f"Training Agent (Seed={seed}) — {total_timesteps:,} timesteps")
    print(f"{'='*60}")

    env = create_env(df_train, features)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        verbose=0,
        seed=seed,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 64], vf=[128, 64])
        ),
    )

    callback = TrainingCallback(check_freq=10000)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"ppo_agent_seed_{seed}"
        model.save(model_path)
        print(f" Saved to {model_path}")

    return model


def train_ensemble(df_train, features, seeds=ENSEMBLE_SEEDS, total_timesteps=TOTAL_TIMESTEPS):
    """Train multiple agents with different seeds for ensemble."""
    models = []
    for seed in seeds:
        model = train_single_agent(df_train, features, seed, total_timesteps)
        models.append(model)

    print(f"\n Ensemble of {len(models)} agents trained (seeds: {seeds})")
    return models


def evaluate_ensemble(models, df_test, features):
    """Run ensemble of agents and average their position recommendations."""
    print(f"\n{'='*60}")
    print(f"Evaluating Ensemble ({len(models)} agents) on Test Data")
    print(f"{'='*60}")

    env = TradingEnv(df_test, feature_cols=features)
    obs, _ = env.reset()

    results = []
    done = False

    while not done:
        # Average actions across all models
        actions = []
        for model in models:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(float(action[0]) if hasattr(action, '__len__') else float(action))

        avg_action = np.mean(actions)
        action_array = np.array([avg_action], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action_array)
        results.append({
            "step": info["step"],
            "net_worth": info["net_worth"],
            "balance": info["balance"],
            "shares_held": info["shares_held"],
            "price": info["price"],
            "action": avg_action,
            "position_frac": info.get("position_frac", 0),
            "daily_return": info.get("daily_return", 0),
            "drawdown": info.get("drawdown", 0),
            "max_drawdown": info.get("max_drawdown", 0),
            "reward": reward,
        })
        done = terminated or truncated

    results_df = pd.DataFrame(results)
    final_nw = results_df["net_worth"].iloc[-1]
    total_return = (final_nw - env.initial_balance) / env.initial_balance

    # ─── Detailed Metrics ───────────────────────────────────────────────────
    daily_returns = results_df["daily_return"].values
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    max_dd = results_df["max_drawdown"].iloc[-1]
    n_trades = env.total_trades

    # Win rate (days with positive return)
    win_days = np.sum(daily_returns > 0)
    total_days = len(daily_returns)
    win_rate = win_days / total_days if total_days > 0 else 0

    # Profit factor
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gains / (losses + 1e-8)

    # Buy & hold
    first_price = float(df_test.iloc[env.window_size]["close"])
    last_price = float(df_test.iloc[-1]["close"])
    bh_return = (last_price - first_price) / first_price
    bh_final = env.initial_balance * (1 + bh_return)

    print(f"\n{'─'*50}")
    print(f"  ENSEMBLE RESULTS (3 agents averaged)")
    print(f"{'─'*50}")
    print(f"  Initial balance:      ${env.initial_balance:>10,.2f}")
    print(f"  Final net worth:      ${final_nw:>10,.2f}")
    print(f"  Total return:         {total_return:>10.2%}")
    print(f"  Sharpe ratio:         {sharpe:>10.2f}")
    print(f"  Max drawdown:         {max_dd:>10.2%}")
    print(f"  Win rate:             {win_rate:>10.2%}")
    print(f"  Profit factor:        {profit_factor:>10.2f}")
    print(f"  Total trades:         {n_trades:>10d}")
    print(f"  Total fees paid:      ${env.total_fees:>10,.2f}")
    print(f"{'─'*50}")
    print(f"  Buy & Hold return:    {bh_return:>10.2%}")
    print(f"  Buy & Hold final:    ${bh_final:>10,.2f}")
    print(f"  Agent vs B&H:         {total_return - bh_return:>+10.2%}")
    print(f"{'─'*50}")

    summary = {
        "initial_balance": env.initial_balance,
        "final_net_worth": final_nw,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": n_trades,
        "total_fees": env.total_fees,
        "bh_return": bh_return,
        "bh_final": bh_final,
        "outperformed": total_return > bh_return,
    }

    return results_df, summary


if __name__ == "__main__":
    df = load_data()
    features = get_feature_columns(df)
    train_df, test_df = split_data(df)

    # Train ensemble
    models = train_ensemble(train_df, features)

    # Evaluate
    results, summary = evaluate_ensemble(models, test_df, features)

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(MODELS_DIR / "backtest_results.csv", index=False)
    joblib.dump(summary, MODELS_DIR / "backtest_summary.joblib")
    print(f"\n Results saved to {MODELS_DIR / 'backtest_results.csv'}")