"""
RL Agent training using Stable-Baselines3 PPO.
Trains a Proximal Policy Optimization agent on the trading environment.
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
)
from data_loader import load_data, get_feature_columns, split_data
from trading_env import TradingEnv


class TrainingCallback(BaseCallback):
    """Custom callback to log training progress."""

    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_net_worths = []

    def _on_step(self):
        # Collect episode info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "net_worth" in info:
                self.episode_net_worths.append(info["net_worth"])

        if self.n_calls % self.check_freq == 0 and self.n_calls > 0:
            if self.episode_net_worths:
                avg_net_worth = np.mean(self.episode_net_worths[-50:])
                print(f"  Step {self.n_calls:>7d} | "
                      f"Avg Net Worth (last 50 episodes): ${avg_net_worth:>10.2f} | "
                      f"Episodes: {len(self.episode_net_worths)}")
        return True


def create_env(df, features):
    """Create a vectorized trading environment for training."""
    def make_env():
        return TradingEnv(df, feature_cols=features)
    env = DummyVecEnv([make_env])
    return env


def train_agent(df_train, features, total_timesteps=TOTAL_TIMESTEPS, save=True):
    """Train PPO agent on the trading environment."""
    print(f"\n{'='*60}")
    print(f"Training PPO Agent — {total_timesteps:,} timesteps")
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
        seed=RANDOM_SEED,
        ent_coef=0.01,       # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    callback = TrainingCallback(check_freq=10000)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "ppo_trading_agent"
        model.save(model_path)
        print(f"\n✅ Model saved to {model_path}")

    return model, env


def evaluate_agent(model, df_test, features):
    """Run the trained agent on test data and return results."""
    print(f"\n{'='*60}")
    print("Evaluating Agent on Test Data")
    print(f"{'='*60}")

    env = TradingEnv(df_test, feature_cols=features)
    obs, _ = env.reset()

    results = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        results.append({
            "step": info["step"],
            "net_worth": info["net_worth"],
            "balance": info["balance"],
            "shares_held": info["shares_held"],
            "price": info["price"],
            "action": info["action"],
            "reward": reward,
        })
        done = terminated or truncated

    results_df = pd.DataFrame(results)
    final_net_worth = results_df["net_worth"].iloc[-1]
    total_return = (final_net_worth - env.initial_balance) / env.initial_balance

    print(f"\nResults:")
    print(f"  Initial balance:    ${env.initial_balance:>10,.2f}")
    print(f"  Final net worth:    ${final_net_worth:>10,.2f}")
    print(f"  Total return:       {total_return:>10.2%}")
    print(f"  Total trades:       {env.total_trades:>10d}")
    print(f"  Total fees paid:    ${env.total_fees:>10.2f}")

    # Buy-and-hold comparison
    first_price = float(df_test.iloc[env.window_size]["close"])
    last_price = float(df_test.iloc[-1]["close"])
    bh_shares = env.initial_balance / first_price
    bh_final = bh_shares * last_price
    bh_return = (bh_final - env.initial_balance) / env.initial_balance
    print(f"\n  Buy & Hold return:  {bh_return:>10.2%}")
    print(f"  Buy & Hold final:  ${bh_final:>10,.2f}")
    print(f"  Agent vs B&H:       {(total_return - bh_return):>+10.2%}")

    return results_df, {
        "initial_balance": env.initial_balance,
        "final_net_worth": final_net_worth,
        "total_return": total_return,
        "total_trades": env.total_trades,
        "total_fees": env.total_fees,
        "bh_return": bh_return,
        "bh_final": bh_final,
        "outperformed": total_return > bh_return,
    }


if __name__ == "__main__":
    df = load_data()
    features = get_feature_columns(df)
    train_df, test_df = split_data(df)

    model, env = train_agent(train_df, features)
    results, summary = evaluate_agent(model, test_df, features)

    # Save results
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(MODELS_DIR / "backtest_results.csv", index=False)
    print(f"\n✅ Results saved to {MODELS_DIR / 'backtest_results.csv'}")