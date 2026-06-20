"""
RL Trader — CLI Entry Point.
Usage:
  rltrader train --symbol AAPL --timesteps 200000
  rltrader backtest --symbol AAPL
  rltrader predict --symbol AAPL --day 100
  rltrader api --host 0.0.0.0 --port 8000
  rltrader ui --port 8501
"""
import argparse
import sys
import logging
from pathlib import Path

from rl_trader.config import AppConfig, setup_logging


def cmd_train(args):
    from rl_trader.data import load_data, get_feature_columns, split_data
    from rl_trader.models import EnsembleAgent

    config = AppConfig.load()
    setup_logging(config.raw)

    df = load_data(symbol=args.symbol)
    features = get_feature_columns(df)
    train_df, _ = split_data(df)

    ensemble = EnsembleAgent(config.agent, seeds=config.agent.ensemble_seeds)
    ensemble.train(train_df, features, save_dir=Path("models"),
                   initial_balance=config.trading.initial_balance,
                   window_size=config.trading.window_size,
                   fee=config.trading.fee,
                   max_position=config.trading.max_position,
                   reward_config=config.reward)


def cmd_backtest(args):
    from rl_trader.data import load_data, get_feature_columns, split_data
    from rl_trader.models import EnsembleAgent, BacktestEngine

    config = AppConfig.load()
    setup_logging(config.raw)

    df = load_data(symbol=args.symbol)
    features = get_feature_columns(df)
    _, test_df = split_data(df)

    ensemble = EnsembleAgent(config.agent, seeds=config.agent.ensemble_seeds)
    ensemble.load(Path("models"))

    results, summary = BacktestEngine.run(
        ensemble, test_df, features,
        initial_balance=config.trading.initial_balance,
        window_size=config.trading.window_size,
        fee=config.trading.fee,
        max_position=config.trading.max_position,
        reward_config=config.reward,
    )

    print(f"\n{'='*50}")
    print(f"Backtest Results for {args.symbol}")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k:>20s}: {v}")


def cmd_predict(args):
    from rl_trader.data import load_data, get_feature_columns, split_data
    from rl_trader.models import EnsembleAgent, TradingEnv

    config = AppConfig.load()
    setup_logging(config.raw)

    df = load_data(symbol=args.symbol)
    features = get_feature_columns(df)
    _, test_df = split_data(df)

    ensemble = EnsembleAgent(config.agent, seeds=config.agent.ensemble_seeds)
    ensemble.load(Path("models"))

    env = TradingEnv(test_df, feature_cols=features,
                     initial_balance=config.trading.initial_balance,
                     window_size=config.trading.window_size)
    obs, _ = env.reset()

    for _ in range(args.day):
        action = ensemble.predict(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            break

    action = ensemble.predict(obs)
    action_val = float(action[0])
    label = "BUY" if action_val > 0.1 else ("SELL" if action_val < -0.1 else "HOLD")
    price = float(test_df.iloc[env.current_step]["close"])

    print(f"\nPrediction for {args.symbol} (day {args.day}):")
    print(f"  Action: {label} ({action_val:.2f})")
    print(f"  Position target: {abs(action_val)*100:.0f}%")
    print(f"  Current price: ${price:.2f}")
    print(f"  Net worth: ${env.net_worth:,.2f}")


def cmd_api(args):
    import uvicorn
    setup_logging(AppConfig.load().raw)
    uvicorn.run("rl_trader.api:app", host=args.host, port=args.port, reload=False)


def cmd_ui(args):
    import subprocess
    subprocess.run([
        "streamlit", "run", "app/app.py",
        "--server.port", str(args.port),
        "--server.headless", "true",
    ])


def main():
    parser = argparse.ArgumentParser(
        prog="rltrader",
        description="RL Trader — Enterprise Reinforcement Learning Trading Bot",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    p_train = subparsers.add_parser("train", help="Train the RL agent ensemble")
    p_train.add_argument("--symbol", default="AAPL", help="Stock symbol")
    p_train.add_argument("--timesteps", type=int, default=200000, help="Training timesteps")
    p_train.set_defaults(func=cmd_train)

    # Backtest
    p_bt = subparsers.add_parser("backtest", help="Run backtest on test data")
    p_bt.add_argument("--symbol", default="AAPL", help="Stock symbol")
    p_bt.set_defaults(func=cmd_backtest)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Get prediction for a specific day")
    p_pred.add_argument("--symbol", default="AAPL", help="Stock symbol")
    p_pred.add_argument("--day", type=int, default=0, help="Day index in test set")
    p_pred.set_defaults(func=cmd_predict)

    # API
    p_api = subparsers.add_parser("api", help="Start the REST API server")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.set_defaults(func=cmd_api)

    # UI
    p_ui = subparsers.add_parser("ui", help="Start the Streamlit dashboard")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()