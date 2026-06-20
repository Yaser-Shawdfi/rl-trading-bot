"""
FastAPI REST API for RL Trader.
Endpoints:
  GET  /api/v1/health          — health check
  GET  /api/v1/market/{symbol} — market data + indicators
  POST /api/v1/predict          — get agent prediction for a given day
  GET  /api/v1/backtest         — run backtest + get metrics
  GET  /api/v1/metrics          — get saved backtest metrics
  POST /api/v1/train             — trigger training (async)
"""
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from rl_trader.config import AppConfig
from rl_trader.data import load_data, get_feature_columns, split_data
from rl_trader.models import EnsembleAgent, BacktestEngine, TradingEnv

logger = logging.getLogger("rl_trader.api")

app = FastAPI(
    title="RL Trader API",
    description="Reinforcement Learning Trading Bot — Enterprise REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Globals ────────────────────────────────────────────────────────────────
config = AppConfig.load()
models_dir = Path("models")
data_cache: dict = {}
ensemble: Optional[EnsembleAgent] = None
backtest_summary: Optional[dict] = None
backtest_results: Optional[pd.DataFrame] = None


def _get_data(symbol: str = "AAPL") -> pd.DataFrame:
    if symbol not in data_cache:
        data_cache[symbol] = load_data(symbol=symbol)
    return data_cache[symbol]


def _load_ensemble() -> EnsembleAgent:
    global ensemble
    if ensemble is None:
        ensemble = EnsembleAgent(config.agent, seeds=config.agent.ensemble_seeds)
        ensemble.load(models_dir)
    return ensemble


# ─── Models (Pydantic) ───────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    symbol: str = "AAPL"
    day_index: int = 0


class PredictResponse(BaseModel):
    action: float
    action_label: str
    position_target: float
    current_price: float
    net_worth: float


class BacktestResponse(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    bh_return: float
    outperformed: bool
    final_net_worth: float


class TrainRequest(BaseModel):
    symbol: str = "AAPL"
    timesteps: int = 200000


class TrainResponse(BaseModel):
    status: str
    message: str


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/v1/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0", "model_loaded": ensemble is not None}


@app.get("/api/v1/market/{symbol}")
async def get_market_data(symbol: str, limit: int = 100):
    """Get market data with technical indicators."""
    df = _get_data(symbol)
    recent = df.tail(limit)
    return {
        "symbol": symbol,
        "total_rows": len(df),
        "returned_rows": len(recent),
        "data": recent.reset_index().rename(columns={"index": "date"}).to_dict(orient="records"),
    }


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Get trading action for a specific day."""
    ens = _load_ensemble()
    df = _get_data(req.symbol)
    features = get_feature_columns(df)
    _, test_df = split_data(df)

    env = TradingEnv(test_df, feature_cols=features,
                     initial_balance=config.trading.initial_balance,
                     window_size=config.trading.window_size,
                     fee=config.trading.fee)
    obs, _ = env.reset()

    # Advance to requested day
    for _ in range(min(req.day_index, env.max_steps - env.window_size - 1)):
        action = ens.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break

    action = ens.predict(obs, deterministic=True)
    action_val = float(action[0])
    price = float(test_df.iloc[env.current_step]["close"])

    return PredictResponse(
        action=action_val,
        action_label="BUY" if action_val > 0.1 else ("SELL" if action_val < -0.1 else "HOLD"),
        position_target=abs(action_val) * 100,
        current_price=price,
        net_worth=env.net_worth,
    )


@app.get("/api/v1/backtest", response_model=BacktestResponse)
async def run_backtest(symbol: str = "AAPL"):
    """Run backtest and return metrics."""
    global backtest_summary, backtest_results

    ens = _load_ensemble()
    df = _get_data(symbol)
    features = get_feature_columns(df)
    _, test_df = split_data(df)

    results, summary = BacktestEngine.run(
        ens, test_df, features,
        initial_balance=config.trading.initial_balance,
        window_size=config.trading.window_size,
        fee=config.trading.fee,
        max_position=config.trading.max_position,
        reward_config=config.reward,
    )

    backtest_summary = summary
    backtest_results = results
    models_dir.mkdir(exist_ok=True)
    results.to_csv(models_dir / "backtest_results.csv", index=False)
    joblib.dump(summary, models_dir / "backtest_summary.joblib")

    return BacktestResponse(**{k: v for k, v in summary.items() if k in BacktestResponse.model_fields})


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get saved backtest metrics."""
    summary_path = models_dir / "backtest_summary.joblib"
    if summary_path.exists():
        return joblib.load(summary_path)
    raise HTTPException(status_code=404, detail="No backtest results found. Run /api/v1/backtest first.")


@app.post("/api/v1/train", response_model=TrainResponse)
async def train_agent(req: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger async training (returns immediately, runs in background)."""

    def _train():
        df = _get_data(req.symbol)
        features = get_feature_columns(df)
        train_df, _ = split_data(df)
        ens = EnsembleAgent(config.agent, seeds=config.agent.ensemble_seeds)
        ens.train(train_df, features, save_dir=models_dir,
                  initial_balance=config.trading.initial_balance,
                  window_size=config.trading.window_size,
                  fee=config.trading.fee,
                  max_position=config.trading.max_position,
                  reward_config=config.reward)
        logger.info("Training complete!")

    background_tasks.add_task(_train)
    return TrainResponse(
        status="started",
        message=f"Training {len(config.agent.ensemble_seeds)} agents on {req.symbol} ({req.timesteps:,} steps each). Check logs."
    )


@app.get("/")
async def root():
    return {"name": "RL Trader API", "version": "2.0.0", "docs": "/docs"}