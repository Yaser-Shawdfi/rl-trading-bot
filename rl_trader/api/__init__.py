"""
RL Trader — REST API (FastAPI).
Endpoints: health, predict, backtest, train, market-data, metrics.
"""
from .main import app

__all__ = ["app"]