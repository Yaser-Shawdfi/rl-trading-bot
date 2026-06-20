"""
Configuration for RL Trading Bot — Improved Edition.
Key upgrades:
  1. Risk-adjusted reward (Sharpe-like + drawdown penalty)
  2. Continuous action space (0-100% position sizing)
  3. More technical indicators (ATR, Stochastic, OBV, VWAP, lags)
  4. Multi-seed ensemble training
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# ─── Trading Config ──────────────────────────────────────────────────────────
SYMBOL = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
INITIAL_BALANCE = 10000
WINDOW_SIZE = 10
FEE = 0.001               # 0.1% per trade
MAX_POSITION = 1.0          # Max 100% invested

# ─── RL Config ────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 200_000
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
N_STEPS = 2048
RANDOM_SEED = 42

# ─── Improved Reward Config ──────────────────────────────────────────────────
REWARD_TYPE = "risk_adjusted"     # "simple" or "risk_adjusted"
DRAWDOWN_PENALTY = 0.5            # Penalty weight for drawdowns
SHARPE_WINDOW = 20                # Window for rolling Sharpe ratio
VOLATILITY_PENALTY = 0.1          # Penalty for return volatility

# ─── Ensemble Config ─────────────────────────────────────────────────────────
ENSEMBLE_SEEDS = [42, 123, 777]   # Train 3 agents with different seeds

# ─── Action Space ────────────────────────────────────────────────────────────
# CONTINUOUS: Box([-1, 1]) → -1=sell everything, 0=hold, +1=buy everything
# The agent outputs a value in [-1, 1] representing target position %