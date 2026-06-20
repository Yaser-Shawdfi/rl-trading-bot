"""
Configuration for RL Trading Bot.
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
INITIAL_BALANCE = 10000  # $10,000 starting capital
WINDOW_SIZE = 10          # Number of past days the agent sees
FEE = 0.001              # 0.1% transaction fee per trade

# ─── RL Config ────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 200_000
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
GAMMA = 0.99             # Discount factor
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
N_STEPS = 2048
RANDOM_SEED = 42

# ─── Action Space ────────────────────────────────────────────────────────────
# 0 = SELL (go short / liquidate position)
# 1 = HOLD (do nothing)
# 2 = BUY (go long / invest all in)