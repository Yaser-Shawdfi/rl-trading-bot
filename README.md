# RL Trader — Enterprise Reinforcement Learning Trading Bot

A production-grade RL trading system with PPO ensemble, 42 technical indicators, REST API, Docker deployment, and CI/CD.

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RL TRADER v2.0                            │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Config   │   │   Data   │   │  Models  │   │     API      │ │
│  │ (YAML +   │   │ (42 ind. │   │ (PPO +   │   │ (FastAPI     │ │
│  │  env vars)│   │  multi-  │   │ ensemble)│   │  REST +      │ │
│  │           │   │  asset)  │   │          │   │  OpenAPI)    │ │
│  └─────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬───────┘ │
│        │               │              │                │         │
│        └───────────────┴──────────────┴────────────────┘         │
│                          │                                       │
│                   ┌────────────┐    ┌──────────┐                 │
│                   │  CLI Tool   │    │ Streamlit│                 │
│                   │ (rltrader)  │    │ Dashboard│                 │
│                   └────────────┘    └──────────┘                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Infrastructure                               │   │
│  │  Docker + CI/CD (GitHub Actions) + Tests (pytest)         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

##  Installation

### From Source (Development)
```bash
git clone https://github.com/Yaser-Shawdfi/rl-trading-bot.git
cd rl-trading-bot
pip install -e ".[dev]"
```

### With Docker (Production)
```bash
docker-compose up
# API at http://localhost:8000
# UI at http://localhost:8501
```

##  Usage

### CLI Commands
```bash
# Train the agent ensemble
rltrader train --symbol AAPL --timesteps 200000

# Run backtest
rltrader backtest --symbol AAPL

# Get prediction for day 100
rltrader predict --symbol AAPL --day 100

# Start REST API
rltrader api --host 0.0.0.0 --port 8000

# Start Streamlit dashboard
rltrader ui --port 8501
```

### REST API
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Get market data
curl http://localhost:8000/api/v1/market/AAPL?limit=10

# Get prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "day_index": 100}'

# Run backtest
curl http://localhost:8000/api/v1/backtest?symbol=AAPL

# Trigger training (async)
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "timesteps": 200000}'
```

API docs available at: `http://localhost:8000/docs`

##  Testing
```bash
# Run all tests
pytest rl_trader/tests/ -v

# With coverage
pytest rl_trader/tests/ --cov=rl_trader --cov-report=term
```

##  Project Structure
```
rl-trading-bot/
├── rl_trader/                    # Python package
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point (rltrader command)
│   ├── config/
│   │   ├── __init__.py           # Config dataclasses + loader
│   │   └── settings.yaml         # YAML configuration
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py        # Multi-asset data + 42 indicators
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trading_env.py        # Gymnasium environment
│   │   └── agent.py              # PPO + Ensemble + BacktestEngine
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py               # FastAPI REST API
│   └── tests/
│       ├── test_data.py          # Data loader tests
│       ├── test_env.py           # Environment tests
│       └── test_api.py           # API endpoint tests
├── app/app.py                    # Streamlit dashboard
├── data/                         # Cached market data
├── models/                       # Saved model artifacts
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # API + UI + test services
├── pyproject.toml                # Package config + dependencies
├── .github/workflows/ci.yml     # GitHub Actions CI/CD
└── README.md
```

##  Configuration

All config in `rl_trader/config/settings.yaml`. Override with env vars:
```bash
export RLTRADER__TRADING__SYMBOL=MSFT
export RLTRADER__AGENT__TOTAL_TIMESTEPS=500000
export RLTRADER__LOGGING__LEVEL=DEBUG
```

##  License

MIT