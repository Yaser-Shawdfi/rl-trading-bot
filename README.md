# 🤖 RL Trading Bot

A Reinforcement Learning trading bot that learns to trade stocks using PPO (Proximal Policy Optimization). The agent observes market data and technical indicators, then decides to **BUY**, **HOLD**, or **SELL** to maximize portfolio returns.

## 📊 Project Overview

| Component | Details |
|-----------|---------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Library | Stable-Baselines3 |
| Environment | Custom OpenAI Gymnasium |
| Data | Yahoo Finance API (AAPL, 2015-2024) |
| Data Points | 2,466 trading days |
| Features | 15 technical indicators |
| Action Space | 3 (Sell, Hold, Buy) |
| Observation | 10-day window × 15 features + position info |
| Initial Capital | $10,000 |

## 🧠 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Market Data  │────▶│   RL Agent   │────▶│   Action    │
│ (15 features)│     │   (PPO)      │     │ Buy/Hold/   │
│ (10-day win) │     │              │     │ Sell        │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌──────────────┐              │
                    │   Reward     │◀─────────────┘
                    │ (P/L change) │
                    └──────────────┘
```

1. **Observation**: Agent sees 10 days of normalized market features + current position
2. **Action**: Agent chooses BUY (invest all), HOLD (do nothing), or SELL (liquidate)
3. **Reward**: Change in portfolio net worth (profit/loss), minus trading fees
4. **Training**: PPO learns optimal policy over 200K timesteps

## 📁 Project Structure

```
rl-trading-bot/
├── config.py                # Configuration (hyperparameters, trading params)
├── requirements.txt         # Python dependencies
├── src/
│   ├── data_loader.py       # Yahoo Finance data + 15 technical indicators
│   ├── trading_env.py       # Custom Gymnasium trading environment
│   └── agent.py             # PPO training + evaluation
├── app/
│   └── app.py               # 4-page Streamlit dashboard
├── data/                    # Cached market data
├── models/                  # Saved model + backtest results
└── reports/                 # Generated charts
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Market Data
```bash
python src/data_loader.py
```

### 3. Train the RL Agent
```bash
PYTHONPATH=. python src/agent.py
```

### 4. Launch the Dashboard
```bash
streamlit run app/app.py --server.port 8501
```

## 📈 Technical Indicators Used

| Indicator | Description |
|-----------|-------------|
| Returns | Daily percentage change |
| SMA 10/30/50 | Simple Moving Averages |
| EMA 12/26 | Exponential Moving Averages |
| MACD | Moving Average Convergence Divergence |
| RSI | Relative Strength Index (14-day) |
| Bollinger Bands | 20-day ± 2σ |
| Volatility | 20-day rolling std of returns |
| Volume Ratio | Volume / 20-day average volume |

## 🎯 Action Space

| Action | Code | Behavior |
|--------|------|----------|
| SELL | 0 | Liquidate all shares → cash |
| HOLD | 1 | Do nothing |
| BUY | 2 | Invest all cash → shares |

## 📊 Dashboard Pages

1. **📊 Overview** — Price history, train/test split, model status
2. **📈 Backtest Results** — Portfolio value vs buy & hold, trading actions
3. **🔮 Live Prediction** — Run agent on test data in real-time
4. **💹 Market Analysis** — Technical indicators, correlations, RSI

## 📝 Academic Context

This project demonstrates:
- Reinforcement Learning (PPO) applied to finance
- Custom OpenAI Gymnasium environment design
- Technical analysis & feature engineering
- Backtesting against buy-and-hold benchmark
- Interactive web deployment

## 📄 License

MIT