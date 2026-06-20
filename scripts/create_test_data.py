"""Create fallback test data if Yahoo Finance download fails in CI."""
import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

if not os.path.exists("data/AAPL_daily.csv"):
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.random.randint(1000, 10000, 200),
            "returns": np.concatenate([[0], np.diff(prices) / prices[:-1]]),
            "rsi": np.random.uniform(20, 80, 200),
            "macd": np.random.randn(200),
            "volatility": np.abs(np.random.randn(200)) * 0.02,
        },
        index=dates,
    )
    df.to_csv("data/AAPL_daily.csv")
    print(f"Created fallback data: {len(df)} rows")
else:
    print("Data already exists, skipping fallback creation")