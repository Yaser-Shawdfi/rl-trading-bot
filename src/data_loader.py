"""
Data loader — downloads historical price data from Yahoo Finance.
Caches locally to avoid repeated API calls.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

from config import DATA_DIR, SYMBOL, START_DATE, END_DATE


def download_data(symbol=SYMBOL, start=START_DATE, end=END_DATE, save=True):
    """Download OHLCV data from Yahoo Finance."""
    print(f"Downloading {symbol} data ({start} to {end})...")
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure standard column names
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    # Add technical indicators
    df = add_technical_indicators(df)

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fname = DATA_DIR / f"{symbol}_daily.csv"
        df.to_csv(fname)
        print(f"✅ Saved {len(df)} rows to {fname}")

    return df


def add_technical_indicators(df):
    """Add commonly used technical indicators as features."""
    df = df.copy()

    # Returns
    df["returns"] = df["close"].pct_change()

    # Moving averages
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    # Exponential moving averages
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std

    # Volatility
    df["volatility"] = df["returns"].rolling(20).std()

    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Drop NaN rows
    df = df.dropna()

    return df


def load_data(filepath=None):
    """Load saved CSV or download fresh data."""
    if filepath is None:
        filepath = DATA_DIR / f"{SYMBOL}_daily.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✅ Loaded {len(df)} rows from {filepath}")
        return df
    else:
        return download_data()


def get_feature_columns(df):
    """Return the list of feature columns for the RL environment."""
    exclude = ["open", "high", "low", "close", "volume", "adj close"]
    return [c for c in df.columns if c.lower() not in exclude]


def split_data(df, train_ratio=0.7):
    """Split data into train and test sets (chronological)."""
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"Train: {len(train)} rows ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"Test:  {len(test)} rows ({test.index[0].date()} → {test.index[-1].date()})")
    return train, test


if __name__ == "__main__":
    df = download_data()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFeature columns: {get_feature_columns(df)}")
    train, test = split_data(df)
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())