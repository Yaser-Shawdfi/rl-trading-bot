"""
Data loader — Improved with more technical indicators.
Adds: ATR, Stochastic Oscillator, OBV, VWAP, Williams %R, lag features, price momentum.
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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    df = add_technical_indicators(df)

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fname = DATA_DIR / f"{symbol}_daily.csv"
        df.to_csv(fname)
        print(f" Saved {len(df)} rows to {fname}")

    return df


def add_technical_indicators(df):
    """Add 25+ technical indicators."""
    df = df.copy()
    df = df.ffill()

    # ─── Returns ─────────────────────────────────────────────────────────
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # ─── Moving Averages ─────────────────────────────────────────────────
    for period in [10, 20, 30, 50]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()

    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # ─── MACD ───────────────────────────────────────────────────────────
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ─── RSI ─────────────────────────────────────────────────────────────
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ─── Bollinger Bands ─────────────────────────────────────────────────
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ─── ATR (Average True Range) ───────────────────────────────────────
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    # ─── Stochastic Oscillator ──────────────────────────────────────────
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ─── Williams %R ─────────────────────────────────────────────────────
    df["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14)

    # ─── OBV (On Balance Volume) ────────────────────────────────────────
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_sma"] = obv.rolling(20).mean()
    df["obv_ratio"] = obv / (df["obv_sma"] + 1)

    # ─── VWAP (Volume Weighted Average Price) ───────────────────────────
    df["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"]

    # ─── Volatility ──────────────────────────────────────────────────────
    df["volatility_10"] = df["returns"].rolling(10).std()
    df["volatility_20"] = df["returns"].rolling(20).std()
    df["volatility_50"] = df["returns"].rolling(50).std()

    # ─── Volume Indicators ───────────────────────────────────────────────
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1)

    # ─── Momentum Indicators ─────────────────────────────────────────────
    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

    # ─── Lag Features (past returns) ─────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["returns"].shift(lag)

    # ─── Price Distance from Moving Averages ────────────────────────────
    df["dist_sma_10"] = (df["close"] - df["sma_10"]) / df["sma_10"]
    df["dist_sma_30"] = (df["close"] - df["sma_30"]) / df["sma_30"]
    df["dist_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

    # Replace inf and drop NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def load_data(filepath=None):
    """Load saved CSV or download fresh data."""
    if filepath is None:
        filepath = DATA_DIR / f"{SYMBOL}_daily.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f" Loaded {len(df)} rows from {filepath}")
        return df
    else:
        return download_data()


def get_feature_columns(df):
    """Return the list of feature columns for the RL environment."""
    exclude = {"open", "high", "low", "close", "volume", "adj close"}
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
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nFeature columns ({len(get_feature_columns(df))}): {get_feature_columns(df)}")