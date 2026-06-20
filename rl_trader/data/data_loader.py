"""
Enterprise data loader — multi-asset support, caching, 42+ indicators.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger("rl_trader.data")


class DataLoader:
    """Multi-asset data loader with caching and technical indicators."""

    def __init__(self, cache_dir: str = "data", symbol: str = "AAPL",
                 start: str = "2015-01-01", end: str = "2024-12-31"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.start = start
        self.end = end

    def load(self, force_download: bool = False) -> pd.DataFrame:
        """Load data from cache or download from Yahoo Finance."""
        cache_path = self.cache_dir / f"{self.symbol}_daily.csv"

        if cache_path.exists() and not force_download:
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            logger.info(f"Downloading {self.symbol} from Yahoo Finance ({self.start} → {self.end})")
            df = self._download()
            self._add_indicators(df)
            df.to_csv(cache_path)
            logger.info(f"Cached {len(df)} rows to {cache_path}")

        return df

    def _download(self) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance."""
        df = yf.download(self.symbol, start=self.start, end=self.end, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df.ffill()
        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 42+ technical indicators in-place."""
        logger.info("Computing technical indicators...")

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Moving averages
        for p in [10, 20, 30, 50]:
            df[f"sma_{p}"] = df["close"].rolling(p).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI
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
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        # Stochastic
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Williams %R
        df["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14)

        # OBV
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_sma"] = obv.rolling(20).mean()
        df["obv_ratio"] = obv / (df["obv_sma"] + 1)

        # VWAP
        df["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"]

        # Volatility
        for p in [10, 20, 50]:
            df[f"volatility_{p}"] = df["returns"].rolling(p).std()

        # Volume
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1)

        # Momentum
        for p in [5, 10, 20]:
            df[f"momentum_{p}"] = df["close"] / df["close"].shift(p) - 1

        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)

        # Distance from SMAs
        for p in [10, 30, 50]:
            df[f"dist_sma_{p}"] = (df["close"] - df[f"sma_{p}"]) / df[f"sma_{p}"]

        # Clean
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        logger.info(f"Added {len(df.columns)} columns ({len(get_feature_columns(df))} features)")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature column names (excludes OHLCV)."""
    exclude = {"open", "high", "low", "close", "volume", "adj close"}
    return [c for c in df.columns if c.lower() not in exclude]


def split_data(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split."""
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


def load_data(symbol: str = "AAPL", cache_dir: str = "data",
              start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Convenience function: load data for a symbol."""
    loader = DataLoader(cache_dir=cache_dir, symbol=symbol, start=start, end=end)
    return loader.load()