from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {c.lower(): c for c in df.columns}

    # Map common alternative names to expected ones
    mapping = {}
    # Timestamp / index
    if "timestamp" in columns:
        ts_col = columns["timestamp"]
    elif "datetime" in columns:
        ts_col = columns["datetime"]
    elif "date" in columns:
        ts_col = columns["date"]
    elif df.index.name is not None:
        ts_col = df.index.name
    else:
        ts_col = df.columns[0]

    # Price columns
    def find_col(*names):
        for name in names:
            if name in columns:
                return columns[name]
        return None

    mapping["open"] = find_col("open")
    mapping["high"] = find_col("high")
    mapping["low"] = find_col("low")
    mapping["close"] = find_col("close", "adj close", "adj_close", "adjusted close")
    mapping["volume"] = find_col("volume", "vol")

    # Build normalized DataFrame
    norm = pd.DataFrame(index=pd.to_datetime(df[ts_col] if ts_col in df.columns else df.index))
    for k in EXPECTED_COLUMNS:
        src = mapping.get(k)
        if src is not None and src in df.columns:
            norm[k] = pd.to_numeric(df[src], errors="coerce")
        else:
            # missing -> fill reasonable defaults
            if k == "volume":
                norm[k] = 0.0
            else:
                norm[k] = np.nan

    # Fill missing OHLC using close where possible
    for k in ["open", "high", "low"]:
        norm[k] = norm[k].fillna(norm["close"])  # if close exists

    norm = norm.sort_index()
    norm.index.name = "timestamp"
    return norm


def load_csv(path: str, tz: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV data from CSV into a normalized DataFrame.

    Expected columns (case-insensitive): timestamp/datetime/date, open, high, low, close, volume.
    Extra columns are ignored. Index is a DateTimeIndex named 'timestamp'.
    """
    df = pd.read_csv(path)
    norm = _normalize_columns(df)
    if tz is not None:
        norm.index = norm.index.tz_localize("UTC").tz_convert(tz) if norm.index.tz is None else norm.index.tz_convert(tz)
    return norm


def generate_gbm(
    start_price: float = 100.0,
    periods: int = 1000,
    freq: str = "1D",
    drift: float = 0.07,
    volatility: float = 0.20,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data via a geometric Brownian motion model.

    Returns a DataFrame with columns: open, high, low, close, volume and a DateTimeIndex.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start="2000-01-01", periods=periods, freq=freq)

    dt = 1.0 / 252.0  # treat as daily by default for drift/vol inputs
    shocks = rng.normal(loc=(drift - 0.5 * volatility ** 2) * dt, scale=volatility * np.sqrt(dt), size=periods)
    log_prices = np.log(start_price) + np.cumsum(shocks)
    close = np.exp(log_prices)

    open_ = np.concatenate(([start_price], close[:-1]))
    # High/Low as a small band around close
    intraday_std = close * (volatility / np.sqrt(252))
    noise = rng.normal(0, 0.25, size=periods)
    high = np.maximum(open_, close) + np.abs(noise) * intraday_std
    low = np.minimum(open_, close) - np.abs(noise) * intraday_std

    volume = rng.lognormal(mean=12.0, sigma=0.5, size=periods)

    data = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=index)
    data.index.name = "timestamp"
    return data
