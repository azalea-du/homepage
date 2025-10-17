from __future__ import annotations

import pandas as pd


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be > 0")
    return series.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be > 0")
    alpha = 2.0 / (window + 1.0)
    return series.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
