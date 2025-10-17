from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    equity = equity_curve.astype(float)
    ret = equity.pct_change().fillna(0.0)
    ret.name = "returns"
    return ret


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    equity = equity_curve.astype(float)
    total_return = float(equity.iloc[-1] / equity.iloc[0])
    n_periods = len(equity) - 1
    if n_periods <= 0:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    ret = compute_returns(equity_curve)
    excess = ret - (risk_free_rate / periods_per_year)
    std = excess.std(ddof=0)
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def max_drawdown(equity_curve: pd.Series) -> float:
    equity = equity_curve.astype(float)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def summarize(equity_curve: pd.Series) -> dict:
    return {
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
        "annual_return": annualized_return(equity_curve),
        "sharpe": sharpe_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
    }
