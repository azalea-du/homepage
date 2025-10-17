from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import numpy as np

from qtrader.broker.paper import PaperBroker
from qtrader.strategy.base import Strategy
from qtrader.risk.manager import RiskManager


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    fills: int
    stats: Dict[str, float]


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    symbol: str = "TEST",
    initial_cash: float = 100_000.0,
    risk_manager: RiskManager | None = None,
) -> BacktestResult:
    if data.empty:
        raise ValueError("data is empty")
    if "close" not in data.columns:
        raise ValueError("data must include 'close' column")

    # Sanitize data: ensure close is numeric and filled
    data = data.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce").ffill().bfill()

    broker = PaperBroker(symbol=symbol, initial_cash=initial_cash)

    # Align weights to data index and fill missing with 0
    weights = strategy.generate_target_weights(data).reindex(data.index).fillna(0.0)

    min_bars = strategy.min_history_bars()
    equity_values = []
    index = data.index

    for i, ts in enumerate(index):
        price = float(data.loc[ts, "close"])  # execution at close for simplicity
        weight = float(weights.loc[ts]) if ts in weights.index else 0.0

        # Skip rebalance if price is invalid or non-positive
        if not np.isfinite(price) or price <= 0:
            equity_values.append(broker.equity(equity_price := data["close"].iloc[max(0, i-1)] if i > 0 else 0.0))
            continue

        if i + 1 >= min_bars:  # allow signals only after enough history
            if risk_manager is not None:
                weight = risk_manager.adjust_weight(
                    portfolio=broker.portfolio,
                    symbol=symbol,
                    price=price,
                    requested_weight=weight,
                )
            broker.rebalance_to_target_weight(weight=weight, price=price)

        equity_values.append(broker.equity(price))

    equity_curve = pd.Series(equity_values, index=index, name="equity")
    # Stats will be filled by metrics module later; provide minimal baseline
    stats = {
        "final_equity": float(equity_curve.iloc[-1]),
        "return_total": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
    }
    return BacktestResult(equity_curve=equity_curve, fills=len(broker.fills), stats=stats)
