from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import numpy as np

from qtrader.broker.paper import PaperBroker
from qtrader.strategy.base import Strategy


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
        bar = data.loc[ts]
        open_price = float(bar.get("open", np.nan))
        high = float(bar.get("high", np.nan))
        low = float(bar.get("low", np.nan))
        close = float(bar.get("close"))
        weight = float(weights.loc[ts]) if ts in weights.index else 0.0

        # Skip rebalance if price is invalid or non-positive
        if not np.isfinite(close) or close <= 0:
            equity_values.append(broker.equity(equity_price := data["close"].iloc[max(0, i-1)] if i > 0 else 0.0))
            continue

        # 1) Intrabar stop processing (if strategy provides stops)
        stops = getattr(strategy, "stops", None)
        if stops is not None:
            position = broker.portfolio.get_or_create_position(symbol)
            qty = position.quantity
            avg_price = position.average_price
            if qty != 0 and np.isfinite(avg_price) and avg_price > 0:
                triggered_side = None  # "stop_loss" or "take_profit"
                exit_price = None

                if qty > 0:  # long
                    if getattr(stops, "stop_loss_pct", None):
                        stop_loss_price = avg_price * (1.0 - float(stops.stop_loss_pct))
                        if np.isfinite(low) and low <= stop_loss_price:
                            triggered_side = triggered_side or "stop_loss"
                            exit_price = stop_loss_price
                    if getattr(stops, "take_profit_pct", None):
                        take_profit_price = avg_price * (1.0 + float(stops.take_profit_pct))
                        if np.isfinite(high) and high >= take_profit_price:
                            # if both fire, prefer stop_loss (more conservative) by only overwriting if not set
                            if triggered_side is None:
                                triggered_side = "take_profit"
                                exit_price = take_profit_price
                else:  # short
                    if getattr(stops, "stop_loss_pct", None):
                        stop_loss_price = avg_price * (1.0 + float(stops.stop_loss_pct))
                        if np.isfinite(high) and high >= stop_loss_price:
                            triggered_side = triggered_side or "stop_loss"
                            exit_price = stop_loss_price
                    if getattr(stops, "take_profit_pct", None):
                        take_profit_price = avg_price * (1.0 - float(stops.take_profit_pct))
                        if np.isfinite(low) and low <= take_profit_price:
                            if triggered_side is None:
                                triggered_side = "take_profit"
                                exit_price = take_profit_price

                if triggered_side is not None and exit_price is not None and exit_price > 0:
                    # Flatten at the stop/take-profit level
                    broker.close_position(price=float(exit_price))

        # 2) Rebalance to target weight at close after stops
        if i + 1 >= min_bars:  # allow signals only after enough history
            broker.rebalance_to_target_weight(weight=weight, price=close)

        equity_values.append(broker.equity(close))

    equity_curve = pd.Series(equity_values, index=index, name="equity")
    # Stats will be filled by metrics module later; provide minimal baseline
    stats = {
        "final_equity": float(equity_curve.iloc[-1]),
        "return_total": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
    }
    return BacktestResult(equity_curve=equity_curve, fills=len(broker.fills), stats=stats)
