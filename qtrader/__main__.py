from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from qtrader.data.loader import load_csv, generate_gbm
from qtrader.engine.backtest import run_backtest
from qtrader.metrics.performance import summarize
from qtrader.strategy.base import SmaCrossStrategy
from qtrader.risk.manager import RiskManager

console = Console()


def _print_stats(stats: dict) -> None:
    table = Table(title="Backtest Summary")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    for k, v in stats.items():
        if isinstance(v, float):
            display = f"{v:,.4f}"
        else:
            display = str(v)
        table.add_row(k, display)
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="qtrader - simple SMA crossover backtester")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV with OHLCV data")
    parser.add_argument("--symbol", type=str, default="TEST", help="Symbol name")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Initial cash")
    parser.add_argument("--short", type=int, default=20, help="Short SMA window")
    parser.add_argument("--long", type=int, default=50, help="Long SMA window")
    parser.add_argument("--periods", type=int, default=1000, help="Synthetic periods if no CSV")
    parser.add_argument("--stop-loss", type=float, default=None, help="Stop-loss fraction (e.g., 0.05 for 5%)")
    parser.add_argument("--take-profit", type=float, default=None, help="Take-profit fraction (e.g., 0.10 for 10%)")
    args = parser.parse_args()

    if args.csv:
        data = load_csv(args.csv)
    else:
        data = generate_gbm(periods=args.periods)

    strategy = SmaCrossStrategy(short_window=args.short, long_window=args.long)
    risk = RiskManager(stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit) if (args.stop_loss or args.take_profit) else None
    result = run_backtest(data=data, strategy=strategy, symbol=args.symbol, initial_cash=args.cash, risk_manager=risk)

    stats = summarize(result.equity_curve)
    stats["fills"] = result.fills
    _print_stats(stats)


if __name__ == "__main__":
    main()
