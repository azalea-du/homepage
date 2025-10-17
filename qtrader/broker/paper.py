from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import pandas as pd

from qtrader.types import Portfolio, OrderSide, Fill


@dataclass
class PaperBroker:
    symbol: str
    initial_cash: float = 100_000.0
    slippage_bps: float = 1.0  # basis points
    commission_rate: float = 0.0005  # fraction of notional

    portfolio: Portfolio = field(init=False)
    fills: List[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.portfolio = Portfolio(cash=self.initial_cash)

    def _apply_trade(self, side: OrderSide, quantity: float, price: float) -> None:
        if quantity == 0:
            return
        side_sign = 1.0 if side == OrderSide.BUY else -1.0
        slip_price = price * (1.0 + side_sign * (self.slippage_bps / 10_000.0))
        notional = abs(quantity) * slip_price
        commission = notional * self.commission_rate

        # Cash adjustment
        cash_delta = -side_sign * notional - commission
        self.portfolio.cash += cash_delta

        # Position update
        position = self.portfolio.get_or_create_position(self.symbol)
        fill = Fill(
            order_id=len(self.fills) + 1,
            symbol=self.symbol,
            side=side,
            quantity=abs(quantity),
            price=slip_price,
            commission=commission,
            slippage=abs(slip_price - price) * abs(quantity),
        )
        position.update_with_fill(fill)
        self.fills.append(fill)

    def rebalance_to_target_weight(self, weight: float, price: float) -> None:
        # Compute desired shares based on current equity and target weight
        last_prices: Dict[str, float] = {self.symbol: price}
        equity = self.portfolio.total_equity(last_prices)
        target_value = weight * equity
        target_shares = int(target_value // price)  # floor to whole shares

        current_shares = self.portfolio.get_or_create_position(self.symbol).quantity
        delta_shares = target_shares - int(current_shares)
        if delta_shares == 0:
            return
        side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
        self._apply_trade(side=side, quantity=abs(delta_shares), price=price)

    def equity(self, price: float) -> float:
        return self.portfolio.total_equity({self.symbol: price})
