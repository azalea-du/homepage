from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


class OrderSide(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()


@dataclass
class Bar:
    timestamp: "pd.Timestamp"  # lazy annotation to avoid hard pandas import
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    symbol: str = ""


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    id: int = 0


@dataclass
class Fill:
    order_id: int
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0

    def update_with_fill(self, fill: Fill) -> None:
        if fill.quantity == 0:
            return
        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        new_quantity = self.quantity + signed_qty
        if new_quantity == 0:
            # fully closed position
            self.quantity = 0.0
            self.average_price = 0.0
            return
        if (self.quantity == 0) or (self.quantity > 0 and signed_qty > 0) or (self.quantity < 0 and signed_qty < 0):
            # adding to same-direction position -> update average price
            total_cost = self.average_price * abs(self.quantity) + fill.price * abs(signed_qty)
            self.quantity = new_quantity
            self.average_price = total_cost / abs(self.quantity)
        else:
            # reducing or reversing: adjust quantity; average price only changes if reversed beyond flat
            self.quantity = new_quantity
            if (self.quantity > 0 and signed_qty > 0) or (self.quantity < 0 and signed_qty < 0):
                # reversed beyond flat: set avg to fill price
                self.average_price = fill.price


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def get_or_create_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def total_equity(self, last_prices: Dict[str, float]) -> float:
        equity = self.cash
        for symbol, position in self.positions.items():
            price = last_prices.get(symbol, 0.0)
            equity += position.quantity * price
        return equity
