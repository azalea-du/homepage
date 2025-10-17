from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qtrader.types import Portfolio


@dataclass
class RiskManager:
    """Simple risk controls: fixed stop-loss and take-profit.

    Pct inputs are expressed as fractions (e.g., 0.05 == 5%). Stops are
    evaluated against the position's average entry price.
    """

    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    def adjust_weight(self, portfolio: Portfolio, symbol: str, price: float, requested_weight: float) -> float:
        position = portfolio.positions.get(symbol)
        if position is None or position.quantity == 0:
            return float(requested_weight)

        avg_price = float(position.average_price)
        if not np.isfinite(avg_price) or avg_price <= 0:
            return float(requested_weight)

        pos_sign = 1.0 if position.quantity > 0 else -1.0
        weight_sign = 0.0 if requested_weight == 0 else (1.0 if requested_weight > 0 else -1.0)

        stop_loss_hit = False
        take_profit_hit = False

        if pos_sign > 0:
            if self.stop_loss_pct is not None and price <= avg_price * (1.0 - self.stop_loss_pct):
                stop_loss_hit = True
            if self.take_profit_pct is not None and price >= avg_price * (1.0 + self.take_profit_pct):
                take_profit_hit = True
        else:
            if self.stop_loss_pct is not None and price >= avg_price * (1.0 + self.stop_loss_pct):
                stop_loss_hit = True
            if self.take_profit_pct is not None and price <= avg_price * (1.0 - self.take_profit_pct):
                take_profit_hit = True

        if stop_loss_hit or take_profit_hit:
            # Allow immediate reversal if strategy requests flipping direction; otherwise flatten
            if weight_sign != 0.0 and weight_sign != pos_sign:
                return float(requested_weight)
            return 0.0

        return float(requested_weight)
