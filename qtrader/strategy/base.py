from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qtrader.indicators.basic import simple_moving_average


class Strategy:
    def min_history_bars(self) -> int:
        return 0

    def generate_target_weights(self, data: pd.DataFrame) -> pd.Series:
        """Return desired target weight ([-1, 1]) per timestamp.

        Positive means long, negative short, zero flat. Index must align to data.index.
        """
        raise NotImplementedError


@dataclass
class SmaCrossStrategy(Strategy):
    short_window: int = 20
    long_window: int = 50

    def min_history_bars(self) -> int:
        return max(self.short_window, self.long_window)

    def generate_target_weights(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].astype(float)
        sma_short = simple_moving_average(close, self.short_window)
        sma_long = simple_moving_average(close, self.long_window)

        signal = (sma_short > sma_long).astype(int) - (sma_short < sma_long).astype(int)
        # Keep last non-NaN signal after enough history; fill NaN with 0
        signal = signal.fillna(0.0)
        signal.name = "weight"
        return signal
