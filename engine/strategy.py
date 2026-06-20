"""
Strategies.

A strategy looks at the price history *up to and including the current bar* and
returns a target weight in [-1, 1] for the symbol:

    +1.0  = go fully long (use the full per-symbol budget)
     0.0  = flat (no position)
    -1.0  = go fully short

The engine, not the strategy, turns a weight into share counts and orders. This
keeps strategies pure and easy to test. Subclass `Strategy` and implement
`target_weight`; set `warmup` to the bars you need before signalling.
"""
from __future__ import annotations

import pandas as pd


class Strategy:
    name = "base"
    warmup = 0          # bars of history required before this strategy signals
    long_only = True    # if True, negative weights are clamped to 0

    def target_weight(self, history: pd.DataFrame) -> float:
        """history: DataFrame[open,high,low,close,volume] ending at the current bar."""
        raise NotImplementedError

    def _clamp(self, w: float) -> float:
        w = max(-1.0, min(1.0, float(w)))
        return max(0.0, w) if self.long_only else w


class MovingAverageCrossover(Strategy):
    """
    Trend / momentum. Go long when the fast MA is above the slow MA (uptrend),
    flat (or short) when it crosses below. Captures sustained trends; whipsaws
    in choppy, sideways markets.
    """
    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = True):
        assert fast < slow, "fast window must be shorter than slow window"
        self.fast, self.slow = fast, slow
        self.warmup = slow + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"]
        fast = close.tail(self.fast).mean()
        slow = close.tail(self.slow).mean()
        return self._clamp(1.0 if fast > slow else -1.0)


class MeanReversion(Strategy):
    """
    Mean reversion via z-score (Bollinger-style). When price stretches far below
    its recent average (z < -entry) we buy the dip; we exit as it reverts toward
    the mean (|z| < exit). Works in range-bound markets; gets run over in strong
    trends — the opposite regime from the crossover above.
    """
    name = "mean_reversion"

    def __init__(self, lookback: int = 20, entry_z: float = 1.0,
                 exit_z: float = 0.25, long_only: bool = True):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.warmup = lookback + 1
        self.long_only = long_only
        self._pos = 0.0   # remember current stance so we can hold until exit

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"].tail(self.lookback)
        mean, std = close.mean(), close.std()
        if std == 0 or pd.isna(std):
            return self._clamp(self._pos)
        z = (close.iloc[-1] - mean) / std

        if z <= -self.entry_z:          # stretched below mean -> buy
            self._pos = 1.0
        elif z >= self.entry_z:         # stretched above mean -> sell/short
            self._pos = -1.0
        elif abs(z) <= self.exit_z:     # reverted to mean -> close out
            self._pos = 0.0
        # else: hold the current stance
        return self._clamp(self._pos)
