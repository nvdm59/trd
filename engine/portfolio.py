"""
Portfolio accounting: cash, positions, equity curve, and a trade blotter.

This is the single source of truth for "how much money do I have and where is
it." The engine asks the portfolio for current equity (to size trades) and tells
it about fills (to update cash and positions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class Position:
    symbol: str
    qty: float = 0.0          # shares held (negative = short)
    avg_price: float = 0.0    # average entry price of the open position

    def market_value(self, price: float) -> float:
        return self.qty * price


@dataclass
class Fill:
    ts: pd.Timestamp
    symbol: str
    qty: float          # +buy / -sell
    price: float
    commission: float


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    equity_curve: List[tuple] = field(default_factory=list)  # (ts, equity)

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol))

    def equity(self, prices: Dict[str, float]) -> float:
        """Total account value = cash + marked-to-market value of all positions."""
        mtm = sum(p.market_value(prices.get(s, p.avg_price))
                  for s, p in self.positions.items())
        return self.cash + mtm

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and the position's average price from an executed trade."""
        pos = self.position(fill.symbol)
        self.cash -= fill.qty * fill.price + fill.commission

        new_qty = pos.qty + fill.qty
        if pos.qty == 0 or (pos.qty > 0) == (fill.qty > 0):
            # opening or adding in the same direction -> blend the average price
            if new_qty != 0:
                pos.avg_price = (pos.avg_price * pos.qty + fill.price * fill.qty) / new_qty
        elif (new_qty != 0) and ((new_qty > 0) != (pos.qty > 0)):
            # flipped through zero -> the remainder is a fresh position
            pos.avg_price = fill.price
        # reducing toward zero keeps the original avg_price
        pos.qty = new_qty
        if abs(pos.qty) < 1e-9:
            pos.qty, pos.avg_price = 0.0, 0.0
        self.fills.append(fill)

    def mark(self, ts: pd.Timestamp, prices: Dict[str, float]) -> None:
        self.equity_curve.append((ts, self.equity(prices)))

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        ts, eq = zip(*self.equity_curve)
        return pd.Series(eq, index=pd.DatetimeIndex(ts), name="equity")
