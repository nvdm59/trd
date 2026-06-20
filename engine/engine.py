"""
The backtest engine — the loop that ties everything together.

For each bar, in time order:
  1. Fill any order placed on the previous bar, at THIS bar's open. (Deciding on
     bar i's close but filling at bar i+1's open is what prevents look-ahead
     bias — you can't trade on information you wouldn't have had yet.)
  2. Hand the strategy all history up to and including this bar's close.
  3. Turn its target weight into a target share count via the risk manager.
  4. Queue the resulting order to fill on the next bar.
  5. Mark the portfolio to market at this bar's close.

The result is an equity curve and a trade blotter you can score with metrics.py.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .broker import SimBroker
from .data import iter_bars
from .metrics import Metrics, compute
from .portfolio import Portfolio
from .risk import RiskManager, RiskLimits
from .strategy import Strategy


class BacktestEngine:
    def __init__(self, strategies: Dict[str, Strategy], starting_cash: float = 100_000.0,
                 limits: RiskLimits | None = None):
        """strategies: {symbol: Strategy instance}. One strategy per symbol."""
        self.strategies = strategies
        self.portfolio = Portfolio(cash=starting_cash)
        self.risk = RiskManager(limits)
        self.broker = SimBroker(self.portfolio, self.risk)
        self._start_cash = starting_cash

    def run(self, history: Dict[str, pd.DataFrame], verbose: bool = False) -> Metrics:
        last_close: Dict[str, float] = {}
        pending: Dict[str, float] = {}   # symbol -> target qty to fill next bar
        seen_count: Dict[str, int] = {s: 0 for s in history}

        for snapshot in iter_bars(history):
            ts = next(iter(snapshot.values())).ts

            for sym, bar in snapshot.items():
                # 1. fill the order queued on this symbol's previous bar, at the open
                if sym in pending:
                    target = pending.pop(sym)
                    delta = target - self.portfolio.position(sym).qty
                    self.broker.submit_market_order(ts, sym, delta, bar.open)

                last_close[sym] = bar.close
                seen_count[sym] += 1

                # 2-4. ask the strategy, size it, and queue for next bar
                strat = self.strategies.get(sym)
                if strat is None or seen_count[sym] < strat.warmup:
                    continue
                hist = history[sym].loc[:ts]
                weight = strat.target_weight(hist)
                equity = self.portfolio.equity(last_close)
                target_qty = self.risk.target_qty(
                    weight, equity, bar.close, self.portfolio.position(sym).qty)
                pending[sym] = target_qty

            # 5. mark to market and update the drawdown kill switch
            self.portfolio.mark(ts, last_close)
            self.risk.update_drawdown(self.portfolio.equity(last_close))
            if verbose:
                print(f"{ts.date()}  equity={self.portfolio.equity(last_close):,.0f}"
                      f"  {'[HALTED]' if self.risk.halted else ''}")

        equity_series = self.portfolio.equity_series()
        return compute(equity_series, num_trades=len(self.portfolio.fills))
