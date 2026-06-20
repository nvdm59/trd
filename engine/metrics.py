"""
Performance metrics computed from an equity curve and trade blotter.

These are how you judge whether a strategy is worth real money. Pay special
attention to max drawdown (the worst peak-to-trough loss — can you stomach it?)
and to results *after* commission and slippage, which the engine already bakes
into fills.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    start_equity: float
    end_equity: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    volatility: float
    num_trades: int

    def report(self) -> str:
        pct = lambda x: f"{x * 100:,.2f}%"
        return (
            "\n──────────── Backtest results ────────────\n"
            f"  Start equity     {self.start_equity:>14,.2f}\n"
            f"  End equity       {self.end_equity:>14,.2f}\n"
            f"  Total return     {pct(self.total_return):>14}\n"
            f"  CAGR             {pct(self.cagr):>14}\n"
            f"  Sharpe (ann.)    {self.sharpe:>14.2f}\n"
            f"  Volatility (ann.){pct(self.volatility):>14}\n"
            f"  Max drawdown     {pct(self.max_drawdown):>14}\n"
            f"  Trades           {self.num_trades:>14}\n"
            "──────────────────────────────────────────\n"
        )


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    median_dt = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
    if median_dt <= 0:
        return 252.0
    seconds_per_year = 365.25 * 24 * 3600
    bars_per_year = seconds_per_year / median_dt
    # cap at trading-day granularity so daily data reads as ~252, not 365
    return min(bars_per_year, 252.0) if median_dt >= 23 * 3600 else bars_per_year


def compute(equity: pd.Series, num_trades: int) -> Metrics:
    equity = equity.dropna()
    if len(equity) < 2:
        raise ValueError("Need at least two equity points to compute metrics.")

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = end / start - 1.0

    ppy = _periods_per_year(equity.index)
    n_years = len(equity) / ppy
    cagr = (end / start) ** (1 / n_years) - 1.0 if n_years > 0 and start > 0 else 0.0

    rets = equity.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(ppy)) if len(rets) else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(ppy)) if rets.std() > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(-drawdown.min()) if len(drawdown) else 0.0

    return Metrics(
        start_equity=start, end_equity=end, total_return=total_return,
        cagr=cagr, sharpe=sharpe, max_drawdown=max_dd,
        volatility=vol, num_trades=num_trades,
    )
