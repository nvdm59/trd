"""
Cross-asset rotation: strategies that look at the WHOLE universe at once.

The single-asset strategies in strategy.py decide each symbol in isolation. A
rotation strategy instead ranks symbols against each other and concentrates
capital in the best — the one family that has historically beaten buy-and-hold on
*both* return and drawdown, because it can flee a falling asset into a rising one
(or into bonds/cash) instead of just sitting in cash.

A `PortfolioStrategy` sees every symbol's history and returns a dict of target
weights {symbol: weight} that sum to ≤ 1 (the remainder is cash). The
`MultiAssetBacktestEngine` runs them with the same risk controls, costs, and
no-look-ahead fills as the single-asset engine.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .data import iter_bars
from .metrics import Metrics, compute
from .portfolio import Portfolio
from .risk import RiskLimits, RiskManager


# ────────────────────────────── strategies ───────────────────────────────────

class PortfolioStrategy:
    name = "base_portfolio"
    warmup = 0

    def target_weights(self, histories: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """histories: {symbol: DataFrame ending at the current bar}. Return
        {symbol: weight}; omitted symbols (and any leftover) are held as cash."""
        raise NotImplementedError


class DualMomentum(PortfolioStrategy):
    """
    Gary Antonacci's Dual Momentum (GEM). Two filters:
      * RELATIVE momentum — among the risky assets, pick the single strongest by
        trailing return (rotate toward what's winning).
      * ABSOLUTE momentum — only hold it if that return is also positive; if even
        the best risky asset is falling, step aside into the safe asset (a bond
        ETF if you put one in `safe_asset`) or cash.
    The absolute filter is what dodges bear markets; the relative filter is what
    lets it ride whichever asset is leading. Holds ONE asset at a time.
    """
    name = "dual_momentum"

    def __init__(self, lookback: int = 252, safe_asset: Optional[str] = None):
        self.lookback = lookback
        self.safe_asset = safe_asset
        self.warmup = lookback + 1

    def target_weights(self, histories):
        rets = {}
        for s, df in histories.items():
            c = df["close"]
            if len(c) >= self.lookback + 1:
                rets[s] = c.iloc[-1] / c.iloc[-(self.lookback + 1)] - 1.0
        risky = {s: r for s, r in rets.items() if s != self.safe_asset}
        if not risky:
            return {}
        best = max(risky, key=risky.get)
        if risky[best] > 0:                       # absolute momentum passes
            return {best: 1.0}
        if self.safe_asset and self.safe_asset in histories:  # flee to bonds
            return {self.safe_asset: 1.0}
        return {}                                  # otherwise cash


class RelativeMomentum(PortfolioStrategy):
    """
    Top-N relative momentum with a trend filter. Rank every symbol by trailing
    return, keep only those also above their own long-term average (trend filter),
    and hold the best `top_n` equally. A diversified cousin of Dual Momentum:
    spreads across several leaders instead of betting on one, smoothing the ride
    at the cost of some concentration upside.
    """
    name = "relative_momentum"

    def __init__(self, lookback: int = 126, top_n: int = 2, trend_window: int = 200):
        self.lookback, self.top_n, self.trend_window = lookback, top_n, trend_window
        self.warmup = max(lookback, trend_window) + 1

    def target_weights(self, histories):
        scored = {}
        for s, df in histories.items():
            c = df["close"]
            if len(c) < self.warmup:
                continue
            mom = c.iloc[-1] / c.iloc[-(self.lookback + 1)] - 1.0
            if c.iloc[-1] > c.tail(self.trend_window).mean():   # trend filter
                scored[s] = mom
        winners = [s for s in sorted(scored, key=scored.get, reverse=True)[:self.top_n]
                   if scored[s] > 0]
        if not winners:
            return {}
        w = 1.0 / len(winners)
        return {s: w for s in winners}


class EqualWeightTrend(PortfolioStrategy):
    """
    Hold every asset that is above its long-term moving average, equally weighted;
    move the rest to cash. The simplest possible "participate in what's trending
    up, avoid what's trending down" rule — a clean baseline to judge the fancier
    rotations against.
    """
    name = "ew_trend"

    def __init__(self, trend_window: int = 200):
        self.trend_window = trend_window
        self.warmup = trend_window + 1

    def target_weights(self, histories):
        chosen = [s for s, df in histories.items()
                  if len(df) >= self.warmup
                  and df["close"].iloc[-1] > df["close"].tail(self.trend_window).mean()]
        if not chosen:
            return {}
        w = 1.0 / len(chosen)
        return {s: w for s in chosen}


class InverseVolatility(PortfolioStrategy):
    """
    Risk-parity-lite: always fully invested across all symbols, but weighted
    INVERSELY to each one's recent volatility, so calm assets get more capital and
    jumpy ones get less. No market timing — it's a diversification benchmark that
    shows what smart *weighting* alone buys you versus the momentum timing above.
    """
    name = "inverse_vol"

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.warmup = lookback + 1

    def target_weights(self, histories):
        inv = {}
        for s, df in histories.items():
            c = df["close"]
            if len(c) < self.lookback + 1:
                continue
            vol = c.pct_change().tail(self.lookback).std()
            if vol and vol > 0:
                inv[s] = 1.0 / vol
        total = sum(inv.values())
        if total == 0:
            return {}
        return {s: v / total for s, v in inv.items()}


ROTATION_REGISTRY = {
    "dual_momentum": DualMomentum,
    "relative_momentum": RelativeMomentum,
    "ew_trend": EqualWeightTrend,
    "inverse_vol": InverseVolatility,
}


# ─────────────────────────────── engine ──────────────────────────────────────

class MultiAssetBacktestEngine:
    """Same loop and risk model as BacktestEngine, but driven by one
    PortfolioStrategy that allocates across all symbols each bar."""

    def __init__(self, strategy: PortfolioStrategy, symbols: List[str],
                 starting_cash: float = 100_000.0, limits: Optional[RiskLimits] = None,
                 rebalance: str = "M"):
        """rebalance: 'M' monthly, 'W' weekly, 'D' daily. Momentum rotation is
        traditionally MONTHLY — re-ranking daily just churns on noise."""
        self.strategy = strategy
        self.symbols = symbols
        self.portfolio = Portfolio(cash=starting_cash)
        # rotations may put 100% in one name, so allow it; gross is capped below
        self.risk = RiskManager(limits or RiskLimits(max_weight_per_symbol=1.0))
        self.rebalance = rebalance

    def _period_key(self, ts: pd.Timestamp):
        if self.rebalance == "D":
            return ts.toordinal()
        if self.rebalance == "W":
            iso = ts.isocalendar()
            return (iso[0], iso[1])
        return (ts.year, ts.month)   # monthly (default)

    def run(self, history: Dict[str, pd.DataFrame]) -> Metrics:
        from .broker import SimBroker
        broker = SimBroker(self.portfolio, self.risk)
        last_close: Dict[str, float] = {}
        pending: Dict[str, float] = {}
        last_period = None

        for snapshot in iter_bars(history):
            ts = next(iter(snapshot.values())).ts

            # 1. fill orders queued on the prior bar, at this bar's open
            for sym, bar in snapshot.items():
                if sym in pending:
                    target = pending.pop(sym)
                    delta = target - self.portfolio.position(sym).qty
                    broker.submit_market_order(ts, sym, delta, bar.open)
                last_close[sym] = bar.close

            # 2. only re-rank on a rebalance boundary; otherwise just hold
            period = self._period_key(ts)
            if period != last_period:
                last_period = period
                histories = {s: df.loc[:ts] for s, df in history.items()}
                weights = self.strategy.target_weights(histories)

                # enforce the gross-exposure cap (never lever past the limit)
                gross = sum(abs(w) for w in weights.values())
                if gross > self.risk.limits.max_gross_exposure and gross > 0:
                    scale = self.risk.limits.max_gross_exposure / gross
                    weights = {s: w * scale for s, w in weights.items()}

                # size every symbol (0 weight => liquidate), queue for next bar
                equity = self.portfolio.equity(last_close)
                for sym in self.symbols:
                    price = last_close.get(sym)
                    if price is None:
                        continue
                    w = weights.get(sym, 0.0)
                    pending[sym] = self.risk.target_qty(
                        w, equity, price, self.portfolio.position(sym).qty)

            # 3. mark to market, update the drawdown kill switch
            self.portfolio.mark(ts, last_close)
            self.risk.update_drawdown(self.portfolio.equity(last_close))

        return compute(self.portfolio.equity_series(), len(self.portfolio.fills))


def run_rotation(name: str, symbols: List[str], history: Dict[str, pd.DataFrame],
                 cash: float = 100_000.0, limits: Optional[RiskLimits] = None,
                 safe_asset: Optional[str] = None):
    """Convenience: build the named rotation strategy, run it, return
    (Metrics, equity_series). dual_momentum gets `safe_asset` if it's in the
    universe."""
    Cls = ROTATION_REGISTRY[name]
    kwargs = {}
    if name == "dual_momentum" and safe_asset and safe_asset in symbols:
        kwargs["safe_asset"] = safe_asset
    engine = MultiAssetBacktestEngine(Cls(**kwargs), symbols, cash, limits)
    metrics = engine.run(history)
    return metrics, engine.portfolio.equity_series()
