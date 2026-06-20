"""
Risk management — the part that actually keeps you solvent.

A strategy says "I want weight w in this symbol." The risk manager decides how
many shares that *really* means, after applying:

  * per-symbol cap         — never bet the whole account on one name
  * gross-exposure cap      — limit total leverage across all positions
  * max-drawdown kill switch — flatten everything if equity falls too far

Tuning these badly is the most common way automated traders blow up. The
defaults here are deliberately conservative.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_weight_per_symbol: float = 0.20   # at most 20% of equity in any one name
    max_gross_exposure: float = 1.0       # 1.0 = no leverage; 2.0 = up to 2x
    max_drawdown: float = 0.25            # flatten everything past a 25% drawdown
    commission_per_share: float = 0.0     # set to your broker's per-share fee
    commission_pct: float = 0.0           # or a percentage of notional (e.g. crypto)
    slippage_pct: float = 0.0005          # 5 bps assumed slippage on every fill
    min_trade_pct: float = 0.01           # ignore rebalances smaller than 1% of
                                          # equity — stops tiny daily-drift churn


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self._peak_equity = 0.0
        self.halted = False

    def update_drawdown(self, equity: float) -> None:
        """Track the high-water mark and trip the kill switch on deep drawdown."""
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity > 0:
            dd = 1.0 - equity / self._peak_equity
            if dd >= self.limits.max_drawdown:
                self.halted = True

    def target_qty(self, weight: float, equity: float, price: float,
                   current_qty: float) -> float:
        """
        Convert a desired weight into a target share count, honoring the
        per-symbol cap. If the kill switch is tripped, the only allowed target
        is zero (flat).
        """
        if self.halted:
            return 0.0
        w = max(-self.limits.max_weight_per_symbol,
                min(self.limits.max_weight_per_symbol, weight))
        if price <= 0:
            return current_qty
        target = (w * equity) / price
        # no-trade band: if the change is tiny, stay put to avoid churn/slippage
        if abs(target - current_qty) * price < self.limits.min_trade_pct * equity:
            return current_qty
        return target

    def fill_price(self, ref_price: float, side: int) -> float:
        """Apply slippage: you buy a touch higher and sell a touch lower."""
        return ref_price * (1 + side * self.limits.slippage_pct)

    def commission(self, qty: float, price: float) -> float:
        return (abs(qty) * self.limits.commission_per_share
                + abs(qty) * price * self.limits.commission_pct)
