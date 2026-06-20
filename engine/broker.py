"""
Brokers.

Two implementations behind one interface:

  * SimBroker   — fills orders against historical/known prices with slippage and
                  commission. Used by the backtester and the paper simulator.
  * AlpacaBroker — sends real orders to Alpaca (use a PAPER account first!).

The engine only knows about `submit_market_order`, so swapping sim for live is a
one-line change.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .portfolio import Fill, Portfolio
from .risk import RiskManager


class SimBroker:
    """Simulated broker: deterministic fills, models slippage + commission."""

    def __init__(self, portfolio: Portfolio, risk: RiskManager):
        self.portfolio = portfolio
        self.risk = risk

    def submit_market_order(self, ts: pd.Timestamp, symbol: str,
                            qty: float, ref_price: float) -> Optional[Fill]:
        """qty: +buy / -sell. ref_price: the price to fill against (e.g. next open)."""
        if abs(qty) < 1e-9:
            return None
        side = 1 if qty > 0 else -1
        price = self.risk.fill_price(ref_price, side)
        commission = self.risk.commission(qty, price)
        fill = Fill(ts=ts, symbol=symbol, qty=qty, price=price, commission=commission)
        self.portfolio.apply_fill(fill)
        return fill


class AlpacaBroker:
    """
    Live/paper broker via alpaca-py. Requires API keys and `pip install alpaca-py`.
    Defaults to the PAPER endpoint — you must explicitly pass paper=False to risk
    real money, and even then this engine will warn you.
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        from alpaca.trading.client import TradingClient
        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.paper = paper
        if not paper:
            print("⚠️  AlpacaBroker is in LIVE mode — real money is at risk.")

    def account_equity(self) -> float:
        return float(self.client.get_account().equity)

    def position_qty(self, symbol: str) -> float:
        try:
            return float(self.client.get_open_position(symbol).qty)
        except Exception:
            return 0.0   # alpaca raises when there is no open position

    def submit_market_order(self, symbol: str, qty: float) -> None:
        """qty: +buy / -sell. Uses a fractional market order, day time-in-force."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if abs(qty) < 1e-9:
            return
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY,
        )
        self.client.submit_order(req)
