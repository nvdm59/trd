#!/usr/bin/env python3
"""
Paper trading on a real broker (Alpaca), using fake money.

This runs ONE decision cycle: pull recent bars, ask the strategy for a target
weight, size it against your (paper) account equity, and submit the order. Run it
on a schedule (cron / Task Scheduler) at the cadence your strategy trades at —
e.g. once a day after the close for a daily strategy.

Setup:
  1. Make a FREE Alpaca account, switch to "Paper" in the dashboard.
  2. Copy your paper API key + secret into a .env file (see .env.example).
  3. pip install -r requirements.txt
  4. python live_paper.py --symbols SPY AAPL --strategy ma_crossover

SAFETY: this defaults to the paper endpoint. It will NEVER touch a live-money
account unless you set ALPACA_PAPER=false yourself. Don't do that until a
strategy has earned it over months of paper trading.
"""
from __future__ import annotations

import argparse
import os

from engine.broker import AlpacaBroker
from engine.data import load_history
from engine.risk import RiskLimits, RiskManager
from engine.strategy import REGISTRY as STRATEGIES


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(description="One paper-trading decision cycle.")
    p.add_argument("--symbols", nargs="+", default=["SPY", "AAPL"])
    p.add_argument("--strategy", default="ma_crossover", choices=list(STRATEGIES))
    p.add_argument("--lookback-days", type=int, default=120,
                   help="how much recent history to pull for the signal")
    p.add_argument("--max-weight", type=float, default=0.20)
    args = p.parse_args()

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_API_SECRET (see .env.example).")

    broker = AlpacaBroker(key, secret, paper=paper)
    risk = RiskManager(RiskLimits(max_weight_per_symbol=args.max_weight))
    equity = broker.account_equity()
    print(f"Account equity: {equity:,.2f}  ({'PAPER' if paper else 'LIVE'})")

    # pull recent daily bars for the signal
    import datetime as dt
    start = (dt.date.today() - dt.timedelta(days=args.lookback_days)).isoformat()
    history = load_history(args.symbols, start=start, interval="1d")

    StrategyCls = STRATEGIES[args.strategy]
    for sym in args.symbols:
        strat = StrategyCls()
        hist = history[sym]
        if len(hist) < strat.warmup:
            print(f"{sym}: not enough history ({len(hist)} < {strat.warmup}); skipping.")
            continue

        weight = strat.target_weight(hist)
        price = float(hist["close"].iloc[-1])
        current = broker.position_qty(sym)
        target = risk.target_qty(weight, equity, price, current)
        delta = round(target - current, 6)

        print(f"{sym}: weight={weight:+.2f} target={target:+.3f} "
              f"current={current:+.3f} -> order {delta:+.3f} sh @ ~{price:,.2f}")
        if abs(delta) > 1e-4:
            broker.submit_market_order(sym, delta)
            print(f"  submitted {('BUY' if delta > 0 else 'SELL')} {abs(delta):.3f} {sym}")


if __name__ == "__main__":
    main()
