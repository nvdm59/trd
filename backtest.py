#!/usr/bin/env python3
"""
Run a historical backtest.

    python backtest.py                      # defaults: SPY+AAPL, MA crossover
    python backtest.py --symbols MSFT NVDA --strategy mean_reversion
    python backtest.py --start 2015-01-01 --strategy ma_crossover --verbose

This is where you SHOULD spend most of your time. A strategy that doesn't make
money in a realistic backtest (with the commission + slippage this engine
applies) will not make money live. Backtests still flatter reality — survivorship
bias, regime changes, and your own curve-fitting all lurk here — so treat a good
backtest as necessary, never sufficient.
"""
from __future__ import annotations

import argparse

from engine.data import load_history
from engine.engine import BacktestEngine
from engine.risk import RiskLimits
from engine.strategy import REGISTRY as STRATEGIES


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest a trading strategy.")
    p.add_argument("--symbols", nargs="+", default=["SPY", "AAPL"])
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d", help="1d, 1h, 15m, 5m, 1m")
    p.add_argument("--strategy", default="ma_crossover", choices=list(STRATEGIES))
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--max-weight", type=float, default=0.20,
                   help="max fraction of equity per symbol")
    p.add_argument("--commission-pct", type=float, default=0.0)
    p.add_argument("--slippage-pct", type=float, default=0.0005)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"Downloading {args.symbols} {args.start}..{args.end or 'today'} "
          f"({args.interval}) ...")
    history = load_history(args.symbols, args.start, args.end, args.interval)

    StrategyCls = STRATEGIES[args.strategy]
    strategies = {sym: StrategyCls() for sym in args.symbols}

    limits = RiskLimits(
        max_weight_per_symbol=args.max_weight,
        commission_pct=args.commission_pct,
        slippage_pct=args.slippage_pct,
    )
    engine = BacktestEngine(strategies, starting_cash=args.cash, limits=limits)

    print(f"Running '{args.strategy}' on {len(args.symbols)} symbol(s) ...")
    metrics = engine.run(history, verbose=args.verbose)
    print(metrics.report())

    if metrics.sharpe < 0.5:
        print("⚠️  Sharpe < 0.5 — this is not a strategy worth trading. Keep iterating.")


if __name__ == "__main__":
    main()
