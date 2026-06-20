#!/usr/bin/env python3
"""
Parameter sweep — test every parameter combination on the full history at once.

    python sweep.py --symbols SPY AAPL --strategy ma_crossover
    python sweep.py --strategy mean_reversion --start 2015-01-01 --top 15

⚠️  READ THIS. This is the most dangerous script in the repo, because it is the
easiest to fool yourself with. Picking the top row of this table and trading it
is textbook overfitting — with enough combinations, *something* always looks
great by luck alone. Use a sweep only to see how SENSITIVE results are to the
parameters (if the top 20 rows are all wildly different and only one shines, it's
noise). To actually choose parameters, use walkforward.py, which grades on data
the optimizer never saw.
"""
from __future__ import annotations

import argparse

from engine.data import load_history
from engine.optimize import (PARAM_GRIDS, STRATEGY_CLASSES, expand_grid, run_once)
from engine.risk import RiskLimits


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep strategy parameters (in-sample).")
    p.add_argument("--symbols", nargs="+", default=["SPY", "AAPL"])
    p.add_argument("--strategy", default="ma_crossover", choices=list(PARAM_GRIDS))
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--max-weight", type=float, default=0.20)
    p.add_argument("--rank", default="sharpe",
                   choices=["sharpe", "total_return", "cagr"])
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    print(f"Downloading {args.symbols} {args.start}..{args.end or 'today'} ...")
    history = load_history(args.symbols, args.start, args.end, args.interval)
    limits = RiskLimits(max_weight_per_symbol=args.max_weight)

    grid = PARAM_GRIDS[args.strategy]
    StrategyCls = STRATEGY_CLASSES[args.strategy]
    combos = expand_grid(grid)
    print(f"Sweeping {len(combos)} combinations of '{args.strategy}' ...\n")

    results = [run_once(StrategyCls, c, args.symbols, history, args.cash, limits)
               for c in combos]
    results = [r for r in results if r is not None]
    results.sort(key=lambda r: getattr(r.metrics, args.rank), reverse=True)

    print(f"  {'params':<36}{'Sharpe':>8}{'Return':>10}{'CAGR':>8}"
          f"{'MaxDD':>8}{'Trades':>8}")
    print("  " + "-" * 78)
    for r in results[:args.top]:
        m = r.metrics
        params = ", ".join(f"{k}={v}" for k, v in r.params.items())
        print(f"  {params:<36}{m.sharpe:>8.2f}{m.total_return * 100:>9.1f}%"
              f"{m.cagr * 100:>7.1f}%{m.max_drawdown * 100:>7.1f}%{m.num_trades:>8}")

    spread = (results[0].metrics.sharpe - results[-1].metrics.sharpe)
    print(f"\nSharpe spread across all {len(results)} combos: "
          f"{results[-1].metrics.sharpe:.2f} .. {results[0].metrics.sharpe:.2f} "
          f"(range {spread:.2f})")
    print("Reminder: the top row is NOT your strategy. Validate it with "
          "walkforward.py before trusting any of these numbers.")


if __name__ == "__main__":
    main()
