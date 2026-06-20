#!/usr/bin/env python3
"""
Walk-forward (out-of-sample) validation — the most important script here.

A normal backtest optimizes and measures on the SAME data, so it always flatters
itself. This instead picks parameters on the past and grades them on the future
they never saw, fold by fold, then stitches those out-of-sample results into one
honest equity curve.

    python walkforward.py --symbols SPY AAPL --strategy ma_crossover
    python walkforward.py --strategy mean_reversion --folds 5 --start 2010-01-01

How to read it: compare the per-fold parameters (do they jump around wildly? that
means the "best" params are noise) and the combined out-of-sample Sharpe (this is
the number that actually predicts live performance — and it's almost always worse
than the in-sample backtest. That gap IS the overfitting).
"""
from __future__ import annotations

import argparse

from engine.data import load_history
from engine.optimize import PARAM_GRIDS, walk_forward
from engine.risk import RiskLimits


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward out-of-sample validation.")
    p.add_argument("--symbols", nargs="+", default=["SPY", "AAPL"])
    p.add_argument("--strategy", default="ma_crossover", choices=list(PARAM_GRIDS))
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=0.5,
                   help="fraction of history used for the initial training window")
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--max-weight", type=float, default=0.20)
    args = p.parse_args()

    print(f"Downloading {args.symbols} {args.start}..{args.end or 'today'} ...")
    history = load_history(args.symbols, args.start, args.end, args.interval)
    limits = RiskLimits(max_weight_per_symbol=args.max_weight)

    print(f"Walk-forward: '{args.strategy}', {args.folds} folds, "
          f"{args.train_frac:.0%} initial train window\n")
    wf = walk_forward(args.strategy, args.symbols, history, n_folds=args.folds,
                      train_frac=args.train_frac, cash=args.cash, limits=limits)

    print("Per-fold (params chosen on the past, scored on the unseen future):")
    print(f"  {'test window':<26}{'chosen params':<34}{'OOS Sharpe':>11}{'OOS ret':>10}")
    for f in wf.folds:
        window = f"{f.test_start.date()}..{f.test_end.date()}"
        params = ", ".join(f"{k}={v}" for k, v in f.chosen.items())
        print(f"  {window:<26}{params:<34}{f.metrics.sharpe:>11.2f}"
              f"{f.metrics.total_return * 100:>9.1f}%")

    distinct = {tuple(sorted(f.chosen.items())) for f in wf.folds}
    print(f"\nParameter stability: {len(distinct)} distinct param set(s) across "
          f"{len(wf.folds)} folds"
          + ("  ← unstable; the 'best' params are likely noise"
             if len(distinct) > 1 else "  ← stable"))

    print("\nCombined OUT-OF-SAMPLE performance (this is the honest number):")
    print(wf.oos_metrics.report())
    if wf.oos_metrics.sharpe < 0.5:
        print("⚠️  Out-of-sample Sharpe < 0.5 — this strategy does NOT generalize. "
              "Do not trade it.")
    else:
        print("✅ Survives out-of-sample. Necessary, not sufficient — paper trade "
              "it next.")


if __name__ == "__main__":
    main()
