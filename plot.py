#!/usr/bin/env python3
"""
Plot a strategy's equity curve and drawdown, against a buy-and-hold benchmark.

    python plot.py --symbols SPY AAPL --strategy ma_crossover --start 2018-01-01
    python plot.py --strategy mean_reversion --out results/mr.png

Saves a PNG (default results/equity.png). The drawdown panel matters as much as
the equity panel: a curve that doubles but spends a year 40% underwater is one
almost nobody can actually hold through. Always compare to buy-and-hold — if your
strategy can't beat just owning the assets, it isn't earning its complexity.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from engine.data import load_history
from engine.optimize import STRATEGY_CLASSES, run_once
from engine.risk import RiskLimits


def buy_and_hold(history, cash: float) -> pd.Series:
    """Equal-weight, invest at the first bar, hold to the end. The benchmark."""
    idx = sorted(set().union(*[df.index for df in history.values()]))
    per_symbol = cash / len(history)
    closes = {s: df["close"].reindex(idx).ffill() for s, df in history.items()}
    shares = {s: per_symbol / float(c.dropna().iloc[0]) for s, c in closes.items()}
    equity = sum(shares[s] * closes[s] for s in history)
    return pd.Series(equity, index=pd.DatetimeIndex(idx), name="buy_and_hold").dropna()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot equity curve and drawdown.")
    p.add_argument("--symbols", nargs="+", default=["SPY", "AAPL"])
    p.add_argument("--strategy", default="ma_crossover", choices=list(STRATEGY_CLASSES))
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--max-weight", type=float, default=0.20)
    p.add_argument("--out", default="results/equity.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # render to file, no display needed
    import matplotlib.pyplot as plt

    print(f"Downloading {args.symbols} {args.start}..{args.end or 'today'} ...")
    history = load_history(args.symbols, args.start, args.end, args.interval)
    limits = RiskLimits(max_weight_per_symbol=args.max_weight)

    StrategyCls = STRATEGY_CLASSES[args.strategy]
    run = run_once(StrategyCls, {}, args.symbols, history, args.cash, limits)
    if run is None:
        raise SystemExit("Backtest produced no results.")
    strat_eq = run.equity
    bench_eq = buy_and_hold(history, args.cash).reindex(strat_eq.index).ffill()

    strat_dd = (strat_eq / strat_eq.cummax() - 1.0) * 100
    bench_dd = (bench_eq / bench_eq.cummax() - 1.0) * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(strat_eq.index, strat_eq.values, label=f"{args.strategy}", lw=1.6)
    ax1.plot(bench_eq.index, bench_eq.values, label="buy & hold",
             lw=1.2, alpha=0.7, ls="--")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"{args.strategy} vs buy & hold  —  {', '.join(args.symbols)}")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.fill_between(strat_dd.index, strat_dd.values, 0, alpha=0.4, label="strategy")
    ax2.plot(bench_dd.index, bench_dd.values, lw=1.0, alpha=0.7,
             ls="--", label="buy & hold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")
    print(run.metrics.report())
    print(f"Buy & hold final equity: {bench_eq.iloc[-1]:,.2f}  "
          f"(strategy: {strat_eq.iloc[-1]:,.2f})")


if __name__ == "__main__":
    main()
