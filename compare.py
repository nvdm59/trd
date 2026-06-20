#!/usr/bin/env python3
"""
Compare every single-asset strategy head-to-head on the same data.

    python compare.py                         # all strategies on SPY since 2005
    python compare.py --symbols QQQ --start 2010-01-01
    python compare.py --symbols SPY AAPL MSFT --start 2015-01-01 --plot

Runs each strategy with its DEFAULT parameters (no tuning — so nobody gets an
unfair, overfit edge), ranks them by Sharpe, and prints what each one does. Use
it to see which *mechanisms* suit an asset, then validate the winner with
walkforward.py before trusting it. Add --plot to also save results/compare.png.

Reminder: these are in-sample, default-parameter results on one asset. A strategy
topping this table is a hypothesis, not a paycheck.
"""
from __future__ import annotations

import argparse

import pandas as pd

from engine.data import load_history
from engine.optimize import run_once
from engine.risk import RiskLimits
from engine.strategy import REGISTRY
from plot import buy_and_hold

# One-line "effect" label per strategy for the results table.
EFFECTS = {
    "ma_crossover":       "trend  | rides trends, whipsaws in chop",
    "trend_momentum":     "trend  | fully long in uptrends, cash in bears",
    "donchian":           "trend  | breakout entries, many false starts",
    "macd":               "trend  | EMA timing, smoother than MA cross",
    "bollinger_breakout": "trend  | buys volatility expansions",
    "roc":                "moment.| pure momentum, no trend filter (riskier)",
    "vol_target_trend":   "trend+ | trend but trims size when volatile",
    "atr_trend":          "trend+ | trend with ATR trailing-stop exits",
    "mean_reversion":     "revert | fades z-score extremes, dies in trends",
    "rsi":                "revert | buys oversold bounces, dies in downtrends",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Compare all single-asset strategies.")
    p.add_argument("--symbols", nargs="+", default=["SPY"])
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--rank", default="sharpe",
                   choices=["sharpe", "cagr", "total_return", "max_drawdown"])
    p.add_argument("--plot", action="store_true", help="save results/compare.png")
    args = p.parse_args()

    print(f"Downloading {args.symbols} {args.start}..{args.end or 'today'} ...")
    history = load_history(args.symbols, args.start, args.end, args.interval)

    # equal budget per symbol so a multi-symbol run never secretly uses leverage
    max_weight = 1.0 / len(args.symbols)
    limits = RiskLimits(max_weight_per_symbol=max_weight)

    rows = []
    equities = {}
    for name, Cls in REGISTRY.items():
        run = run_once(Cls, {}, args.symbols, history, args.cash, limits)
        if run is None:
            continue
        rows.append((name, run.metrics))
        equities[name] = run.equity

    bench = buy_and_hold(history, args.cash)
    from engine.metrics import compute
    bench_m = compute(bench, 0)

    # max_drawdown ranks ascending (smaller is better); others descending
    reverse = args.rank != "max_drawdown"
    rows.sort(key=lambda r: getattr(r[1], args.rank), reverse=reverse)

    print(f"\nAll strategies on {', '.join(args.symbols)}, "
          f"{args.start}..{args.end or 'today'} (default params, ranked by {args.rank}):\n")
    print(f"  {'strategy':<20}{'CAGR':>7}{'Sharpe':>8}{'MaxDD':>8}{'Return':>9}"
          f"{'Trades':>7}   effect")
    print("  " + "-" * 96)
    for name, m in rows:
        print(f"  {name:<20}{m.cagr*100:>6.1f}%{m.sharpe:>8.2f}{m.max_drawdown*100:>7.1f}%"
              f"{m.total_return*100:>8.1f}%{m.num_trades:>7}   {EFFECTS.get(name, '')}")
    print("  " + "-" * 96)
    print(f"  {'buy_and_hold':<20}{bench_m.cagr*100:>6.1f}%{bench_m.sharpe:>8.2f}"
          f"{bench_m.max_drawdown*100:>7.1f}%{bench_m.total_return*100:>8.1f}%"
          f"{0:>7}   benchmark | just own the asset(s)")

    beats_sharpe = [n for n, m in rows if m.sharpe > bench_m.sharpe]
    beats_return = [n for n, m in rows if m.total_return > bench_m.total_return]
    print(f"\n  Beat buy & hold on Sharpe: {beats_sharpe or 'none'}")
    print(f"  Beat buy & hold on return: {beats_return or 'none'}")
    print("  (Beating on Sharpe = smoother ride; beating on return is much rarer "
          "in a bull market.)")

    if args.plot:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 7))
        for name, m in rows:
            ax.plot(equities[name].index, equities[name].values, lw=1.1, label=name)
        ax.plot(bench.index, bench.values, lw=2.0, color="black", ls="--",
                label="buy & hold")
        ax.set_yscale("log")
        ax.set_ylabel("Equity ($, log scale)")
        ax.set_xlabel("Date")
        ax.set_title(f"Single-asset strategies vs buy & hold — {', '.join(args.symbols)}")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(alpha=0.3, which="both")
        os.makedirs("results", exist_ok=True)
        fig.tight_layout()
        fig.savefig("results/compare.png", dpi=120)
        print("\n  Saved results/compare.png")


if __name__ == "__main__":
    main()
