#!/usr/bin/env python3
"""
Run a cross-asset ROTATION strategy on a universe of ETFs.

    # Dual Momentum across stocks/intl/bonds/gold, fleeing to bonds in bear markets
    python rotate.py --strategy dual_momentum --safe TLT

    python rotate.py --strategy relative_momentum --symbols SPY QQQ EFA EEM TLT GLD
    python rotate.py --strategy ew_trend --plot

Unlike the single-asset tools, these rank the symbols against each other and
rotate capital into the leaders (or into the safe asset / cash when nothing is
trending up). Compared against an equal-weight buy-and-hold of the same universe.

Why a universe matters: rotation has somewhere to GO when stocks fall (bonds,
gold), which is how it can beat buy-and-hold on both return and drawdown — the
thing single-asset timing usually can't do.
"""
from __future__ import annotations

import argparse

from engine.data import load_history
from engine.metrics import compute
from engine.multi_asset import ROTATION_REGISTRY, run_rotation
from engine.risk import RiskLimits
from plot import buy_and_hold

# A classic diversified rotation universe: US large, US tech, developed intl,
# emerging, long Treasuries (the "safe" asset), and gold.
DEFAULT_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "TLT", "GLD"]


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-asset rotation backtest.")
    p.add_argument("--strategy", default="dual_momentum", choices=list(ROTATION_REGISTRY))
    p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--safe", default="TLT",
                   help="safe asset for dual_momentum to flee to (must be in --symbols)")
    p.add_argument("--start", default="2007-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    print(f"Downloading {args.symbols} {args.start}..{args.end or 'today'} ...")
    history = load_history(args.symbols, args.start, args.end, "1d")
    limits = RiskLimits(max_weight_per_symbol=1.0)  # rotations may go 100% in one name

    print(f"Running rotation '{args.strategy}' on {len(args.symbols)} assets "
          f"(safe asset: {args.safe})\n")
    metrics, equity = run_rotation(args.strategy, args.symbols, history,
                                   args.cash, limits, safe_asset=args.safe)

    bench = buy_and_hold(history, args.cash)
    bench_m = compute(bench, 0)

    print(metrics.report())
    pct = lambda x: f"{x*100:,.1f}%"
    print(f"  {'':16}{'rotation':>12}{'equal-weight B&H':>18}")
    print(f"  {'CAGR':<16}{pct(metrics.cagr):>12}{pct(bench_m.cagr):>18}")
    print(f"  {'Sharpe':<16}{metrics.sharpe:>12.2f}{bench_m.sharpe:>18.2f}")
    print(f"  {'Max drawdown':<16}{pct(metrics.max_drawdown):>12}{pct(bench_m.max_drawdown):>18}")
    print(f"  {'Final equity':<16}{equity.iloc[-1]:>12,.0f}{bench.iloc[-1]:>18,.0f}")

    better_ret = metrics.total_return > bench_m.total_return
    better_dd = metrics.max_drawdown < bench_m.max_drawdown
    verdict = ("✅ beat buy & hold on BOTH return and drawdown — the rotation edge"
               if better_ret and better_dd else
               "↗ beat buy & hold on return only" if better_ret else
               "🛡 beat buy & hold on drawdown only (smoother, less total return)"
               if better_dd else "❌ lost to buy & hold on both")
    print(f"\n  {verdict}")

    if args.plot:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eq = equity.reindex(bench.index).ffill()
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(eq.index, eq.values, lw=1.6, label=f"{args.strategy} (rotation)")
        ax.plot(bench.index, bench.values, lw=1.2, ls="--", label="equal-weight buy & hold")
        ax.set_yscale("log")
        ax.set_ylabel("Equity ($, log)")
        ax.set_title(f"{args.strategy} vs buy & hold — {', '.join(args.symbols)}")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3, which="both")
        os.makedirs("results", exist_ok=True)
        fig.tight_layout()
        out = f"results/rotate_{args.strategy}.png"
        fig.savefig(out, dpi=120)
        print(f"  Saved {out}")


if __name__ == "__main__":
    main()
