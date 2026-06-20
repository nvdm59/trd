"""
Parameter search and walk-forward validation.

This module exists to fight the single biggest way backtests lie to you:
**overfitting**. If you try enough parameter combinations on one fixed stretch
of history, some will look brilliant by pure luck. The only honest test is to
choose parameters on past data and measure them on *future* data the optimizer
never saw — that's what `walk_forward` does.

Shared helpers:
  * expand_grid   — turn a {param: [values]} grid into concrete combos
  * run_once      — run one backtest, return metrics + equity curve
  * best_params   — pick the highest-Sharpe combo on a given slice
  * walk_forward  — anchored out-of-sample validation across rolling folds
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import pandas as pd

from .engine import BacktestEngine
from .metrics import Metrics, compute
from .risk import RiskLimits
from .strategy import REGISTRY, Strategy

# Parameter grids to search per strategy. Keep these SMALL — every extra combo
# is another lottery ticket for overfitting.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "ma_crossover": {"fast": [10, 20, 30, 50], "slow": [50, 100, 150, 200]},
    "mean_reversion": {"lookback": [10, 20, 30], "entry_z": [1.0, 1.5, 2.0],
                       "exit_z": [0.25, 0.5]},
    "trend_momentum": {"trend_window": [100, 150, 200], "lookback": [63, 126, 252]},
    "donchian": {"entry": [20, 55, 100], "exit": [10, 20, 55]},
    "macd": {"fast": [8, 12], "slow": [21, 26], "signal": [9]},
    "rsi": {"period": [7, 14], "oversold": [25, 30], "overbought": [70, 75]},
    "bollinger_breakout": {"lookback": [20, 50], "num_std": [1.5, 2.0, 2.5]},
    "roc": {"lookback": [63, 126, 252], "threshold": [0.0, 0.05]},
    "vol_target_trend": {"trend_window": [100, 200], "vol_window": [20],
                         "target_vol": [0.10, 0.15, 0.20]},
    "atr_trend": {"sma_window": [100, 200], "atr_window": [14],
                  "atr_mult": [2.0, 3.0, 4.0]},
    "triple_ma": {"fast": [10, 20], "mid": [50], "slow": [100, 200]},
    "stochastic": {"period": [14], "oversold": [15, 20], "overbought": [80, 85]},
    "cci": {"period": [14, 20], "entry": [100, 150]},
    "keltner_breakout": {"ema": [20], "atr_window": [10], "mult": [1.5, 2.0, 2.5]},
    "supertrend": {"atr_window": [7, 10], "mult": [2.0, 3.0]},
    "tsmom_multi": {},  # tuple lookbacks aren't grid-friendly; default only
    "bollinger_reversion": {"lookback": [20], "num_std": [1.5, 2.0, 2.5]},
    "volatility_breakout": {"k": [0.3, 0.5, 0.8]},
    "connors_rsi2": {"oversold": [5, 10], "exit_level": [50, 70]},
    "adx_trend": {"period": [14], "adx_min": [20, 25]},
    "psar": {"af_step": [0.01, 0.02], "af_max": [0.2]},
    "aroon": {"period": [14, 25]},
    "dmi_cross": {"period": [10, 14]},
    "trix": {"period": [9, 15]},
    "kama_trend": {"er_period": [10], "fast": [2], "slow": [30]},
    "williams_r": {"period": [14], "oversold": [-80, -85], "overbought": [-20]},
    "obv_trend": {"ma_window": [20, 50]},
    "vwma_trend": {"fast": [10, 20], "slow": [50]},
    "mfi": {"period": [14], "oversold": [20], "overbought": [80]},
    "roc_accel": {"lookback": [63, 126], "accel_lag": [21]},
}

# The strategy registry lives in strategy.py; re-exported here so the analysis
# tools have a single import for both the classes and their grids.
STRATEGY_CLASSES: Dict[str, Type[Strategy]] = REGISTRY


def expand_grid(grid: Dict[str, list]) -> List[dict]:
    """{'a':[1,2],'b':[3]} -> [{'a':1,'b':3}, {'a':2,'b':3}]."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def slice_history(history: Dict[str, pd.DataFrame],
                  start: Optional[pd.Timestamp] = None,
                  end: Optional[pd.Timestamp] = None) -> Dict[str, pd.DataFrame]:
    return {sym: df.loc[start:end] for sym, df in history.items()}


def union_index(history: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    return sorted(set().union(*[df.index for df in history.values()]))


@dataclass
class RunResult:
    params: dict
    metrics: Metrics
    equity: pd.Series


def run_once(StrategyCls: Type[Strategy], params: dict, symbols: List[str],
             history: Dict[str, pd.DataFrame], cash: float = 100_000.0,
             limits: Optional[RiskLimits] = None) -> Optional[RunResult]:
    """
    Run a single backtest with one parameter set. Returns None if the params are
    invalid (e.g. fast >= slow) or there isn't enough data to score.
    """
    try:
        strategies = {sym: StrategyCls(**params) for sym in symbols}
    except (AssertionError, ValueError):
        return None
    engine = BacktestEngine(strategies, starting_cash=cash, limits=limits)
    try:
        metrics = engine.run(history)
    except ValueError:
        return None  # too few bars to compute metrics
    return RunResult(params=params, metrics=metrics,
                     equity=engine.portfolio.equity_series())


def best_params(StrategyCls: Type[Strategy], grid: Dict[str, list],
                symbols: List[str], history: Dict[str, pd.DataFrame],
                cash: float = 100_000.0, limits: Optional[RiskLimits] = None,
                rank: str = "sharpe") -> Optional[RunResult]:
    """Search the grid on `history`, return the best result by `rank` metric."""
    results = [run_once(StrategyCls, p, symbols, history, cash, limits)
               for p in expand_grid(grid)]
    results = [r for r in results if r is not None]
    if not results:
        return None
    return max(results, key=lambda r: getattr(r.metrics, rank))


def max_warmup(StrategyCls: Type[Strategy], grid: Dict[str, list]) -> int:
    """Largest warmup any combo in the grid needs (for the lookback buffer)."""
    warm = 0
    for p in expand_grid(grid):
        try:
            warm = max(warm, StrategyCls(**p).warmup)
        except (AssertionError, ValueError):
            continue
    return warm


@dataclass
class Fold:
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen: dict
    metrics: Metrics


@dataclass
class WalkForwardResult:
    folds: List[Fold]
    oos_equity: pd.Series        # stitched out-of-sample equity curve
    oos_metrics: Metrics         # metrics on the stitched curve


def walk_forward(strategy_name: str, symbols: List[str],
                 history: Dict[str, pd.DataFrame], n_folds: int = 4,
                 train_frac: float = 0.5, cash: float = 100_000.0,
                 limits: Optional[RiskLimits] = None) -> WalkForwardResult:
    """
    Anchored walk-forward validation.

    The first `train_frac` of the timeline is the initial training window. The
    remaining data is split into `n_folds` equal test segments. For each fold we
    pick the best parameters on everything BEFORE the segment (expanding window),
    then trade that segment with those frozen parameters. The per-fold
    out-of-sample returns are chained into one continuous equity curve — the
    closest thing to "what you'd actually have experienced."
    """
    StrategyCls = STRATEGY_CLASSES[strategy_name]
    grid = PARAM_GRIDS[strategy_name]
    limits = limits or RiskLimits()

    idx = union_index(history)
    if len(idx) < 100:
        raise ValueError("Not enough history for a meaningful walk-forward.")

    buffer = max_warmup(StrategyCls, grid)
    train_end_pos = int(len(idx) * train_frac)
    test_positions = list(range(train_end_pos, len(idx)))
    fold_size = max(1, len(test_positions) // n_folds)

    folds: List[Fold] = []
    stitched: List[tuple] = []
    equity_level = cash

    for f in range(n_folds):
        test_start_pos = train_end_pos + f * fold_size
        test_end_pos = (train_end_pos + (f + 1) * fold_size
                        if f < n_folds - 1 else len(idx))
        if test_start_pos >= len(idx):
            break

        train_hist = slice_history(history, idx[0], idx[test_start_pos - 1])
        chosen = best_params(StrategyCls, grid, symbols, train_hist, cash, limits)
        if chosen is None:
            continue

        # run the test segment with a warmup buffer so the strategy isn't flat at
        # the start, then keep only the genuine out-of-sample portion
        buf_pos = max(0, test_start_pos - buffer)
        test_hist = slice_history(history, idx[buf_pos], idx[test_end_pos - 1])
        run = run_once(StrategyCls, chosen.params, symbols, test_hist, cash, limits)
        if run is None or run.equity.empty:
            continue

        test_start_ts = idx[test_start_pos]
        oos = run.equity[run.equity.index >= test_start_ts]
        rets = oos.pct_change().dropna()
        for ts, r in rets.items():
            equity_level *= (1.0 + r)
            stitched.append((ts, equity_level))

        folds.append(Fold(
            train_start=idx[0], test_start=test_start_ts,
            test_end=idx[test_end_pos - 1], chosen=chosen.params,
            metrics=run.metrics,
        ))

    if not stitched:
        raise ValueError("Walk-forward produced no out-of-sample equity.")
    ts, eq = zip(*stitched)
    oos_equity = pd.Series(eq, index=pd.DatetimeIndex(ts), name="oos_equity")
    total_trades = sum(f.metrics.num_trades for f in folds)
    return WalkForwardResult(folds=folds, oos_equity=oos_equity,
                             oos_metrics=compute(oos_equity, total_trades))
