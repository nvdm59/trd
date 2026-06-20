"""
Market data feeds.

A feed yields OHLCV bars in time order. Backtests read history from yfinance;
live trading would pull the latest bar from the broker. Everything downstream
(strategy, risk, portfolio) only sees one bar at a time, so a strategy can never
accidentally peek at the future.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List

import pandas as pd


@dataclass
class Bar:
    """One OHLCV candle for one symbol at one timestamp."""
    symbol: str
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_history(
    symbols: List[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Download historical OHLCV for each symbol from Yahoo Finance.

    interval: '1d', '1h', '15m', '5m', '1m' (intraday is limited to recent
    history by Yahoo — minutes only go back ~30 days, hours ~2 years).

    Returns {symbol: DataFrame[open, high, low, close, volume]} indexed by time.
    """
    import yfinance as yf

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.download(
            sym, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False,
        )
        if df is None or df.empty:
            raise ValueError(f"No data returned for {sym} ({start}..{end}, {interval})")
        # yfinance may return MultiIndex columns when given one ticker; flatten.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.dropna()
        out[sym] = df
    return out


def iter_bars(history: Dict[str, pd.DataFrame]) -> Iterator[Dict[str, Bar]]:
    """
    Walk every symbol's history forward in lock-step on a shared timeline.

    Yields {symbol: Bar} for each timestamp where at least one symbol trades.
    Symbols missing a bar at a given timestamp are simply absent from that dict.
    """
    union = sorted(set().union(*[df.index for df in history.values()]))
    for ts in union:
        snapshot: Dict[str, Bar] = {}
        for sym, df in history.items():
            if ts in df.index:
                row = df.loc[ts]
                snapshot[sym] = Bar(
                    symbol=sym, ts=ts,
                    open=float(row.open), high=float(row.high),
                    low=float(row.low), close=float(row.close),
                    volume=float(row.volume),
                )
        if snapshot:
            yield snapshot
