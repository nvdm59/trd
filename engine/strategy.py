"""
Strategies.

A strategy looks at the price history *up to and including the current bar* and
returns a target weight in [-1, 1] for the symbol:

    +1.0  = go fully long (use the full per-symbol budget)
     0.0  = flat (no position)
    -1.0  = go fully short

The engine, not the strategy, turns a weight into share counts and orders. This
keeps strategies pure and easy to test. Subclass `Strategy` and implement
`target_weight`; set `warmup` to the bars you need before signalling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Strategy:
    name = "base"
    warmup = 0          # bars of history required before this strategy signals
    long_only = True    # if True, negative weights are clamped to 0

    def target_weight(self, history: pd.DataFrame) -> float:
        """history: DataFrame[open,high,low,close,volume] ending at the current bar."""
        raise NotImplementedError

    def _clamp(self, w: float) -> float:
        w = max(-1.0, min(1.0, float(w)))
        return max(0.0, w) if self.long_only else w


class MovingAverageCrossover(Strategy):
    """
    Trend / momentum. Go long when the fast MA is above the slow MA (uptrend),
    flat (or short) when it crosses below. Captures sustained trends; whipsaws
    in choppy, sideways markets.
    """
    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = True):
        assert fast < slow, "fast window must be shorter than slow window"
        self.fast, self.slow = fast, slow
        self.warmup = slow + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"]
        fast = close.tail(self.fast).mean()
        slow = close.tail(self.slow).mean()
        return self._clamp(1.0 if fast > slow else -1.0)


class MeanReversion(Strategy):
    """
    Mean reversion via z-score (Bollinger-style). When price stretches far below
    its recent average (z < -entry) we buy the dip; we exit as it reverts toward
    the mean (|z| < exit). Works in range-bound markets; gets run over in strong
    trends — the opposite regime from the crossover above.
    """
    name = "mean_reversion"

    def __init__(self, lookback: int = 20, entry_z: float = 1.0,
                 exit_z: float = 0.25, long_only: bool = True):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.warmup = lookback + 1
        self.long_only = long_only
        self._pos = 0.0   # remember current stance so we can hold until exit

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"].tail(self.lookback)
        mean, std = close.mean(), close.std()
        if std == 0 or pd.isna(std):
            return self._clamp(self._pos)
        z = (close.iloc[-1] - mean) / std

        if z <= -self.entry_z:          # stretched below mean -> buy
            self._pos = 1.0
        elif z >= self.entry_z:         # stretched above mean -> sell/short
            self._pos = -1.0
        elif abs(z) <= self.exit_z:     # reverted to mean -> close out
            self._pos = 0.0
        # else: hold the current stance
        return self._clamp(self._pos)


class TimeSeriesMomentum(Strategy):
    """
    Absolute (time-series) momentum with a long-term trend filter — built to beat
    buy-and-hold by capturing the upside while sidestepping deep bear markets.

    Unlike the MA crossover (which the per-symbol cap leaves mostly in cash), this
    goes FULLY long (weight 1.0) during healthy uptrends, so it can actually keep
    pace with — and then pull ahead of — buy-and-hold by avoiding the big crashes:

      Long (1.0) only when BOTH hold:
        * price is above its long-term moving average (trend_window), AND
        * the trailing return over `lookback` bars is positive (momentum).
      Otherwise flat (cash). It never shorts — in a downtrend it simply waits.

    To let it be fully invested you must raise the risk cap: run tools with
    --max-weight 1.0 for a single symbol (or 1/N per symbol for N symbols).
    """
    name = "trend_momentum"

    def __init__(self, trend_window: int = 200, lookback: int = 126):
        self.trend_window = trend_window
        self.lookback = lookback
        self.warmup = max(trend_window, lookback) + 1
        self.long_only = True

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"]
        sma = close.tail(self.trend_window).mean()
        price = close.iloc[-1]
        past = close.iloc[-(self.lookback + 1)]
        momentum = price / past - 1.0
        return self._clamp(1.0 if (price > sma and momentum > 0) else 0.0)


class DonchianBreakout(Strategy):
    """
    Channel breakout (the original "Turtle Traders" trend system). Go long when
    price makes a new `entry`-bar high (a breakout — momentum is asserting), and
    exit when it sags to a new `exit`-bar low. Catches big sustained moves; the
    cost is many small false-breakout losses in sideways markets ("death by a
    thousand whipsaws"). Reacts to *price levels*, where MA crossover reacts to
    *averages* — so it tends to enter trends earlier and rougher.
    """
    name = "donchian"

    def __init__(self, entry: int = 55, exit: int = 20, long_only: bool = True):
        self.entry, self.exit = entry, exit
        self.warmup = max(entry, exit) + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"]
        price = close.iloc[-1]
        upper = close.iloc[-(self.entry + 1):-1].max()  # prior highs, exclude now
        lower = close.iloc[-(self.exit + 1):-1].min()
        if price >= upper:
            self._pos = 1.0
        elif price <= lower:
            self._pos = 0.0
        return self._clamp(self._pos)


class MACDTrend(Strategy):
    """
    MACD trend filter. Long while the MACD line (fast EMA − slow EMA) is above its
    signal line, flat otherwise. Smoother and faster-reacting than a simple MA
    crossover because it uses exponential averages, but it whipsaws in the same
    choppy regimes all trend systems hate. A momentum/trend cousin, included so
    you can see whether EMA-based timing beats SMA-based timing on your data.
    """
    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 long_only: bool = True):
        assert fast < slow, "fast span must be shorter than slow span"
        self.fast, self.slow, self.signal = fast, slow, signal
        self.warmup = slow + signal + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"]
        macd = (close.ewm(span=self.fast, adjust=False).mean()
                - close.ewm(span=self.slow, adjust=False).mean())
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        return self._clamp(1.0 if macd.iloc[-1] > signal.iloc[-1] else 0.0)


class RSIReversion(Strategy):
    """
    RSI oscillator mean-reversion (COUNTER-trend). RSI measures how one-sided
    recent moves have been, 0–100. Buy when RSI falls below `oversold` (the asset
    has been beaten down and tends to bounce); exit when it climbs back above
    `overbought`. Profits from short-term overreactions in range-bound markets;
    its weakness is the mirror of trend systems — in a strong downtrend it keeps
    "buying the dip" all the way down. Different engine from the z-score
    `mean_reversion`: RSI reacts to the *streak* of up vs down days, not distance
    from a mean.
    """
    name = "rsi"

    def __init__(self, period: int = 14, oversold: float = 30.0,
                 overbought: float = 70.0, long_only: bool = True):
        self.period = period
        self.oversold, self.overbought = oversold, overbought
        self.warmup = period + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        delta = history["close"].diff()
        gain = delta.clip(lower=0).tail(self.period).mean()
        loss = (-delta.clip(upper=0)).tail(self.period).mean()
        rsi = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)
        if rsi <= self.oversold:
            self._pos = 1.0
        elif rsi >= self.overbought:
            self._pos = 0.0
        return self._clamp(self._pos)


class BollingerBreakout(Strategy):
    """
    Bollinger-band breakout (TREND, not reversion). The bands sit `num_std`
    standard deviations around a moving average, so they widen when volatility
    rises. Go long when price closes *above* the upper band — a volatility
    expansion that often marks the start of a move — and exit when it falls back
    to the middle band. Note this uses the bands the OPPOSITE way to mean
    reversion: it buys strength instead of fading it. Great when breakouts run;
    poor in choppy ranges where the breakout immediately fails.
    """
    name = "bollinger_breakout"

    def __init__(self, lookback: int = 20, num_std: float = 2.0,
                 long_only: bool = True):
        self.lookback, self.num_std = lookback, num_std
        self.warmup = lookback + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        window = history["close"].tail(self.lookback)
        mean, std = window.mean(), window.std()
        price = history["close"].iloc[-1]
        if price >= mean + self.num_std * std:
            self._pos = 1.0
        elif price <= mean:
            self._pos = 0.0
        return self._clamp(self._pos)


class RateOfChange(Strategy):
    """
    Pure rate-of-change momentum. Long whenever the trailing `lookback` return is
    above `threshold`, flat otherwise. This is `trend_momentum` stripped of its
    moving-average trend filter — included on purpose so you can measure what that
    filter is actually worth. Expect ROC to get in and out faster and to take
    deeper drawdowns than the filtered version, since nothing stops it re-entering
    during a bear-market bounce.
    """
    name = "roc"

    def __init__(self, lookback: int = 126, threshold: float = 0.0,
                 long_only: bool = True):
        self.lookback, self.threshold = lookback, threshold
        self.warmup = lookback + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"]
        roc = close.iloc[-1] / close.iloc[-(self.lookback + 1)] - 1.0
        return self._clamp(1.0 if roc > self.threshold else 0.0)


class VolatilityTargetTrend(Strategy):
    """
    Volatility-targeted trend. Same up/down decision as `trend_momentum` (long
    only above the long-term average), but instead of betting a flat 100% it sizes
    the position to hit a constant `target_vol` (annualized). In calm uptrends it
    rides near-full size; when the market gets jumpy it automatically trims — so
    it tends to deliver a smoother equity curve and shallower drawdowns than a
    fixed-size trend system, at the price of lagging in calm-but-strong rallies.
    Returns a *continuous* weight in (0, 1], not just 0/1.
    """
    name = "vol_target_trend"

    def __init__(self, trend_window: int = 200, vol_window: int = 20,
                 target_vol: float = 0.15, long_only: bool = True):
        self.trend_window, self.vol_window = trend_window, vol_window
        self.target_vol = target_vol
        self.warmup = max(trend_window, vol_window) + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        close = history["close"]
        if close.iloc[-1] <= close.tail(self.trend_window).mean():
            return 0.0  # downtrend -> cash
        daily_vol = close.pct_change().tail(self.vol_window).std()
        if not daily_vol or pd.isna(daily_vol):
            return self._clamp(1.0)
        ann_vol = daily_vol * (252 ** 0.5)
        return self._clamp(min(self.target_vol / ann_vol, 1.0))


class ATRTrendStop(Strategy):
    """
    Trend entry with an ATR trailing-stop exit — the "let winners run, cut losers"
    discipline written in code. Enter long when price is above its `sma_window`
    average; then trail a stop `atr_mult` × ATR below the highest price seen.
    Exit when price hits that ratcheting stop (or breaks the average). ATR (average
    true range) measures recent volatility, so the stop sits wider in wild markets
    and tighter in calm ones. Tends to give back less at the end of a trend than a
    plain MA exit, but a too-tight multiple stops you out of perfectly good trends.
    Uses high/low, not just close.
    """
    name = "atr_trend"

    def __init__(self, sma_window: int = 100, atr_window: int = 14,
                 atr_mult: float = 3.0, long_only: bool = True):
        self.sma_window, self.atr_window = sma_window, atr_window
        self.atr_mult = atr_mult
        self.warmup = max(sma_window, atr_window) + 1
        self.long_only = long_only
        self._pos = 0.0
        self._stop = None

    def target_weight(self, history: pd.DataFrame) -> float:
        high, low, close = history["high"], history["low"], history["close"]
        prev_close = close.shift(1)
        true_range = pd.concat([high - low, (high - prev_close).abs(),
                                (low - prev_close).abs()], axis=1).max(axis=1)
        atr = true_range.tail(self.atr_window).mean()
        price = close.iloc[-1]
        sma = close.tail(self.sma_window).mean()

        if self._pos == 0.0:
            if price > sma:
                self._pos = 1.0
                self._stop = price - self.atr_mult * atr
        else:
            self._stop = max(self._stop, price - self.atr_mult * atr)  # ratchet up
            if price <= self._stop or price < sma:
                self._pos, self._stop = 0.0, None
        return self._clamp(self._pos)


# ───────────────────────── second wave: 10 more ──────────────────────────────

class TripleMA(Strategy):
    """Trend. Long only when all three SMAs are stacked bullishly (fast > mid >
    slow) — a stricter, later, but cleaner trend confirmation than a 2-MA cross.
    Fewer false signals, but it gives back more at tops."""
    name = "triple_ma"

    def __init__(self, fast: int = 10, mid: int = 50, slow: int = 100,
                 long_only: bool = True):
        assert fast < mid < slow, "need fast < mid < slow"
        self.fast, self.mid, self.slow = fast, mid, slow
        self.warmup = slow + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        c = history["close"]
        f, m, s = c.tail(self.fast).mean(), c.tail(self.mid).mean(), c.tail(self.slow).mean()
        return self._clamp(1.0 if f > m > s else 0.0)


class StochasticReversion(Strategy):
    """Reversion. The stochastic oscillator (%K) measures where price sits within
    its recent high-low range, 0–100. Buy when %K is oversold (near the bottom of
    the range), exit when overbought. Counter-trend; good in ranges, bad in
    downtrends."""
    name = "stochastic"

    def __init__(self, period: int = 14, oversold: float = 20.0,
                 overbought: float = 80.0, long_only: bool = True):
        self.period, self.oversold, self.overbought = period, oversold, overbought
        self.warmup = period + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        hi = history["high"].tail(self.period).max()
        lo = history["low"].tail(self.period).min()
        close = history["close"].iloc[-1]
        k = 50.0 if hi == lo else (close - lo) / (hi - lo) * 100.0
        if k <= self.oversold:
            self._pos = 1.0
        elif k >= self.overbought:
            self._pos = 0.0
        return self._clamp(self._pos)


class CCITrend(Strategy):
    """Trend/breakout. The Commodity Channel Index flags how far price has pushed
    from its average in volatility units. Go long on a strong positive thrust
    (CCI > entry), exit when it falls back to zero. Catches momentum surges; whips
    in quiet markets."""
    name = "cci"

    def __init__(self, period: int = 20, entry: float = 100.0, long_only: bool = True):
        self.period, self.entry = period, entry
        self.warmup = period + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        tp = (history["high"] + history["low"] + history["close"]) / 3.0
        w = tp.tail(self.period)
        sma = w.mean()
        mean_dev = (w - sma).abs().mean()
        cci = 0.0 if mean_dev == 0 else (tp.iloc[-1] - sma) / (0.015 * mean_dev)
        if cci >= self.entry:
            self._pos = 1.0
        elif cci <= 0:
            self._pos = 0.0
        return self._clamp(self._pos)


class KeltnerBreakout(Strategy):
    """Trend. Like Bollinger breakout, but the channel is built from an EMA
    midline ± ATR (volatility) rather than standard deviation, so it's smoother
    and less spiky. Long on a close above the upper channel, exit back at the
    midline."""
    name = "keltner_breakout"

    def __init__(self, ema: int = 20, atr_window: int = 10, mult: float = 2.0,
                 long_only: bool = True):
        self.ema, self.atr_window, self.mult = ema, atr_window, mult
        self.warmup = max(ema, atr_window) + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        seg = history.tail(self.ema * 5 + 5)
        h, l, c = seg["high"], seg["low"], seg["close"]
        mid = c.ewm(span=self.ema, adjust=False).mean().iloc[-1]
        prev = c.shift(1)
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        atr = tr.tail(self.atr_window).mean()
        price = c.iloc[-1]
        if price >= mid + self.mult * atr:
            self._pos = 1.0
        elif price <= mid:
            self._pos = 0.0
        return self._clamp(self._pos)


class SuperTrend(Strategy):
    """Trend. The popular SuperTrend indicator: ATR bands that ratchet in the
    trend's direction and act as a trailing line, flipping long/flat when price
    closes through them. Volatility-aware and responsive — stays long through
    normal pullbacks but exits on a real break. Computed with the standard
    recursive band-carry logic over a trailing window."""
    name = "supertrend"

    def __init__(self, atr_window: int = 10, mult: float = 3.0, long_only: bool = True):
        self.atr_window, self.mult = atr_window, mult
        self.warmup = atr_window + 2
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        seg = history.tail(self.atr_window * 12 + 5)
        h = seg["high"].to_numpy(); l = seg["low"].to_numpy(); c = seg["close"].to_numpy()
        n, w = len(c), self.atr_window
        if n < w + 2:
            return self._clamp(self._pos)

        tr = np.empty(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        atr = np.convolve(tr, np.ones(w) / w, mode="valid")  # rolling mean, len n-w+1
        hl2 = (h[w - 1:] + l[w - 1:]) / 2.0
        upper = hl2 + self.mult * atr
        lower = hl2 - self.mult * atr
        cc = c[w - 1:]  # align closes with the band arrays

        fu, fl = upper.copy(), lower.copy()
        trend = 1
        for i in range(1, len(cc)):
            fu[i] = upper[i] if (upper[i] < fu[i - 1] or cc[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lower[i] if (lower[i] > fl[i - 1] or cc[i - 1] < fl[i - 1]) else fl[i - 1]
            if cc[i] > fu[i - 1]:
                trend = 1
            elif cc[i] < fl[i - 1]:
                trend = -1
        self._pos = 1.0 if trend > 0 else 0.0
        return self._clamp(self._pos)


class MultiTimeframeMomentum(Strategy):
    """Momentum. Votes the sign of returns over several horizons (1, 3, 6, 12
    months by default) and goes long only when a majority are positive. More
    robust than single-lookback momentum because it needs agreement across
    timeframes, so it sidesteps some one-off blips."""
    name = "tsmom_multi"

    def __init__(self, lookbacks=(21, 63, 126, 252), long_only: bool = True):
        self.lookbacks = tuple(lookbacks)
        self.warmup = max(self.lookbacks) + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        c = history["close"]
        votes = sum(1 for lb in self.lookbacks
                    if c.iloc[-1] / c.iloc[-(lb + 1)] - 1.0 > 0)
        return self._clamp(1.0 if votes / len(self.lookbacks) > 0.5 else 0.0)


class BollingerReversion(Strategy):
    """Reversion. The textbook Bollinger use: buy when price is stretched DOWN to
    the lower band, exit as it reverts to the middle. The mirror image of
    bollinger_breakout — buys weakness instead of strength. Good in ranges, gets
    steamrolled in downtrends."""
    name = "bollinger_reversion"

    def __init__(self, lookback: int = 20, num_std: float = 2.0, long_only: bool = True):
        self.lookback, self.num_std = lookback, num_std
        self.warmup = lookback + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        w = history["close"].tail(self.lookback)
        mean, std = w.mean(), w.std()
        price = history["close"].iloc[-1]
        if price <= mean - self.num_std * std:
            self._pos = 1.0
        elif price >= mean:
            self._pos = 0.0
        return self._clamp(self._pos)


class VolatilityBreakout(Strategy):
    """Trend (Larry Williams style). Go long when today's price exceeds yesterday's
    close by k × yesterday's range — i.e. a decisive expansion off the prior bar.
    Reacts very fast to momentum bursts; correspondingly noisy."""
    name = "volatility_breakout"

    def __init__(self, k: float = 0.5, long_only: bool = True):
        self.k = k
        self.warmup = 2
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        c, h, l = history["close"], history["high"], history["low"]
        if len(c) < 2:
            return 0.0
        prev_range = h.iloc[-2] - l.iloc[-2]
        return self._clamp(1.0 if c.iloc[-1] > c.iloc[-2] + self.k * prev_range else 0.0)


class ConnorsRSI2(Strategy):
    """Reversion (Connors "buy the dip in an uptrend"). Only acts when price is
    above its 200-day average (long-term uptrend), then buys very short-term
    oversold spikes (RSI-2 below ~10) and sells once RSI normalizes. The trend
    filter is what keeps it from catching falling knives like plain RSI does — a
    deliberately better-designed reversion strategy to compare against `rsi`."""
    name = "connors_rsi2"

    def __init__(self, period: int = 2, oversold: float = 10.0, exit_level: float = 50.0,
                 trend_window: int = 200, long_only: bool = True):
        self.period, self.oversold, self.exit_level = period, oversold, exit_level
        self.trend_window = trend_window
        self.warmup = trend_window + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        c = history["close"]
        delta = c.diff()
        gain = delta.clip(lower=0).tail(self.period).mean()
        loss = (-delta.clip(upper=0)).tail(self.period).mean()
        rsi = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)
        uptrend = c.iloc[-1] > c.tail(self.trend_window).mean()
        if not uptrend:
            self._pos = 0.0
        elif rsi <= self.oversold:
            self._pos = 1.0
        elif rsi >= self.exit_level:
            self._pos = 0.0
        return self._clamp(self._pos)


class ADXTrend(Strategy):
    """Trend with a strength filter. ADX gauges how strong a trend is (regardless
    of direction); +DI/-DI give direction. Go long only when the trend is both
    strong (ADX ≥ adx_min) and up (+DI > -DI). The filter keeps it OUT of weak,
    ch0ppy markets where trend systems bleed — its whole purpose is saying 'no
    trade' more often."""
    name = "adx_trend"

    def __init__(self, period: int = 14, adx_min: float = 20.0, long_only: bool = True):
        self.period, self.adx_min = period, adx_min
        self.warmup = 3 * period + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        seg = history.tail(4 * self.period + 5)
        h, l, c = seg["high"], seg["low"], seg["close"]
        up, down = h.diff(), -l.diff()
        plus_dm = ((up > down) & (up > 0)) * up
        minus_dm = ((down > up) & (down > 0)) * down
        prev = c.shift(1)
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.period).mean()
        plus_di = 100.0 * plus_dm.rolling(self.period).mean() / atr
        minus_di = 100.0 * minus_dm.rolling(self.period).mean() / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
        adx = dx.rolling(self.period).mean()
        if pd.isna(adx.iloc[-1]) or pd.isna(plus_di.iloc[-1]):
            return self._clamp(0.0)
        strong = adx.iloc[-1] >= self.adx_min
        bull = plus_di.iloc[-1] > minus_di.iloc[-1]
        return self._clamp(1.0 if (strong and bull) else 0.0)


# ───────────────────────── third wave: 10 more ───────────────────────────────

class ParabolicSAR(Strategy):
    """Trend (stop-and-reverse). The Parabolic SAR is a dot that trails price and
    accelerates as a trend extends; when price crosses it, the trend is presumed
    over. Long while price is above the SAR. Excellent at riding and then exiting
    clean trends; in choppy markets it flips constantly."""
    name = "psar"

    def __init__(self, af_step: float = 0.02, af_max: float = 0.2, long_only: bool = True):
        self.af_step, self.af_max = af_step, af_max
        self.warmup = 5
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        seg = history.tail(250)
        h = seg["high"].to_numpy(); l = seg["low"].to_numpy()
        n = len(h)
        if n < 3:
            return self._clamp(self._pos)
        up = True
        sar, ep, af = l[0], h[0], self.af_step
        for i in range(1, n):
            sar = sar + af * (ep - sar)
            if up:
                sar = min(sar, l[i - 1], l[max(i - 2, 0)])
                if h[i] > ep:
                    ep, af = h[i], min(af + self.af_step, self.af_max)
                if l[i] < sar:
                    up, sar, ep, af = False, ep, l[i], self.af_step
            else:
                sar = max(sar, h[i - 1], h[max(i - 2, 0)])
                if l[i] < ep:
                    ep, af = l[i], min(af + self.af_step, self.af_max)
                if h[i] > sar:
                    up, sar, ep, af = True, ep, h[i], self.af_step
        self._pos = 1.0 if up else 0.0
        return self._clamp(self._pos)


class Aroon(Strategy):
    """Trend timing. Aroon-Up/Down track how recently the highest high / lowest
    low occurred within a window. Long when Aroon-Up > Aroon-Down (a new high came
    more recently than a new low), i.e. an uptrend is in control. Good at catching
    trend *changes* early; noisy in flat markets."""
    name = "aroon"

    def __init__(self, period: int = 25, long_only: bool = True):
        self.period = period
        self.warmup = period + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        hh = history["high"].tail(self.period + 1).to_numpy()
        ll = history["low"].tail(self.period + 1).to_numpy()
        # most-recent extreme: larger argmax/argmin index = more recent
        return self._clamp(1.0 if int(hh.argmax()) > int(ll.argmin()) else 0.0)


class DMICross(Strategy):
    """Trend. The Directional Movement +DI/−DI lines measure the strength of up
    vs down moves. Long when +DI is above −DI (buyers dominating). Same machinery
    as `adx_trend` but WITHOUT the ADX strength gate, so it trades more often and
    in weaker trends — included to show what that gate is worth."""
    name = "dmi_cross"

    def __init__(self, period: int = 14, long_only: bool = True):
        self.period = period
        self.warmup = 2 * period + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        seg = history.tail(3 * self.period + 5)
        h, l, c = seg["high"], seg["low"], seg["close"]
        up, down = h.diff(), -l.diff()
        plus_dm = ((up > down) & (up > 0)) * up
        minus_dm = ((down > up) & (down > 0)) * down
        prev = c.shift(1)
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.period).mean()
        plus_di = 100.0 * plus_dm.rolling(self.period).mean() / atr
        minus_di = 100.0 * minus_dm.rolling(self.period).mean() / atr
        if pd.isna(plus_di.iloc[-1]) or pd.isna(minus_di.iloc[-1]):
            return self._clamp(0.0)
        return self._clamp(1.0 if plus_di.iloc[-1] > minus_di.iloc[-1] else 0.0)


class TRIX(Strategy):
    """Momentum oscillator. TRIX is the rate of change of a triple-smoothed EMA —
    the triple smoothing strips out short-term noise, so it whipsaws far less than
    raw momentum. Long when TRIX is positive (smoothed momentum rising). Smooth and
    late; best on persistent trends."""
    name = "trix"

    def __init__(self, period: int = 15, long_only: bool = True):
        self.period = period
        self.warmup = 4 * period
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        seg = history["close"].tail(self.period * 6 + 5)
        e1 = seg.ewm(span=self.period, adjust=False).mean()
        e2 = e1.ewm(span=self.period, adjust=False).mean()
        e3 = e2.ewm(span=self.period, adjust=False).mean()
        trix = e3.pct_change().iloc[-1]
        return self._clamp(1.0 if trix > 0 else 0.0)


class KAMATrend(Strategy):
    """Trend (adaptive). Kaufman's Adaptive Moving Average speeds up in clean
    trends and slows down in noise, using an 'efficiency ratio' of directional
    travel vs total travel. Long when price is above its KAMA. The adaptivity is
    meant to cut the whipsaws that kill fixed-length MAs in choppy markets."""
    name = "kama_trend"

    def __init__(self, er_period: int = 10, fast: int = 2, slow: int = 30,
                 long_only: bool = True):
        self.er_period, self.fast, self.slow = er_period, fast, slow
        self.warmup = er_period + 50
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        c = history["close"].tail(self.er_period + 60).to_numpy()
        n = len(c)
        if n < self.er_period + 2:
            return self._clamp(0.0)
        fast_sc, slow_sc = 2.0 / (self.fast + 1), 2.0 / (self.slow + 1)
        kama = c[0]
        for i in range(1, n):
            if i >= self.er_period:
                change = abs(c[i] - c[i - self.er_period])
                vol = np.abs(np.diff(c[i - self.er_period:i + 1])).sum()
                er = change / vol if vol != 0 else 0.0
            else:
                er = 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama = kama + sc * (c[i] - kama)
        return self._clamp(1.0 if c[-1] > kama else 0.0)


class WilliamsR(Strategy):
    """Reversion. Williams %R is where price sits in its recent high-low range,
    scaled −100 (at the lows) to 0 (at the highs). Buy when deeply oversold
    (below oversold), exit when it recovers above overbought. A close cousin of
    the stochastic oscillator; counter-trend, so it suffers in downtrends."""
    name = "williams_r"

    def __init__(self, period: int = 14, oversold: float = -80.0,
                 overbought: float = -20.0, long_only: bool = True):
        self.period, self.oversold, self.overbought = period, oversold, overbought
        self.warmup = period + 1
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        hh = history["high"].tail(self.period).max()
        ll = history["low"].tail(self.period).min()
        close = history["close"].iloc[-1]
        wr = -50.0 if hh == ll else -100.0 * (hh - close) / (hh - ll)
        if wr <= self.oversold:
            self._pos = 1.0
        elif wr >= self.overbought:
            self._pos = 0.0
        return self._clamp(self._pos)


class OBVTrend(Strategy):
    """Trend, VOLUME-based. On-Balance Volume adds the day's volume on up days and
    subtracts it on down days, so a rising OBV means volume is flowing in. Long
    when OBV is above its own moving average — using *participation*, not price, to
    confirm a trend. The first volume-driven strategy in the set."""
    name = "obv_trend"

    def __init__(self, ma_window: int = 20, long_only: bool = True):
        self.ma_window = ma_window
        self.warmup = ma_window + 2
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        seg = history.tail(self.ma_window * 3 + 5)
        c, v = seg["close"], seg["volume"]
        obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
        return self._clamp(1.0 if obv.iloc[-1] > obv.tail(self.ma_window).mean() else 0.0)


class VWMATrend(Strategy):
    """Trend, VOLUME-weighted. A volume-weighted moving-average crossover: bars
    with heavier volume count more toward the average, so the signal leans on
    prices that actually traded size. Long when the fast VWMA is above the slow
    VWMA. Like `ma_crossover`, but volume-aware."""
    name = "vwma_trend"

    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = True):
        assert fast < slow, "fast window must be shorter than slow"
        self.fast, self.slow = fast, slow
        self.warmup = slow + 1
        self.long_only = long_only

    def _vwma(self, c, v, n) -> float:
        cc, vv = c.tail(n), v.tail(n)
        s = vv.sum()
        return float((cc * vv).sum() / s) if s > 0 else float(cc.mean())

    def target_weight(self, history: pd.DataFrame) -> float:
        c, v = history["close"], history["volume"]
        return self._clamp(1.0 if self._vwma(c, v, self.fast) > self._vwma(c, v, self.slow)
                           else 0.0)


class MoneyFlowIndex(Strategy):
    """Reversion, VOLUME-weighted. The Money Flow Index is essentially a
    volume-weighted RSI: it weights up/down days by their dollar volume to gauge
    buying vs selling pressure. Buy when oversold, exit when overbought. Tends to
    give cleaner reversion signals than plain RSI because thin moves count less."""
    name = "mfi"

    def __init__(self, period: int = 14, oversold: float = 20.0,
                 overbought: float = 80.0, long_only: bool = True):
        self.period, self.oversold, self.overbought = period, oversold, overbought
        self.warmup = period + 2
        self.long_only = long_only
        self._pos = 0.0

    def target_weight(self, history: pd.DataFrame) -> float:
        seg = history.tail(self.period + 1)
        tp = (seg["high"] + seg["low"] + seg["close"]) / 3.0
        rmf = tp * seg["volume"]
        change = tp.diff()
        pos = rmf[change > 0].sum()
        neg = rmf[change < 0].sum()
        mfi = 100.0 if neg == 0 else 100.0 - 100.0 / (1.0 + pos / neg)
        if mfi <= self.oversold:
            self._pos = 1.0
        elif mfi >= self.overbought:
            self._pos = 0.0
        return self._clamp(self._pos)


class ROCAcceleration(Strategy):
    """Momentum acceleration (2nd derivative). Not just 'is price rising' but 'is
    the rise speeding UP': long only when trailing momentum is positive AND higher
    than it was `accel_lag` bars ago. Aims to catch trends while they're
    strengthening and step aside as they decelerate — earlier exits than plain
    momentum, at the cost of more false alarms."""
    name = "roc_accel"

    def __init__(self, lookback: int = 63, accel_lag: int = 21, long_only: bool = True):
        self.lookback, self.accel_lag = lookback, accel_lag
        self.warmup = lookback + accel_lag + 1
        self.long_only = long_only

    def target_weight(self, history: pd.DataFrame) -> float:
        c = history["close"]
        roc_now = c.iloc[-1] / c.iloc[-(self.lookback + 1)] - 1.0
        roc_past = (c.iloc[-(self.accel_lag + 1)]
                    / c.iloc[-(self.lookback + self.accel_lag + 1)] - 1.0)
        return self._clamp(1.0 if (roc_now > 0 and roc_now - roc_past > 0) else 0.0)


# Single source of truth for strategy names -> classes. Every CLI tool reads
# this, so adding a strategy here wires it into backtest, sweep, walk-forward,
# plot, and live trading at once. (These are SINGLE-asset strategies; the
# cross-asset rotation strategies live in engine/multi_asset.py.)
REGISTRY = {
    "ma_crossover": MovingAverageCrossover,
    "mean_reversion": MeanReversion,
    "trend_momentum": TimeSeriesMomentum,
    "donchian": DonchianBreakout,
    "macd": MACDTrend,
    "rsi": RSIReversion,
    "bollinger_breakout": BollingerBreakout,
    "roc": RateOfChange,
    "vol_target_trend": VolatilityTargetTrend,
    "atr_trend": ATRTrendStop,
    "triple_ma": TripleMA,
    "stochastic": StochasticReversion,
    "cci": CCITrend,
    "keltner_breakout": KeltnerBreakout,
    "supertrend": SuperTrend,
    "tsmom_multi": MultiTimeframeMomentum,
    "bollinger_reversion": BollingerReversion,
    "volatility_breakout": VolatilityBreakout,
    "connors_rsi2": ConnorsRSI2,
    "adx_trend": ADXTrend,
    "psar": ParabolicSAR,
    "aroon": Aroon,
    "dmi_cross": DMICross,
    "trix": TRIX,
    "kama_trend": KAMATrend,
    "williams_r": WilliamsR,
    "obv_trend": OBVTrend,
    "vwma_trend": VWMATrend,
    "mfi": MoneyFlowIndex,
    "roc_accel": ROCAcceleration,
}
