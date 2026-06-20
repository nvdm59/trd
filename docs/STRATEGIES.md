# Strategy catalog

24 strategies: **20 single-asset** (decide each symbol on its own) and **4
cross-asset rotation** (rank symbols against each other and rotate capital).

- Single-asset live in `engine/strategy.py`, registered in `REGISTRY`. Run with
  `backtest.py`, `compare.py`, `sweep.py`, `walkforward.py`, `plot.py`.
- Rotation strategies live in `engine/multi_asset.py`, registered in
  `ROTATION_REGISTRY`. Run with `rotate.py` (and they appear in `compare.py` when
  you pass more than one symbol).

Families: **trend** (buy strength, sell weakness), **momentum** (buy what has
risen most), **reversion** (buy what has fallen, bet on a bounce), **rotation**
(cross-asset allocation). Measured results are in [REPORT.md](REPORT.md).

## Single-asset (20)

| name | family | what it does | shines when | fails when |
|------|--------|--------------|-------------|------------|
| `trend_momentum` | trend | long above 200-SMA *and* positive momentum, else cash | strong trends with deep bears to dodge | choppy/range-bound markets |
| `ma_crossover` | trend | long when fast SMA > slow SMA | sustained trends | sideways chop (whipsaws) |
| `triple_ma` | trend | long when fast > mid > slow SMAs all stacked up | clean, established trends | choppy markets; late at tops |
| `donchian` | trend | long on N-bar high breakout, exit on M-bar low | big breakouts that run | false breakouts in ranges |
| `macd` | trend | long when MACD line > signal line | smooth trending moves | choppy regimes |
| `bollinger_breakout` | trend | long when price breaks above the upper band | volatility-expansion breakouts | failed breakouts in ranges |
| `keltner_breakout` | trend | breakout above an EMA±ATR channel | smooth trending breakouts | whippy ranges |
| `supertrend` | trend | ATR trailing line that flips long/flat | trending markets, rides pullbacks | sideways chop |
| `cci` | trend | long on a strong positive CCI thrust | momentum surges | quiet markets |
| `adx_trend` | trend | long only when ADX says the trend is strong & up | strong, persistent trends | weak/rangey markets (sits out) |
| `volatility_breakout` | trend | long on a big expansion vs the prior bar | fast intraday-style momentum | **daily data — very noisy, avoid** |
| `roc` | momentum | long when trailing return > threshold (no trend filter) | persistent momentum | bear bounces (nothing filters it) |
| `tsmom_multi` | momentum | majority vote of 1/3/6/12-month momentum signs | robust, multi-horizon trends | sharp regime turns |
| `vol_target_trend` | trend+ | trend, but sizes the position to a target volatility | smoother ride, shallower drawdowns | calm-but-strong rallies (under-sized) |
| `atr_trend` | trend+ | trend entry, ATR trailing-stop exit | letting winners run, cutting losers | tight stops shake you out |
| `mean_reversion` | reversion | fades z-score extremes back to the mean | range-bound markets | strong trends (fades the move) |
| `rsi` | reversion | buys RSI-oversold, sells RSI-overbought | choppy ranges | sustained downtrends (catches knives) |
| `stochastic` | reversion | buys stochastic-oversold within the range | ranges | downtrends |
| `bollinger_reversion` | reversion | buys the lower band, exits at the middle | mean-reverting assets | trending assets |
| `connors_rsi2` | reversion | dip-buy (RSI-2) **only** when above the 200-SMA | pullbacks inside an uptrend | choppy bear markets |

## Cross-asset rotation (4)

These need a *universe* (several symbols) and rebalance **monthly** by default.
A bond ETF (e.g. `TLT`) in the universe gives momentum a safe harbor in bear
markets. The reason rotation can beat buy-and-hold on *both* return and drawdown:
when stocks fall it has somewhere to go (bonds/gold/cash) instead of just sitting.

| name | family | what it does | trade-off |
|------|--------|--------------|-----------|
| `dual_momentum` | rotation | hold the single strongest asset; flee to bonds/cash if even the best is falling | highest return, but concentrated → volatile, deeper drawdown |
| `relative_momentum` | rotation | hold the top-N strongest assets (that are also trending up), equal weight | more diversified than dual, usually lower return |
| `ew_trend` | rotation | hold every asset above its 200-SMA, equal weight; rest to cash | smoothest, shallowest drawdown; lags in big rallies |
| `inverse_vol` | rotation | always fully invested, each asset weighted by 1/volatility | best risk-adjusted (Sharpe); no market timing |

## Adding your own

Single-asset: subclass `Strategy`, set `warmup`, implement `target_weight(history)
-> weight in [-1, 1]`, add it to `REGISTRY` (and a grid to `PARAM_GRIDS` for
sweep/walk-forward). Rotation: subclass `PortfolioStrategy`, implement
`target_weights(histories) -> {symbol: weight}`, add it to `ROTATION_REGISTRY`.
