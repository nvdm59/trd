# trd_nvd — a small, honest algorithmic trading engine

A backtest-first, paper-trade-first framework for systematic trading. It executes
trades correctly; it does **not** promise profit. Read this whole file before you
risk a cent.

## ⚠️ Read this first

- **This is not High-Frequency Trading, and it can't be.** Real HFT needs servers
  co-located inside the exchange, direct data feeds costing tens of thousands a
  month, and C++/FPGA systems measured in nanoseconds. A Python script on a home
  laptop is ~50–200 ms from the market — you'd lose every race. HFT is closed to
  retail. What this engine does is *lower-frequency systematic trading* (minutes
  to days), which is real and accessible.
- **A working engine is not a working edge.** This code places and tracks trades
  flawlessly. Whether a *strategy* makes money is a separate, much harder problem.
  Most simple strategies break even or lose after fees and slippage.
- **Anyone selling a guaranteed "money bot" is scamming you.** If a strategy
  reliably printed money, they'd run it, not sell it.
- **Order of operations:** backtest → paper trade for months → only then consider
  real money, with capital you can afford to lose. Most people should stop after
  step two.

## How automated trading works

```
   data  ──►  strategy  ──►  risk  ──►  broker  ──►  portfolio
 (prices)   (buy/sell?)   (how much?)  (execute)   (P&L, cash)
```

- **data** (`engine/data.py`) — OHLCV bars, one at a time, in time order.
- **strategy** (`engine/strategy.py`) — looks at history up to *now* and returns a
  target weight in [-1, 1]. Ships with three: `ma_crossover` (trend), `mean_reversion`
  (z-score), and `trend_momentum` (absolute momentum + 200-day trend filter,
  designed to dodge bear markets — see below).
- **risk** (`engine/risk.py`) — turns a weight into a share count under caps:
  per-symbol size, gross exposure, and a max-drawdown kill switch.
- **broker** (`engine/broker.py`) — `SimBroker` for backtests/paper sim,
  `AlpacaBroker` for live/paper trading on a real broker.
- **portfolio** (`engine/portfolio.py`) — cash, positions, equity curve, blotter.
- **engine** (`engine/engine.py`) — the loop. Decides on bar *i*'s close, fills on
  bar *i+1*'s open, so a strategy can never peek at the future.
- **metrics** (`engine/metrics.py`) — return, CAGR, Sharpe, max drawdown.

## Quick start

```bash
cd ~/trd_nvd
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Backtest on free historical data
python backtest.py --symbols SPY AAPL --strategy ma_crossover --start 2018-01-01
python backtest.py --symbols SPY AAPL --strategy mean_reversion --start 2018-01-01

# 2) Validate it out-of-sample — the step that stops you fooling yourself
python walkforward.py --symbols SPY AAPL --strategy ma_crossover

# 3) (optional) See how sensitive it is to parameters
python sweep.py --symbols SPY AAPL --strategy ma_crossover

# 4) Picture is worth a thousand numbers — equity + drawdown vs buy & hold
python plot.py --symbols SPY AAPL --strategy ma_crossover   # -> results/equity.png

# 5) Cross-asset ROTATION (dual momentum: hold the strongest asset, flee to bonds)
python rotate.py --strategy dual_momentum --safe TLT        # -> beats buy & hold

# 6) Paper trade with fake money on a real broker
#    - make a free Alpaca account, switch to Paper, generate keys
cp .env.example .env          # then paste your PAPER keys into .env
python live_paper.py --symbols SPY AAPL --strategy ma_crossover
```

## The six analysis tools, and why they exist

| Script | Question it answers | Trap it protects you from |
|--------|--------------------|---------------------------|
| `backtest.py` | Did this make money on history? | — (start here, but never trust it alone) |
| `compare.py` | Which of the 24 strategies suits this asset/universe? | Falling for one idea before seeing the field |
| `rotate.py` | Can cross-asset rotation beat buy & hold? | Believing single-asset timing is the only option |
| `sweep.py` | How sensitive is it to its parameters? | Cherry-picking — the best row is usually luck |
| `walkforward.py` | Does it work on data it was *never tuned on*? | **Overfitting** — the #1 account killer |
| `plot.py` | What does the ride actually feel like vs just holding? | Ignoring drawdown; failing to beat buy & hold |

## Strategies and results — see `docs/`

There are **24 strategies**: 20 single-asset (trend / momentum / reversion) and 4
cross-asset rotation (dual momentum, relative momentum, equal-weight trend,
inverse-vol). Full catalog with what each does, when it shines and when it fails:
**[docs/STRATEGIES.md](docs/STRATEGIES.md)**. Measured, reproducible results with
realistic costs: **[docs/REPORT.md](docs/REPORT.md)**.

Race them all yourself:

```bash
python compare.py --symbols SPY --start 2008-01-01 --plot                 # single-asset shootout
python compare.py --symbols SPY QQQ EFA EEM TLT GLD --start 2007-01-01    # + rotation, on a universe
```

**The headline finding** (in-sample, default params — a hypothesis, not a
forecast): on **risk-adjusted** terms several trend/momentum strategies and
`inverse_vol` beat buy-and-hold (better Sharpe, roughly half the drawdown); on
**raw return**, only `dual_momentum` rotation beats it — because when stocks fall
it rotates into bonds/gold instead of sitting in cash. There is no free lunch:
every strategy that lowered drawdown also lowered return.

> Risk note: the drawdown **kill switch is OFF by default** (`max_drawdown=1.0`).
> It permanently flattens the book once tripped — a sensible *live* safety, but it
> freezes a backtest forever, so it's off for research. Set it (e.g. `0.25`) only
> for live/paper trading. The per-symbol weight cap and no-trade band are always on.

**Read results in this order:** a strategy must (1) be profitable in `backtest.py`,
(2) hold up in `walkforward.py` (out-of-sample Sharpe is the number that predicts
live results — it's almost always worse than the in-sample backtest, and that gap
is the overfitting), and (3) actually beat buy-and-hold in `plot.py`. Most
strategies fail at least one of these. That's the point — finding out *here*,
with fake money, is the whole game.

> Note on cash drag: with `--max-weight 0.20` and 2 symbols, the engine is at most
> 40% invested (the rest sits in cash), which caps both returns and drawdown.
> Raise `--max-weight` or add symbols to put more capital to work — at the cost of
> bigger swings.

## Can anything beat buy-and-hold? (`trend_momentum`)

Short answer: **not on raw return, in a bull market — and that's reality, not a
bug.** The `trend_momentum` strategy goes fully long only when price is above its
200-day average *and* trailing momentum is positive, otherwise it sits in cash.
It's the classic "ride the trend, dodge the crash" approach. Run it fully
invested on SPY:

```bash
python plot.py --symbols SPY --strategy trend_momentum --start 2005-01-01 --max-weight 1.0
python walkforward.py --symbols SPY --strategy trend_momentum --start 2005-01-01 --max-weight 1.0
```

Result over SPY since 2005 (and it survives out-of-sample):

| metric | trend_momentum | buy & hold |
|--------|---------------:|-----------:|
| CAGR | 7.1% | 10.9% |
| Sharpe | **0.67** | 0.64 |
| Max drawdown | **24%** | 55% |

It wins on **risk** (higher Sharpe, *half* the drawdown — it stepped aside in
2008) but loses on **raw return**, because in a long bull market it occasionally
whipsaws out and misses sharp recoveries. This is the honest takeaway of the
whole repo: a strategy's job isn't always more money — often it's *the same money
with a ride you can actually survive*. If you can't beat buy-and-hold on return,
the question becomes whether you can beat it on drawdown enough to matter to you.

Run `live_paper.py` on a schedule (cron / Task Scheduler) at your strategy's
cadence — e.g. once daily after the close for a daily strategy.

## Writing your own strategy

Subclass `Strategy`, set `warmup`, implement `target_weight`:

```python
from engine.strategy import Strategy

class Breakout(Strategy):
    name = "breakout"
    def __init__(self, lookback=55):
        self.lookback = lookback
        self.warmup = lookback + 1
    def target_weight(self, history):
        close = history["close"]
        return 1.0 if close.iloc[-1] >= close.tail(self.lookback).max() else 0.0
```

Then register it in the `STRATEGIES` dict in `backtest.py` / `live_paper.py`.

## What this does NOT include (on purpose)

Live tick streaming, options/futures, portfolio optimization, ML models, and a
GUI. Get a simple strategy working and honestly evaluated first — that's 90% of
the value and 100% of the discipline.

## Disclaimer

Educational software. Not financial advice. Trading involves substantial risk of
loss. You are solely responsible for any orders this code places. The authors
accept no liability.
