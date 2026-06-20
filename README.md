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
  target weight in [-1, 1]. Ships with a trend strategy (MA crossover) and a
  mean-reversion strategy (z-score).
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

# 5) Paper trade with fake money on a real broker
#    - make a free Alpaca account, switch to Paper, generate keys
cp .env.example .env          # then paste your PAPER keys into .env
python live_paper.py --symbols SPY AAPL --strategy ma_crossover
```

## The four analysis tools, and why they exist

| Script | Question it answers | Trap it protects you from |
|--------|--------------------|---------------------------|
| `backtest.py` | Did this make money on history? | — (start here, but never trust it alone) |
| `sweep.py` | How sensitive is it to its parameters? | Cherry-picking — the best row is usually luck |
| `walkforward.py` | Does it work on data it was *never tuned on*? | **Overfitting** — the #1 account killer |
| `plot.py` | What does the ride actually feel like vs just holding? | Ignoring drawdown; failing to beat buy & hold |

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
