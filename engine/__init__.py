"""
trd_nvd — a small, honest algorithmic trading engine.

Pipeline:  data -> strategy -> risk -> broker -> portfolio

Start with backtest.py (historical), then live_paper.py (fake money on a real
broker). Never point this at a live-money account until a strategy has survived
realistic backtests AND a long paper-trading run. See README.md.
"""

__version__ = "0.1.0"
