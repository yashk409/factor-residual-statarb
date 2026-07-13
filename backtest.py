"""
backtest.py -- cross-sectional weight-vector P&L engine.

This is a REWRITE, not a port. The predecessor traded discrete share blocks per pair; the
object here is a single weight vector over the whole universe, rebalanced daily.

Two things are non-negotiable, both learned the hard way:

  * GROSS AND NET ARE ALWAYS REPORTED SEPARATELY. The pairs project spent months without
    knowing that its gross edge (+0.73%) was smaller than its cost bill. Never again.
  * WEIGHTS AT t ARE APPLIED TO RETURNS AT t+1. The weight vector for date t is decided at
    the close of t from data <= t; it earns the return of t+1. Off-by-one here is the single
    easiest way to manufacture a fake Sharpe.
"""
import numpy as np
import pandas as pd

from metrics import compute_metrics


def run(weights, returns, capital, t_cost):
    """
    weights : DataFrame (dates x tickers), gross-normalized. Decided at close of each date.
    returns : DataFrame (dates x tickers) of daily simple/log returns.
    t_cost  : cost per unit of turnover, one-way (e.g. 0.0005 = 5bps).

    Returns dict with gross/net cumulative P&L, per-day turnover, and metrics.
    """
    idx = weights.index
    cols = weights.columns

    # Align: weight for day t earns return for day t+1.
    fwd = returns.reindex(idx).shift(-1).reindex(columns=cols).fillna(0.0)
    W = weights.fillna(0.0)

    port_ret = (W.values * fwd.values).sum(axis=1)      # gross return per day, frac of book

    # Turnover: how much of the book we had to trade to get from yesterday's weights to
    # today's. Costs scale with this, not with the number of "trades".
    Wp = np.vstack([np.zeros((1, W.shape[1])), W.values[:-1]])
    turnover = np.abs(W.values - Wp).sum(axis=1)
    cost_ret = turnover * t_cost

    # Drop the final day (no forward return to earn).
    port_ret = port_ret[:-1]
    cost_ret = cost_ret[:-1]
    turnover = turnover[:-1]
    dates = idx[:-1]

    gross_pnl = np.concatenate([[0.0], np.cumsum(port_ret) * capital])
    net_pnl = np.concatenate([[0.0], np.cumsum(port_ret - cost_ret) * capital])

    mg = compute_metrics(gross_pnl, capital)
    mn = compute_metrics(net_pnl, capital, turnover=turnover)

    return {
        "dates": dates,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "turnover": turnover,
        "gross": mg,
        "net": mn,
        "gross_pct": mg["ret_pct"],
        "net_pct": mn["ret_pct"],
        "cost_pct": round(mg["ret_pct"] - mn["ret_pct"], 2),
        "gross_sharpe": mg["sharpe"],
        "sharpe": mn["sharpe"],
        "max_dd_pct": mn["max_dd_pct"],
        "ann_vol_pct": mn["ann_vol_pct"],
        "avg_daily_turnover": mn.get("avg_daily_turnover", 0.0),
    }


def benchmark(prices, ticker, dates, capital):
    """Buy & hold `ticker` over the same window, using real downloaded prices."""
    px = prices[ticker].reindex(dates).ffill()
    sh = capital / float(px.iloc[0])
    pnl = np.array([sh * (float(px.iloc[i]) - float(px.iloc[0])) for i in range(len(dates))])
    m = compute_metrics(pnl, capital)
    return {"gross_pct": m["ret_pct"], "net_pct": m["ret_pct"], "cost_pct": 0.0,
            "sharpe": m["sharpe"], "max_dd_pct": m["max_dd_pct"],
            "ann_vol_pct": m["ann_vol_pct"], "pnl": pnl}
