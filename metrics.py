"""
metrics.py -- performance metrics. Ported from the Kalman project (verified correct there).

Sharpe here is benchmarked against ZERO, not against cash. At a ~4-5% risk-free rate a
strategy returning 1%/yr has a deeply negative Sharpe versus T-bills even though this
function reports a positive number. Read every Sharpe below with that in mind.
"""
import numpy as np
import pandas as pd

ANN = np.sqrt(252)


def compute_metrics(pnl, capital, turnover=None):
    """
    pnl: cumulative P&L series (starts at 0.0), in currency units.
    capital: book size, for converting to percentages.
    turnover: optional per-day gross turnover (sum |dw|), for reporting.
    """
    pnl = np.asarray(pnl, dtype=float)
    daily = np.diff(pnl)
    sd = daily.std()
    equity = capital + pnl
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    out = {
        "final_pnl": round(float(pnl[-1]), 0),
        "ret_pct": round(100 * float(pnl[-1]) / capital, 2),
        "sharpe": round(float(daily.mean() / sd * ANN), 2) if sd > 0 else 0.0,
        "ann_vol_pct": round(100 * float(sd * ANN / capital), 1) if capital else 0.0,
        "max_dd_pct": round(100 * float(dd.min()), 1),
        "days": len(pnl) - 1,
    }
    if turnover is not None:
        t = np.asarray(turnover, dtype=float)
        out["avg_daily_turnover"] = round(float(t.mean()), 3)
        out["ann_turnover_x"] = round(float(t.mean()) * 252, 1)
    return out


def summarize(rows, by="strategy"):
    """Aggregate per-window rows into a verdict table with a t-test against zero."""
    from scipy import stats
    df = pd.DataFrame(rows)
    out = []
    for name, g in df.groupby(by, sort=False):
        r = g["net_pct"].values
        t, p = stats.ttest_1samp(r, 0) if len(r) > 1 else (np.nan, np.nan)
        out.append({
            by: name,
            "gross_pct": round(g["gross_pct"].mean(), 2),
            "net_pct": round(r.mean(), 2),
            "sharpe": round(g["sharpe"].mean(), 2),
            "max_dd_pct": round(g["max_dd_pct"].mean(), 1),
            "win": f"{int((r > 0).sum())}/{len(r)}",
            "t": round(float(t), 2),
            "p": round(float(p), 3),
        })
    return pd.DataFrame(out)
