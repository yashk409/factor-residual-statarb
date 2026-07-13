"""
data.py -- price download + local cache.

Ported from the Kalman project's data_loader.download_prices. The `threads=False` flag is
NOT optional: yfinance's concurrent writes corrupt its own SQLite timezone cache on large
batch downloads ("unable to open database file"), which surfaces as FALSE "no price data"
errors on perfectly valid tickers. That bug silently shrank the universe for weeks.

Results are cached to parquet so the 460-ticker pull happens once, not once per experiment.
"""
import os

import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


def download_prices(tickers, start, end, use_cache=True):
    """Adjusted closes for `tickers`. Failed/empty tickers are dropped, not faked."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"px_{len(tickers)}_{start}_{end}.parquet"
    path = os.path.join(CACHE_DIR, key)
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)

    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False, threads=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all", axis=1).sort_index()
    if use_cache:
        close.to_parquet(path)
    return close


def log_returns(prices):
    """Daily log returns. Index-aligned, first row dropped."""
    import numpy as np
    return np.log(prices).diff().iloc[1:]


def tradeable_mask(prices, min_history):
    """
    Boolean (dates x tickers): True where a ticker has at least `min_history` prior
    observations AND a price today. Used to exclude names that had not listed yet, so the
    backtest never trades a stock before it existed.
    """
    valid = prices.notna()
    have = valid.cumsum() >= min_history
    return valid & have
