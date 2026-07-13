"""
residuals.py -- Avellaneda & Lee (2010) PCA factor-residual mean reversion.

WHY THIS EXISTS (read before changing anything):

The predecessor project traded cointegrated PAIRS. It failed for two reasons, both of which
this module structurally eliminates:

  1. SELECTION BIAS. Pairs had to be screened, and screening looked at data that overlapped
     the test period. Sharpe went 1.55 -> -0.20 the moment the screen was made honest.
     ==> Here there is NO selection. Every stock in the universe gets a signal, every day.
         There is no screener, so there is nothing to bias.

  2. NO BREADTH. 30 pairs, IC ~= 0.14. IR = IC * sqrt(breadth) leaves nothing.
     ==> Here every stock is a bet: ~460 signals instead of 30.

THE METHOD (Avellaneda & Lee 2010, "Statistical Arbitrage in the U.S. Equities Market"):

  1. Rolling `pca_win` days of returns -> correlation matrix -> top `n_factors` eigenvectors.
  2. Eigenportfolio weights Q[j,i] = V[i,j] / sigma_i  (risk-scaled, per the paper).
     Factor returns F_j = sum_i Q[j,i] * R_i  -- these are TRADEABLE portfolios.
  3. Over a shorter `reg_win`, regress each stock on the factors:
         R_i = beta_i0 + sum_j beta_ij F_j + eps_i
     By APT, the factor portfolio is stock i's "fair price"; eps is the temporary deviation.
  4. Cumulate the residual: X_i(t) = sum_{s<=t} eps_i(s).  Fit an Ornstein-Uhlenbeck
     process to X_i and form the s-score  s_i = (X_i - m_eq) / sigma_eq.
  5. Trade the s-score: buy the residual when it is far below equilibrium, short it when far
     above. Reject stocks whose reversion is too slow to round-trip (kappa filter).
  6. The position is the stock MINUS its factor hedge, so the book is factor-neutral by
     construction rather than by a fitted beta on one other stock.

EVERYTHING IS ROLLING. Every parameter at time t is estimated from a trailing window ending
at t. There is no train/test split, no fitted hyperparameter, and no lookahead available --
which is exactly why this is a fair test in a way the pairs project never was.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def eigen_portfolios(R_win, n_factors):
    """
    PCA on the correlation matrix of returns (Avellaneda-Lee eq. 8-10).

    R_win : (T, N) returns, no NaN.
    Returns Q : (n_factors, N) eigenportfolio weights in ORIGINAL return space, and the
    fraction of variance explained.
    """
    sigma = R_win.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-12, np.nan, sigma)
    Y = (R_win - R_win.mean(axis=0)) / sigma          # standardized returns
    C = np.corrcoef(Y, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)

    vals, vecs = np.linalg.eigh(C)                    # ascending
    order = np.argsort(vals)[::-1][:n_factors]
    V = vecs[:, order]                                # (N, m)
    explained = float(vals[order].sum() / max(vals.sum(), 1e-12))

    # Risk-scaled eigenportfolio weights: Q[j, i] = V[i, j] / sigma_i
    Q = (V / sigma[:, None]).T                        # (m, N)
    return Q, explained


def ou_scores(resid_cum):
    """
    Fit an OU process to each column of the cumulated residual and return (s, kappa).

    Discretized OU is an AR(1):  X(t) = a + b*X(t-1) + zeta
        kappa     = -log(b) * 252          (mean-reversion speed, per year)
        m_eq      = a / (1 - b)            (equilibrium level)
        sigma_eq  = sqrt(var(zeta) / (1 - b^2))
        s         = (X(T) - m_eq) / sigma_eq

    Vectorized across all stocks at once. Columns with b outside (0, 1) are not
    mean-reverting and get kappa = 0, which the caller's filter rejects.
    """
    X = resid_cum
    x0, x1 = X[:-1], X[1:]                            # (T-1, N)
    n = x0.shape[0]

    m0, m1 = x0.mean(axis=0), x1.mean(axis=0)
    cov = ((x0 - m0) * (x1 - m1)).sum(axis=0) / (n - 1)
    var0 = ((x0 - m0) ** 2).sum(axis=0) / (n - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(var0 > 1e-16, cov / var0, np.nan)
        a = m1 - b * m0
        zeta = x1 - (a + b * x0)
        var_z = (zeta ** 2).sum(axis=0) / max(n - 2, 1)

        ok = np.isfinite(b) & (b > 1e-6) & (b < 1.0 - 1e-6)
        kappa = np.where(ok, -np.log(np.clip(b, 1e-9, 1 - 1e-9)) * TRADING_DAYS, 0.0)
        m_eq = np.where(ok, a / (1.0 - b), 0.0)
        sig_eq = np.where(ok, np.sqrt(np.clip(var_z / (1.0 - b ** 2), 1e-16, None)), np.nan)
        s = np.where(ok & (sig_eq > 1e-12), (X[-1] - m_eq) / sig_eq, np.nan)

    return s, kappa


def decide(s, prev, sbo=1.25, sso=1.25, sbc=0.50, ssc=0.75):
    """
    Avellaneda-Lee trading rules, with hysteresis (hold between the open and close bands).

        s < -sbo  -> open LONG residual   (+1)
        s > +sso  -> open SHORT residual  (-1)
        long  closes when s > -sbc
        short closes when s < +ssc

    `s` and `prev` are vectors; returns the new signal vector.
    """
    sig = prev.copy()
    nan = ~np.isfinite(s)

    sig = np.where(s < -sbo, 1.0, sig)                # open long
    sig = np.where(s > sso, -1.0, sig)                # open short
    sig = np.where((prev > 0) & (s > -sbc), 0.0, sig)  # close long
    sig = np.where((prev < 0) & (s < ssc), 0.0, sig)   # close short
    sig = np.where(nan, 0.0, sig)
    return sig


def build_weights(signal, betas, Q):
    """
    Turn per-stock residual signals into a factor-neutral portfolio weight vector.

    Position for stock i is:  signal_i * ( e_i  -  sum_j beta_ij * Q_j )
    i.e. long/short the stock AGAINST its factor-mimicking portfolio. Summing over stocks:

        w = signal - (betas @ signal) @ Q

    Then normalized to gross exposure 1.0. Net exposure is returned as a diagnostic -- the
    factor hedge should drive it near zero via the first principal component (the market).
    """
    c = betas @ signal                                # (m,) net factor exposure
    w = signal - (c @ Q)                              # (N,)
    gross = np.abs(w).sum()
    if gross < 1e-12:
        return np.zeros_like(w), 0.0
    w = w / gross
    return w, float(w.sum())


def residual_signals(returns, dates, pca_win=252, reg_win=60, n_factors=15,
                     kappa_min=TRADING_DAYS / 30.0, thresholds=None):
    """
    Run the full Avellaneda-Lee pipeline over `dates`, producing a weight vector per day.

    returns : DataFrame (dates x tickers) of daily log returns.
    dates   : the rebalance dates to produce weights for (must be inside `returns.index`).

    Returns (W, diag):
      W    : DataFrame (dates x tickers) of portfolio weights, gross-normalized to 1.
      diag : DataFrame of per-day diagnostics (n_tradeable, n_active, net_exposure, ...).

    At date t, EVERYTHING uses only returns up to and including t. No lookahead exists.
    """
    th = thresholds or {}
    cols = list(returns.columns)
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    prev = pd.Series(0.0, index=cols)
    diag = []

    R_all = returns
    for dt in dates:
        hist = R_all.loc[:dt]
        if len(hist) < pca_win:
            diag.append({"date": dt, "n_tradeable": 0, "n_active": 0,
                         "net_exposure": 0.0, "var_explained": np.nan})
            continue

        win = hist.iloc[-pca_win:]
        # Only stocks with a complete return history across the PCA window. This silently
        # excludes names that had not listed yet -- we never trade a stock before it existed.
        live = win.columns[win.notna().all(axis=0)]
        if len(live) < n_factors + 5:
            diag.append({"date": dt, "n_tradeable": len(live), "n_active": 0,
                         "net_exposure": 0.0, "var_explained": np.nan})
            continue

        R_win = win[live].values                              # (pca_win, n_live)
        Q, explained = eigen_portfolios(R_win, n_factors)     # (m, n_live)

        R_reg = R_win[-reg_win:]                              # (reg_win, n_live)
        F = R_reg @ Q.T                                       # (reg_win, m) factor returns
        X = np.column_stack([np.ones(len(F)), F])             # design with intercept

        B, *_ = np.linalg.lstsq(X, R_reg, rcond=None)         # (m+1, n_live)
        resid = R_reg - X @ B                                 # (reg_win, n_live)
        resid_cum = np.cumsum(resid, axis=0)

        s, kappa = ou_scores(resid_cum)
        # Reject residuals that revert too slowly to round-trip inside a month. Without this
        # the book fills up with drifting, non-stationary names.
        s = np.where(kappa >= kappa_min, s, np.nan)

        p = prev.reindex(live).fillna(0.0).values
        sig = decide(s, p,
                     th.get("sbo", 1.25), th.get("sso", 1.25),
                     th.get("sbc", 0.50), th.get("ssc", 0.75))

        w, net = build_weights(sig, B[1:, :], Q)

        prev = pd.Series(0.0, index=cols)
        prev.loc[live] = sig
        W.loc[dt, live] = w

        diag.append({"date": dt, "n_tradeable": len(live),
                     "n_active": int((sig != 0).sum()),
                     "net_exposure": round(net, 4),
                     "var_explained": round(explained, 3)})

    return W, pd.DataFrame(diag).set_index("date")
