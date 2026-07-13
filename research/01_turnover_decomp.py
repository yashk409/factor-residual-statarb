"""
Decompose the 0.362/day turnover. PRE-COMMITTED QUESTION, asked before the answer:

  Of the total daily turnover, how much comes from the SIGNAL vector changing (real
  trading decisions) vs the HEDGE re-estimating (PCA + betas refit daily)?

If hedge churn dominates, the cost bill is an implementation artifact and is fixable.
If signal churn dominates, the strategy is intrinsically expensive and Phase A is dead.
"""
import numpy as np, pandas as pd
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import eigen_portfolios, ou_scores, decide, build_weights

tk = all_tickers(UNIVERSE_US)
px = download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets = log_returns(px.drop(columns=['SPY']))
dates = rets.index[rets.index >= pd.Timestamp('2016-01-04')]

cols = list(rets.columns)
prev = pd.Series(0.0, index=cols)
W = pd.DataFrame(0.0, index=dates, columns=cols)          # actual weights (hedge refit daily)
Wsig = pd.DataFrame(0.0, index=dates, columns=cols)       # signal-only weights (no hedge)
sig_hist = pd.DataFrame(0.0, index=dates, columns=cols)   # raw +-1 signal
n_flip = []

for dt in dates:
    hist = rets.loc[:dt]
    if len(hist) < 252: continue
    win = hist.iloc[-252:]
    live = win.columns[win.notna().all(axis=0)]
    if len(live) < 20: continue
    R_win = win[live].values
    Q,_ = eigen_portfolios(R_win, 15)
    R_reg = R_win[-60:]
    F = R_reg @ Q.T
    X = np.column_stack([np.ones(len(F)), F])
    B,*_ = np.linalg.lstsq(X, R_reg, rcond=None)
    resid = R_reg - X@B
    s, kappa = ou_scores(np.cumsum(resid, axis=0))
    s = np.where(kappa >= 252/30, s, np.nan)
    p = prev.reindex(live).fillna(0.0).values
    sig = decide(s, p, 1.25, 1.25, 0.50, 0.75)
    n_flip.append(int((sig != p).sum()))
    w, net = build_weights(sig, B[1:,:], Q)
    # signal-only book: just the +-1 signals, gross-normalized, NO factor hedge
    gs = np.abs(sig).sum()
    ws = sig/gs if gs > 1e-12 else sig*0
    prev = pd.Series(0.0, index=cols); prev.loc[live] = sig
    W.loc[dt, live] = w
    Wsig.loc[dt, live] = ws
    sig_hist.loc[dt, live] = sig

def turn(D):
    V = D.values
    Vp = np.vstack([np.zeros((1,V.shape[1])), V[:-1]])
    return np.abs(V-Vp).sum(axis=1)

t_full = turn(W)          # actual: signal changes + hedge refit
t_sig  = turn(Wsig)       # signal changes only, unhedged book
t_raw  = turn(sig_hist)/2 # count of +-1 signal flips (÷2 -> per-name units)

print('='*80)
print('TURNOVER DECOMPOSITION  (mean per day, after 252d warmup)')
print('='*80)
m = slice(252, None)
print(f'  actual book (hedge refit daily) : {t_full[m].mean():.4f}/day  ({t_full[m].mean()*252:.0f}x/yr)')
print(f'  signal-only book (NO hedge)     : {t_sig[m].mean():.4f}/day  ({t_sig[m].mean()*252:.0f}x/yr)')
print(f'  -> hedge churn share of turnover: {100*(1 - t_sig[m].mean()/t_full[m].mean()):.0f}%')
print()
print(f'  avg # names flipping signal/day : {np.mean(n_flip[252:]):.1f}')
print(f'  avg # active positions          : {(sig_hist!=0).sum(axis=1)[m].mean():.0f}')
print(f'  implied avg holding period      : {(sig_hist!=0).sum(axis=1)[m].mean()/max(np.mean(n_flip[252:]),1e-9):.1f} days')
np.save('/private/tmp/claude-501/-Users-katiyar-Desktop-Projects-StatArb/9b3d37e1-fc7b-4aca-a611-938f21051c27/scratchpad/t_full.npy', t_full)
W.to_parquet('/private/tmp/claude-501/-Users-katiyar-Desktop-Projects-StatArb/9b3d37e1-fc7b-4aca-a611-938f21051c27/scratchpad/W_base.parquet')
Wsig.to_parquet('/private/tmp/claude-501/-Users-katiyar-Desktop-Projects-StatArb/9b3d37e1-fc7b-4aca-a611-938f21051c27/scratchpad/W_sig.parquet')
sig_hist.to_parquet('/private/tmp/claude-501/-Users-katiyar-Desktop-Projects-StatArb/9b3d37e1-fc7b-4aca-a611-938f21051c27/scratchpad/sig.parquet')
