"""
PHASE A ROBUSTNESS SWEEP.  These are CHECKS, not tuning.

PRE-COMMITTED, written before the run:
  (a) If gross Sharpe is positive at SOME settings and negative at others, the edge is
      FITTED and therefore fake. A real structural edge is positive across the grid.
  (b) Coverage = edge_per_unit_turnover / t_cost must exceed 1.0 SOMEWHERE, or the
      strategy cannot pay for itself at any setting and Phase A is dead in substance.
  (c) Selectivity raising coverage is MECHANICAL, not evidence of a better model.
      Judge it only by whether coverage clears 1.0 while return stays meaningful.

One eigendecomposition per date, shared across all configs.
"""
import numpy as np, pandas as pd, itertools, time
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import ou_scores, decide, build_weights

TC = 0.0005
tk = all_tickers(UNIVERSE_US)
px = download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets = log_returns(px.drop(columns=['SPY']))
dates = list(rets.index[rets.index >= pd.Timestamp('2016-01-04')])
R_np = rets.values; cols = list(rets.columns)
col_ix = {c:i for i,c in enumerate(cols)}
date_ix = {d:i for i,d in enumerate(rets.index)}

NF   = [5, 10, 15, 20]
RW   = [30, 60, 120]
THR  = [1.0, 1.25, 1.5, 2.0, 2.5]
PCAW = 252

configs = list(itertools.product(NF, RW, THR))
state = {c: {"prev_sig": None, "prev_w": np.zeros(len(cols)), "r": [], "t": [], "na": []}
         for c in configs}

t0 = time.time()
for di, dt in enumerate(dates[:-1]):
    i = date_ix[dt]
    if i < PCAW: continue
    win = R_np[i-PCAW+1:i+1]                       # trailing pca_win days INCLUDING today
    livem = ~np.isnan(win).any(axis=0)
    live = np.where(livem)[0]
    if len(live) < 30: continue
    Rw = win[:, live]
    fwd = R_np[i+1, live]                          # weights at t earn returns at t+1
    fwd = np.nan_to_num(fwd, nan=0.0)

    sigma = Rw.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-12, np.nan, sigma)
    Y = (Rw - Rw.mean(axis=0)) / sigma
    C = np.nan_to_num(np.corrcoef(Y, rowvar=False), nan=0.0); np.fill_diagonal(C, 1.0)
    vals, vecs = np.linalg.eigh(C)                 # ONE eigh, reused by every n_factors
    order = np.argsort(vals)[::-1]

    for nf in NF:
        V = vecs[:, order[:nf]]
        Q = (V / sigma[:, None]).T                 # (nf, n_live)
        for rw in RW:
            Rr = Rw[-rw:]
            F = Rr @ Q.T
            X = np.column_stack([np.ones(len(F)), F])
            B,*_ = np.linalg.lstsq(X, Rr, rcond=None)
            resid = Rr - X @ B
            s, kappa = ou_scores(np.cumsum(resid, axis=0))
            s = np.where(kappa >= 252/30, s, np.nan)
            betas = B[1:, :]
            for th in THR:
                st = state[(nf, rw, th)]
                p = st["prev_sig"]
                p = np.zeros(len(live)) if p is None else p.get(live, 0.0) if False else \
                    np.array([st["prev_map"].get(c, 0.0) for c in live]) if "prev_map" in st else np.zeros(len(live))
                sig = decide(s, p, th, th, 0.50, 0.75)
                w_live, net = build_weights(sig, betas, Q)
                w = np.zeros(len(cols)); w[live] = w_live
                st["r"].append(float(w_live @ fwd))
                st["t"].append(float(np.abs(w - st["prev_w"]).sum()))
                st["na"].append(int((sig != 0).sum()))
                st["prev_w"] = w
                st["prev_map"] = {c: v for c, v in zip(live, sig) if v != 0}
    if di % 400 == 0:
        print(f"  {dt.date()}  {di}/{len(dates)}  {time.time()-t0:.0f}s", flush=True)

rows = []
for (nf, rw, th), st in state.items():
    r = np.array(st["r"]); t = np.array(st["t"])
    if len(r) < 100: continue
    n = np.array(st["na"]).mean()
    gross_d, cost_d = r.mean(), t.mean()*TC
    sd = r.std()
    yrs = len(r)/252
    rows.append({
        "n_factors": nf, "reg_win": rw, "thr": th,
        "gross_pct_yr": round(100*gross_d*252, 2),
        "net_pct_yr":   round(100*(gross_d-cost_d)*252, 2),
        "gross_sharpe": round(gross_d/sd*np.sqrt(252), 2) if sd>0 else 0,
        "net_sharpe":   round((gross_d-cost_d)/sd*np.sqrt(252), 2) if sd>0 else 0,
        "turn_day":     round(t.mean(), 3),
        "edge_bps":     round(1e4*gross_d/max(t.mean(),1e-9), 2),   # per unit turnover
        "cost_bps":     round(1e4*TC, 2),
        "coverage":     round((gross_d/max(t.mean(),1e-9))/TC, 2),
        "breakeven_bps":round(1e4*gross_d/max(t.mean(),1e-9), 2),
        "n_active":     round(n, 0),
    })
df = pd.DataFrame(rows).sort_values(["n_factors","reg_win","thr"])
df.to_csv('/Users/katiyar/Desktop/Projects/StatArb/results/phase_a_sweep.csv', index=False)
pd.set_option('display.width', 200)
print()
print(df.to_string(index=False))
print()
print("gross Sharpe: min %.2f  max %.2f  | POSITIVE in %d/%d configs"
      % (df.gross_sharpe.min(), df.gross_sharpe.max(), (df.gross_sharpe>0).sum(), len(df)))
print("coverage    : min %.2f  max %.2f  | >1.0 in %d/%d configs"
      % (df.coverage.min(), df.coverage.max(), (df.coverage>1).sum(), len(df)))
print("net Sharpe  : min %.2f  max %.2f  | >0 in %d/%d configs"
      % (df.net_sharpe.min(), df.net_sharpe.max(), (df.net_sharpe>0).sum(), len(df)))
