"""
PHASE B PRE-FLIGHT, BRANCH 1 (signal extraction): IS THE s-SCORE ALREADY SUFFICIENT?

The predecessor's DL died because its target had a 2% R^2 ceiling AND that signal was
already sitting in one of its own input features (z_spread). The nets could only match the
linear rule or add noise. Test for the identical failure mode BEFORE building anything.

Phase B's CNN+Transformer reads the RESIDUAL PATH (60 days) and emits a signal.
The OU s-score is a hand-crafted SUMMARY of that same path (s = -m_eq/sigma_eq).

So the correct linear control for Phase B's signal extractor is:
    a LINEAR model on the full 60-day residual path, i.e. the optimally-weighted
    combination of exactly the inputs the network would see.

TARGET = what the strategy actually earns: the forward RESIDUAL return,
    eps_fwd_i = r_i(t+1) - sum_j beta_ij(t) * F_j(t+1)     [betas/Q frozen at t]

MODELS (all fit WALK-FORWARD, refit yearly, 5-day purge at every boundary):
    (a) s-score only            <- the Phase A / OU linear baseline
    (b) Ridge on 60-day path    <- THE LINEAR CONTROL for the CNN+Transformer
    (c) Ridge on path + OU extras
    (d) Gradient-boosted trees on the path  <- cheap NONLINEAR probe (proxy for the net)

PRE-COMMITTED, before the run:
  Phase B's signal branch is worth building ONLY if a richer model on the residual path
  beats the s-score OUT OF SAMPLE by a margin that is (i) positive in a MAJORITY of
  walk-forward years and (ii) large enough to matter economically.
  If (b) does NOT beat (a), the s-score is already the sufficient statistic of the path,
  a nonlinear model on the same input is curve-fitting, and PHASE B'S SIGNAL BRANCH IS DEAD
  -- for exactly the reason the predecessor's DL branch was dead.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import ou_scores

PCAW, RW, NF = 252, 60, 10
tk=all_tickers(UNIVERSE_US)
px=download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets=log_returns(px.drop(columns=['SPY']))
R_np=rets.values; all_d=list(rets.index)
start=next(i for i,d in enumerate(all_d) if d>=pd.Timestamp('2016-01-04'))

rows_path=[]; rows_s=[]; rows_meta=[]
STEP=3   # sample every 3rd day: keeps ~200k samples, limits target overlap
for i in range(start, len(all_d)-2, STEP):
    win=R_np[i-PCAW+1:i+1]
    live=np.where(~np.isnan(win).any(axis=0))[0]
    if len(live)<30: continue
    Rw=win[:,live]
    sigma=Rw.std(axis=0,ddof=1); sigma=np.where(sigma<1e-12,np.nan,sigma)
    Y=(Rw-Rw.mean(axis=0))/sigma
    C=np.nan_to_num(np.corrcoef(Y,rowvar=False),nan=0.0); np.fill_diagonal(C,1.0)
    vals,vecs=np.linalg.eigh(C); order=np.argsort(vals)[::-1][:NF]
    Q=(vecs[:,order]/sigma[:,None]).T
    Rr=Rw[-RW:]; F=Rr@Q.T; X=np.column_stack([np.ones(len(F)),F])
    B,*_=np.linalg.lstsq(X,Rr,rcond=None)
    resid=Rr-X@B
    rc=np.cumsum(resid,axis=0)                    # (RW, n_live) cumulative residual path
    s,kap=ou_scores(rc)
    betas=B[1:,:]

    # forward residual return: stock return minus its FROZEN factor hedge, t+1
    r_next=R_np[i+1,live]; f_next=np.nan_to_num(R_np[i+1,live],nan=0.0)@Q.T   # factor rets t+1
    eps_fwd = r_next - (betas.T @ f_next)

    # normalize the path per-stock so it's comparable across names & time
    sd_path = rc.std(axis=0); sd_path=np.where(sd_path<1e-12,np.nan,sd_path)
    path = (rc/sd_path).T                          # (n_live, RW)

    ok = np.isfinite(eps_fwd) & np.isfinite(s) & np.isfinite(path).all(axis=1) & (kap>0)
    if ok.sum()<30: continue
    rows_path.append(path[ok]); rows_s.append(s[ok])
    d=all_d[i]
    rows_meta.append(pd.DataFrame({"date":d,"y":eps_fwd[ok],"kap":kap[ok],
                                   "sdp":sd_path[ok]}))

P=np.vstack(rows_path); S=np.concatenate(rows_s); M=pd.concat(rows_meta,ignore_index=True)
M["s"]=S
print(f"panel: {P.shape[0]:,} stock-days | path dim {P.shape[1]} | "
      f"{M.date.min().date()} -> {M.date.max().date()}\n")

years=sorted(M.date.dt.year.unique())
res=[]
for yr in years[2:]:                                # need >=2y of training history
    tr = M.date.dt.year < yr
    te = M.date.dt.year == yr
    # PURGE: drop the last 5 trading days of training to kill label overlap at the boundary
    cut = M.loc[tr,"date"].max() - pd.Timedelta(days=8)
    tr = tr & (M.date <= cut)
    if tr.sum()<5000 or te.sum()<500: continue
    ytr, yte = M.y[tr].values, M.y[te].values
    ysd = ytr.std()
    out={"year":yr,"n_test":int(te.sum())}

    # (a) s-score baseline: prediction is -s (buy when s is very negative)
    pred_s = -M.s[te].values
    out["IC_s"] = stats.spearmanr(pred_s, yte).correlation

    # (b) Ridge on the 60-day path -- THE LINEAR CONTROL for the CNN+Transformer
    rg = Ridge(alpha=100.0).fit(P[tr.values], ytr/ysd)
    out["IC_path"] = stats.spearmanr(rg.predict(P[te.values]), yte).correlation

    # (c) Ridge on path + OU extras
    Xtr=np.column_stack([P[tr.values], M.s[tr].values, M.kap[tr].values, M.sdp[tr].values])
    Xte=np.column_stack([P[te.values], M.s[te].values, M.kap[te].values, M.sdp[te].values])
    rg2=Ridge(alpha=100.0).fit(Xtr, ytr/ysd)
    out["IC_path_ou"] = stats.spearmanr(rg2.predict(Xte), yte).correlation

    # (d) nonlinear probe: GBT on the path (stand-in for the CNN+Transformer)
    gb=HistGradientBoostingRegressor(max_iter=120,max_depth=4,learning_rate=0.05,
                                     random_state=0).fit(P[tr.values], ytr/ysd)
    out["IC_gbt"] = stats.spearmanr(gb.predict(P[te.values]), yte).correlation
    res.append(out)

df=pd.DataFrame(res)
df.to_csv('/Users/katiyar/Desktop/Projects/StatArb/results/phase_b_signal_ceiling.csv',index=False)
pd.set_option('display.width',200)
print("="*92)
print("OUT-OF-SAMPLE INFORMATION COEFFICIENT on the forward RESIDUAL return")
print("(Spearman, per walk-forward year. This is the number Phase B has to beat.)")
print("="*92)
print(df.round(4).to_string(index=False))
print()
mean=df[["IC_s","IC_path","IC_path_ou","IC_gbt"]].mean()
print("MEAN OOS IC across walk-forward years:")
for k,v in mean.items(): print(f"   {k:12s} {v:+.4f}")
print()
b=df.IC_path - df.IC_s
g=df.IC_gbt  - df.IC_s
print(f"LINEAR path model beats s-score in {int((b>0).sum())}/{len(df)} years  (mean delta {b.mean():+.4f})")
print(f"NONLINEAR (GBT)     beats s-score in {int((g>0).sum())}/{len(df)} years  (mean delta {g.mean():+.4f})")
print()
print("VERDICT ON PHASE B SIGNAL BRANCH:")
if b.mean() > 0 and (b>0).sum() > len(df)/2:
    print("  Residual path carries information the s-score misses -> headroom EXISTS.")
else:
    print("  The residual path carries NO extra linearly-extractable information.")
    print("  The OU s-score is already a sufficient statistic of the path.")
    print("  => a CNN+Transformer on the same input has nothing to learn. SIGNAL BRANCH DEAD.")
