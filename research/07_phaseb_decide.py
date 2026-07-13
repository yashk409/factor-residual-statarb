"""
PHASE B PRE-FLIGHT, DECISIVE.  Two questions, properly powered.

Q1 (statistical): does a model on the residual PATH have a higher cross-sectional IC than
    the OU s-score?  Test on DAILY cross-sectional ICs (~880 paired obs), not 9 annual ones.
    The 9-year test had t=0.58 (linear) / t=1.05 (GBT) -- underpowered and insignificant.

Q2 (economic): plug the learned signal into the SAME allocator + SAME cost model as Phase A.
    Does it beat the s-score NET of costs?  This bypasses all IC->Sharpe guesswork.
    Rule 6: the model must beat the linear baseline ON THE SAME TASK.

GBT is the stand-in for the CNN+Transformer: a strong, low-variance nonlinear learner on the
exact same 60-day residual-path input. It is a GENEROUS proxy -- it has no seed dispersion
problem, and the predecessor's nets could not even match their own linear rule.

PRE-COMMITTED, before the run:
  * If GBT's daily IC does not beat the s-score's with t > 2, there is no signal headroom.
  * If GBT's NET Sharpe (same allocator, same costs) does not beat the s-score's, Phase B's
    signal branch is DEAD -- a CNN+Transformer would be curve-fitting on a target whose
    sufficient statistic is already the s-score. This is the predecessor's failure exactly.
  * A net Sharpe that clears 0.5 must do so ROBUSTLY, not in one cell.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import ou_scores

PCAW,RW,NF,TC = 252,60,10,0.0005
tk=all_tickers(UNIVERSE_US)
px=download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets=log_returns(px.drop(columns=['SPY']))
R_np=rets.values; cols=list(rets.columns); all_d=list(rets.index)
start=next(i for i,d in enumerate(all_d) if d>=pd.Timestamp('2016-01-04'))

# ---- build the panel: path, s-score, forward residual return, and hedge pieces ----
snap=[]
for i in range(start,len(all_d)-1):
    win=R_np[i-PCAW+1:i+1]
    live=np.where(~np.isnan(win).any(axis=0))[0]
    if len(live)<30: continue
    Rw=win[:,live]
    sigma=Rw.std(axis=0,ddof=1); sigma=np.where(sigma<1e-12,np.nan,sigma)
    Yv=(Rw-Rw.mean(axis=0))/sigma
    C=np.nan_to_num(np.corrcoef(Yv,rowvar=False),nan=0.0); np.fill_diagonal(C,1.0)
    vals,vecs=np.linalg.eigh(C); order=np.argsort(vals)[::-1][:NF]
    Q=(vecs[:,order]/sigma[:,None]).T
    Rr=Rw[-RW:]; F=Rr@Q.T; X=np.column_stack([np.ones(len(F)),F])
    B,*_=np.linalg.lstsq(X,Rr,rcond=None)
    rc=np.cumsum(Rr-X@B,axis=0)
    s,kap=ou_scores(rc)
    betas=B[1:,:]
    rn=np.nan_to_num(R_np[i+1,live],nan=0.0)
    eps_fwd = rn - (betas.T @ (rn@Q.T))
    sdp=rc.std(axis=0); sdp=np.where(sdp<1e-12,np.nan,sdp)
    path=(rc/sdp).T
    ok=np.isfinite(eps_fwd)&np.isfinite(s)&np.isfinite(path).all(axis=1)&(kap>0)
    if ok.sum()<30: continue
    snap.append({"i":i,"date":all_d[i],"live":live,"ok":ok,"path":path[ok],"s":s[ok],
                 "y":eps_fwd[ok],"betas":betas,"Q":Q,"fwd":rn})
print(f"panel: {len(snap)} days, {sum(len(x['s']) for x in snap):,} stock-days\n")

# ---- walk-forward: refit yearly on all prior data, 8-day purge ----
yrs=sorted({x["date"].year for x in snap})
preds={}
for yr in yrs[2:]:
    tr=[x for x in snap if x["date"].year<yr]
    if not tr: continue
    cut=max(x["date"] for x in tr)-pd.Timedelta(days=8)     # PURGE the boundary
    tr=[x for x in tr if x["date"]<=cut]
    te=[x for x in snap if x["date"].year==yr]
    if len(tr)<400 or not te: continue
    Ptr=np.vstack([x["path"] for x in tr]); ytr=np.concatenate([x["y"] for x in tr])
    ysd=ytr.std()
    gb=HistGradientBoostingRegressor(max_iter=120,max_depth=4,learning_rate=0.05,
                                     random_state=0).fit(Ptr,ytr/ysd)
    rg=Ridge(alpha=100.0).fit(Ptr,ytr/ysd)
    for x in te:
        preds[x["i"]]={"gbt":gb.predict(x["path"]),"ridge":rg.predict(x["path"])}
    print(f"  refit {yr}: train {Ptr.shape[0]:,} -> test {sum(len(x['s']) for x in te):,}",flush=True)

test=[x for x in snap if x["i"] in preds]
print(f"\nOOS test set: {len(test)} days, {sum(len(x['s']) for x in test):,} stock-days\n")

# ---- Q1: DAILY cross-sectional IC, paired ----
ic={"s":[],"ridge":[],"gbt":[]}
for x in test:
    y=x["y"]
    if len(y)<30: continue
    ic["s"].append(stats.spearmanr(-x["s"],y).correlation)
    ic["ridge"].append(stats.spearmanr(preds[x["i"]]["ridge"],y).correlation)
    ic["gbt"].append(stats.spearmanr(preds[x["i"]]["gbt"],y).correlation)
ic={k:np.array(v) for k,v in ic.items()}
n=len(ic["s"])
print("="*92); print(f"Q1  DAILY CROSS-SECTIONAL IC  (n = {n} days, paired)"); print("="*92)
print(f"{'model':10s} {'mean IC':>9s} {'t vs 0':>8s} {'IC-IR':>7s}")
for k in ["s","ridge","gbt"]:
    t0=ic[k].mean()/(ic[k].std()/np.sqrt(n))
    print(f"{k:10s} {ic[k].mean():+9.5f} {t0:+8.2f} {ic[k].mean()/ic[k].std():+7.3f}")
print()
for k in ["ridge","gbt"]:
    d=ic[k]-ic["s"]
    t=d.mean()/(d.std()/np.sqrt(n))
    print(f"  {k:6s} - s : delta IC {d.mean():+.5f}   PAIRED t = {t:+.2f}   p = {2*(1-stats.t.cdf(abs(t),n-1)):.3f}"
          f"   beats s on {100*(d>0).mean():.0f}% of days")

# ---- Q2: economic. Same allocator, same costs, head to head. ----
print()
print("="*92); print("Q2  NET SHARPE, SAME ALLOCATOR + SAME COSTS (5bps/unit turnover)"); print("="*92)
print(f"{'signal':8s} {'lam':>5s} {'gross%/yr':>10s} {'net%/yr':>9s} {'gSharpe':>8s} {'nSharpe':>8s} {'net t':>6s} {'turn/yr':>8s} {'cover':>6s}")
out=[]
for name in ["s","ridge","gbt"]:
    for lam in [1.0,0.5,0.25,0.1,0.05]:
        pw=np.zeros(len(cols)); rl=[];tl=[]
        for x in test:
            live=x["live"]; ok=x["ok"]
            v = -x["s"] if name=="s" else preds[x["i"]][name]
            v = (v-v.mean())/(v.std()+1e-12)              # cross-sectional z
            raw=np.zeros(len(live)); raw[ok]=v            # full live set, 0 where invalid
            g=np.abs(raw).sum()
            tgt=raw/g if g>1e-12 else raw*0
            c_=x["betas"]@tgt; wl=tgt-(c_@x["Q"])
            gg=np.abs(wl).sum(); wl=wl/gg if gg>1e-12 else wl*0
            wt=np.zeros(len(cols)); wt[live]=wl
            w=(1-lam)*pw+lam*wt
            gs=np.abs(w).sum()
            if gs>1e-12: w=w/gs
            rl.append(float(w[live]@x["fwd"])); tl.append(float(np.abs(w-pw).sum())); pw=w
        r=np.array(rl);t_=np.array(tl); sd=r.std(); yy=len(r)/252
        gd,cd=r.mean(),t_.mean()*TC
        gsh=gd/sd*np.sqrt(252); nsh=(gd-cd)/sd*np.sqrt(252)
        cov=(gd/max(t_.mean(),1e-9))/TC
        out.append({"signal":name,"lam":lam,"gross_yr":100*gd*252,"net_yr":100*(gd-cd)*252,
                    "gSharpe":gsh,"nSharpe":nsh,"net_t":nsh*np.sqrt(yy),
                    "turn_yr":t_.mean()*252,"cover":cov})
        print(f"{name:8s} {lam:>5} {100*gd*252:>10.2f} {100*(gd-cd)*252:>9.2f} {gsh:>8.2f} "
              f"{nsh:>8.2f} {nsh*np.sqrt(yy):>6.2f} {t_.mean()*252:>8.1f} {cov:>6.2f}")
df=pd.DataFrame(out); df.to_csv('/Users/katiyar/Desktop/Projects/StatArb/results/phase_b_decide.csv',index=False)
print()
bs=df[df.signal=="s"].nSharpe.max(); bg=df[df.signal=="gbt"].nSharpe.max(); br=df[df.signal=="ridge"].nSharpe.max()
print(f"best NET Sharpe   s-score: {bs:+.2f}   ridge: {br:+.2f}   GBT: {bg:+.2f}")
print(f"does the learned signal beat the OU s-score net of costs? "
      f"{'YES' if max(bg,br) > bs + 0.10 else 'NO -- it does not clear the linear baseline'}")
