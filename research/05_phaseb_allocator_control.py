"""
PHASE B PRE-FLIGHT, BRANCH 2 (allocation): the LINEAR CONTROL for the allocator network.

Phase B proposes a network mapping signals -> weights, trained on -Sharpe with a turnover
penalty. Rule 6 says: it must beat a linear baseline on the same task. Run the baseline FIRST
-- if the linear version already hits the ceiling, the network is curve-fitting.

The linear allocator that a turnover-penalized net would converge toward:
    w_target  ∝  -s          (continuous, no threshold -- no discrete flips)
    w_t = (1-lam)*w_{t-1} + lam*w_target      (EWMA smoothing == a turnover penalty)
lam=1.0 is no smoothing; lam->0 is heavy turnover damping. This spans the SAME
signal/turnover frontier the network is being asked to learn, using zero parameters.

PRE-COMMITTED, before the run:
  A network cannot invent gross alpha that is not in the signal; it can only harvest the
  existing signal at lower turnover. So this frontier is an UPPER BOUND on what Phase B's
  allocation branch can deliver from the OU s-score.
    - If NO (config, lam) reaches net Sharpe > 0.5 robustly (positive in a MAJORITY of
      structural configs, not just the argmax), Phase B's allocation branch cannot clear
      the Phase C bar and is DEAD.
    - Judge by the MEDIAN across configs, never the max. The max of a 36-cell grid is a
      selection-biased statistic -- that is how the last project died.
"""
import numpy as np, pandas as pd, itertools
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import ou_scores

TC=0.0005
tk=all_tickers(UNIVERSE_US)
px=download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets=log_returns(px.drop(columns=['SPY']))
R_np=rets.values; cols=list(rets.columns); all_d=list(rets.index)
start=next(i for i,d in enumerate(all_d) if d>=pd.Timestamp('2016-01-04'))
PCAW=252
NF=[5,10,15,20]; RW=[30,60,120]; LAM=[1.0,0.5,0.25,0.1,0.05]
cfgs=list(itertools.product(NF,RW,LAM))
st={c:{"pw":np.zeros(len(cols)),"r":[],"t":[]} for c in cfgs}

for i in range(start,len(all_d)-1):
    win=R_np[i-PCAW+1:i+1]
    live=np.where(~np.isnan(win).any(axis=0))[0]
    if len(live)<30: continue
    fwd=np.nan_to_num(R_np[i+1,live],nan=0.0)
    Rw=win[:,live]
    sigma=Rw.std(axis=0,ddof=1); sigma=np.where(sigma<1e-12,np.nan,sigma)
    Y=(Rw-Rw.mean(axis=0))/sigma
    C=np.nan_to_num(np.corrcoef(Y,rowvar=False),nan=0.0); np.fill_diagonal(C,1.0)
    vals,vecs=np.linalg.eigh(C); order=np.argsort(vals)[::-1]
    for nf in NF:
        Q=(vecs[:,order[:nf]]/sigma[:,None]).T
        for rw in RW:
            Rr=Rw[-rw:]; F=Rr@Q.T; X=np.column_stack([np.ones(len(F)),F])
            B,*_=np.linalg.lstsq(X,Rr,rcond=None)
            s,kap=ou_scores(np.cumsum(Rr-X@B,axis=0))
            s=np.where(kap>=252/30,s,np.nan)
            sc=np.nan_to_num(s,nan=0.0)
            raw=-sc                                   # LINEAR allocator: long when s<0
            g=np.abs(raw).sum()
            tgt_live = raw/g if g>1e-12 else raw*0
            # factor-neutralize the target book, same hedge as Phase A
            c_=B[1:,:]@tgt_live
            wl=tgt_live-(c_@Q)
            gg=np.abs(wl).sum()
            wl=wl/gg if gg>1e-12 else wl*0
            wt=np.zeros(len(cols)); wt[live]=wl
            for lam in LAM:
                d=st[(nf,rw,lam)]
                w=(1-lam)*d["pw"]+lam*wt              # EWMA smoothing == turnover penalty
                gsum=np.abs(w).sum()
                if gsum>1e-12: w=w/gsum               # keep gross exposure at 1.0
                d["r"].append(float(w[live]@fwd))
                d["t"].append(float(np.abs(w-d["pw"]).sum()))
                d["pw"]=w

rows=[]
for (nf,rw,lam),d in st.items():
    r=np.array(d["r"]); t=np.array(d["t"]); sd=r.std(); yrs=len(r)/252
    gd,cd=r.mean(),t.mean()*TC
    gsh=gd/sd*np.sqrt(252); nsh=(gd-cd)/sd*np.sqrt(252)
    rows.append({"nf":nf,"rw":rw,"lam":lam,
      "gross_yr":round(100*gd*252,2),"net_yr":round(100*(gd-cd)*252,2),
      "gSharpe":round(gsh,2),"nSharpe":round(nsh,2),"n_t":round(nsh*np.sqrt(yrs),2),
      "turn_yr":round(t.mean()*252,1),"cover":round((gd/max(t.mean(),1e-9))/TC,2)})
df=pd.DataFrame(rows); df.to_csv('/Users/katiyar/Desktop/Projects/StatArb/results/phase_b_allocator_control.csv',index=False)
pd.set_option('display.width',250)
print("="*104)
print("LINEAR ALLOCATOR (w ∝ -s) + EWMA TURNOVER PENALTY  -- the control Phase B must beat")
print("="*104)
print(f"{'lam':>5} {'turn/yr':>8} {'gSharpe med':>12} {'nSharpe med':>12} {'nSharpe max':>12} {'net%/yr med':>12} {'cover med':>10} {'nSharpe>0':>10} {'n_t>2':>7}")
for lam in LAM:
    s_=df[df.lam==lam]
    print(f"{lam:>5} {s_.turn_yr.median():>8.1f} {s_.gSharpe.median():>12.2f} {s_.nSharpe.median():>12.2f} "
          f"{s_.nSharpe.max():>12.2f} {s_.net_yr.median():>12.2f} {s_.cover.median():>10.2f} "
          f"{str(int((s_.nSharpe>0).sum()))+'/'+str(len(s_)):>10} {str(int((s_.n_t>2).sum()))+'/'+str(len(s_)):>7}")
print()
print("BEST 8 CELLS BY NET SHARPE (remember: this is the ARGMAX of a 60-cell grid):")
print(df.sort_values("nSharpe",ascending=False).head(8).to_string(index=False))
print()
best=df.nSharpe.max()
maj=((df.groupby('lam').nSharpe.median()>0.5).any())
print(f"Any lam with MEDIAN net Sharpe > 0.5 across structural configs? {'YES' if maj else 'NO'}")
print(f"Best single-cell net Sharpe anywhere: {best:.2f}   (t = {df.loc[df.nSharpe.idxmax(),'n_t']:.2f})")
