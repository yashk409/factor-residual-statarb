"""
HOLDING PERIOD x STRUCTURAL GRID.  Is the K=5 coverage>1 result real, or one lucky cell?

PRE-COMMITTED, before the run:
  The K=5 result is only believable if positive coverage at slow rebalance is a PROPERTY OF
  THE FRONTIER, not of one (n_factors, reg_win, thr) cell. Specifically:
    - gross Sharpe must stay positive across the grid at K=5  (else the edge is fitted)
    - if net Sharpe is positive in only a minority of cells, the K=5 net result is NOISE
      and must be reported as such.
  I also report the t-stat of every Sharpe. Sharpe over Y years has t = Sharpe*sqrt(Y).
  With Y=10.5, a Sharpe of 0.26 has t=0.84 -> NOT significant. Say so.
"""
import numpy as np, pandas as pd, itertools
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import ou_scores, decide, build_weights

TC=0.0005
tk=all_tickers(UNIVERSE_US)
px=download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets=log_returns(px.drop(columns=['SPY']))
R_np=rets.values; cols=list(rets.columns); all_d=list(rets.index)
start=next(i for i,d in enumerate(all_d) if d>=pd.Timestamp('2016-01-04'))
PCAW=252
NF=[5,10,15,20]; RW=[30,60,120]; THR=[1.0,1.25,1.5]; KS=[1,5,10]
cfgs=list(itertools.product(NF,RW,THR,KS))
st={c:{"pm":{}, "pw":np.zeros(len(cols)), "r":[], "t":[]} for c in cfgs}

for i in range(start, len(all_d)-1):
    win=R_np[i-PCAW+1:i+1]
    live=np.where(~np.isnan(win).any(axis=0))[0]
    if len(live)<30: continue
    fwd=np.nan_to_num(R_np[i+1,live],nan=0.0)
    Rw=win[:,live]
    sigma=Rw.std(axis=0,ddof=1); sigma=np.where(sigma<1e-12,np.nan,sigma)
    Y=(Rw-Rw.mean(axis=0))/sigma
    C=np.nan_to_num(np.corrcoef(Y,rowvar=False),nan=0.0); np.fill_diagonal(C,1.0)
    vals,vecs=np.linalg.eigh(C); order=np.argsort(vals)[::-1]
    cache={}
    for nf in NF:
        Q=(vecs[:,order[:nf]]/sigma[:,None]).T
        for rw in RW:
            Rr=Rw[-rw:]; F=Rr@Q.T; X=np.column_stack([np.ones(len(F)),F])
            B,*_=np.linalg.lstsq(X,Rr,rcond=None)
            s,kap=ou_scores(np.cumsum(Rr-X@B,axis=0))
            cache[(nf,rw)]=(np.where(kap>=252/30,s,np.nan), B[1:,:], Q)
    for (nf,rw,th,K) in cfgs:
        d=st[(nf,rw,th,K)]
        if (i-start)%K==0:
            s,betas,Q=cache[(nf,rw)]
            p=np.array([d["pm"].get(c,0.0) for c in live])
            sig=decide(s,p,th,th,0.50,0.75)
            wl,_=build_weights(sig,betas,Q)
            w=np.zeros(len(cols)); w[live]=wl
            d["pm"]={c:v for c,v in zip(live,sig) if v!=0}
        else:
            w=d["pw"]
        d["r"].append(float(w[live]@fwd))
        d["t"].append(float(np.abs(w-d["pw"]).sum()))
        d["pw"]=w

rows=[]
for (nf,rw,th,K),d in st.items():
    r=np.array(d["r"]); t=np.array(d["t"]); sd=r.std(); yrs=len(r)/252
    gd,cd=r.mean(),t.mean()*TC
    gsh=gd/sd*np.sqrt(252); nsh=(gd-cd)/sd*np.sqrt(252)
    rows.append({"nf":nf,"rw":rw,"thr":th,"K":K,
        "gross_yr":round(100*gd*252,2),"net_yr":round(100*(gd-cd)*252,2),
        "gSharpe":round(gsh,2),"g_t":round(gsh*np.sqrt(yrs),2),
        "nSharpe":round(nsh,2),"n_t":round(nsh*np.sqrt(yrs),2),
        "turn_yr":round(t.mean()*252,0),"cover":round((gd/max(t.mean(),1e-9))/TC,2)})
df=pd.DataFrame(rows); df.to_csv('/Users/katiyar/Desktop/Projects/StatArb/results/phase_a_grid_hold.csv',index=False)
pd.set_option('display.width',250)
for K in KS:
    sub=df[df.K==K]
    print("="*104)
    print(f"REBALANCE EVERY {K} DAY(S)   -- {len(sub)} structural configs")
    print("="*104)
    print(f"  gross Sharpe : min {sub.gSharpe.min():+.2f}  med {sub.gSharpe.median():+.2f}  max {sub.gSharpe.max():+.2f}   | >0 in {(sub.gSharpe>0).sum()}/{len(sub)}   | t>2 in {(sub.g_t>2).sum()}/{len(sub)}")
    print(f"  NET   Sharpe : min {sub.nSharpe.min():+.2f}  med {sub.nSharpe.median():+.2f}  max {sub.nSharpe.max():+.2f}   | >0 in {(sub.nSharpe>0).sum()}/{len(sub)}   | t>2 in {(sub.n_t>2).sum()}/{len(sub)}")
    print(f"  coverage     : min {sub.cover.min():.2f}  med {sub.cover.median():.2f}  max {sub.cover.max():.2f}   | >1 in {(sub.cover>1).sum()}/{len(sub)}")
    print(f"  net %/yr     : min {sub.net_yr.min():+.2f}  med {sub.net_yr.median():+.2f}  max {sub.net_yr.max():+.2f}")
    print()
print("="*104)
print("K=5 detail (all 36 structural configs):")
print(df[df.K==5].sort_values("nSharpe",ascending=False).to_string(index=False))
