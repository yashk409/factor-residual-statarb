"""
THE STEELMAN FOR PHASE B.

Objection to the last result: the GBT was trained on MSE at a 1-DAY horizon -- a PREDICTION
objective. That is exactly the predecessor's S12.6 error. Phase B trains END-TO-END on Sharpe
WITH A TURNOVER PENALTY, so it is incentivized to emit a signal that is both INFORMATIVE and
PERSISTENT. My smoothed-GBT test does not give it that chance.

Fair. So give it that chance, the cheap way: train the model to predict the k-DAY forward
residual return. A longer-horizon target IS a turnover penalty expressed through the target
instead of the loss -- it forces the model to find slow, persistent structure rather than
one-day noise. Then hold the position for k days. Sweep k.

Also measure signal AUTOCORRELATION directly, to confirm the persistence mechanism.

PRE-COMMITTED, before the run:
  If NO horizon k produces a learned signal that beats the OU s-score NET of costs, then the
  persistent, tradeable component of the residual path is ALREADY what the s-score captures.
  The extra information a network finds is real but economically unreachable, and PHASE B IS
  DEAD -- not because DL cannot predict, but because what it predicts cannot be afforded.
  Judge on NET Sharpe, head to head, same allocator, same costs.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import ou_scores

PCAW,RW,NF,TC = 252,60,10,0.0005
HOR=[1,5,10,21]
tk=all_tickers(UNIVERSE_US)
px=download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets=log_returns(px.drop(columns=['SPY']))
R_np=rets.values; cols=list(rets.columns); all_d=list(rets.index)
start=next(i for i,d in enumerate(all_d) if d>=pd.Timestamp('2016-01-04'))
MAXH=max(HOR)

snap=[]
for i in range(start,len(all_d)-MAXH-1):
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
    s,kap=ou_scores(rc); betas=B[1:,:]
    # k-day forward residual returns, betas/Q FROZEN at t (that is what the book earns)
    ys={}
    for h in HOR:
        rh=np.nan_to_num(R_np[i+1:i+1+h,live],nan=0.0).sum(axis=0)     # h-day stock return
        fh=rh@Q.T
        ys[h]=rh-(betas.T@fh)
    rn=np.nan_to_num(R_np[i+1,live],nan=0.0)
    sdp=rc.std(axis=0); sdp=np.where(sdp<1e-12,np.nan,sdp)
    path=(rc/sdp).T
    ok=np.isfinite(s)&np.isfinite(path).all(axis=1)&(kap>0)&np.isfinite(ys[1])
    if ok.sum()<30: continue
    snap.append({"i":i,"date":all_d[i],"live":live,"ok":ok,"path":path[ok],"s":s[ok],
                 "ys":{h:ys[h][ok] for h in HOR},"betas":betas,"Q":Q,"fwd":rn})
print(f"panel: {len(snap)} days\n")

yrs=sorted({x["date"].year for x in snap})
preds={h:{} for h in HOR}
for yr in yrs[2:]:
    tr=[x for x in snap if x["date"].year<yr]
    if not tr: continue
    cut=max(x["date"] for x in tr)-pd.Timedelta(days=40)    # PURGE >= max horizon
    tr=[x for x in tr if x["date"]<=cut]
    te=[x for x in snap if x["date"].year==yr]
    if len(tr)<400 or not te: continue
    Ptr=np.vstack([x["path"] for x in tr])
    for h in HOR:
        ytr=np.concatenate([x["ys"][h] for x in tr])
        gb=HistGradientBoostingRegressor(max_iter=120,max_depth=4,learning_rate=0.05,
                                         random_state=0).fit(Ptr,ytr/ytr.std())
        for x in te: preds[h][x["i"]]=gb.predict(x["path"])
    print(f"  refit {yr}",flush=True)
test=[x for x in snap if x["i"] in preds[1]]
print(f"\nOOS: {len(test)} days\n")

# ---- signal persistence: 1-day autocorrelation of the cross-sectional signal ----
def autocorr(getter):
    ac=[]
    for a,b in zip(test[:-1],test[1:]):
        if b["i"]-a["i"]!=1: continue
        la,lb=a["live"][a["ok"]], b["live"][b["ok"]]
        common=np.intersect1d(la,lb)
        if len(common)<30: continue
        ma={t:v for t,v in zip(la,getter(a))}; mb={t:v for t,v in zip(lb,getter(b))}
        va=np.array([ma[t] for t in common]); vb=np.array([mb[t] for t in common])
        ac.append(np.corrcoef(va,vb)[0,1])
    return float(np.nanmean(ac))
print("="*92); print("SIGNAL PERSISTENCE (1-day autocorrelation of the cross-sectional signal)"); print("="*92)
print(f"  OU s-score            : {autocorr(lambda x: -x['s']):+.3f}   <- slow, cheap to hold")
for h in HOR:
    print(f"  GBT trained on {h:2d}-day : {autocorr(lambda x,h=h: preds[h][x['i']]):+.3f}")

# ---- head-to-head net Sharpe: allocate on each signal, hold k days ----
def run(getter, K, lam=1.0):
    pw=np.zeros(len(cols)); rl=[];tl=[]
    for n,x in enumerate(test):
        live=x["live"]; ok=x["ok"]
        if n%K==0:
            v=getter(x); v=(v-v.mean())/(v.std()+1e-12)
            raw=np.zeros(len(live)); raw[ok]=v
            g=np.abs(raw).sum(); tgt=raw/g if g>1e-12 else raw*0
            c_=x["betas"]@tgt; wl=tgt-(c_@x["Q"])
            gg=np.abs(wl).sum(); wl=wl/gg if gg>1e-12 else wl*0
            wt=np.zeros(len(cols)); wt[live]=wl
            w=(1-lam)*pw+lam*wt
            gs=np.abs(w).sum()
            if gs>1e-12: w=w/gs
        else:
            w=pw
        rl.append(float(w[live]@x["fwd"])); tl.append(float(np.abs(w-pw).sum())); pw=w
    r=np.array(rl);t_=np.array(tl); sd=r.std(); yy=len(r)/252
    gd,cd=r.mean(),t_.mean()*TC
    return dict(gross=100*gd*252, net=100*(gd-cd)*252, gS=gd/sd*np.sqrt(252),
                nS=(gd-cd)/sd*np.sqrt(252), t=((gd-cd)/sd*np.sqrt(252))*np.sqrt(yy),
                turn=t_.mean()*252, cov=(gd/max(t_.mean(),1e-9))/TC)

print()
print("="*92); print("HEAD TO HEAD -- signal trained on k-day target, HELD k days. Same costs."); print("="*92)
print(f"{'signal':22s} {'hold':>5s} {'gross%':>7s} {'net%':>7s} {'gSharpe':>8s} {'nSharpe':>8s} {'net t':>6s} {'turn/yr':>8s} {'cover':>6s}")
rows=[]
for K in HOR:
    r=run(lambda x:-x["s"], K)
    rows.append({"signal":"OU s-score","hold":K,**r})
    print(f"{'OU s-score':22s} {K:>5d} {r['gross']:>7.2f} {r['net']:>7.2f} {r['gS']:>8.2f} {r['nS']:>8.2f} {r['t']:>6.2f} {r['turn']:>8.1f} {r['cov']:>6.2f}")
for h in HOR:
    r=run(lambda x,h=h: preds[h][x["i"]], h)
    rows.append({"signal":f"GBT {h}d-target","hold":h,**r})
    print(f"{'GBT '+str(h)+'d-target':22s} {h:>5d} {r['gross']:>7.2f} {r['net']:>7.2f} {r['gS']:>8.2f} {r['nS']:>8.2f} {r['t']:>6.2f} {r['turn']:>8.1f} {r['cov']:>6.2f}")
df=pd.DataFrame(rows); df.to_csv('/Users/katiyar/Desktop/Projects/StatArb/results/phase_b_steelman.csv',index=False)
bs=df[df.signal=="OU s-score"].nS.max(); bg=df[df.signal.str.startswith("GBT")].nS.max()
print()
print(f"best NET Sharpe -- OU s-score: {bs:+.2f}   |   best learned signal: {bg:+.2f}")
print()
if bg > bs + 0.10:
    print("The learned signal BEATS the linear baseline net of costs. Phase B has a job.")
else:
    print("PHASE B IS DEAD.")
    print("At every horizon, a strong nonlinear learner on the residual path -- with the")
    print("turnover incentive built into its target -- FAILS to beat the OU s-score net of")
    print("costs. The persistent, affordable component of the residual path is exactly what")
    print("the s-score already extracts. The rest is real information that cannot be traded.")
