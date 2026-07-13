"""
THE DECIDING TEST: does the gross edge survive a slower rebalance?

Turnover (91x/yr) is the binding constraint. Selectivity cannot fix it (proven: it
concentrates the book and raises turnover). The only remaining lever is HOLDING PERIOD.

Guijarro-Ordonez/Pelger/Zanotti claim "around half of the Sharpe ratio can persist for a
holding period of one week." The predecessor's post-mortem (S12.3) blamed daily re-deciding
of a multi-day forecast for manufacturing turnover. Both point the same way.

PRE-COMMITTED, written before the run:
  Phase B's turnover penalty can only ever discover a point on THIS frontier -- a network
  cannot trade less often than it rebalances. So:

  If coverage stays < 1.0 at EVERY rebalance frequency, then no turnover-aware model can
  make this pay, Phase B is curve-fitting, and the strategy is DEAD.
  If coverage clears 1.0 at some frequency with a still-meaningful gross return, Phase B
  has a real job and the project continues.
"""
import numpy as np, pandas as pd
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import ou_scores, decide, build_weights

TC = 0.0005
tk = all_tickers(UNIVERSE_US)
px = download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets = log_returns(px.drop(columns=['SPY']))
R_np = rets.values; cols=list(rets.columns)
all_d = list(rets.index)
start = next(i for i,d in enumerate(all_d) if d >= pd.Timestamp('2016-01-04'))
PCAW, RW, NF, THR = 252, 60, 10, 1.25     # mid-grid, NOT the argmax -- avoids tuning
KS = [1, 2, 3, 5, 10, 21]

print(f"config: n_factors={NF} reg_win={RW} thr={THR} (mid-grid, not the sweep argmax)\n")
rows=[]
for K in KS:
    prev_map, prev_w = {}, np.zeros(len(cols))
    r_l, t_l, na_l = [], [], []
    for i in range(start, len(all_d)-1):
        win = R_np[i-PCAW+1:i+1]
        live = np.where(~np.isnan(win).any(axis=0))[0]
        if len(live) < 30: continue
        fwd = np.nan_to_num(R_np[i+1, live], nan=0.0)

        if (i - start) % K == 0:                       # REBALANCE day: recompute weights
            Rw = win[:, live]
            sigma = Rw.std(axis=0, ddof=1); sigma=np.where(sigma<1e-12,np.nan,sigma)
            Y=(Rw-Rw.mean(axis=0))/sigma
            C=np.nan_to_num(np.corrcoef(Y,rowvar=False),nan=0.0); np.fill_diagonal(C,1.0)
            vals,vecs=np.linalg.eigh(C); order=np.argsort(vals)[::-1][:NF]
            Q=(vecs[:,order]/sigma[:,None]).T
            Rr=Rw[-RW:]; F=Rr@Q.T; X=np.column_stack([np.ones(len(F)),F])
            B,*_=np.linalg.lstsq(X,Rr,rcond=None)
            s,kappa=ou_scores(np.cumsum(Rr-X@B,axis=0))
            s=np.where(kappa>=252/30,s,np.nan)
            p=np.array([prev_map.get(c,0.0) for c in live])
            sig=decide(s,p,THR,THR,0.50,0.75)
            w_live,_=build_weights(sig,B[1:,:],Q)
            w=np.zeros(len(cols)); w[live]=w_live
            prev_map={c:v for c,v in zip(live,sig) if v!=0}
            na_l.append(int((sig!=0).sum()))
        else:                                          # HOLD: same weights, no trade
            w = prev_w
        r_l.append(float(w[live] @ fwd))
        t_l.append(float(np.abs(w-prev_w).sum()))
        prev_w = w

    r=np.array(r_l); t=np.array(t_l); sd=r.std()
    gd, cd = r.mean(), t.mean()*TC
    cov = (gd/max(t.mean(),1e-9))/TC
    rows.append({"rebal_days":K,
        "gross_pct_yr":round(100*gd*252,2), "net_pct_yr":round(100*(gd-cd)*252,2),
        "gross_sharpe":round(gd/sd*np.sqrt(252),2), "net_sharpe":round((gd-cd)/sd*np.sqrt(252),2),
        "turn_day":round(t.mean(),3), "turn_yr_x":round(t.mean()*252,0),
        "breakeven_bps":round(1e4*gd/max(t.mean(),1e-9),2), "coverage":round(cov,2),
        "n_active":round(np.mean(na_l),0)})
    print(f"  K={K:2d}d done", flush=True)

df=pd.DataFrame(rows); pd.set_option('display.width',200)
print()
print("="*100)
print("REBALANCE FREQUENCY FRONTIER   (cost = 5bps/unit turnover)")
print("="*100)
print(df.to_string(index=False))
df.to_csv('/Users/katiyar/Desktop/Projects/StatArb/results/phase_a_holding.csv', index=False)
print()
print("coverage > 1.0 at ANY frequency? ", "YES" if (df.coverage>1).any() else "NO")
print("net Sharpe > 0 at ANY frequency? ", "YES" if (df.net_sharpe>0).any() else "NO")
