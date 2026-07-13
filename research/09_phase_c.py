"""
PHASE C -- REALITY CHECK on the winner (the OU s-score; the learned signal lost).

The backtest is more generous than reality. Add what it omits, and report the strategy
against CASH, not zero.

PRE-COMMITTED: net-of-everything Sharpe < 0.5, or it cannot beat cash => NOT TRADEABLE.

I report the MEDIAN across the structural grid AND the argmax. The argmax of a grid is a
selection-biased statistic -- quoting it as "the result" is how the last project died.
"""
import numpy as np, pandas as pd, itertools
from data import download_prices, log_returns
from universe import UNIVERSE_US, all_tickers
from residuals import ou_scores

tk=all_tickers(UNIVERSE_US)
px=download_prices(tk+['SPY'],'2015-01-01',pd.Timestamp.today().strftime('%Y-%m-%d'))
rets=log_returns(px.drop(columns=['SPY']))
R_np=rets.values; cols=list(rets.columns); all_d=list(rets.index)
start=next(i for i,d in enumerate(all_d) if d>=pd.Timestamp('2016-01-04'))
PCAW=252
NF=[5,10,15]; RW=[60,120]; HOLD=[5,10,21]
cfgs=list(itertools.product(NF,RW,HOLD))
st={c:{"pw":np.zeros(len(cols)),"r":[],"t":[],"shrt":[],"d":[]} for c in cfgs}

for n_,i in enumerate(range(start,len(all_d)-1)):
    win=R_np[i-PCAW+1:i+1]
    live=np.where(~np.isnan(win).any(axis=0))[0]
    if len(live)<30: continue
    fwd=np.nan_to_num(R_np[i+1,live],nan=0.0)
    Rw=win[:,live]
    sigma=Rw.std(axis=0,ddof=1); sigma=np.where(sigma<1e-12,np.nan,sigma)
    Yv=(Rw-Rw.mean(axis=0))/sigma
    C=np.nan_to_num(np.corrcoef(Yv,rowvar=False),nan=0.0); np.fill_diagonal(C,1.0)
    vals,vecs=np.linalg.eigh(C); order=np.argsort(vals)[::-1]
    for nf in NF:
        Q=(vecs[:,order[:nf]]/sigma[:,None]).T
        for rw in RW:
            Rr=Rw[-rw:]; F=Rr@Q.T; X=np.column_stack([np.ones(len(F)),F])
            B,*_=np.linalg.lstsq(X,Rr,rcond=None)
            s,kap=ou_scores(np.cumsum(Rr-X@B,axis=0))
            v=np.nan_to_num(np.where(kap>0,-s,np.nan),nan=0.0)
            if v.std()>1e-12: v=(v-v.mean())/v.std()
            g=np.abs(v).sum(); tgt=v/g if g>1e-12 else v*0
            c_=B[1:,:]@tgt; wl=tgt-(c_@Q)
            gg=np.abs(wl).sum(); wl=wl/gg if gg>1e-12 else wl*0
            wt=np.zeros(len(cols)); wt[live]=wl
            for K in HOLD:
                d=st[(nf,rw,K)]
                w = wt if (n_%K==0) else d["pw"]
                d["r"].append(float(w[live]@fwd))
                d["t"].append(float(np.abs(w-d["pw"]).sum()))
                d["shrt"].append(float(np.abs(w[w<0]).sum()))   # short-leg gross exposure
                d["d"].append(all_d[i]); d["pw"]=w

TCS=[0,1,2,3,5,10]                 # bps per unit turnover, one-way
BORROW=[0.0,0.003,0.005,0.01]      # annual, on the short leg
RF=0.045                           # ~T-bill over the sample

rows=[]
for (nf,rw,K),d in st.items():
    r=np.array(d["r"]); t=np.array(d["t"]); sh=np.array(d["shrt"]); sd=r.std()
    yrs=len(r)/252; gd=r.mean()
    row={"nf":nf,"rw":rw,"hold":K,"gross_yr":100*gd*252,
         "gSharpe":gd/sd*np.sqrt(252),"turn_yr":t.mean()*252,"short_frac":sh.mean(),
         "breakeven_bps":1e4*gd/max(t.mean(),1e-9)}
    for tc in TCS:
        for bw in BORROW:
            nd = gd - t.mean()*tc*1e-4 - sh.mean()*bw/252
            row[f"nS_tc{tc}_bw{int(bw*1e4)}"]=nd/sd*np.sqrt(252)
            row[f"net_tc{tc}_bw{int(bw*1e4)}"]=100*nd*252
    rows.append(row)
df=pd.DataFrame(rows); df.to_csv('/Users/katiyar/Desktop/Projects/StatArb/results/phase_c.csv',index=False)
pd.set_option('display.width',250)

print("="*96); print("C1. BREAK-EVEN TRANSACTION COST  (pre-committed: <10bps/leg = too thin for reality)"); print("="*96)
print(f"  across {len(df)} structural configs:  median {df.breakeven_bps.median():.2f} bps   "
      f"max {df.breakeven_bps.max():.2f} bps   min {df.breakeven_bps.min():.2f} bps")
print(f"  configs with break-even > 10 bps: {(df.breakeven_bps>10).sum()}/{len(df)}")
print(f"  short leg = {df.short_frac.median():.1%} of gross book (borrow applies to this)")

print()
print("="*96); print("C2. NET SHARPE vs COST and SHORT BORROW   (median across structural configs)"); print("="*96)
print(f"{'t_cost':>7s} " + " ".join(f"{'bw='+str(int(b*1e4))+'bp':>10s}" for b in BORROW))
for tc in TCS:
    cells=" ".join(f"{df[f'nS_tc{tc}_bw{int(b*1e4)}'].median():>10.2f}" for b in BORROW)
    print(f"{tc:>5d}bp {cells}")
print()
print("  same table, but the ARGMAX config (selection-biased -- shown only for honesty):")
print(f"{'t_cost':>7s} " + " ".join(f"{'bw='+str(int(b*1e4))+'bp':>10s}" for b in BORROW))
for tc in TCS:
    cells=" ".join(f"{df[f'nS_tc{tc}_bw{int(b*1e4)}'].max():>10.2f}" for b in BORROW)
    print(f"{tc:>5d}bp {cells}")

print()
print("="*96); print("C3. SHARPE vs CASH"); print("="*96)
best=df.loc[df["nS_tc5_bw50"].idxmax()]
print(f"  This book is DOLLAR-NEUTRAL (short leg {best.short_frac:.0%} of gross), so it is")
print(f"  SELF-FINANCING: the reported return is ALREADY an excess return over cash. The")
print(f"  README's claim that 'a 1%/yr strategy has a deeply negative Sharpe vs T-bills' is")
print(f"  TRUE for a long-only book and FALSE for this one. Correcting it does NOT save the")
print(f"  strategy -- the excess-return Sharpe is simply the number reported above.")
print(f"    total return to an investor ~= rf({RF:.1%}) + alpha - borrow")
b5=best['net_tc5_bw50']
print(f"    best config @5bp+50bp borrow: alpha {b5:+.2f}%/yr  ->  total ~{100*RF+b5:.2f}%/yr vs cash {100*RF:.1f}%")
print(f"    excess-return Sharpe = {best['nS_tc5_bw50']:+.2f}")

print()
print("="*96); print("C4. SUBPERIOD STABILITY (gross, median config -- does it work in the bear?)"); print("="*96)
d=st[(10,60,10)]
sr=pd.Series(d["r"],index=pd.DatetimeIndex(d["d"])); tt=pd.Series(d["t"],index=sr.index)
for lab,a,b in [("2016-2019 (bull)","2016","2019"),("2020-2021 (covid)","2020","2021"),
                ("2022      (bear)","2022","2022"),("2023-2026 (bull)","2023","2026")]:
    x=sr[a:b]; y=tt[a:b]
    if len(x)<50: continue
    gs=x.mean()/x.std()*np.sqrt(252)
    ns=(x.mean()-y.mean()*5e-4)/x.std()*np.sqrt(252)
    print(f"  {lab}: gross {100*x.mean()*252:+6.2f}%/yr  gSharpe {gs:+5.2f}  |  net {100*(x.mean()-y.mean()*5e-4)*252:+6.2f}%/yr  nSharpe {ns:+5.2f}")

print()
print("="*96); print("C5. CAPACITY"); print("="*96)
tv=df.turn_yr.median()/252
for aum in [1e8,1e9,5e9,2e10]:
    dollars=aum*tv; per=dollars/450
    print(f"  AUM ${aum/1e9:>5.1f}B: ${dollars/1e6:>7.1f}M traded/day  = ${per/1e6:>5.2f}M/name/day"
          f"  ~= {100*per/1e8:>5.2f}% of a $100M ADV name")
print("  Capacity is NOT the binding constraint -- the raw edge is. The strategy dies at")
print("  $10M for the same reason it dies at $10B: the edge does not cover the spread.")
