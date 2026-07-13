"""
run_phase_a.py -- PHASE A: the Avellaneda-Lee baseline, and the go/no-go for this project.

WHAT THIS TESTS
  Does mean reversion in PCA factor residuals have a real edge on ~460 US large-caps in
  2016-2026 -- BEFORE any deep learning is involved?

WHY IT IS A FAIR TEST IN A WAY THE PAIRS PROJECT NEVER WAS
  There is no pair selection, no screener, and no fitted hyperparameter. Every quantity at
  time t (PCA factors, betas, OU parameters, s-scores) is estimated from a trailing window
  ending at t. The whole post-warmup period is therefore ONE CONTINUOUS out-of-sample
  backtest with no lookahead available anywhere.

PRE-COMMITTED CRITERION -- written before the first run, and not to be moved afterwards:

      GROSS Sharpe < 0.5  ->  STOP THE PROJECT.

  Gross, not net. If the residual signal has no edge BEFORE costs, then no neural network
  can manufacture one -- Phase B would be pure curve-fitting. The predecessor project died
  because its gross edge (0.17 bps/trade) was smaller than its costs (0.18 bps/trade); if
  that arithmetic repeats here, it dies for the same reason and we stop.

USAGE
  python3 run_phase_a.py
  python3 run_phase_a.py --n-factors 15 --pca-win 252 --reg-win 60
  python3 run_phase_a.py --t-cost 0.0005 --start 2015-01-01
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest
from data import download_prices, log_returns
from metrics import summarize
from residuals import residual_signals, TRADING_DAYS
from universe import UNIVERSE_US, all_tickers

BENCH = "SPY"
GO_NO_GO_GROSS_SHARPE = 0.5


def main():
    ap = argparse.ArgumentParser(description="Phase A: Avellaneda-Lee PCA residual baseline")
    ap.add_argument("--start", default="2015-01-01", help="history start (warmup included)")
    ap.add_argument("--test-start", default="2016-01-04", help="first traded day")
    ap.add_argument("--pca-win", type=int, default=252, help="PCA/correlation window")
    ap.add_argument("--reg-win", type=int, default=60, help="factor-regression window")
    ap.add_argument("--n-factors", type=int, default=15)
    ap.add_argument("--kappa-min", type=float, default=TRADING_DAYS / 30.0,
                    help="min OU reversion speed (default: reverts inside ~30d)")
    ap.add_argument("--t-cost", type=float, default=0.0005, help="one-way cost per unit turnover")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--sbo", type=float, default=1.25)
    ap.add_argument("--sso", type=float, default=1.25)
    ap.add_argument("--sbc", type=float, default=0.50)
    ap.add_argument("--ssc", type=float, default=0.75)
    ap.add_argument("--out", default="results/phase_a.csv")
    args = ap.parse_args()

    tk = all_tickers(UNIVERSE_US)
    print("=" * 92)
    print("PHASE A -- AVELLANEDA-LEE PCA RESIDUAL MEAN REVERSION")
    print("=" * 92)
    print(f"universe={len(tk)} tickers | factors={args.n_factors} | pca_win={args.pca_win} "
          f"| reg_win={args.reg_win} | t_cost={args.t_cost*1e4:.0f}bps")
    print(f"NO pair selection, NO screener, NO fitted hyperparameter -> no lookahead exists.")
    print(f"PRE-COMMITTED: gross Sharpe < {GO_NO_GO_GROSS_SHARPE} => STOP THE PROJECT.\n")

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    print("Downloading universe (cached after first run)...")
    px = download_prices(tk + [BENCH], args.start, today)
    print(f"  {px.shape[1]} tickers, {px.index[0].date()} -> {px.index[-1].date()}")

    rets = log_returns(px.drop(columns=[BENCH], errors="ignore"))
    dates = rets.index[rets.index >= pd.to_datetime(args.test_start)]
    print(f"  trading {len(dates)} days from {dates[0].date()} to {dates[-1].date()}\n")

    print("Computing rolling factor residuals + OU s-scores...")
    W, diag = residual_signals(
        rets, dates,
        pca_win=args.pca_win, reg_win=args.reg_win, n_factors=args.n_factors,
        kappa_min=args.kappa_min,
        thresholds={"sbo": args.sbo, "sso": args.sso, "sbc": args.sbc, "ssc": args.ssc},
    )
    act = diag["n_active"].mean()
    print(f"  avg tradeable {diag['n_tradeable'].mean():.0f} | avg active positions {act:.0f}"
          f" | avg |net exposure| {diag['net_exposure'].abs().mean():.3f}"
          f" | var explained {diag['var_explained'].mean():.1%}\n")

    if act < 1:
        print("No active positions -- thresholds or kappa filter are too strict. Aborting.")
        return

    res = backtest.run(W, rets, args.capital, args.t_cost)

    # Per-calendar-year breakdown: a real edge should not live in one lucky year.
    print("=" * 92)
    print("PER-YEAR (gross vs net)")
    print("=" * 92)
    print(f"{'year':6s} {'GROSS%':>8s} {'NET%':>8s} {'cost%':>7s} {'gSharpe':>8s} {'Sharpe':>7s} "
          f"{'maxDD':>7s} {'turn/d':>7s} {'SPY%':>7s}")
    rows = []
    d = pd.Series(res["net_pnl"][1:], index=res["dates"])
    g = pd.Series(res["gross_pnl"][1:], index=res["dates"])
    for yr, grp in d.groupby(d.index.year):
        sl = res["dates"].year == yr
        gr = np.concatenate([[0.0], g[sl].values])
        nr = np.concatenate([[0.0], d[sl].values])
        gr = gr - gr[0]; nr = nr - nr[0]
        # re-base each year to start at zero
        gy = np.concatenate([[0.0], np.diff(g[sl].values, prepend=g[sl].values[0])]).cumsum()
        ny = np.concatenate([[0.0], np.diff(d[sl].values, prepend=d[sl].values[0])]).cumsum()
        from metrics import compute_metrics
        mg = compute_metrics(gy, args.capital)
        mn = compute_metrics(ny, args.capital)
        turn = res["turnover"][sl].mean()
        bpx = px[BENCH].reindex(res["dates"][sl]).ffill()
        spy = 100 * (float(bpx.iloc[-1]) / float(bpx.iloc[0]) - 1)
        print(f"{yr:<6d} {mg['ret_pct']:+7.2f}% {mn['ret_pct']:+7.2f}% "
              f"{mg['ret_pct']-mn['ret_pct']:6.2f}% {mg['sharpe']:+8.2f} {mn['sharpe']:+7.2f} "
              f"{mn['max_dd_pct']:+6.1f}% {turn:7.3f} {spy:+6.1f}%")
        rows.append({"year": yr, "gross_pct": mg["ret_pct"], "net_pct": mn["ret_pct"],
                     "gross_sharpe": mg["sharpe"], "sharpe": mn["sharpe"],
                     "max_dd_pct": mn["max_dd_pct"], "avg_turnover": round(float(turn), 3),
                     "spy_pct": round(spy, 2)})

    bm = backtest.benchmark(px, BENCH, res["dates"], args.capital)

    print("\n" + "=" * 92)
    print("FULL PERIOD")
    print("=" * 92)
    print(f"{'':16s} {'return':>9s} {'Sharpe':>8s} {'maxDD':>8s} {'ann vol':>8s}")
    print(f"{'Residual GROSS':16s} {res['gross_pct']:+8.2f}% {res['gross_sharpe']:+8.2f} "
          f"{res['gross']['max_dd_pct']:+7.1f}% {res['gross']['ann_vol_pct']:7.1f}%")
    print(f"{'Residual NET':16s} {res['net_pct']:+8.2f}% {res['sharpe']:+8.2f} "
          f"{res['max_dd_pct']:+7.1f}% {res['ann_vol_pct']:7.1f}%")
    print(f"{'SPY buy&hold':16s} {bm['gross_pct']:+8.2f}% {bm['sharpe']:+8.2f} "
          f"{bm['max_dd_pct']:+7.1f}% {bm['ann_vol_pct']:7.1f}%")
    print(f"\ncost drag {res['cost_pct']:.2f}%  |  avg daily turnover "
          f"{res['avg_daily_turnover']:.3f} ({res['net'].get('ann_turnover_x', 0):.0f}x/yr)")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out}")

    gs = res["gross_sharpe"]
    print("\n" + "=" * 92)
    if gs < GO_NO_GO_GROSS_SHARPE:
        print(f"NO-GO: gross Sharpe {gs:.2f} < {GO_NO_GO_GROSS_SHARPE}.")
        print("The residual signal has no edge even BEFORE costs. No neural network can")
        print("manufacture one. Phase B would be curve-fitting. STOP.")
    else:
        print(f"GO: gross Sharpe {gs:.2f} >= {GO_NO_GO_GROSS_SHARPE}.")
        print("A real pre-cost edge exists in the residuals. Phase B (CNN+Transformer signal")
        print("extraction with a Sharpe objective and turnover penalty) has something to learn.")
        if res["sharpe"] < 0:
            print(f"\nBUT net Sharpe is {res['sharpe']:.2f}: costs currently eat the entire edge.")
            print("Phase B must optimize NET Sharpe with a turnover penalty, not raw signal.")
    print("=" * 92)


if __name__ == "__main__":
    main()
