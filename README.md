# StatArb — Factor-Residual Statistical Arbitrage

Mean reversion in **PCA factor residuals** across ~460 US large-caps.
Successor to the [Kalman pairs-trading project](../Kalman/PROJECT_REPORT.md), which was **closed as a null result**.

---

## Why this project exists

The predecessor traded cointegrated **pairs**. It failed, and the post-mortem is the entire design brief here:

| What killed the pairs project | How this project structurally eliminates it |
|---|---|
| **Selection bias.** Pairs were screened on data overlapping the test period. Sharpe went **1.55 → −0.20** the moment the screen was made honest. | **There is no selection.** No screener, no pairs. Every stock gets a signal every day. Nothing to bias. |
| **No breadth.** 30 pairs with IC ≈ 0.14. `IR = IC × √breadth` leaves nothing. | **~460 signals**, one per stock. |
| **Wrong objective.** Nets minimized Huber loss on a 2%-R² target, then a threshold was bolted on. The trading objective was never optimized. | **Phase B optimizes Sharpe directly**, with a turnover penalty. |
| **edge/trade (0.17bps) < cost/trade (0.18bps).** Coverage 0.94×. | **Turnover is a first-class constraint**, not an afterthought. |
| **2% R² ceiling** — the DL target's signal was already in `z_spread`. | Factor residuals are a cleaner, higher-amplitude reverting series. |
| Screener selected **GOOG/GOOGL** (spread amplitude < round-trip cost). | No screener exists. |

The method follows [Avellaneda & Lee (2010)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505), with [Guijarro-Ordonez, Pelger & Zanotti (2022)](https://arxiv.org/abs/2106.04028) as the Phase B target.

---

## The method (Phase A)

1. Rolling 252d returns → correlation matrix → top 15 **eigenportfolios** (tradeable factor portfolios).
2. Regress each stock on those factors over 60d. By APT, the factor portfolio is the stock's **fair price**; the residual is the temporary deviation.
3. Cumulate the residual, fit an **Ornstein-Uhlenbeck** process, form the **s-score** `s = (X − m_eq) / σ_eq`.
4. Trade the s-score (long when far below equilibrium, short when far above), rejecting names whose reversion is too slow to round-trip (`kappa` filter).
5. Hold the stock **minus its factor hedge** → the book is factor-neutral by construction.

**Everything is rolling.** Every parameter at time *t* comes from a trailing window ending at *t*. There is no train/test split and no fitted hyperparameter, so the whole post-warmup period is one continuous out-of-sample backtest with **no lookahead available anywhere**.

---

## The rules of this project

These are not style preferences. Each one is a scar.

1. **Pre-commit the falsification criterion before every run.** Write it into the script. When the number comes back, you do not get to move it.
2. **Gross and net, always, side by side.** The pairs project spent months not knowing its gross edge was smaller than its cost bill.
3. **Point-in-time everything — *including selection*.** The old protocol rigorously prevented leakage in *fitting* and was completely blind to leakage in *selection*. That one hole invalidated nine stages of work.
4. **Never read a conclusion off a 2-seed ablation.** Seed σ was 1–2%; every ±2–4% swing across scenarios was noise. That is how two false positives were manufactured.
5. **Robustness to a parameter ≠ robustness to the protocol.** A Sharpe flat across every threshold felt like proof. It was flat in the wrong dimension.
6. **Any model must beat a linear baseline on the same task**, or it dies. We never ran that control and paid for it.

---

## Phase A — the go/no-go

```bash
pip install -r requirements.txt
python3 run_phase_a.py
```

> ### PRE-COMMITTED: **gross Sharpe < 0.5 → STOP THE PROJECT.**
>
> **Gross, not net.** If the residual signal has no edge *before* costs, no neural network can manufacture one — Phase B would be pure curve-fitting. The pairs strategy died because gross edge < costs. If that arithmetic repeats, it dies the same way.

---

## Phase B — only if Phase A passes

Replace the OU s-score with a **CNN+Transformer** signal extractor, and the threshold rule with a network mapping signals → portfolio weights, trained on **negative Sharpe with a turnover penalty**.

This is the one framing with published evidence, and it gives deep learning a job it can actually do — unlike the predecessor, where it was asked to rediscover a coefficient already sitting in its own input features.

**Honest prior: ~25–30%** this ends profitable after realistic costs. The architecture is sound and independently evidenced, but the cost problem does not vanish because the signal is cleaner, and 2016–2026 is a far more arbitraged sample than the papers' 1998–2016.

---

## Known limitations (stated up front)

- **Survivorship bias.** `universe.py` lists names liquid *today*; the bankrupt and delisted are absent. Unfixable without point-in-time data (CRSP). Biases results **optimistically**.
- **Unmodelled costs.** Short borrow (0.3–1%/yr), market impact, and financing are *not* in the cost model. `t_cost` covers commission + spread only.
- **Sharpe is benchmarked against zero, not cash.** At a 4–5% risk-free rate, a 1%/yr strategy has a deeply negative Sharpe versus T-bills.
- **The literature's headline numbers are not credible.** Replications report Sharpe > 10 pre-cost; the authors themselves note these *"have not yet incorporated realistic transaction costs, market impact, or short-sale fees."* Trust the architecture, not the numbers.

---

## Layout

| File | Role |
|---|---|
| `universe.py` | 463 US tickers, 40 groups. Groups are **diagnostics only** — nothing selects on them. |
| `data.py` | yfinance download + parquet cache. `threads=False` is mandatory (SQLite cache corruption). |
| `residuals.py` | **Core.** PCA eigenportfolios → factor regression → OU fit → s-score → factor-neutral weights. |
| `backtest.py` | Weight-vector P&L engine. Gross/net, turnover-based costs. Weights at *t* earn returns at *t+1*. |
| `metrics.py` | Sharpe / max-DD / vol / turnover. |
| `run_phase_a.py` | Phase A runner with the go/no-go wired in. |
| `models/` | Phase B (CNN+Transformer). Empty until Phase A passes. |
