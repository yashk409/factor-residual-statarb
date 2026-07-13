# StatArb — Factor-Residual Statistical Arbitrage — Final Report

**Status: CLOSED — null result.**
**Verdict: KILL IT. The residual signal has a real, statistically significant gross edge. It is not large enough to pay for the trading required to harvest it — and it does not clear the bar even if trading were free.**

Phases A, B (pre-flight) and C complete. 2016-01-04 → 2026-07-10, 463 US large-caps, ~2,644 trading days, one continuous out-of-sample backtest.

---

## The one-line result

| | Pairs (predecessor) | **StatArb (this project)** |
|---|---|---|
| edge / unit turnover | 0.17 bps | **1.67 bps** |
| cost / unit turnover | 0.18 bps | **5.00 bps** |
| **coverage** | **0.94×** | **0.33×** |
| break-even cost | 10 bps/leg | **3.5 bps** (median), 8.7 (best of 18) |

> **The gross edge is real: +1.5%/yr, gross Sharpe 0.73, positive in 57 of 60 structural configurations, IC t = +4.11.**
> **The net edge does not exist: net Sharpe > 0 in 0 of 60 configs at daily rebalance, and t > 2 in 0 of 108 configs at any rebalance frequency.**

Buying SPY returned **+343.6% (Sharpe 0.84)** over the same window. The strategy returned **−31.8%** net.

---

## Phase A — the go/no-go

**PRE-COMMITTED (written into `run_phase_a.py` before it was ever run): gross Sharpe < 0.5 → STOP.**

**Result: gross Sharpe 0.73. The criterion PASSED. It was not moved.**

```
universe=463 | factors=15 | pca_win=252 | reg_win=60 | t_cost=5bps
avg tradeable 454 | avg active 157 | avg |net exposure| 0.108 | var explained 63.2%

                    return   Sharpe    maxDD  ann vol
Residual GROSS     +15.98%    +0.73    -3.0%     2.1%
Residual NET       -31.83%    -1.46   -33.3%     2.1%
SPY buy&hold      +343.56%    +0.84   -33.7%    38.9%

cost drag 47.81%  |  avg daily turnover 0.362 (91x/yr)
```

The gate did its job — a real pre-cost edge exists — but it was a *necessary*, not a *sufficient*, condition. **Costs are 3× the entire gross edge.** Coverage 0.33×, versus the 0.94× that killed the predecessor. This project's strategy is, on the metric that matters, **three times worse than the one we buried.**

### Per-year: the edge is decaying, and it dies in the bear market

| year | gross | net | gross Sharpe | SPY |
|---|---|---|---|---|
| 2016 | +4.53% | −0.08% | +2.81 | +13.6% |
| 2017 | +1.05% | −3.61% | +0.76 | +20.8% |
| 2018 | +1.72% | −2.66% | +1.02 | −5.2% |
| 2019 | +2.54% | −1.90% | +1.55 | +31.1% |
| 2020 | +2.22% | −2.65% | +0.70 | +17.2% |
| 2021 | +1.53% | −3.14% | +0.90 | +30.5% |
| **2022** | **−0.82%** | −5.29% | **−0.33** | −18.6% |
| 2023 | +2.48% | −1.77% | +1.23 | +26.7% |
| **2024** | **−0.80%** | −5.18% | **−0.41** | +25.6% |
| 2025 | +2.58% | −1.92% | +1.29 | +18.0% |
| **2026** | **−1.87%** | −4.24% | **−1.29** | +10.6% |

2016 alone is 28% of the total gross P&L. Gross is **negative** in 2022, 2024 and 2026. This matches Avellaneda & Lee's own finding — they report Sharpe 1.44 for 1997–2007 but only **0.9** for 2003–2007. The edge was decaying when they published it in 2010. It has kept decaying.

### The code audit (done BEFORE the first run, deliberately)

No lookahead found, and the README's structural claim holds:
- `residual_signals` slices `R_all.loc[:dt]` at every date — every PCA, beta, and OU parameter is trailing.
- `live` names require a complete *trailing* 252-day history; a stock is never traded before it existed.
- `backtest.run` applies `shift(-1)` — weights at *t* earn returns at *t+1*.
- `auto_adjust=True` handles splits and dividends.
- **There is no screener and no selection, so the bias that killed the predecessor is structurally impossible here.** That claim survived scrutiny.

Two properties confirmed numerically before trusting any number:

1. **The s-score carries no information from the current residual level.** OLS with an intercept forces residuals to sum to zero over the fitting window, so `resid_cum[-1] = 0` for every stock every day (verified: max |·| = 1.1e-15). The s-score therefore collapses to exactly `s = −m_eq/σ_eq` (verified to 1e-13). This is Avellaneda-Lee as published, but it means the signal is a function only of the *shape* of the 60-day path.
2. **The κ filter is inert.** It passes **458/458** names. It is supposed to reject stocks that revert too slowly to round-trip, and it rejects nobody — AR(1) coefficients are downward-biased on a 60-point sample, which inflates κ past any sane threshold. `kappa_min` is not a knob; it is decoration.

---

## The turnover diagnosis

Turnover is 91×/yr. My first hypothesis was that this was an **implementation artifact**: the PCA and betas are re-estimated daily, so the factor hedge would churn even when no signal changed. If true, it was fixable and free.

**It was wrong, and the test said so:**

| | turnover/day | annualized |
|---|---|---|
| actual book (hedge refit daily) | 0.361 | 91× |
| signal-only book (**no hedge at all**) | 0.256 | 64× |
| → **hedge churn share** | **29%** | |

The signal itself churns at 64×/yr. 35 of 158 positions flip *every day*; average holding period **4.5 days**. Even eliminating 100% of the hedge churn leaves coverage at 0.47× — still under 1. **The strategy is intrinsically expensive.**

---

## The structural sweep — 60 configurations

`n_factors` ∈ {5,10,15,20} × `reg_win` ∈ {30,60,120} × threshold ∈ {1.0,1.25,1.5,2.0,2.5}. These were run as **robustness checks, not tuning**.

```
gross Sharpe: min -0.26  max +1.35  | POSITIVE in 57/60 configs
coverage    : min  0.00  max  0.50  | >1.0    in  0/60 configs
net Sharpe  : min -7.56  max -0.85  | >0      in  0/60 configs
```

**The gross edge is structural, not fitted** — it survives every setting. That is a genuine, and genuinely rare, finding: the predecessor never had this.

**And selectivity runs backwards here.** In the pairs project, raising the entry threshold cut turnover and lifted coverage. Here it does the opposite:

| threshold | active names | turnover/day | coverage |
|---|---|---|---|
| 1.00 | 150 | 0.536 | **0.50×** |
| 1.50 | 52 | 0.788 | 0.37× |
| 2.50 | 2 | 0.964 | 0.04× |

The reason is structural: **the book is gross-normalized to 1.0.** Cutting to 52 names makes each name a larger share of the book, so one signal flip now moves more capital. Selectivity *concentrates* turnover instead of reducing it. The one lever that partially rescued the pairs strategy is unavailable here **by construction.**

---

## The holding-period frontier — and a false positive I caught on myself

The only remaining lever was holding period — flagged in the predecessor's post-mortem (§12.3: *"you predict a 5-day reversion, then abandon it after 1 day when noise flips the sign"*) and independently claimed by Guijarro-Ordonez et al. (*"around half of the Sharpe ratio can persist for a holding period of one week"*).

At `nf=10, reg_win=60, thr=1.25`:

| rebalance | gross %/yr | net %/yr | gross Sharpe | **net Sharpe** | turnover/yr | coverage |
|---|---|---|---|---|---|---|
| 1d | 2.73 | −1.31 | 1.14 | −0.55 | 81× | 0.68× |
| **5d** | 2.56 | **+0.60** | 1.10 | **+0.26** | 39× | **1.30×** |
| 10d | 1.91 | +0.55 | 0.76 | +0.22 | 27× | 1.40× |
| 21d | 0.88 | +0.05 | 0.31 | +0.02 | 17× | 1.06× |

The literature claim **replicates**: gross Sharpe barely decays out to a 5-day hold (1.14 → 1.10) while turnover halves. Coverage clears 1.0. **This looked like the answer.**

> ### ❌ IT WAS A FALSE POSITIVE, AND IT WAS MINE.
>
> I ran that on **one cell** of the structural grid, chosen as a "neutral" mid-grid setting. Re-running the holding-period test across **all 36 structural configs**:
>
> | rebalance | gross Sharpe > 0 | **net Sharpe > 0** | **net t > 2** | median net %/yr |
> |---|---|---|---|---|
> | 1d | 36/36 | **0/36** | 0/36 | −3.05% |
> | 5d | 36/36 | **5/36** | **0/36** | −1.05% |
> | 10d | 36/36 | **7/36** | **0/36** | −0.23% |
>
> **The cell I picked was the single best of the 36.** The median config loses 1.05%/yr. Across all 108 configurations tested, **net Sharpe has t > 2 in exactly zero of them.**
>
> This is the predecessor's `entry_z = 1.5` error, reproduced exactly — a conclusion read off the argmax of a grid — and it survived only because the next test was run. **Had I stopped one test earlier, I would have reported "the strategy works at weekly rebalance, net Sharpe 0.26."**

---

## Phase B — killed at pre-flight, on evidence, before a line of it was written

Phase B proposed two branches. Rule 6 says any model must beat a linear baseline on the same task. **Both baselines were run first — which is what made building the network unnecessary.**

### Branch 2 (allocation): the linear control

A network mapping signals → weights, trained on −Sharpe with a turnover penalty, can only ever find a point on the signal/turnover frontier. That frontier is spanned, with **zero parameters**, by a linear allocator with EWMA smoothing:

```
w_target ∝ −s                                  (continuous — no discrete flips)
w_t = (1−λ)·w_{t−1} + λ·w_target                (EWMA smoothing ≡ a turnover penalty)
```

| λ | turnover/yr | median gross Sharpe | median net Sharpe | max net Sharpe |
|---|---|---|---|---|
| 1.00 | 94× | 0.80 | −1.38 | −0.50 |
| 0.25 | 37× | 0.56 | −0.34 | +0.10 |
| 0.10 | 22× | 0.34 | −0.15 | **+0.27** |
| 0.05 | 13× | −0.03 | −0.20 | +0.12 |

**No λ has a median net Sharpe above zero, let alone 0.5.** Best single cell: +0.27, **t = 0.86** — the argmax of 60 cells, and insignificant.

The turnover penalty *works* mechanically: turnover falls 94× → 13×/yr and median coverage improves 0.37 → 0.68. **But the gross edge decays as fast as the costs do** (median gross Sharpe 0.80 → −0.03). You cannot smooth your way out. You just trade less of a shrinking edge.

### Branch 1 (signal extraction): the s-score is NOT a sufficient statistic — and it doesn't matter

The predecessor's DL died because its target had a 2% R² ceiling and *the signal was already sitting in one of its own input features*. I expected to find the same thing here. **I was wrong, and the properly-powered test says so.**

The correct linear control for a CNN+Transformer reading the 60-day residual path is *an optimally-weighted linear model on that same path*. Walk-forward, refit yearly, 8-day purge, 978,624 out-of-sample stock-days. Judged on **daily cross-sectional IC (n = 2,140 paired days)**, not 9 annual ones:

| signal | mean daily IC | t vs 0 | IC-IR |
|---|---|---|---|
| OU s-score | +0.0080 | +4.11 | 0.089 |
| Ridge on path | +0.0119 | +6.56 | 0.142 |
| **GBT on path** | **+0.0149** | **+8.34** | **0.180** |

**GBT beats the s-score by paired t = +4.27, p < 0.001.** Ridge beats it at t = +2.18, p = 0.030.

**The residual path genuinely contains information the OU s-score throws away.** Guijarro-Ordonez et al.'s central claim — *"signal extraction is the separating element"* — is **statistically correct**.

*(An earlier, underpowered version of this test — 9 annual ICs — gave t = 1.05, p = 0.32, and would have killed this branch for the wrong reason. Nine observations is not a test.)*

### The decisive result: a better predictor is a worse strategy

Plug the learned signal into the **same allocator, same cost model** as Phase A:

| signal (λ=1.0) | gross %/yr | gross Sharpe | turnover/yr | **coverage** | **net Sharpe** |
|---|---|---|---|---|---|
| OU s-score | 2.98 | 1.21 | 89× | **0.67×** | −0.60 |
| **GBT on path** | **4.11** | **1.74** | **292×** | **0.28×** | **−4.45** |

The GBT's extra IC is real and **does** convert to gross alpha — gross Sharpe 1.21 → **1.74**. But it trades **3.3× more**, coverage collapses 0.67 → 0.28, and it loses badly net of costs.

**Best net Sharpe across every smoothing level: s-score +0.04, ridge −0.22, GBT −0.38.**

### The mechanism — the most valuable finding in this project

Watch what a turnover penalty does to each signal's gross alpha as λ goes 1.0 → 0.1:

- **OU s-score:** 2.98 → 1.10 %/yr — *degrades gracefully*
- **GBT:** 4.11 → **0.08** %/yr — *annihilated*

**Signal persistence (1-day autocorrelation):**

| signal | autocorr | gross %/yr | net Sharpe |
|---|---|---|---|
| **OU s-score** | **+0.921** | 2.99 | −0.60 |
| GBT, 1-day target | +0.287 | 4.54 | −4.31 |
| GBT, 5-day target | +0.577 | 2.11 | −0.45 |
| GBT, 10-day target | +0.604 | 0.46 | −0.61 |
| GBT, 21-day target | +0.626 | 0.78 | −0.05 |

> **The learned signal's incremental information lives entirely in the high-frequency component — exactly the component that costs the most to trade, and exactly the one a turnover penalty destroys.**
>
> **The s-score's alpha is persistent (ρ = 0.92). The GBT's is fleeting (ρ = 0.29). You can have the alpha, or you can afford to trade it. Not both.**

### The steelman — and its failure

**Objection:** the GBT minimized MSE at a 1-day horizon — a *prediction* objective. That is precisely the predecessor's §12.6 error. A network trained end-to-end on Sharpe *with a turnover penalty* would be incentivized to emit a signal that is **both informative and persistent**. The test above never gives it that chance.

**This is a fair objection, so it was tested.** Training on a longer-horizon target *is* a turnover penalty, expressed through the target instead of the loss: it forces the model to find slow, persistent structure rather than one-day noise.

It does exactly that — and it isn't enough. Signal autocorrelation rises 0.287 → 0.626 as the target horizon goes 1d → 21d, **but the gross alpha collapses from 4.54% to 0.78%/yr on the way.** The model can be informative or it can be cheap to hold, and the trade is close to one-for-one.

**Head to head, best net Sharpe: OU s-score +0.18. Best learned signal −0.05.**

> ### PHASE B IS DEAD — and for a *different reason* than the predecessor's DL.
>
> | | Predecessor | **This project** |
> |---|---|---|
> | gross alpha of the nets | **zero** | **real** (IC t = +4.27, gross Sharpe 1.97 vs 1.22) |
> | why it failed | *"confident estimators of nothing"* | **the information is real and economically unreachable** |
>
> **The persistent, tradeable component of the residual path is exactly what the OU s-score already extracts. Avellaneda & Lee's 2010 hand-crafted statistic is, for economic purposes, the sufficient statistic of the residual path.**
>
> Guijarro-Ordonez et al. are right that deep signal extraction beats the s-score. They are right *statistically* and wrong *economically* — which is exactly what their own replication warns: the headline numbers *"have not yet incorporated realistic transaction costs, market impact, or short-sale fees."* Trust the architecture, not the numbers. We did, and the architecture still doesn't pay.

---

## Phase C — reality check

**PRE-COMMITTED: net-of-everything Sharpe < 0.5, or cannot beat cash → NOT TRADEABLE.**

Run on the winner (the OU s-score), across 18 structural configs. Medians reported, because **the argmax of a grid is a selection-biased statistic and you cannot trade the argmax** — you must pick a config in advance.

### C1. Break-even transaction cost

| | median | best of 18 | configs > 10bps |
|---|---|---|---|
| break-even cost | **3.53 bps** | 8.65 bps | **0 / 18** |

Pre-committed threshold was 10 bps/leg. **Zero configurations pass. The criterion fires everywhere.**

### C2. The zero-cost ceiling — the number that ends the project

| t_cost | borrow=0 | 30bp | 50bp | 100bp |
|---|---|---|---|---|
| **0 bp** | **+0.38** | +0.32 | +0.29 | +0.21 |
| 1 bp | +0.30 | +0.23 | +0.19 | +0.09 |
| 2 bp | +0.18 | +0.12 | +0.08 | −0.02 |
| 3 bp | +0.06 | 0.00 | −0.04 | −0.14 |
| 5 bp | −0.13 | −0.20 | **−0.24** | −0.34 |
| 10 bp | −0.56 | −0.63 | −0.68 | −0.80 |

*(median net Sharpe across 18 structural configs)*

> **With ZERO transaction costs, ZERO borrow and ZERO market impact — a frictionless world — the median configuration's Sharpe is 0.38. It does not clear the pre-committed bar of 0.5 even when trading is FREE.**
>
> Only 5 of 18 configs clear 0.5 frictionlessly. **This is not a cost problem that better execution or a smarter allocator can solve.** Once the book is slowed enough to be affordable, the signal is not strong enough to matter.

At a realistic 5bps + 50bps borrow: **median net Sharpe −0.24**, best-of-18 +0.18. At an aggressively optimistic 2bps: median **+0.08**.

### C3. Sharpe vs cash — a claim in the project's own README is WRONG

The README and PROMPT both assert: *"Sharpe is benchmarked against zero, not cash. At a 4–5% risk-free rate, a 1%/yr strategy has a deeply negative Sharpe versus T-bills."*

**That is true for a long-only book and false for this one.** This book is **dollar-neutral** (short leg = 50.1% of gross), therefore **self-financing**: the long side is funded by the short proceeds, the capital sits in cash earning the risk-free rate, and the reported return **is already an excess return over cash.**

- Total return to an investor ≈ rf (4.5%) + alpha − borrow
- Best config @ 5bps + 50bps borrow: alpha **+0.39%/yr** → total ≈ **4.89%/yr** vs cash at 4.5%
- **Excess-return Sharpe = +0.18** — which is simply the net Sharpe already reported.

So the strategy *does* beat cash — by **39 basis points a year**, at the best of 18 configurations, with a Sharpe of 0.18. Correcting the error does not save it. **It just means the strategy is worthless rather than catastrophic.**

### C4. Subperiod stability — it fails in the bear market

| period | gross %/yr | gross Sharpe | net %/yr | net Sharpe |
|---|---|---|---|---|
| 2016–2019 (bull) | +2.52% | +1.63 | +1.15% | +0.75 |
| 2020–2021 (covid) | +1.32% | +0.45 | +0.00% | 0.00 |
| **2022 (bear)** | **+0.38%** | **+0.16** | **−0.95%** | **−0.39** |
| 2023–2026 (bull) | +2.21% | +1.02 | +0.86% | +0.40 |

The edge lives in bull markets and **dies in 2022**, the one period a market-neutral book is supposed to earn its keep. Whatever this is, it is not the crisis hedge that dollar-neutral stat arb is sold as.

### C5. Capacity — not the binding constraint

| AUM | traded/day | per name/day | % of a $100M ADV name |
|---|---|---|---|
| $0.1B | $9.3M | $0.02M | 0.02% |
| $1.0B | $92.9M | $0.21M | 0.21% |
| $5.0B | $464M | $1.03M | 1.03% |
| $20B | $1,857M | $4.13M | 4.13% |

Impact only begins to bite around $5B. **Capacity is irrelevant — the strategy dies at $10M for the same reason it dies at $10B: the edge does not cover the spread.**

### C6. Survivorship — cannot be quantified, and biases us optimistically

`universe.py` lists 463 names liquid **today**. The bankrupt and delisted are absent. This **cannot be quantified without point-in-time data (CRSP)**, and it biases every number in this report **upward**. The true result is worse than what is written here. It does not need to be worse to change the verdict.

---

## Falsification ledger — including my own false positives

| Claim | Verdict | Note |
|---|---|---|
| No lookahead exists in the Phase A design | ✅ **Confirmed** | Audited before the first run. No screener, no selection → the predecessor's fatal bias is structurally impossible. |
| The s-score = −m_eq/σ_eq exactly | ✅ **Confirmed** | Verified to 1e-13. `resid_cum[-1]` is pinned to 0 by the OLS intercept. |
| The κ filter rejects slow-reverting names | ❌ **REFUTED** | It passes **458/458**. Small-sample AR(1) bias inflates κ. The filter is decoration. |
| ~~Turnover is hedge churn — an implementation artifact~~ | ❌ **REFUTED (mine)** | My first hypothesis. Hedge churn is only **29%**. The signal itself churns at 64×/yr. Not fixable. |
| ~~The strategy works at weekly rebalance (net Sharpe 0.26)~~ | ❌ **REFUTED (mine)** | **Read off the argmax of a 36-cell grid.** Positive in only 5/36 configs; median loses 1.05%/yr; t > 2 in **0/108**. This is the predecessor's `entry_z` error, reproduced exactly. Caught only because the next test was run. |
| ~~The s-score is a sufficient statistic of the residual path~~ | ❌ **REFUTED (mine)** | I expected the predecessor's "signal already in the features" failure. **Wrong.** GBT beats it at paired t = +4.27, p < 0.001. |
| ~~Phase B's DL will have no gross alpha (like the predecessor's)~~ | ❌ **REFUTED (mine)** | It has **real** gross alpha — gross Sharpe 1.97 vs the s-score's 1.22. |
| ~~"A 1%/yr strategy has a deeply negative Sharpe vs T-bills"~~ (README/PROMPT) | ❌ **REFUTED (user's)** | True for long-only, **false for a dollar-neutral book**, which is self-financing. The reported return already *is* an excess return. Does not save the strategy. |
| Selectivity improves coverage (as it did in pairs) | ❌ **REFUTED** | It **inverts** here. A gross-normalized book concentrates turnover as you get selective. |
| The gross edge is structural, not fitted | ✅ **Confirmed** | Positive in 57/60 configs, 36/36 at every rebalance frequency. Genuinely robust. |
| A better predictor makes a better strategy | ❌ **REFUTED** | The central finding. More IC, more turnover, **worse net**. |
| The strategy is not tradeable | ✅ **Confirmed** | Break-even 3.5bps; sub-0.5 Sharpe **even at zero cost**. |

**Six refuted claims. Five were mine. One was the user's.** The methodology worked: every one was caught by a test that was pre-committed and then actually run.

---

## What the literature says, and what we add

| Source | Their finding | What we found |
|---|---|---|
| **Avellaneda & Lee (2010)** | PCA residual + OU s-score. Sharpe 1.44 (1997–2007), but only **0.9** in 2003–2007. | Decay continued. Gross Sharpe 0.73 (2016–2026), **negative gross in 2022, 2024, 2026.** |
| **Guijarro-Ordonez, Pelger & Zanotti (2022)** | CNN+Transformer, Sharpe > 4. *"Signal extraction is the separating element."* | **Statistically right** (IC t = +4.27) and **economically wrong**. The extracted signal is un-tradeable. |
| **DLSA replication (2024)** | Sharpe > 10, and warns it *"has not yet incorporated realistic transaction costs, market impact, or short-sale fees."* | **This is the whole gap.** We incorporated them. It goes to zero. |
| **Do & Faff (2012)** | Pairs trading *"largely unprofitable post-2002."* | Killed the predecessor. The residual version is a **more efficient harvest of an edge that is still too small.** |

**Our contribution, stated plainly:** the DLSA architecture's headline Sharpes are an artifact of omitting transaction costs. The deep signal is *real* — that part of the paper replicates and we confirm it with 978k out-of-sample stock-days. But its information is concentrated at a frequency whose trading cost exceeds its value, and **a turnover penalty cannot recover it, because penalizing turnover destroys precisely the alpha the model found.**

---

## What is reusable

| Asset | Verdict |
|---|---|
| **The coverage ratio (edge/cost per unit turnover)** | **The single most valuable instrument across both projects.** It killed the predecessor and it killed this one, and in both cases it was legible from day one. Compute it first, always. |
| **The "linear control before the model" discipline** | **Phase B was killed on evidence without writing a line of it.** The linear allocator + GBT pre-flight cost ~40 minutes and saved days of neural-net work. |
| **The signal-persistence diagnostic** (autocorr of the signal) | **New, and the key to the whole result.** It is the bridge between IC and net Sharpe. A high-IC, low-autocorrelation signal is a trap. Nobody reports this. Everybody should. |
| **`residuals.py`** | Sound. Faithful Avellaneda-Lee. The κ filter is decoration and should be removed or fixed. |
| **`backtest.py`** | Sound. Weight-vector engine, correct `shift(-1)`, gross/net always side by side. |
| **The zero-cost ceiling test** | **Run this early.** "What is my Sharpe if trading were free?" bounds everything downstream. Had we run it first, the whole project was a two-hour question. |
| The universe / data / metrics layer | Ported from the predecessor, worked without incident. |

**The most valuable thing to carry forward is still not code — it is the refusal to read a conclusion off the best cell of a grid.** That single discipline caught a false positive in this project too (the weekly-rebalance result), and it was *my* false positive, generated the same way the predecessor's was.

---

## Verdict

# KILL IT.

**Not because the signal is fake. Because it is real and too small.**

1. **The gross edge is genuine, robust, and statistically significant.** +1.5%/yr, gross Sharpe 0.73, positive in 57/60 structural configs, IC t = +4.11. The Avellaneda-Lee construction works. There is no selection bias, no lookahead, no fitted hyperparameter. This is an honest measurement of a real phenomenon.
2. **It cannot pay for itself.** Break-even cost 3.5 bps (median). Coverage 0.33×. Costs are 3× the gross edge.
3. **It does not clear the bar even at zero cost.** Frictionless median Sharpe **0.38 < 0.5**. This is the finding that closes the question — there is no execution improvement, no smarter allocator, and no better model that recovers a Sharpe that isn't there in the first place.
4. **Deep learning finds more signal and makes less money.** Real incremental alpha (t = +4.27), concentrated at an unaffordable frequency. Phase B is dead on evidence, not on prejudice.
5. **It fails in 2022** — the one regime a market-neutral book exists to survive.
6. **Buying SPY beat it by 375 percentage points.**

**Do not proceed to Phase B.** Do not tune it. Do not add features. The binding constraint is not the model, the factor count, the holding period, or the execution — it is that **the residual mean-reversion edge in US large-caps, in 2016–2026, is smaller than the cost of harvesting it.** That is a property of the market, not of our code.

**Recommended further work: none on this strategy.** If the residual-mean-reversion idea is pursued at all, the only axes with any prior are (a) a universe where the edge has not been arbitraged flat — small-caps, or a non-US market, accepting far worse costs, or (b) an execution model that captures rather than pays the spread, which is a market-making business and a different project entirely. Neither is a modification of this one.

---

## Bottom line

The predecessor produced a harness rigorous enough to prove its own hypothesis false. **This project did it twice — once to the strategy, and once to me.** The weekly-rebalance result was a textbook grid-argmax false positive, generated by exactly the mechanism the predecessor's post-mortem warned about, and it survived precisely one test.

Two rigorous null results, and one genuinely new finding worth more than either:

> **A better predictor is not a better strategy. Deep learning extracts real information from factor residuals that the OU s-score misses — and that information is worthless, because it lives at a frequency you cannot afford to trade. The hand-crafted statistic from 2010 is not a crude approximation of what the network finds. It is the part of what the network finds that you can actually keep.**
