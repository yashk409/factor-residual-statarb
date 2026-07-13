# Master Prompt — StatArb (all phases)

Paste the block below into a fresh session in `~/Desktop/Projects/StatArb`.
Trim the phases you've already completed.

---

```
You are helping me build a factor-residual statistical arbitrage strategy in
~/Desktop/Projects/StatArb. Read README.md and ../Kalman/PROJECT_REPORT.md first —
the second is the post-mortem of the predecessor project and it defines everything
about how I want you to work.

## CONTEXT: THE PREDECESSOR FAILED, AND WHY

I spent a full project on cointegration PAIRS trading. It produced a null result:

  - Sharpe collapsed 1.55 -> -0.20 the moment pair selection was made point-in-time.
    Selection bias hid for nine stages behind a "clean OOS" protocol that rigorously
    policed FITTING and was completely blind to leakage in SELECTION.
  - edge/trade = 0.17 bps vs cost/trade = 0.18 bps. Coverage 0.94x. The average trade
    earned less than it cost. That was the whole story.
  - The deep-learning branch had ZERO gross alpha. Its losses were almost exactly its
    transaction costs. Seed dispersion did NOT shrink under 4x breadth (1.00 -> 1.04),
    proving the error was coherent, not diversifiable — confident estimators of nothing.
  - The DL target had a 2% R^2 ceiling, and that signal was already sitting in one of its
    own input features (z_spread). The nets could only match a linear rule or add noise.
  - Do & Faff (2012) independently found pairs trading is "largely unprofitable post-2002."
    We reproduced a published null result without knowing it.

Two of my own confident claims during that project were REFUTED by later tests
(an entry_z "sweet spot" that inverted across markets, and a "hidden LSTM alpha"
hypothesis). Assume I will generate more false positives. Your job is to catch them.

## THE RULES — these are non-negotiable, each one is a scar

1. PRE-COMMIT the falsification criterion BEFORE each run. Write it into the script.
   When the number comes back, neither of us gets to move it.
2. GROSS AND NET, always, side by side. Never report one without the other.
3. POINT-IN-TIME EVERYTHING — INCLUDING SELECTION. Not just fitting.
4. NEVER read a conclusion off a 2-seed ablation. If seed sigma is 1-2%, then a
   +-3% swing across scenarios is NOISE, not a finding.
5. Robustness to a PARAMETER is not robustness to the PROTOCOL. A flat Sharpe across
   thresholds proves nothing about lookahead.
6. Any model must BEAT A LINEAR BASELINE on the same task, or it dies. Always run the
   control.

Tell me when I'm fooling myself. If a result looks good, your first instinct should be
to attack it, not to celebrate it. A rigorous null result is a success.

## PHASE A — Avellaneda-Lee baseline (go/no-go for the whole project)

Already implemented in run_phase_a.py. Run it:

    python3 run_phase_a.py

PRE-COMMITTED CRITERION: gross Sharpe < 0.5 => STOP THE PROJECT.
Gross, not net. If there is no edge BEFORE costs, no neural net can manufacture one and
Phase B is pure curve-fitting.

Then:
  - Interpret per-year gross/net, turnover, cost drag, active positions, net exposure.
  - Sanity-check: is the edge concentrated in one year? one sector? the COVID vol spike?
    If it lives in a single regime, say so.
  - Sweep the structural knobs (n_factors 5/10/15/20, reg_win 30/60/120,
    kappa_min, s-score thresholds) but treat these as ROBUSTNESS CHECKS, not tuning.
    If the result only works at one setting, it is fitted and therefore fake.
  - Report edge/trade vs cost/trade (the coverage ratio). This number killed the last
    project; I want it on every report.

If NO-GO: write it up honestly and stop. Do not "just try one more thing."

## PHASE B — Deep learning, done right (only if Phase A passes)

The predecessor's DL failed because it minimized Huber loss on a near-noise target and
then bolted a threshold on top. It NEVER optimized the trading objective. Fix that.

Follow Guijarro-Ordonez, Pelger & Zanotti (2022), "Deep Learning Statistical Arbitrage"
(arxiv 2106.04028):

  1. Signal extraction: CNN + Transformer over the residual time series (replaces the
     OU s-score). Port models/cnn_transformer.py from ../Kalman as the starting point,
     but SHRINK IT — the old one had ~15k params for a 2% R^2 signal, which is why every
     seed learned a different confident fiction. Target ~500-2000 params.
  2. Allocation: a network mapping signals -> portfolio weights (replaces the threshold
     rule).
  3. THE KEY CHANGE: train end-to-end on NEGATIVE SHARPE OF THE RESULTING PORTFOLIO,
     with an explicit TURNOVER PENALTY in the loss. The trading objective is different
     from a prediction objective. This is the whole point.

Mandatory controls, or the phase is worthless:
  - Linear baseline (the OU s-score from Phase A) on the same windows.
  - Seed-variance luck test: >=8 seeds, per-model, NO ensembling. If mean/std < ~0.5 and
    ~half the seeds are profitable, it is a coin flip. Report the seed dispersion.
  - Purge/embargo the label overlap at every split boundary.

PRE-COMMITTED CRITERIA:
  - DL must beat the OU s-score baseline NET of costs. If not, DL is dead permanently.
  - If seed dispersion does not shrink with breadth, the error is coherent, not noise,
    and the models are worthless (this is exactly how the last DL branch died).

## PHASE C — Reality check (only if B passes)

The backtest is more generous than reality. Before believing anything:

  - SHORT BORROW: 0.3-1%/yr on the short leg, currently unmodelled. Add it.
  - SLIPPAGE / MARKET IMPACT: t_cost currently covers commission + spread only.
  - COST SENSITIVITY: find the break-even t_cost. If break-even is under ~10bps/leg,
    the strategy is too thin to survive reality.
  - SHARPE VS CASH: our Sharpe is benchmarked against ZERO. At a 4-5% risk-free rate,
    a 2%/yr strategy has a NEGATIVE Sharpe versus T-bills. Report it both ways.
  - SURVIVORSHIP: universe.py lists names liquid today. Quantify the likely bias or
    state clearly that it cannot be quantified without point-in-time (CRSP) data.
  - SUBPERIOD STABILITY: does it work in 2022 (bear) as well as 2023-25 (bull)?
  - CAPACITY: at what AUM does market impact eat the edge?

PRE-COMMITTED CRITERION: if net-of-everything Sharpe < 0.5 or the strategy cannot beat
cash, it is not tradeable. Say so plainly.

## PHASE D — Decide

Write a final report (markdown) covering: what was built, every result, every claim that
was refuted (including yours and mine), what's reusable, and a clear verdict:
TRADE IT / KILL IT / needs specific further work.

If the honest answer is "buying SPY beat this," say that. The last project's most
valuable output was a harness rigorous enough to prove its own hypothesis false.

## KNOWN TRAPS — do not fall into these again

  - Tuning a parameter on 4 windows and calling it "evidence-backed."
  - Reading a finding off a 2-seed ablation.
  - Assuming a clean OOS protocol covers selection just because it covers fitting.
  - Believing an extraordinary Sharpe. The literature's replications report Sharpe > 10
    pre-cost and the authors themselves admit no realistic costs are modelled. Trust the
    architecture, not the numbers.
  - Spending overnight compute on the wrong axis (more epochs) when the binding
    constraint is costs, breadth, or bias.
```
