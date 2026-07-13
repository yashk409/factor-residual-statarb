# research/ — every number in PROJECT_REPORT.md, reproducible

Run from the repo root:

    PYTHONPATH=. ./venv/bin/python research/<script>.py

Each script has its PRE-COMMITTED falsification criterion in its module docstring,
written before it was run. Outputs land in `../results/`.

| # | script | question | answer |
|---|---|---|---|
| — | `../run_phase_a.py` | Is there a gross edge? | **gross Sharpe 0.73 — GO** (criterion: <0.5 → stop) |
| 01 | `01_turnover_decomp.py` | Is the 91x/yr turnover hedge churn (fixable) or signal churn? | signal. Hedge is only 29%. Not fixable. |
| 02 | `02_structural_sweep.py` | Is the gross edge fitted? Does coverage ever clear 1.0? | gross +ve in 57/60. **coverage >1 in 0/60.** |
| 03 | `03_holding_frontier.py` | Does a slower rebalance fix coverage? | looks like YES at K=5 — **this was a false positive** |
| 04 | `04_holding_x_grid.py` | ...does that survive the full grid? | **NO.** 5/36 configs. net t>2 in 0/108. |
| 05 | `05_phaseb_allocator_control.py` | Linear control for Phase B's allocator net | best net Sharpe +0.27 (t=0.86). Dead. |
| 06 | `06_phaseb_signal_ceiling.py` | Linear control for Phase B's signal net (9-year IC) | **underpowered — t=1.05. Do not trust.** |
| 07 | `07_phaseb_decide.py` | ...redone with daily ICs (n=2140) + economics | IC: GBT beats s-score **t=+4.27**. Net: **GBT loses.** |
| 08 | `08_phaseb_steelman.py` | Does a horizon-trained (persistent) model beat the s-score? | **No, at any horizon.** Phase B dead. |
| 09 | `09_phase_c.py` | Reality check: borrow, impact, break-even, cash, regimes | break-even 3.5bps. **Sharpe 0.38 at ZERO cost.** |

**Read 03 and 04 together.** 03 alone produces a believable, wrong answer. That is the
whole methodology in one pair of files.
