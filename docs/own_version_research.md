# Own version: research log

Baseline: the paper's final strategy (replicated). All design work on train 2016–2022; the test
period 2023 – Oct 2026 will be used once more for the frozen candidate and reported as a SECOND
USE (first use: paper versions + the retired GEX attempt, docs/own_strategy_gex.md).

## Stage 1: diagnosis (train only) — 2026-10-05
Generated tables and figures: `results/diagnose_report.md`, `results/figures/diagnose_*.png`
(`scripts/04b_diagnose.py`).

### 1. Where good and bad years come from
Each year = opportunity (share of days closing outside the noise area) × capture (mean return on
those days) + cost (mean return on the other days).
- Opportunity is stable: 37–45% of days every year. It explains little of the year-to-year spread
  (effects within ±3 bps/day).
- Capture and cost drive the results: 2017 lost through weak capture (29 bps on trend days vs 47
  average; −7.9 bps/day effect), 2016 through expensive failed breakouts (−32 bps vs −23; −5.8),
  2018 won through strong capture (+10.0).
- The strategy keeps about 21% of a perfect-hindsight open-to-close trade on trend days.

**Conclusion: work on trade execution (which breakouts to take, and how large), not on picking days.**

### 2. What separates good trades from bad ones at entry (1,546 trades)
- No feature predicts *whether* a trade wins: every AUC is between 0.47 and 0.53, none significant
  after correcting for 10 tests.
- Several features predict the *payoff* (rank correlation with the trade's 1x return, significant
  after correction), and they share one theme — **trades entered when today is already more
  turbulent than normal do worse**:

| Feature at entry | Rank correlation | Same sign in |
|---|---|---|
| Realized volatility so far today vs normal | −0.146 | 7/7 years |
| First-30-minute range vs normal | −0.120 | 7/7 |
| VIX at the open | −0.102 | 6/7 |
| Volume in the last 30 minutes vs normal | −0.093 | 7/7 |
| Distance from VWAP (in band widths) | −0.083 | 6/7 |
| VIX vs recent realized volatility | +0.074 | 4/7 |

- Contradicted ideas: volume confirmation (high volume is *worse*), an active open (a wide opening
  range is *worse*). Breakout size, path efficiency and gap direction carry no reliable signal.
- Entry time: midday entries (12:00–14:00) average −0.2 to −6.0 bps at 1x; 10:30 (+10.4) and
  15:00 (+16.1) are the best. Consistent with the paper's FAQ Q18 (trends pause over lunch).

### 3. The 50 worst trades
- 60% were entered on days already in their top 20% of turbulence (vs 20% of all trades): 3×
  over-represented. 54% were stopped within 30 minutes (vs 35%).
- Midday entries 1.34× and re-entries 1.22× over-represented. Not over-represented: FOMC-time
  Wednesdays (0.44×), breakouts against the gap (0.84×), days at the 4x leverage cap (0.99×).
- Reading the charts by hand: most are **entries right after a sharp news-driven spike** — buying
  near the top or selling near the bottom, then a V-shaped reversal (2017-12-01 crash on the Flynn
  report, breakout 6.7 band widths, turbulence 4.3× normal; 2021-02-26; 2016-08-26 Jackson Hole) —
  or **entries into a scheduled release** (2019-10-01: long at exactly 10:00, the ISM release;
  2016-05-18: long held into the 14:00 FOMC minutes).

### Mechanism behind the main finding
The noise area is calibrated on the previous 14 days. On a day that is already far more
turbulent, ordinary noise crosses that band easily, so breakouts are more often false and the
entry comes late, near the extreme of a spike.

## Stage 2: hypotheses (to be screened in walk-forward blocks before building)
- **H1, turbulence-adjusted band (main):** widen today's band by how turbulent today already is
  (e.g. sigma × max(1, realized volatility so far / normal)). Acts on entries and stops; at most
  one parameter.
- **H2, overextension:** reduce size when price is far beyond VWAP at entry (likely overlaps H1).
- **H3, midday entries:** reduce size for entries 12:00–14:00.
- **H4, implied vs realized volatility:** size up when VIX is high relative to recent realized
  volatility (weaker: same sign in only 4/7 years).
- Not pursued without new data: scheduled-release times (10:00 macro data, 14:00 FOMC).

ML note: individual features are weak (|rank correlation| ≤ 0.15), and they mostly measure one
common factor (turbulence vs normal). A simple rule on that factor may capture most of the value;
an ML trade filter would be compared against it on the same walk-forward blocks.

## Stage 3: ML screen and the sizing hypothesis (train only) — 2026-10-05

### Screen: ML entry filter vs ML sizing
Quick screen on the final strategy's 1,546 train trades. Each model was fitted on the years before
a walk-forward block and scored on that block (4 blocks, 2018–2022; final strategy Sharpe 1.69
there). Shallow gradient boosting (depth 2–3) unless stated. Exact accounting: shares are fixed
per day and costs are per share, so a trade scaled by m contributes m × its P&L.

| Version | Model predicts | Sharpe gain vs final [95% CI] | Blocks won |
|---|---|---|---|
| Entry filter (skip 30%) | win/loss | −0.13 to −0.33 | 1–2 of 4 |
| Entry filter (skip 30%) | trade return | −0.75 [−1.22, −0.32] | 0 of 4 |
| Entry filter, rule: skip turbulent | — | +0.03 [−0.62, +0.60] | 1 of 4 |
| Sizing 0.5–1.5x | win/loss | −0.05 to −0.17 | 0–2 of 4 |
| Sizing | trade return | −0.11 to −0.15 | 1 of 4 |
| Sizing | held to close vs stopped | +0.05 [−0.14, +0.26] | 2 of 4 |
| Sizing, monotone, turbulence features | trade return (rank) | +0.12 [−0.06, +0.30] | 3 of 4 |
| Sizing | rest-of-day volatility | +0.23 [+0.04, +0.44] | 3 of 4 |
| Sizing, rule: 1 / today's vol so far vs normal | — | +0.28 [+0.06, +0.49] | 3 of 4 |

18 variants in total (scratch scripts, not in the repo); they count as tries.

- **Filters fail.** The best 10% of trades make 324% of the profit, and nothing at entry tells
  them apart, so skipping trades removes some of the few that pay for everything.
- **Win/loss and trade return cannot be learned** from ~1,500 trades (AUC 0.46–0.54).
- **"Held to close" is mostly the clock.** Entry time alone predicts it with AUC 0.71 (the model:
  0.72). 15:30 entries are 100% held to close and earn +0.07 bps on average.
- **Volatility can be forecast.** Rest-of-day volatility has out-of-sample R² of about 0.5, and
  that model trains on every day × decision time (~21,000 rows), not just the trades.

### Evidence: the problem is risk, not outcome
Trades split into five equal groups by today's turbulence at entry (realized volatility since
the open vs its 14-day normal at the same time of day):

| | Q1 calmest | Q2 | Q3 | Q4 | Q5 most turbulent |
|---|---|---|---|---|---|
| Volatility so far vs normal | 0.64× | 0.83× | 1.00× | 1.22× | 1.67× |
| Leverage used | 2.5× | 2.7× | 2.8× | 2.8× | 2.8× |
| Rest-of-day volatility vs normal | 0.66× | 0.82× | 0.95× | 1.13× | 1.55× |
| Spread of trade results (std, 1x) | 29 bps | 41 | 42 | 57 | 66 bps |
| Average trade (1x) | +1.5 bps | +6.7 | +4.8 | +2.3 | −2.3 bps |
| Share of total risk (sum of squared results) | 5% | 12% | 16% | 26% | 41% |
| Share of total profit | 8% | 36% | 22% | 27% | 7% |

1. Leverage is flat because it is set at the open from the previous 14 days (rank correlation
   with today's turbulence: 0.08). Risk is not flat: a turbulent-morning trade has 2.3× the spread.
2. Volatility persists within the day (rank correlation of vol so far with rest-of-day vol 0.71,
   R² 0.51); direction does not (AUC 0.53 for the rest of the day going our way).
3. Sizing by 1 / (vol so far vs normal) evens out the risk actually taken vs the 2% plan: from
   0.66×–1.55× across the five groups to 0.84×–0.93×.
4. Turbulence and trade result are negatively rank-correlated in 7 of 7 years (−0.02 to −0.30).
5. Permutation test: the same multipliers shuffled across trades 2,000 times give a Sharpe change
   of −0.01 on average (99th percentile +0.17); the rule gives +0.22 (p = 0.001).

Correction to the intuition "too small on expansion days": expansion days are MORE turbulent at
entry (65% of Q5 trades vs 50% of Q1), so the rule cuts them too (−0.28 of profit there, +0.29 saved
on other days). Net profit is about unchanged; the gain is lower risk. Examples: 2017-12-01 (vol
4.3× normal, −2.5% → −1.2% of the account) and 2016-08-26 (−2.6% → −1.3%) are helped; the best
train trade, 2018-10-10 (vol 2.0×, +9.0% → +4.6%), is hurt.

Limits: the average trade result does not differ significantly between Q1 and Q5 (t = 0.93), so
the case rests on risk-adjusted payoff; and the hypothesis was formed by looking at train data.

### Hypothesis
**The final strategy takes its biggest risks on turbulent days, where it earns the least per unit
of risk. A trade's rest-of-day volatility is predictable at entry and its outcome is not, so
sizing each trade inversely to predicted volatility, on top of the daily volatility target,
raises the Sharpe ratio.** Theory: volatility-managed portfolios (Moreira & Muir 2017), the same
logic as the paper's own volatility target (Sharpe 0.80 → 1.07), applied again within the day.

### The two versions
Both keep the final strategy's entries, stops and daily leverage `min(4, 2% / 14-day vol)`. Each
trade is scaled at entry by a multiplier in [0.5, 1.5], held to its exit, total leverage ≤ 4x,
whole shares (`config/strategies.yaml`, `src/engine/sizing.py`).
- **A. `own_turbulence` (main candidate):** multiplier = 1 / (realized vol so far today / its
  14-day normal at the same time of day). No fitted parameters; bounds fixed in advance.
- **B. `own_ml_vol` (challenger):** multiplier = 1 / gradient-boosting forecast of (rest-of-day
  realized vol / its normal) (`src/strategies/ml_sizing.py`). Settings chosen on train only by
  forecast accuracy, never by P&L (`scripts/05a_tune_ml_sizing.py`). For the test period the model
  is fitted once on the whole train period (2016–2022) and frozen.
- GEX is not an input: it failed out of sample (own_strategy_gex.md), an end-of-day value cannot
  see same-day (0DTE) gamma, and its data was removed.

### Fixed before the test run
- Both versions are tested once on 2023 – Oct 2026 (the SECOND use of the test period) and on the
  post-publication slice, and both are reported whatever the result.
- Reading of the paired Sharpe difference vs final (block bootstrap): **improvement** if the 95%
  interval is above zero; **points the right way, not significant** if the gain is positive but
  the interval includes zero; **no improvement** otherwise.
- The challenger counts as better than the main candidate only if its test Sharpe is higher.
- The research.yaml success criteria still apply (Sharpe ≥ 0.5, beat buy & hold and the placebo).

## Stage 4: the two versions built and judged on train; freeze — 2026-10-05
Engine: each trade is scaled at entry and held to exit (`hold_from_entry`), whole shares per
trade, peak leverage ≤ 4x. The paper versions reproduce their earlier results to the last digit.
Tests: `tests/test_intraday_sizing.py` (sizing, accounting, no look-ahead in the ML features, and
the frozen ML model gives identical forecasts with or without the test period present).

### ML settings (train only, `results/ml_sizing_tuning.md`)
Objective: validation error of the log rest-of-day volatility forecast, never trading P&L.
1. Grid of 72 settings on all 11 candidate features, one-SE rule.
2. Backward elimination (drop a feature while removing it costs < 0.25% of the error). Kept 4:
   volatility so far, volatility in the last 30 minutes, volume since the open (each vs normal),
   and VIX vs realized volatility. Dropped: VIX level, opening range, band width vs normal, gap,
   time of day, first-30-minute volatility, last-30-minute volume.
3. A finer grid of 288 settings reaching past the first grid's edges, one-SE rule. Chosen: depth-1
   trees (an additive model), 250 iterations, learning rate 0.05, leaves ≥ 200, L2 = 10,
   monotone constraints. The best setting in the grid was within one paired standard error.

| Forecast R², validation blocks | Block 1 | 2 | 3 | 4 | Mean | At trade entries |
|---|---|---|---|---|---|---|
| Rule's implicit forecast (vol so far) | 0.53 | 0.48 | 0.52 | 0.31 | 0.46 | 0.50 |
| Linear (HAR-style) | 0.58 | 0.55 | 0.59 | 0.44 | 0.54 | 0.57 |
| Gradient boosting, final settings | 0.57 | 0.56 | 0.58 | 0.46 | 0.54 | 0.56 |

### Train results (`results/own_vs_final_train.md`)
The walk-forward blocks are the only span where all three are out of sample.

| Version | Span | Annual return | Volatility | Sharpe | Max drawdown | Sharpe gain vs final [95% CI] |
|---|---|---|---|---|---|---|
| final | blocks 2018–2022 | 27.6% | 15.1% | 1.69 | 9.9% | |
| own_turbulence | blocks 2018–2022 | 26.6% | 12.4% | 1.97 | 7.2% | +0.28 [+0.06, +0.49] |
| own_ml_vol | blocks 2018–2022 | 27.7% | 13.1% | 1.93 | 8.0% | +0.24 [+0.04, +0.44] |
| final | whole train 2016–2022 | 15.8% | 14.7% | 1.07 | 24.7% | |
| own_turbulence | whole train 2016–2022 | 16.3% | 12.2% | 1.30 | 20.8% | +0.23 [+0.05, +0.40] |

- ML vs rule on the blocks: −0.03 [−0.10, +0.04], a tie.
- Both lose in block 2 (Apr 2019 – Jul 2020, COVID) and win the other three.
- Cost stress at $0.005/share slippage: gains +0.26 (rule) and +0.23 (ML), intervals above zero.

### Freeze
Frozen on 2026-10-05 with the git commit that adds this section: `config/strategies.yaml`
(own_turbulence, own_ml_vol), `config/ml_sizing.yaml`, `src/engine/sizing.py`,
`src/strategies/ml_sizing.py`, `src/intraday.py`. Roles and test criteria as fixed in Stage 3.
The next step is a single run of `scripts/06_run_test.py`.

## Stage 5: test result (SECOND use of 2023 – Oct 2026) — 2026-10-05
One run of `scripts/06_run_test.py` after the freeze commit (aa5acc2). Net of costs. Full tables:
`results/own_vs_final_test.md`, `results/test_report.md`, `results/post_publication_report.md`.

| Test 2023-01-03 – 2026-10-02 | Annual return | Volatility | Sharpe | Max drawdown | Sharpe gain vs final [95% CI] |
|---|---|---|---|---|---|
| final (paper, our implementation) | 16.3% | 14.4% | 1.12 | 18.8% | |
| own_turbulence (rule, main) | 17.6% | 12.5% | 1.35 | 11.5% | +0.23 [−0.01, +0.44] |
| own_ml_vol (ML, challenger) | 15.8% | 12.6% | 1.22 | 15.0% | +0.10 [−0.16, +0.32] |
| SPY buy & hold | 22.1% | 14.9% | 1.42 | 18.8% | |

| Post-publication 2024-05-10 – 2026-10-02 | Annual return | Volatility | Sharpe | Max drawdown | Sharpe gain vs final [95% CI] |
|---|---|---|---|---|---|
| final | 5.5% | 14.8% | 0.44 | 18.8% | |
| own_turbulence | 6.3% | 12.0% | 0.57 | 11.6% | +0.14 [−0.26, +0.51] |
| own_ml_vol | 4.8% | 12.2% | 0.44 | 15.0% | +0.00 [−0.39, +0.38] |

### Verdicts, as fixed in Stage 3
- **Rule (main): points the right way, not significant.** +0.23 Sharpe, interval [−0.01, +0.44],
  just touching zero. Same size of gain as on train (+0.23 whole train, +0.28 blocks), with lower
  volatility (14.4% → 12.5%) and a much smaller drawdown (18.8% → 11.5%) at a slightly higher return.
- **ML (challenger): points the right way, not significant**, and worse than the rule:
  −0.13 [−0.24, −0.02] vs own_turbulence. The rule stays the main version.
- research.yaml criteria: Sharpe ≥ 0.5 met; placebo beaten (p = 0.006 rule, 0.009 ML); buy & hold
  NOT beaten on Sharpe (1.35 vs 1.42), as for the paper's final (1.12). All three have beta ≈ 0
  (−0.01 to −0.03) and alpha ≈ 16–17% a year, so they diversify a stock portfolio rather than replace it.

### Reading
- Most of the test gain comes from 2026 (+0.58 Sharpe, a partial year); 2023 +0.05, 2024 −0.04,
  2025 +0.12. The sign was right in 3 of 4 years, but the gain is not spread evenly.
- The ML forecast was better than the rule's implicit forecast on train (R² 0.54 vs 0.46) and still
  traded worse on test. Better volatility forecasts did not translate into better sizing; the
  simpler rule generalized better.
- After publication the strategy remains weak (final 0.44); sizing reduces risk but does not
  restore the faded edge.

## Stage 6: VWAP-only stop — tried and retired, 2026-10-06
Hypothesis: once an expansion starts it keeps going, so the band + VWAP stop exits too early.
Tested by exiting only when a half-hour check finds the price on the wrong side of VWAP (paper FAQ
Q22), for final, own_turbulence and own_ml_vol, everything else unchanged. Its script and report
were removed; this note and its runs in `results/experiment_log.csv` (strategies `*_vwap_stop`,
note "VWAP-only stop variant") are the record.

| Sharpe: band + VWAP → VWAP only | Train blocks 2018–2022 | Whole train | Test (third look) |
|---|---|---|---|
| final | 1.69 → 1.78 (+0.09 [−0.21, +0.36]) | 1.07 → 1.13 | 1.12 → 1.03 |
| own_turbulence | 1.97 → 2.02 (+0.05 [−0.31, +0.39]) | 1.30 → 1.34 | 1.35 → 1.31 |
| own_ml_vol | 1.93 → 2.00 (+0.07 [−0.29, +0.40]) | — | 1.22 → 1.20 |

Expansions do run further (trades held to the close 33% → 48%; about +7 bps a day on expansion
days), but failed breakouts are held longer too (3–6 bps a day worse on the other ~59% of days).
Return and volatility both rise; Sharpe does not change significantly. Not adopted.

## Stage 7: options data and the release calendar as a day-level sizing signal — feasibility, 2026-10-06
Hypothesis: the strategy profits when today's move beats what the noise bands expect (the
"volatility surprise", |open-to-close| / band sigma at the close). Bands look back 14 days, options
look forward, so options data and scheduled releases known at the open should predict the surprise,
and sizing the day by that forecast should help. Train 2016–2022 only; walk-forward blocks 2018–2022.

Data (free, in `data/raw/options/`, git-ignored): Cboe VIX9D and VIX3M (9:30 opens), VVIX and SKEW
(previous close), SqueezeMetrics GEX (previous close, percentile of its trailing year), FOMC decision
days (federalreserve.gov), payroll days (BLS scheduling rule; the BLS site blocks automated access).
CPI days are missing: no free source reachable automatically.

**Step 1, forecast: passed, modestly.** On their own, VIX / 14-day realized vol (rank correlation with
the surprise +0.19, same sign 7/7 years), the overnight gap in vol units (+0.18, 7/7), GEX (−0.14,
7/7), VIX9D / VIX (+0.13, 6/7) and VIX9D / 5-day realized vol (+0.11, 7/7) all predict it. FOMC days
expand more often (54% vs 41%), payroll days too (52%). Out of sample: AUC for an expansion day 0.62
with VIX / realized vol alone, 0.64 with all inputs (gradient boosting or ridge); R² of the surprise
2.5% → 4–5%.

**Step 2, sizing: failed.** Day multiplier 0.5–1.5x by the forecast's rank, on final and on
own_turbulence, three forecast models: Sharpe change −0.02 to −0.22 in all 6 variants (ridge
significantly negative), annual return up, volatility up more.

**Why.** By fifths of the out-of-sample forecast, the expansion-day rate rises from 31% to 60% and the
strategy's daily spread from 40 to 63 bps, but its mean per unit of risk stays flat (about 0.1). The
forecast predicts the strategy's risk (rank correlation with its absolute daily result +0.20), not its
return (+0.02). Predictable volatility (priced by options, scheduled events) comes without extra edge;
the profitable expansions are the unexpected ones. FOMC days illustrate it: more expansion, but −1.0
bps a day vs +2.4 on other days (whipsaw around 14:00).

Not adopted as a return signal. Open idea, formed AFTER seeing these results: use the same inputs as a
risk forecast (smaller size on predicted big-move days). It would need a fresh holdout (other ETFs or
paper trading). Screen: 6 sizing variants + 1 diagnostic (scratch scripts, not in the repo).

**Follow-up, same day: options inputs inside the own_ml_vol volatility model (size = 1 / forecast).**
Forward selection from the frozen 4 features (same settings, same 0.25% tolerance) added FOMC-ahead
(14:00 statement still to come; −9.1% forecast error) and VIX9D / VIX (−1.6%). Out-of-sample R² of
rest-of-day volatility 0.543 → 0.594 (at 10:00: 0.482 → 0.551). Trading, walk-forward blocks
2018–2022: Sharpe 1.93 → 1.93 (+0.00 [−0.04, +0.04]); the turbulence rule 1.97. A better volatility
forecast did not change the result. Not adopted. 1 more variant (7 in this stage).

## Stage 8: opportunity map — where is there still something to gain? (train only, 2026-10-06)
Method: ceiling analysis (Sharpe at 1x if one decision were perfect) × predictability (out-of-sample
IC / AUC of what follows each decision, from every input known at that moment; shallow GBT, 4
walk-forward blocks), plus plain conditional means. The final strategy at 1x has Sharpe 0.81.

| Lever | Ceiling (perfect) | Predictability found | Verdict |
|---|---|---|---|
| Direction at entry | 8.5 | breakout continuation: AUC 0.44–0.48, IC ≤ 0 | none |
| Day sizing (good vs bad days) | 4.3 | expansion AUC 0.64, but return per unit risk flat (Stage 7) | only as volatility |
| Exit: stop or hold | 3.8 | open trade, rest of day: IC 0.057 (3/4 blocks), AUC 0.54; trades already losing keep losing (t = −2.3, 5/7 years) | the one weak signal |
| Re-entries | 2.5 | +1.0 to +1.5 bps vs +3.3 for first trades; negative in 4/7 years | weak |
| New source: reversion inside the band | — | IC 0.01, AUC 0.51 | none |
| New source: hold winners overnight | — | −0.8 bps, t = −0.2 | none |
| Size of the next move (volatility) | — | IC 0.40, 4/4 blocks | already used |

Reading: from price, volume, volatility and options state, the direction of SPY at every decision
point is unpredictable; only the size of moves is. The edge is the payoff shape (small losses, large
trend-day wins), not a forecast. With per-bet IC near zero, Grinold's fundamental law (IR ≈ IC × √breadth)
points to breadth (more independent markets) and to new information not contained in price.
