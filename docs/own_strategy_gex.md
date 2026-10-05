# Own strategy attempt 1: dealer-gamma (GEX) sizing — RETIRED

**Status:** frozen before testing, tested once on the test period, failed out of sample.
Retired on 2026-10-05: all its code, data and outputs were removed from the project. This page
and its runs in `results/experiment_log.csv` (strategy `own_gex`) are the record.

## Idea
The paper explains intraday momentum partly by dealer hedging: when dealers are short gamma they
trade with the move (trends continue); when long gamma they trade against it (breakouts fade). The
paper had no gamma data and used RSI(5) as a stand-in, which did not replicate on our 2016–2022
data (t = −0.96). We tested the mechanism with real data: SqueezeMetrics' free daily GEX estimate
for S&P 500 index options (2011 onward), used as yesterday's value in percentile of its trailing year.

## Evidence on the train period (2016–2022), before building the rule
- High GEX predicted fewer expansion days (days closing outside the noise area): 33% in the top
  quintile vs 50% in the bottom; −20 percentage points from lowest to highest GEX, t = −4.8 after
  controlling for volatility and VIX. Consistent in every year.
- The strategy lost money on top-20% GEX days (−1.8 bps/day vs +6 to +12 bps elsewhere), lower
  returns in 6 of 7 years — but this return link was weak (t = −1.75, p = 0.08).
- Negative GEX (dealers short gamma) brought more expansion but no profit (crisis whipsaws).

## Rule (frozen 2026-10-05)
Final strategy, with leverage × 0.5 when yesterday's GEX percentile > 0.8. Entries and stops unchanged.

## Results (net of costs)

| | Final (paper) | Own (GEX cut) |
|---|---|---|
| Train 2016–2022: annual return / Sharpe | 15.8% / 1.07 | 16.7% / 1.20 |
| Train: Sharpe gain, paired bootstrap 95% CI | | +0.13 [−0.01, +0.28] |
| Train: walk-forward blocks won | | 3 of 4 |
| **Test 2023 – Oct 2026: annual return / Sharpe** | **16.3% / 1.12** | **12.8% / 1.03** |
| Test: Sharpe gain, paired bootstrap 95% CI | | −0.09 [−0.34, +0.19] |
| Post-publication (May 2024 →): Sharpe | 0.44 | 0.20 |

## Why it failed
- The market-side mechanism held out of sample: high-GEX days still expanded less often (38% vs 45%).
- The return side reversed: the final strategy earned more on high-GEX test days (+7.7 vs +5.7
  bps/day, notably 2024), so halving them removed profit. The rule rested on the weaker half of the
  train evidence (the return link, p = 0.08), with a threshold chosen after seeing the data.
- Possible contributors: same-day (0DTE) options dominate intraday gamma since 2023 and are invisible
  to end-of-day GEX; with GEX trending up, "top 20% of the trailing year" covered 33% of test days.

## Lesson
Predicting whether breakouts continue is not the same as profiting from it: high-gamma expansion
days can still whipsaw, and the strategy's P&L is concentrated in a few days that a size cut can miss.
