# Own strategy: every version tried

Baseline: the paper's final strategy, our replication. Sharpe 1.69 in the train walk-forward blocks
(2018–2022), 1.07 on the whole train period (2016–2022), 1.12 on the test period (2023 – Oct 2026).
All design work on train; Sharpe gains are paired block bootstrap differences with a 95% interval.
Every run is logged in `results/experiment_log.csv`.

| # | Attempt | Best on train | Test | End |
|:-:|:--|:--|:--|:--|
| 1 | GEX sizing | +0.13 [−0.01, +0.28] | −0.09 [−0.34, +0.19] | Retired |
| 2 | ML entry filters | +0.03 at best | not run | Retired |
| 3 | ML sizing by trade outcome | +0.12 at best | not run | Retired |
| 4 | **Own A: turbulence sizing** | **+0.28 [+0.06, +0.49]** | **+0.23 [−0.01, +0.44]** | **Kept, main version** |
| 5 | Own B: ML volatility sizing | +0.24 [+0.04, +0.44] | +0.10 [−0.16, +0.32] | Challenger, lost to Own A |
| 6 | VWAP-only stop | +0.09 [−0.21, +0.36] | 1.12 → 1.03 | Retired |
| 7 | Day sizing by options data and release calendar | −0.02 to −0.22 | not run | Retired |
| 8 | Options inputs in the ML model | +0.00 [−0.04, +0.04] | not run | Retired |
| 9 | Multi-asset, 12 ETFs | portfolio Sharpe 0.56 vs 1.69 for SPY alone | not run | Stopped after train |

## 1. GEX sizing (`own_gex`), first own attempt
- **Idea:** halve leverage on days after high dealer gamma (SqueezeMetrics GEX in the top 20% of its
  trailing year), when dealers hedge against moves and breakouts should fade.
- **Train:** high-GEX days expanded less often (33% vs 50%, t = −4.8); Sharpe 1.07 → 1.20,
  +0.13 [−0.01, +0.28], 3 of 4 blocks.
- **Test:** 1.12 → 1.03, −0.09 [−0.34, +0.19]; post-publication 0.44 → 0.20.
- **Why it failed:** the market effect held, but the strategy earned *more* on high-GEX test days, so
  cutting them removed profit; the rule rested on a weak train link (p = 0.08).

## 2. ML entry filters (skip the 30% of trades a model rates worst)
- **Idea:** predict at entry which breakouts fail and skip them.
- **Train blocks:** win/loss model −0.13 to −0.33; trade-return model −0.75 [−1.22, −0.32]; simple
  rule "skip turbulent" +0.03 [−0.62, +0.60].
- **Why it failed:** the best 10% of trades make 324% of the profit and nothing at entry tells them
  apart (AUC 0.46–0.54), so every filter also skips some of the trades that pay for everything.

## 3. ML sizing by predicted trade outcome
- **Idea:** size trades 0.5–1.5× by a model's view of whether they will win.
- **Train blocks:** by win/loss −0.05 to −0.17; by trade return −0.11 to −0.15; by held-to-close
  +0.05 [−0.14, +0.26]; monotone turbulence model +0.12 [−0.06, +0.30].
- **Why it failed:** trade outcomes cannot be learned from ~1,500 trades; "held to close" is mostly
  predicted by the clock (AUC 0.71 from entry time alone).

## 4. Own A: turbulence sizing (`own_turbulence`), kept
- **Idea:** outcome is unpredictable, risk is not; size each trade by 1 / (realized volatility so far
  today ÷ its 14-day normal at that time of day), 0.5–1.5×, no fitted parameters.
- **Train:** blocks 1.69 → 1.97, +0.28 [+0.06, +0.49]; whole train 1.07 → 1.30; permutation test
  p = 0.001.
- **Test:** 1.12 → 1.35, +0.23 [−0.01, +0.44]; volatility 14.4% → 12.5%; max drawdown 18.8% → 11.5%;
  post-publication 0.44 → 0.57.
- **Why not a clear success:** same gain as on train, but the interval just touches zero and most of
  the test gain came in 2026, a partial year.

## 5. Own B: ML volatility sizing (`own_ml_vol`), challenger
- **Idea:** same as Own A, with a gradient-boosting forecast of rest-of-day volatility instead of
  volatility so far (4 features, depth-1 trees, chosen by forecast accuracy only).
- **Train:** forecast R² 0.54 vs 0.46 for Own A's implicit forecast; blocks 1.69 → 1.93,
  +0.24 [+0.04, +0.44].
- **Test:** 1.12 → 1.22, +0.10 [−0.16, +0.32]; against Own A −0.13 [−0.24, −0.02].
- **Why it failed:** a more accurate volatility forecast did not give better sizing; the simple rule
  generalized better.

## 6. VWAP-only stop
- **Idea:** trend days run further, so exit only on a VWAP cross instead of band or VWAP.
- **Train blocks:** final 1.69 → 1.78, +0.09 [−0.21, +0.36]; Own A 1.97 → 2.02.
- **Test:** final 1.12 → 1.03; Own A 1.35 → 1.31.
- **Why it failed:** winners ran further (held to close 33% → 48%), but losers were held longer too,
  so return and risk rose together.

## 7. Day sizing by options data and the release calendar
- **Idea:** VIX9D, VIX3M, VVIX, SKEW, GEX, FOMC and payroll days known at the open predict big-move
  days; size the whole day up or down by that forecast.
- **Train:** the forecast worked (AUC 0.64 for a big-move day; FOMC days 54% vs 41%), but sizing
  changed Sharpe by −0.02 to −0.22 in all 6 variants.
- **Why it failed:** it predicts the strategy's risk, not its return; predictable volatility brings
  no extra edge (FOMC days: −1.0 vs +2.4 bps a day, whipsaw at 14:00).

## 8. Options inputs inside the ML model
- **Idea:** add FOMC-ahead and VIX9D/VIX to Own B's volatility model.
- **Train blocks:** forecast R² 0.543 → 0.594, Sharpe 1.93 → 1.93, +0.00 [−0.04, +0.04].
- **Why it failed:** a better volatility forecast again did not change the trading result.

## 9. Multi-asset: the three versions on 12 ETFs
- **Idea:** the same rules, equal capital on 12 liquid ETFs (equity indices, sector, Treasuries,
  credit, metals, real estate) for more independent bets.
- **Train blocks:** portfolio Sharpe 0.56 (final), 0.66 (Own A) vs 1.69 for SPY alone; average
  correlation 0.16 (about 4.4 independent bets); only SPY and QQQ have a clear edge; at $0.005
  slippage the portfolio loses money.
- **Why it failed:** the edge is specific to the S&P 500 and Nasdaq-100; other ETFs have little gross
  edge and costs 4–8× larger relative to their moves.

## What all of this says
Across every attempt, the direction of SPY at a decision point was unpredictable (breakout
continuation AUC 0.44–0.48); only the size of moves was. The edge is the payoff shape (many small
losses, a few large trend days), so the one change that held up is sizing by risk (Own A), which
lowers risk rather than raising profit.
