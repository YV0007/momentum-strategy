# Own strategy: every version tried

| # | Attempt | Train blocks 2018–22 | Whole train 2016–22 | Test 2023–26 | End |
|:-:|:--|:-:|:-:|:-:|:--|
| | Paper's final strategy (baseline) | 1.69 | 1.07 | 1.12 | |
| 1 | GEX sizing | — | 1.20 | 1.03 | Retired |
| 2 | ML entry filters | 1.72 | — | — | Retired |
| 3 | ML sizing by trade outcome | 1.81 | — | — | Retired |
| 4 | **Own A: turbulence sizing** | **1.97** | **1.30** | **1.35** | **Kept, main version** |
| 5 | Own B: ML volatility sizing | 1.93 | — | 1.22 | Challenger, lost to Own A |
| 6 | VWAP-only stop (on Own A) | 2.02 | 1.34 | 1.31 | Retired |
| 7 | Day sizing by options data and release calendar | — | — | — | Retired |
| 8 | Options inputs in the ML model | 1.93 | — | — | Retired |
| 9 | Multi-asset, 12 ETFs (Own A portfolio) | 0.66 | 0.27 | — | Stopped after train |

## Ideas

1. **GEX sizing:** halve leverage on days after high dealer gamma, when breakouts should fade.
2. **ML entry filters:** skip the 30% of breakouts a model rates most likely to fail.
3. **ML sizing by trade outcome:** size each trade 0.5–1.5× by a model's prediction of whether it wins.
4. **Own A:** size each trade by 1 / (volatility so far today ÷ its normal at that time of day).
5. **Own B:** the same, with a gradient-boosting forecast of the rest of the day's volatility.
6. **VWAP-only stop:** exit only on a VWAP cross, so trend days can run further.
7. **Day sizing by options data:** size the whole day by a forecast of big-move days from VIX9D,
   VIX3M, VVIX, SKEW, GEX, FOMC and payroll days.
8. **Options inputs in the ML model:** add FOMC-ahead and VIX9D/VIX to Own B's volatility model.
9. **Multi-asset:** the same rules with equal capital on 12 liquid ETFs, for more independent bets.
