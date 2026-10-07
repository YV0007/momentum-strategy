# Project skeleton

Reminder of where everything lives. `P#` = phase in the project plan.

```
SPY_Momentum_str/
│
├── README.md                    # P9  hypothesis, method, results, how to run
├── SKELETON.md                  #     this file
├── requirements.txt             #     libraries
├── .env.example                 #     template for Alpaca keys (real .env is gitignored)
├── .gitignore                   #     .env, data/, results/, caches
│
├── config/
│   ├── research.yaml            # P0  fixed rules: split dates, costs, metrics, success criteria
│   ├── strategies.yaml          # P3  named configs: base, vwap_stop, final, own_turbulence, own_ml_vol
│   ├── ml_sizing.yaml           # P6  ML volatility forecast settings (written by 05a, frozen)
│   ├── universe.yaml            # P9  the 16 ETFs of the multi-asset test and their asset class
│   ├── ablation.yaml            # P3  ladder: base → final, one decision at a time
│   └── paper_monthly_returns.csv  #   paper's published monthly returns (replication reference)
│
├── data/                        #     gitignored, rebuilt by scripts
│   ├── raw/                     # P1  <SYMBOL>_1min_<year>.parquet, calendar, Yahoo daily + dividends, VIX,
│   │                            #     options/ (Cboe VIX family, SqueezeMetrics GEX; Stage 7 screen)
│   └── processed/<SYMBOL>/      # P1–P2 minute, daily, features_minute, features_daily (.parquet)
│
├── src/
│   ├── config.py                # P0  loads YAML into StrategyConfig / ResearchConfig dataclasses
│   ├── data/                    # P1
│   │   ├── alpaca.py            #     download 1-min SPY bars + trading calendar
│   │   ├── yahoo.py             #     official daily close + dividends (reference source)
│   │   ├── vix.py               #     download CBOE VIX daily
│   │   ├── clean.py             #     regular hours, half-days, missing minutes, dividends
│   │   └── quality.py           #     data-quality report (gaps, outliers, cross-check)
│   ├── features.py              # P2  sigma, noise bands, VWAP, daily vol, RSI, VIX, NR4
│   ├── intraday.py              # P6  today vs normal at each decision; ML feature panel + target
│   ├── engine/                  # P3
│   │   ├── backtest.py          #     day loop, decision points, positions, AUM update
│   │   ├── rules.py             #     entry/stop logic per stop type
│   │   ├── sizing.py            #     1x vs volatility-target sizing; own versions' per-trade multiplier
│   │   └── costs.py             #     commission + slippage per share; tiered commission, I-Star impact
│   ├── evaluation/              # P4
│   │   ├── metrics.py           #     Sharpe, ann. return/vol, MDD, hit ratio, alpha/beta
│   │   ├── baselines.py         #     buy & hold, open-to-close, random-entry placebo
│   │   ├── stats.py             #     Sharpe SE, block bootstrap, Deflated Sharpe
│   │   ├── split.py             # P5  train/test periods, time-series CV folds
│   │   ├── diagnostics.py       # P5  regime, time-of-day, side, trade-sequence breakdowns
│   │   ├── trade_features.py    # P6  market situation at each trade's entry (+ ML dataset later)
│   │   ├── replication.py       #     our returns vs the paper's monthly table
│   │   ├── paper_tables.py      # P8  paper Section 4 + FAQ analyses (patterns, weekday, legs, Table 4)
│   │   ├── portfolio.py         # P9  equal-capital portfolio of single-ETF sleeves, diversification
│   │   ├── ablation.py          # P3  runs the ladder → results/train_ablation.md
│   │   └── report.py            # P4  full evaluation of one period → tables, figures, report
│   ├── strategies/
│   │   └── ml_sizing.py         # P6  gradient-boosting rest-of-day volatility forecast (own_ml_vol)
│   ├── experiment_log.py        # P0  appends each run's config + metrics to a log
│   └── plots.py                 # P4–P8 equity, drawdown, metric bars, heatmaps
│
├── scripts/                     #     one command per step, run in order
│   ├── 01_download_data.py      # P1  every symbol of the universe (or --symbols)
│   ├── 02_build_dataset.py      # P1–P2 per symbol; quality reports: docs/ (SPY), results/data_quality/
│   ├── 03_run_backtests.py      # P3–P4 run + evaluate + ablation, train only
│   ├── 04_checkpoint.py         # P5  diagnostics on train only → results/checkpoint_report.md
│   ├── 04b_diagnose.py          # P6  Stage 1: P&L split, entry features, worst trades (train)
│   ├── 05a_tune_ml_sizing.py    # P6  choose the ML settings on train (forecast accuracy only)
│   ├── 05_own_strategy.py       # P6  own versions vs final: walk-forward blocks, years, cost stress
│   ├── 06_run_test.py           # P7  test + post-publication evaluation, replication report
│   ├── 07_robustness.py         # P8  every paper variation and analysis, train + test
│   └── 08_multi_asset.py        # P9  three versions on 16 ETFs, equal-capital portfolio (--period)
│
├── tests/                       # run with `pytest`
│   ├── conftest.py              #     synthetic market + real-data fixture
│   ├── test_features.py         #     each feature vs brute-force paper formula
│   ├── test_lookahead.py        #     perturb future data → past outputs unchanged
│   ├── test_synthetic_days.py   #     hand-made price paths with known trades
│   ├── test_accounting.py       #     flat at close, leverage ≤ 4x, P&L reconciles
│   ├── test_paper_days.py       #     2022-01-20, 01-31, 04-29 + yearly returns match the paper
│   ├── test_evaluation.py       #     metrics, Sharpe stats, placebo, CV folds
│   ├── test_diagnostics.py      #     diagnostics reconcile with the backtest
│   ├── test_trade_features.py   #     entry features have no look-ahead
│   ├── test_intraday_sizing.py  #     per-trade sizing, whole-share accounting, ML sees no test data
│   ├── test_paper_variants.py   #     tiered commission, I-Star impact, daily patterns
│   └── test_multi_asset.py      #     portfolio arithmetic, pooled ML sees no future blocks
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # scratch work
│   └── report.ipynb               # P9  final report: loads results, shows plots
│
├── results/                     #     gitignored, regenerated by scripts
│   ├── experiment_log.csv
│   ├── backtests/               #     daily returns + trade logs per config
│   └── figures/
│
└── docs/
    ├── data_quality.md          # P1
    ├── own_strategy_gex.md      # P6  record of the retired GEX attempt (code removed)
    ├── own_version_research.md  # P6  research log of the new own version (stages, findings)
    ├── paper_coverage.md        # P8  every element of the paper: implemented where, result vs paper
    └── talking_points.md        # P9
```

## Data flow

```
alpaca.py / vix.py → clean.py → features.py → backtest.py ← StrategyConfig
                                                  │
                                       daily returns + trade log
                                                  │
                     metrics.py / baselines.py / stats.py → plots.py → report.ipynb
```

## Principles
- Logic lives in `src/`; `scripts/` only call it; notebooks only display results.
- One engine, many configs: base / middle / final / own differ only in `strategies.yaml`.
- `research.yaml` is fixed in Phase 0; later changes show up in git history.
- Every run is logged in `experiment_log.csv` (feeds the Deflated Sharpe).
- `data/` and `results/` are always rebuildable; only code and configs are committed.
