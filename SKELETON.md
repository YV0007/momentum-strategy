# Project skeleton

Where everything lives.

```
SPY_Momentum_str/
│
├── README.md                    # hypothesis, method, results, how to run
├── SKELETON.md                  # this file
├── requirements.txt             # libraries; the .env format for the Alpaca keys is in its header
├── .gitignore                   # .env, data/, results/, caches
│
├── config/
│   ├── research.yaml            # fixed rules: split dates, costs, metrics, success criteria
│   ├── strategies.yaml          # named configs: base, vwap_stop, final, own_turbulence, own_ml_vol
│   ├── ml_sizing.yaml           # ML volatility forecast settings (written by 05a, frozen)
│   ├── universe.yaml            # the 12 ETFs of the multi-asset test and their asset class
│   ├── ablation.yaml            # ladder: base → final, one decision at a time
│   └── paper_monthly_returns.csv  # paper's published monthly returns (replication reference)
│
├── data/                        # gitignored, rebuilt by scripts
│   ├── raw/                     # <SYMBOL>_1min_<year>.parquet, calendar, Yahoo daily + dividends, VIX,
│   │                            # options/ (Cboe VIX family, SqueezeMetrics GEX)
│   └── processed/<SYMBOL>/      # minute, daily, features_minute, features_daily (.parquet)
│
├── src/
│   ├── config.py                # loads YAML into StrategyConfig / ResearchConfig dataclasses
│   ├── data/
│   │   ├── alpaca.py            # download 1-min bars + trading calendar
│   │   ├── yahoo.py             # official daily close + dividends (reference source)
│   │   ├── vix.py               # download CBOE VIX daily
│   │   ├── clean.py             # regular hours, half-days, missing minutes, dividends
│   │   └── quality.py           # data-quality report (gaps, outliers, cross-check)
│   ├── features.py              # sigma, noise bands, VWAP, daily vol, RSI, VIX, NR4
│   ├── intraday.py              # today vs normal at each decision; ML feature panel + target
│   ├── engine/
│   │   ├── backtest.py          # day loop, decision points, positions, AUM update
│   │   ├── rules.py             # entry/stop logic per stop type
│   │   ├── sizing.py            # 1x vs volatility-target sizing; own versions' per-trade multiplier
│   │   └── costs.py             # commission + slippage per share; tiered commission, I-Star impact
│   ├── evaluation/
│   │   ├── metrics.py           # Sharpe, ann. return/vol, MDD, hit ratio, alpha/beta
│   │   ├── baselines.py         # buy & hold, open-to-close, random-entry placebo
│   │   ├── stats.py             # Sharpe SE, block bootstrap, Deflated Sharpe
│   │   ├── split.py             # train/test periods, time-series CV folds
│   │   ├── diagnostics.py       # regime, time-of-day, side, trade-sequence breakdowns
│   │   ├── trade_features.py    # market situation at each trade's entry
│   │   ├── replication.py       # our returns vs the paper's monthly table
│   │   ├── paper_tables.py      # paper Section 4 + FAQ analyses (patterns, weekday, legs, Table 4)
│   │   ├── portfolio.py         # equal-capital portfolio of single-ETF sleeves, diversification
│   │   ├── ablation.py          # runs the ladder → results/train_ablation.md
│   │   └── report.py            # full evaluation of one period → tables, figures, report
│   ├── strategies/
│   │   └── ml_sizing.py         # gradient-boosting rest-of-day volatility forecast (own_ml_vol)
│   ├── experiment_log.py        # appends each run's config + metrics to a log
│   └── plots.py                 # equity, drawdown, metric bars, heatmaps
│
├── scripts/                     # one command per step, run in order
│   ├── 01_download_data.py      # every symbol of the universe (or --symbols)
│   ├── 02_build_dataset.py      # per symbol; data-quality report → results/data_quality/
│   ├── 03_run_backtests.py      # run + evaluate + ablation, train only
│   ├── 04_checkpoint.py         # diagnostics on train only → results/checkpoint_report.md
│   ├── 04b_diagnose.py          # P&L split, entry features, worst trades (train)
│   ├── 05a_tune_ml_sizing.py    # choose the ML settings on train (forecast accuracy only)
│   ├── 05_own_strategy.py       # own versions vs final: walk-forward blocks, years, cost stress
│   ├── 06_run_test.py           # test + post-publication evaluation, replication report
│   ├── 07_robustness.py         # every paper variation and analysis, train + test
│   └── 08_multi_asset.py        # three versions on 12 ETFs, equal-capital portfolio (train; stopped)
│
├── tests/                       # run with `pytest`
│   ├── conftest.py              # synthetic market + real-data fixture
│   ├── test_features.py         # each feature vs brute-force paper formula
│   ├── test_lookahead.py        # perturb future data → past outputs unchanged
│   ├── test_synthetic_days.py   # hand-made price paths with known trades
│   ├── test_accounting.py       # flat at close, leverage ≤ 4x, P&L reconciles
│   ├── test_paper_days.py       # 2022-01-20, 01-31, 04-29 + yearly returns match the paper
│   ├── test_evaluation.py       # metrics, Sharpe stats, placebo, CV folds
│   ├── test_diagnostics.py      # diagnostics reconcile with the backtest
│   ├── test_trade_features.py   # entry features have no look-ahead
│   ├── test_intraday_sizing.py  # per-trade sizing, whole-share accounting, ML sees no test data
│   ├── test_paper_variants.py   # tiered commission, I-Star impact, daily patterns
│   └── test_multi_asset.py      # portfolio arithmetic, pooled ML sees no future blocks
│
├── results/                     # gitignored, regenerated by scripts
│   ├── experiment_log.csv
│   ├── backtests/               # daily returns + trade logs per config
│   └── figures/
│
└── docs/
    ├── own_strategy_attempts.md # every own version tried, its numbers and why it was kept or retired
    └── results/                 # snapshot of the key reports and figures from results/
```
