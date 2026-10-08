# Paper coverage: every element of the paper and where it is in this project

Checked against the full paper (Zarattini, Aziz & Barbon, version of 3 Feb 2025: Sections 3–4,
Tables 1–6 and A1, FAQ Q1–Q25) on 2026-10-05.

**Data.** Paper: SPY and VIX 1-minute data from IQFeed, May 2007 – April 2024. Ours: SPY 1-minute
data from Alpaca (SIP), which starts in 2016; train 2016–2022, test 2023-01-01 – 2026-10-02. The
paper's 2007–2015 years (incl. 2008) cannot be reproduced with this data.

**Status.** ✅ implemented and run · ➖ not applicable or out of scope (reason given).

## A. The strategy (Section 3)

| Paper element | Paper ref. | Status | Code / config | Checked by |
|---|---|---|---|---|
| Noise area: average absolute move from the open at each minute over the previous 14 days | eq. 1–2 | ✅ | `features.noise_sigma` | `test_features.py` (vs brute force), `test_lookahead.py` |
| Bands around the open | eq. 3 | ✅ | `features.noise_bands(gap_adjust=False)`; ablation `s1` | `test_synthetic_days.py` |
| Gap adjustment: bands around max/min(open, previous close) | §3 | ✅ | `features.noise_bands` (previous close dividend-adjusted) | `test_features.py` |
| Decisions only at HH:00 / HH:30 from 10:00, stops only then | §3 | ✅ | `backtest.decision_minutes` | `test_accounting.py`, `test_synthetic_days.py` |
| Observe the price at the decision, fill at the next minute's open | §3, §4.6 | ✅ | `backtest.prepare` | `test_synthetic_days.py` |
| Everything closed at 16:00 | §3 | ✅ | last leg ends at the official close | `test_accounting.py` |
| Base: stop at the opposite band, then reverse | Table 1 | ✅ | `rules.opposite_band`; config `base` | `test_paper_days.py` (2022-01-20) |
| Current band as trailing stop | Fig. 5a | ✅ | `rules.band_only`; ablation `s3` | `test_synthetic_days.py` |
| Current band + VWAP trailing stop | Table 2 | ✅ | `rules.band_vwap`; config `vwap_stop` | `test_paper_days.py` |
| VWAP-only trailing stop, 1x and with vol sizing | FAQ Q22 | ✅ | `rules.vwap_only`; ablation `s3b`, `s6b` | `test_synthetic_days.py` |
| VWAP from regular-hours data only | footnote 2 | ✅ | `features.vwap` | `test_features.py` |
| 100% of AUM, whole shares at the open | eq. 5 | ✅ | `sizing.shares` | `test_accounting.py` |
| Volatility target min(4, 2% / 14-day sample std) | eq. 6 | ✅ | `sizing.leverage`, `features.daily_volatility`; config `final` | `test_features.py`, `test_synthetic_days.py` |
| $100,000 start, $0.0035 commission + $0.001 slippage per share | §3, §4.6 | ✅ | `config/research.yaml`, `engine/costs.py` | `test_synthetic_days.py`, `test_accounting.py` |
| IB tiered commission ($0.002 above 300k shares a month) | §4.6 | ✅ option | `costs.commission_rate` (`commission_tiered`) | `test_paper_variants.py` |
| I-Star market impact, large-cap parameters | FAQ Q15 | ✅ option | `costs.slippage` (`slippage_model: istar`) | `test_paper_variants.py` |
| Volatility multiplier VM | §4.4 | ✅ | `StrategyConfig.vm` | |
| Noise-area lookback | FAQ Q6 | ✅ | `StrategyConfig.lookback` | |

## B. Results the paper reports, and where ours are

| Paper item | Paper ref. | Status | Our output |
|---|---|---|---|
| Total return, IRR, vol, Sharpe, hit ratio, max drawdown, alpha, beta, skew, worst/best day | Tables 1–3, A1 | ✅ | `results/train_report.md`, `test_report.md`, `post_publication_report.md` |
| Equity curves vs buy & hold | Figs. 3, 6, 7 | ✅ | `figures/train_equity.png`, `test_equity.png` |
| Trade-level statistics | Table 4 | ✅ | `robustness_report.md` §9 |
| Monthly return table | FAQ Q4, Q24 | ✅ | `replication_report.md` (vs the paper's months) |
| Sharpe, annual return, annual volatility charts (case-study brief) | — | ✅ | `figures/*_metric_bars.png`, `robustness_paper_versions.png` |

## C. Further investigations (Section 4) and FAQ

| Paper item | Paper ref. | Status | Our output |
|---|---|---|---|
| Sharpe on days with VIX above thresholds | §4.1, Fig. 8 | ✅ | `robustness_report.md` §6, `checkpoint_report.md` 1.6 |
| 8 daily patterns (NR4, NR7, ID, OD, Triangle, Trend, Big tail, Strong/weak closure) | §4.2, Table 5 | ✅ | `robustness_report.md` §7 |
| Day of the week | §4.3, Table 6 | ✅ | `robustness_report.md` §8 |
| VM sweep | §4.4, Fig. 9 | ✅ | `robustness_report.md` §2, `figures/robustness_vm.png` |
| RSI(5) regression (gamma-imbalance proxy) | §4.5 | ✅ | `checkpoint_report.md` 1.2, 1.7; real GEX data: `docs/own_strategy_attempts.md` (attempt 1) |
| Commission sensitivity | §4.6, Fig. 10 | ✅ | `robustness_report.md` §4, `figures/robustness_commission.png` |
| Long and short legs | FAQ Q5 | ✅ | `robustness_report.md` §11, `figures/robustness_legs.png` |
| Lookback sweep | FAQ Q6 | ✅ | `robustness_report.md` §3, `figures/robustness_lookback.png` |
| SPY's 10 worst quarters | FAQ Q7 | ✅ | `robustness_report.md` §14 |
| Overfitting (few parameters; VM and lookback not at their best) | FAQ Q14 | ✅ | sweeps §2–3 |
| I-Star slippage | FAQ Q15 | ✅ | `robustness_report.md` §5 |
| Intraday seasonality | FAQ Q18 | ✅ | `checkpoint_report.md` 2.1–2.2 |
| Short trades vs VIX (regression; only above a VIX level) | FAQ Q19, Q20 | ✅ | `robustness_report.md` §12 |
| Shorts only below the 100/150/200-day SMA | FAQ Q21 | ✅ | `robustness_report.md` §13 |
| VWAP-only stop | FAQ Q22 | ✅ | `robustness_report.md` §1 (`s6b`) |
| Profit per share by year | FAQ Q23 | ✅ | `robustness_report.md` §10 |
| Hedge for a long portfolio (discussion) | FAQ Q8 | ✅ | beta and worst quarters above |
| Other ETFs and stocks; 33 futures markets | FAQ Q12, Q13 | ➖ | out of scope: the case study is SPY only and our data is SPY only |
| Code, data source, TradingView bands, software, automation, management | FAQ Q1–Q3, Q9–Q11, Q16, Q17, Q25 | ➖ | not results |

## D. What our data says compared with the paper

Paper numbers: 2007–2024. Ours: train 2016–2022 / test 2023 – Oct 2026, net of costs.

| Item | Paper | Ours: train | Ours: test | Agrees? |
|---|---|---|---|---|
| Base, opposite-band stop, 1x: Sharpe | 0.61 | 0.57 | 0.79 | yes |
| Band + VWAP stop, 1x: Sharpe | 1.24 | 0.80 | 0.92 | direction yes, level lower |
| Final (band + VWAP, vol sizing): annual return / Sharpe | 19.6% / 1.33 | 15.8% / 1.07 | 16.3% / 1.12 | yes |
| Monthly returns vs the paper's table, Jan 2016 – Jan 2025 | | correlation 0.988, slope 0.999, no bias | | yes |
| VWAP-only stop with vol sizing (Q22): Sharpe vs final | 1.17 vs 1.35 | 1.13 vs 1.07 | 1.03 vs 1.12 | test yes, train no |
| Best VM | ~1.5 (Sharpe 1.55) | 1.5 (1.34 vs 1.07 at VM 1) | 1.0 (VM 1.5: 0.83) | the VM = 1 default held up better out of sample |
| Best lookback | 90 days (1.50) | 20 days (1.10); range 0.85–1.10 | 5 / 14 / 90 days (1.11–1.13) | flat in both: not a sensitive parameter |
| Commission $0 → $0.01: Sharpe | falls steadily | 1.17 → 0.88 | 1.17 → 1.02 | yes |
| I-Star impact: Sharpe | 1.33 → 1.17 | 1.07 → 1.04 | 1.12 → 1.08 | yes; smaller hit (smaller account) |
| IB tiered commission | used on ~20% of days | never reached | never reached | our account trades < 300k shares a month |
| Higher VIX, higher Sharpe | rises to ~3.5 at VIX > 40 | rises to ~2 at VIX ≥ 15–25, negative at ≥ 30 (100 days, mostly 2020) | rises with VIX (few days above 30) | partly |
| NR4 the best pattern | 22 bps, t 5.1 | 20.9 bps, t 2.9 | 17.0 bps, t 2.0 | yes |
| Trend day after a trend day | −2 bps, n.s. | −11.5 bps, n.s. | −5.4 bps, n.s. | yes |
| Wednesday the best weekday | 18 bps, t 3.4 | 22.6 bps, t 2.4 | 14.9 bps, t 1.3 | yes on train |
| Trades per day (paper) = our orders per day | 1.3 base / 1.8 band + VWAP | 1.22 / 1.75 | 1.25 / 1.79 | yes |
| Trade hit ratio, base / band + VWAP | 53% / 37% | 53% / 38% | 56% / 42% | yes |
| Average profit per share, base / band + VWAP | $0.10 / $0.09 | $0.12 / $0.10 | $0.20 / $0.10 | yes |
| Both legs profitable | yes | long 8.9% a year (Sharpe 1.05), short 6.4% (0.58) | long 9.1% (1.03), short 6.5% (0.58) | yes |
| Positive in SPY's worst quarters | 10 of 10 | 8 of 10 (2016 – Oct 2026, e.g. 2020Q1 +4.2%, 2022Q2 +12.4%, 2018Q4 +29.2%) | | mostly |
| Short-trade return unrelated to VIX (Q19) | p = 0.91 | p = 0.23 | positive, p < 0.01 | train yes, test no |
| Shorts only at high VIX lose money overall (Q20) | yes | yes | yes | yes |
| Shorts only below the SMA lose money overall (Q21) | yes | total return 179% → 77–90% | 76% → 41–47% | yes |
| Profit per share grows with the SPY price (Q23) | yes | 2018–2023 high, 2016–2017 negative | 2026 negative | partly |
| RSI(5) predicts lower returns (gamma proxy) | slope −3.25, p = 0.001 | −1.4 bps per 10 points, p = 0.34 | | no |
