"""Compare the own versions with the paper's final strategy, inside the TRAIN period only. (Phase 6)

    python -m scripts.05_own_strategy                         (own_turbulence and own_ml_vol)
    python -m scripts.05_own_strategy --candidate own_turbulence

Every number for the ML version is out of sample: in each walk-forward block it is sized by a
model fitted only on the days before that block, so it has no result for the first two train
years. The rule has no fitted parameters and is shown on the whole train period as well.
Shows the walk-forward blocks together and one by one, each year, and a cost stress test, with a
paired block bootstrap of every Sharpe difference. The test period is never touched here.
Writes results/own_vs_final_train.md and results/figures/own_train_*.png.
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from src import plots
from src.config import RESULTS_DIR, load_research, load_strategies
from src.engine import sizing
from src.engine.backtest import MarketData, decision_minutes, prepare, simulate
from src.evaluation import metrics, stats
from src.evaluation.split import walk_forward_folds
from src.experiment_log import log_run
from src.strategies import ml_sizing

STRESS_SLIPPAGE = 0.005          # $/share, the level independent replications assume (vs our 0.001)


def block_metrics(ret: pd.Series) -> dict:
    return {"annual_return": metrics.annual_return(ret), "annual_volatility": metrics.annual_volatility(ret),
            "sharpe": metrics.sharpe(ret), "max_drawdown": metrics.max_drawdown(ret)}


def gain(a: pd.Series, b: pd.Series) -> str:
    d = stats.sharpe_difference(a, b)
    return f"{d['difference']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", nargs="+", default=["own_turbulence", "own_ml_vol"],
                        help="strategies in config/strategies.yaml")
    args = parser.parse_args()

    research = load_research()
    strategies = load_strategies()
    data = MarketData.load()
    start, end = research.train_start, research.train_end
    days = data.daily.loc[start:end].index
    folds = walk_forward_folds(days)
    blocks_days = pd.DatetimeIndex(np.concatenate([val for _, val in folds]))
    names = ["final", *args.candidate]

    # ---- size multipliers: the rule's from the config, the ML's from walk-forward fits
    multipliers = {}
    for name in args.candidate:
        cfg = strategies[name]
        if cfg.intraday_sizing == "ml_vol":
            with np.errstate(divide="ignore", invalid="ignore"):
                forecast = ml_sizing.walk_forward_forecast(data, decision_minutes(cfg), folds)
                multipliers[name] = sizing.clip_multiplier(1 / forecast, cfg)
    oos_only = {name for name in multipliers}            # no result before the first block

    def backtest(name: str, research_cfg=research):
        prep = prepare(strategies[name], data, start, end, multiplier=multipliers.get(name))
        return simulate(prep, prep.positions(), research_cfg)

    results = {name: backtest(name) for name in names}
    ret = {name: r.daily["ret"] for name, r in results.items()}
    stressed = {name: backtest(name, replace(research, slippage=STRESS_SLIPPAGE)).daily["ret"] for name in names}

    for name in args.candidate:
        span = blocks_days if name in oos_only else days
        log_run(strategies[name], "train", str(span[0].date()), end, metrics.summary(results[name].daily.loc[span]),
                note="walk-forward blocks only (ML fitted before each block)" if name in oos_only else "")

    # ---- 1. headline: the walk-forward blocks together (every version out of sample there)
    rows = []
    for name in names:
        rows.append({"version": name, "period": "blocks 2018–2022", **block_metrics(ret[name].loc[blocks_days]),
                     "sharpe_gain_vs_final [95% CI]": "" if name == "final" else
                     gain(ret[name].loc[blocks_days], ret["final"].loc[blocks_days])})
    for name in names:
        if name in oos_only:
            continue
        rows.append({"version": name, "period": "whole train 2016–2022", **block_metrics(ret[name]),
                     "sharpe_gain_vs_final [95% CI]": "" if name == "final" else gain(ret[name], ret["final"])})
    headline = pd.DataFrame(rows)
    pairs = [(a, b) for i, a in enumerate(args.candidate) for b in args.candidate[i + 1:]]
    pair_lines = [f"- {b} vs {a} (blocks 2018–2022): Sharpe difference "
                  f"{gain(ret[b].loc[blocks_days], ret[a].loc[blocks_days])}" for a, b in pairs]

    # ---- 2. each walk-forward block and each year
    per_block = pd.DataFrame({
        f"block {i}: {val[0]:%Y-%m}–{val[-1]:%Y-%m}": {f"sharpe {n}": metrics.sharpe(ret[n].loc[val]) for n in names}
        for i, (_, val) in enumerate(folds, 1)}).T
    years = sorted(set(days.year))
    per_year = pd.DataFrame({y: {f"sharpe {n}": metrics.sharpe(ret[n][ret[n].index.year == y]) for n in names}
                             for y in years}).T
    for n in args.candidate:
        per_block[f"gain {n}"] = per_block[f"sharpe {n}"] - per_block["sharpe final"]
        per_year[f"gain {n}"] = per_year[f"sharpe {n}"] - per_year["sharpe final"]
        if n in oos_only:
            per_year.loc[per_year.index < blocks_days[0].year, [f"sharpe {n}", f"gain {n}"]] = np.nan

    # ---- 3. cost stress and sizing profile
    stress = pd.DataFrame([{"version": n, "sharpe at $0.001": metrics.sharpe(ret[n].loc[blocks_days]),
                            f"sharpe at ${STRESS_SLIPPAGE}": metrics.sharpe(stressed[n].loc[blocks_days]),
                            "gain vs final at stress [95% CI]": "" if n == "final" else
                            gain(stressed[n].loc[blocks_days], stressed["final"].loc[blocks_days])} for n in names])
    profile = []
    for n in names:
        t = results[n].trades
        t = t[t["date"].isin(blocks_days)]
        d = results[n].daily.loc[blocks_days]
        peak = d["peak_shares"] * data.daily.loc[blocks_days, "open"] / d["aum_start"]
        profile.append({"version": n, "trades": len(t), "mean size": t["size"].mean(),
                        "at floor 0.5": (t["size"] <= 0.5 + 1e-9).mean(), "above 1": (t["size"] > 1 + 1e-9).mean(),
                        "max leverage used": peak.max()})
    profile = pd.DataFrame(profile)

    # ---- figures and report
    plots.equity_curves({n: ret[n].loc[blocks_days] for n in names},
                        "Own versions vs final, walk-forward blocks 2018–2022 (train)", "own_train_equity")
    plots.drawdowns({n: ret[n].loc[blocks_days] for n in names},
                    "Drawdowns, walk-forward blocks 2018–2022 (train)", "own_train_drawdowns")

    pct = lambda df: df.assign(**{c: df[c].map("{:.1%}".format) for c in
                                  ("annual_return", "annual_volatility", "max_drawdown") if c in df})
    md = lambda df, **kw: df.to_markdown(floatfmt=".2f", **kw)
    report = "\n\n".join([
        "# Own versions vs final, train period only\n\n_Generated by `scripts/05_own_strategy.py`. The test "
        "period is not used. Sharpe differences: paired block bootstrap, 95% interval._",
        "## Headline\n\nThe walk-forward blocks (2018–2022) are the only span where every version is out of "
        "sample: the ML is fitted on the days before each block only.\n\n"
        + md(pct(headline), index=False) + "\n\n" + "\n".join(pair_lines),
        "## Each walk-forward block\n\n" + md(per_block),
        "## Each year\n\n" + md(per_year),
        f"## Cost stress: slippage ${STRESS_SLIPPAGE}/share instead of $0.001 (blocks 2018–2022)\n\n"
        + md(stress, index=False),
        "## How the versions size their trades (blocks 2018–2022)\n\n"
        + md(profile.assign(**{c: profile[c].map("{:.0%}".format) for c in ("at floor 0.5", "above 1")}), index=False),
    ]) + "\n"
    (RESULTS_DIR / "own_vs_final_train.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
