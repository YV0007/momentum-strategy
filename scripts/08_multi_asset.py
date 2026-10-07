"""The strategy on the 16-ETF universe of config/universe.yaml: the paper's final strategy, the
turbulence sizing rule and the ML volatility sizing, each run on every ETF and combined into one
equal-capital portfolio. (Phase 9)

    python -m scripts.08_multi_asset                  # train 2016-2022 (design)
    python -m scripts.08_multi_asset --period test    # 2023 - last SPY day; run ONCE, after the freeze

Same rules, costs and settings as on SPY; nothing is tuned per asset. The ML model is ONE
gradient-boosting model pooled over all ETFs, with the frozen SPY settings minus the VIX input (VIX
describes the S&P 500, not the traded asset). On train it is fitted walk-forward, so the three
versions are compared on the 2018-2022 blocks; for the test it is fitted once on 2016-2022. Train
also reports the same model fitted per asset, and a cost stress at $0.005 slippage per share.
Writes results/multi_asset_<period>.md and results/figures/multi_asset_<period>_equity.png.
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from src import plots
from src.config import RESULTS_DIR, load_research, load_strategies, load_universe
from src.engine import sizing
from src.engine.backtest import MarketData, decision_minutes, prepare, simulate
from src.evaluation import metrics, portfolio, stats
from src.evaluation.split import walk_forward_folds
from src.experiment_log import log_run
from src.intraday import decision_panel
from src.strategies import ml_sizing

VERSIONS = ["final", "own_turbulence", "own_ml_vol"]
STRESS_SLIPPAGE = 0.005


def gain(a: pd.Series, b: pd.Series) -> str:
    d = stats.sharpe_difference(a, b)
    return f"{d['difference']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"


def row(ret: pd.Series) -> dict:
    return {"annual_return": f"{metrics.annual_return(ret):.1%}", "annual_volatility": f"{metrics.annual_volatility(ret):.1%}",
            "sharpe": metrics.sharpe(ret), "max_drawdown": f"{metrics.max_drawdown(ret):.1%}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=["train", "test"], default="train")
    period = parser.parse_args().period

    research = load_research()
    strategies = load_strategies()
    universe = load_universe()
    data = {sym: MarketData.load(sym) for sym in universe}
    last_day = data["SPY"].daily.index[-1]                 # every ETF stops where the SPY data stops
    start, end = ((research.train_start, research.train_end) if period == "train"
                  else (research.test_start, f"{last_day:%Y-%m-%d}"))
    ks = decision_minutes(strategies["final"])
    cfg_ml, settings = strategies["own_ml_vol"], ml_sizing.asset_settings()

    # ---- ML size multipliers from one model pooled over all ETFs
    panels = {sym: decision_panel(d, ks) for sym, d in data.items()}
    train_days = data["SPY"].daily.loc[research.train_start:research.train_end].index
    folds = walk_forward_folds(train_days)
    blocks = pd.DatetimeIndex(np.concatenate([val for _, val in folds]))
    to_mult = lambda f: sizing.clip_multiplier(1 / f, cfg_ml)
    with np.errstate(divide="ignore", invalid="ignore"):
        if period == "train":
            ml_mult = {s: to_mult(f) for s, f in ml_sizing.walk_forward_pooled(panels, folds, settings).items()}
            ml_mult_own = {s: to_mult(ml_sizing.walk_forward_pooled({s: panels[s]}, folds, settings)[s]) for s in universe}
        else:
            model = ml_sizing.fit_pooled(panels, train_days, settings)
            ml_mult = {s: to_mult(ml_sizing.forecast(model, panels[s], settings)) for s in universe}

    def sleeves(version: str, multipliers: dict | None = None, research_cfg=research) -> dict[str, pd.DataFrame]:
        out = {}
        for sym in universe:
            prep = prepare(strategies[version], data[sym], start, end,
                           multiplier=None if multipliers is None else multipliers[sym])
            out[sym] = simulate(prep, prep.positions(), research_cfg, with_trades=False).daily
        return out

    runs = {v: sleeves(v, ml_mult if v == "own_ml_vol" else None) for v in VERSIONS}
    ret = {v: {s: d["ret"] for s, d in runs[v].items()} for v in VERSIONS}
    port = {v: portfolio.combine(ret[v]) for v in VERSIONS}
    spy_alone = ret["final"]["SPY"]
    span = blocks if period == "train" else port["final"].index      # where all three are out of sample
    span_name = "walk-forward blocks 2018–2022" if period == "train" else f"test {start} – {end}"

    for v in VERSIONS:                                              # one logged trial per portfolio version
        daily = pd.DataFrame({"ret": port[v], "trades": sum(d["trades"] for d in runs[v].values())}).loc[span]
        label = "train_blocks" if period == "train" else "test"
        log_run(replace(strategies[v], name=f"{v}_multi16"), label, f"{span[0]:%Y-%m-%d}", f"{span[-1]:%Y-%m-%d}",
                metrics.summary(daily), note="multi-asset portfolio, 16 ETFs")

    # ---- 1. portfolio
    rows = [{"version": "SPY alone (paper's final)", **row(spy_alone.loc[span]), "gain [95% CI]": ""}]
    for v in VERSIONS:
        rows.append({"version": f"portfolio: {v}", **row(port[v].loc[span]),
                     "gain [95% CI]": gain(port[v].loc[span], spy_alone.loc[span]) + " vs SPY alone"
                     + ("" if v == "final" else f"; {gain(port[v].loc[span], port['final'].loc[span])} vs portfolio final")})
    if period == "train":
        rows.append({"version": "portfolio: final, whole train 2016–2022", **row(port["final"]),
                     "gain [95% CI]": gain(port["final"], spy_alone) + " vs SPY alone"})
        rows.append({"version": "portfolio: own_turbulence, whole train 2016–2022", **row(port["own_turbulence"]),
                     "gain [95% CI]": gain(port["own_turbulence"], port["final"]) + " vs portfolio final"})
    headline = pd.DataFrame(rows).set_index("version")

    # ---- 2. each ETF, and how diversified the sleeves are
    per_etf = pd.DataFrame({sym: {"class": universe[sym], "valid days": int(data[sym].daily.loc[start:end, "is_valid"].sum()),
                                  "trades": int(runs["final"][sym]["trades"].sum()),
                                  **{f"sharpe {v}": metrics.sharpe(ret[v][sym].loc[span]) for v in VERSIONS}}
                            for sym in universe}).T
    div = portfolio.diversification({s: r.loc[span] for s, r in ret["final"].items()})
    corr = pd.DataFrame({s: r.loc[span] for s, r in ret["final"].items()}).fillna(0).corr()
    corr = corr.where(~np.eye(len(corr), dtype=bool))                  # leave out each ETF with itself
    classes = pd.Series(universe)
    class_corr = corr.groupby(classes).mean().T.groupby(classes).mean()

    sections = [f"# Multi-asset test, {period}\n\n_Generated by `scripts/08_multi_asset.py`. 16 ETFs "
                "(config/universe.yaml), same rules, costs and settings as on SPY, equal capital per ETF. "
                f"Main span: {span_name}. Sharpe gains: paired block bootstrap, 95% interval._",
                "## Portfolio\n\n" + headline.to_markdown(floatfmt=".2f"),
                f"## Diversification (paper's final, {span_name})\n\n"
                f"- Average pairwise correlation of daily returns: {div['average_correlation']:.2f}\n"
                f"- Worth about {div['independent_bets']:.1f} independent bets out of {div['assets']}\n\n"
                "Average correlation between and within asset classes:\n\n" + class_corr.to_markdown(floatfmt=".2f"),
                f"## Each ETF ({span_name})\n\n" + per_etf.to_markdown(floatfmt=".2f")]

    if period == "train":
        own = portfolio.combine({s: d["ret"] for s, d in sleeves("own_ml_vol", ml_mult_own).items()})
        stressed = {v: portfolio.combine({s: d["ret"] for s, d in
                                          sleeves(v, ml_mult if v == "own_ml_vol" else None,
                                                  replace(research, slippage=STRESS_SLIPPAGE)).items()}) for v in VERSIONS}
        sections += [
            "## ML: one pooled model vs one model per ETF (blocks 2018–2022)\n\n"
            f"- pooled: Sharpe {metrics.sharpe(port['own_ml_vol'].loc[blocks]):.2f}; per ETF: "
            f"{metrics.sharpe(own.loc[blocks]):.2f}; pooled minus per ETF: {gain(port['own_ml_vol'].loc[blocks], own.loc[blocks])}",
            f"## Cost stress: slippage ${STRESS_SLIPPAGE}/share instead of $0.001 (blocks 2018–2022)\n\n"
            + pd.DataFrame({v: {"sharpe": metrics.sharpe(stressed[v].loc[blocks]),
                                "annual_return": f"{metrics.annual_return(stressed[v].loc[blocks]):.1%}"}
                            for v in VERSIONS}).T.to_markdown(floatfmt=".2f")]

    plots.equity_curves({"spy_alone": spy_alone.loc[span], **{v: port[v].loc[span] for v in VERSIONS}},
                        f"16-ETF portfolio (three versions) vs SPY alone, {span_name}", f"multi_asset_{period}_equity")
    report = "\n\n".join(sections) + "\n"
    (RESULTS_DIR / f"multi_asset_{period}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
