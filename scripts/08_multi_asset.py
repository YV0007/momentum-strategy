"""Runs the strategies on 12 ETFs as a portfolio."""

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
    research = load_research()
    strategies = load_strategies()
    universe = load_universe()
    data = {sym: MarketData.load(sym) for sym in universe}
    start, end = research.train_start, research.train_end
    ks = decision_minutes(strategies["final"])
    cfg_ml, settings = strategies["own_ml_vol"], ml_sizing.asset_settings()

    panels = {sym: decision_panel(d, ks) for sym, d in data.items()}
    folds = walk_forward_folds(data["SPY"].daily.loc[start:end].index)
    blocks = pd.DatetimeIndex(np.concatenate([val for _, val in folds]))
    to_mult = lambda f: sizing.clip_multiplier(1 / f, cfg_ml)
    with np.errstate(divide="ignore", invalid="ignore"):
        ml_mult = {s: to_mult(f) for s, f in ml_sizing.walk_forward_pooled(panels, folds, settings).items()}
        ml_mult_own = {s: to_mult(ml_sizing.walk_forward_pooled({s: panels[s]}, folds, settings)[s]) for s in universe}

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

    for v in VERSIONS:
        daily = pd.DataFrame({"ret": port[v], "trades": sum(d["trades"] for d in runs[v].values())}).loc[blocks]
        log_run(replace(strategies[v], name=f"{v}_multi{len(universe)}"), "train_blocks", f"{blocks[0]:%Y-%m-%d}",
                f"{blocks[-1]:%Y-%m-%d}", metrics.summary(daily), note=f"multi-asset portfolio, {len(universe)} ETFs")

    rows = [{"version": "SPY alone (paper's final)", **row(spy_alone.loc[blocks]), "gain [95% CI]": ""}]
    for v in VERSIONS:
        rows.append({"version": f"portfolio: {v}", **row(port[v].loc[blocks]),
                     "gain [95% CI]": gain(port[v].loc[blocks], spy_alone.loc[blocks]) + " vs SPY alone"
                     + ("" if v == "final" else f"; {gain(port[v].loc[blocks], port['final'].loc[blocks])} vs portfolio final")})
    rows.append({"version": "portfolio: final, whole train 2016–2022", **row(port["final"]),
                 "gain [95% CI]": gain(port["final"], spy_alone) + " vs SPY alone"})
    rows.append({"version": "portfolio: own_turbulence, whole train 2016–2022", **row(port["own_turbulence"]),
                 "gain [95% CI]": gain(port["own_turbulence"], port["final"]) + " vs portfolio final"})
    headline = pd.DataFrame(rows).set_index("version")

    per_etf = pd.DataFrame({sym: {"class": universe[sym], "valid days": int(data[sym].daily.loc[start:end, "is_valid"].sum()),
                                  "trades": int(runs["final"][sym]["trades"].sum()),
                                  **{f"sharpe {v}": metrics.sharpe(ret[v][sym].loc[blocks]) for v in VERSIONS}}
                            for sym in universe}).T
    div = portfolio.diversification({s: r.loc[blocks] for s, r in ret["final"].items()})
    corr = pd.DataFrame({s: r.loc[blocks] for s, r in ret["final"].items()}).fillna(0).corr()
    corr = corr.where(~np.eye(len(corr), dtype=bool))
    classes = pd.Series(universe)
    class_corr = corr.groupby(classes).mean().T.groupby(classes).mean()

    own = portfolio.combine({s: d["ret"] for s, d in sleeves("own_ml_vol", ml_mult_own).items()})
    stressed = {v: portfolio.combine({s: d["ret"] for s, d in sleeves(v, ml_mult if v == "own_ml_vol" else None,
                                                                      replace(research, slippage=STRESS_SLIPPAGE)).items()})
                for v in VERSIONS}

    sections = [
        f"# Multi-asset test, train\n\n_Generated by `scripts/08_multi_asset.py`. {len(universe)} ETFs "
        "(config/universe.yaml), same rules, costs and settings as on SPY, equal capital per ETF. Main span: "
        "walk-forward blocks 2018–2022. Sharpe gains: paired block bootstrap, 95% interval. Stopped after train "
        "(finding: the edge does not carry over); the test period was not used._",
        "## Portfolio\n\n" + headline.to_markdown(floatfmt=".2f"),
        "## Diversification (paper's final, blocks 2018–2022)\n\n"
        f"- Average pairwise correlation of daily returns: {div['average_correlation']:.2f}\n"
        f"- Worth about {div['independent_bets']:.1f} independent bets out of {div['assets']}\n\n"
        "Average correlation between and within asset classes:\n\n" + class_corr.to_markdown(floatfmt=".2f"),
        "## Each ETF (blocks 2018–2022)\n\n" + per_etf.to_markdown(floatfmt=".2f"),
        "## ML: one pooled model vs one model per ETF (blocks 2018–2022)\n\n"
        f"- pooled: Sharpe {metrics.sharpe(port['own_ml_vol'].loc[blocks]):.2f}; per ETF: "
        f"{metrics.sharpe(own.loc[blocks]):.2f}; pooled minus per ETF: {gain(port['own_ml_vol'].loc[blocks], own.loc[blocks])}",
        f"## Cost stress: slippage ${STRESS_SLIPPAGE}/share instead of $0.001 (blocks 2018–2022)\n\n"
        + pd.DataFrame({v: {"sharpe": metrics.sharpe(stressed[v].loc[blocks]),
                            "annual_return": f"{metrics.annual_return(stressed[v].loc[blocks]):.1%}"}
                        for v in VERSIONS}).T.to_markdown(floatfmt=".2f"),
    ]
    plots.equity_curves({"spy_alone": spy_alone.loc[blocks], **{v: port[v].loc[blocks] for v in VERSIONS}},
                        f"{len(universe)}-ETF portfolio (three versions) vs SPY alone, blocks 2018–2022",
                        "multi_asset_train_equity")
    report = "\n\n".join(sections) + "\n"
    (RESULTS_DIR / "multi_asset_train.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
