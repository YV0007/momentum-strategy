"""Chooses the ML model's settings on the train period."""

import argparse
import itertools
from dataclasses import asdict, replace
from datetime import date

import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

from src.config import RESULTS_DIR, load_research, load_strategies
from src.engine.backtest import MarketData, clock, decision_minutes, run
from src.evaluation.split import walk_forward_folds
from src.intraday import PANEL_FEATURES, decision_panel
from src.strategies import ml_sizing
from src.strategies.ml_sizing import MLSettings

GRID = {"max_depth": [2, 3, 4], "max_iter": [150, 400], "min_samples_leaf": [50, 200, 800],
        "l2_regularization": [1.0, 10.0], "monotone": [True, False]}
FINE_GRID = {"max_depth": [1, 2, 3], "max_iter": [50, 100, 150, 250], "min_samples_leaf": [20, 50, 100, 200],
             "l2_regularization": [1.0, 10.0, 30.0], "monotone": [True, False]}
DROP_TOLERANCE = 0.0025
HAR = ["log_rel_vol_so_far", "log_rel_vol_last30", "log_rel_vol_first30"]


class CV:
    def __init__(self, panel: pd.DataFrame, days: pd.Index, entries: pd.MultiIndex):
        dates = panel.index.get_level_values("date")
        self.folds = []
        for fit_days, val_days in walk_forward_folds(days):
            fit_rows = panel[dates.isin(fit_days) & panel["target"].notna()]
            val_rows = panel[dates.isin(val_days) & panel["target"].notna()]
            self.folds.append((fit_rows, val_rows, val_rows.index.isin(entries)))

    def mse(self, predict) -> np.ndarray:
        return np.array([np.mean((v["target"] - predict(f, v)) ** 2) for f, v, _ in self.folds])

    def r2(self, predict, entries_only: bool = False) -> np.ndarray:
        out = []
        for f, v, at_entry in self.folds:
            v = v[at_entry] if entries_only else v
            y, p = v["target"].to_numpy(), predict(f, v)
            out.append(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
        return np.array(out)


def gbt(s: MLSettings):
    def predict(fit_rows, val_rows):
        model = ml_sizing.make_model(s).fit(fit_rows[list(s.features)], fit_rows["target"])
        return model.predict(val_rows[list(s.features)])
    return predict


def rule_forecast(fit_rows, val_rows):
    return val_rows["log_rel_vol_so_far"].fillna(0.0).to_numpy()


def linear_forecast(fit_rows, val_rows):
    x_fit, x_val = fit_rows[HAR].fillna(0.0), val_rows[HAR].fillna(0.0)
    return LinearRegression().fit(x_fit, fit_rows["target"]).predict(x_val)


def complexity(s: MLSettings) -> tuple:
    return (not s.monotone, s.max_depth, s.max_iter, -s.min_samples_leaf, -s.l2_regularization)


def grid_search(cv: CV, features: tuple[str, ...], grid: dict = GRID) -> tuple[MLSettings, pd.DataFrame]:
    rows = {}
    for values in itertools.product(*grid.values()):
        s = MLSettings(features=features, **dict(zip(grid, values)))
        rows[s] = cv.mse(gbt(s))
    best = min(rows, key=lambda s: rows[s].mean())
    table = []
    for s, m in rows.items():
        diff = m - rows[best]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if s != best else 0.0
        table.append({**{k: getattr(s, k) for k in GRID}, "cv_mse": m.mean(), "vs_best": diff.mean(),
                      "paired_se": se, "within_1se": diff.mean() <= se, "settings": s})
    table = pd.DataFrame(table).sort_values("cv_mse")
    chosen = min(table.loc[table["within_1se"], "settings"], key=complexity)
    return chosen, table


def eliminate(cv: CV, s: MLSettings) -> tuple[MLSettings, list[dict]]:
    current = list(s.features)
    cur = cv.mse(gbt(s)).mean()
    path = [{"step": "all features", "n_features": len(current), "cv_mse": cur}]
    while len(current) > 1:
        trials = {f: cv.mse(gbt(replace(s, features=tuple(x for x in current if x != f)))).mean()
                  for f in current}
        f, mse = min(trials.items(), key=lambda t: t[1])
        if mse > cur * (1 + DROP_TOLERANCE):
            path.append({"step": f"stop: cheapest drop ({f}) would cost {mse / cur - 1:+.2%}",
                         "n_features": len(current), "cv_mse": cur})
            break
        current.remove(f)
        path.append({"step": f"drop {f} ({mse / cur - 1:+.2%})", "n_features": len(current), "cv_mse": mse})
        cur = mse
    return replace(s, features=tuple(current)), path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write config/ml_sizing.yaml")
    args = parser.parse_args()

    research = load_research()
    data = MarketData.load()
    final = load_strategies()["final"]
    ks = decision_minutes(final)
    days = data.daily.loc[research.train_start:research.train_end].index
    panel = decision_panel(data, ks)
    panel = panel[panel.index.get_level_values("date").isin(days)]

    trades = run(final, research, data, research.train_start, research.train_end).trades
    j = trades["entry_time"].map({t: i for i, t in enumerate(clock(ks))})
    entries = pd.MultiIndex.from_arrays([trades["date"], j])
    cv = CV(panel, days, entries)
    print(f"rows per fold (fit / validation): {[(len(f), len(v)) for f, v, _ in cv.folds]}")

    print("stage 1: grid on all candidate features ...", flush=True)
    s1, grid1 = grid_search(cv, tuple(PANEL_FEATURES))
    print("  chosen:", {k: getattr(s1, k) for k in GRID})
    print("stage 2: backward elimination ...", flush=True)
    s2, path = eliminate(cv, s1)
    print("  kept:", s2.features)
    print("stage 3: finer grid on the kept features ...", flush=True)
    final_s, grid3 = grid_search(cv, s2.features, FINE_GRID)
    print("  chosen:", {k: getattr(final_s, k) for k in GRID})

    models = {"constant (train mean)": lambda f, v: np.full(len(v), f["target"].mean()),
              "rule: vol so far": rule_forecast, "linear (HAR-style)": linear_forecast,
              "GBT, all features (stage 1)": gbt(s1), "GBT, final settings": gbt(final_s)}
    quality = pd.DataFrame({name: {**{f"R2 block {i + 1}": r for i, r in enumerate(cv.r2(p))},
                                   "R2 mean": cv.r2(p).mean(),
                                   "R2 at trade entries": cv.r2(p, entries_only=True).mean()}
                            for name, p in models.items()}).T

    model = ml_sizing.fit(panel, days, final_s)
    rows = panel[panel["target"].notna()].sample(8000, random_state=0)
    imp = permutation_importance(model, rows[list(final_s.features)], rows["target"], n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, index=final_s.features).sort_values(ascending=False)

    show = lambda t: t.drop(columns="settings").head(10).to_markdown(index=False, floatfmt=".5f")
    report = "\n\n".join([
        "# ML volatility forecast: settings chosen on the train period\n\n"
        f"_Generated by `scripts/05a_tune_ml_sizing.py` on {date.today()}. Train {research.train_start} – "
        f"{research.train_end} only; walk-forward blocks of `walk_forward_folds`; objective = validation MSE of "
        "log(rest-of-day vol / normal). Trading P&L is not used here._",
        "## Forecast quality out of sample (R², validation blocks)\n\n" + quality.to_markdown(floatfmt=".3f"),
        "## Stage 1: grid on all features (top 10 by CV error)\n\n" + show(grid1),
        "## Stage 2: backward elimination (tolerance 0.25% of CV error)\n\n"
        + pd.DataFrame(path).to_markdown(index=False, floatfmt=".5f"),
        "## Stage 3: finer grid on the kept features (top 10)\n\n" + show(grid3),
        "## Final settings\n\n```yaml\n" + yaml.safe_dump({**asdict(final_s), "features": list(final_s.features)},
                                                       sort_keys=False) + "```",
        "## Permutation importance (final model fitted on the whole train period)\n\n"
        + importance.to_frame("importance").to_markdown(floatfmt=".4f"),
    ]) + "\n"
    (RESULTS_DIR / "ml_sizing_tuning.md").write_text(report)
    print(quality.round(3).to_string())
    print(importance.round(4).to_string())

    if args.write:
        header = (f"# ML volatility forecast settings (own_ml_vol). Chosen on the TRAIN period only by\n"
                  f"# scripts/05a_tune_ml_sizing.py on {date.today()}; see results/ml_sizing_tuning.md.\n"
                  f"# Frozen with the own versions: do not edit after the test run.\n")
        settings = {**asdict(final_s), "features": list(final_s.features)}
        ml_sizing.SETTINGS_FILE.write_text(header + yaml.safe_dump(settings, sort_keys=False))
        print(f"wrote {ml_sizing.SETTINGS_FILE}")


if __name__ == "__main__":
    main()
