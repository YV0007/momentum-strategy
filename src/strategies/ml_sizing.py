"""Gradient-boosting forecast of rest-of-day volatility for own_ml_vol."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import CONFIG_DIR, load_research
from src.intraday import PANEL_FEATURES, decision_panel

SETTINGS_FILE = CONFIG_DIR / "ml_sizing.yaml"

MONOTONE = {"log_rel_vol_so_far": 1, "log_rel_vol_first30": 1, "log_rel_vol_last30": 1,
            "log_rel_opening_range": 1, "log_rel_volume_30m": 1, "log_rel_volume_so_far": 1,
            "vix_open": 0, "log_vix_vs_realized": 1, "log_rel_band_width": 0, "abs_gap_vol": 1,
            "decision_minute": 0}
assert set(MONOTONE) == set(PANEL_FEATURES)

SPX_ONLY = {"log_vix_vs_realized", "vix_open"}


@dataclass(frozen=True)
class MLSettings:
    features: tuple[str, ...]
    max_depth: int = 3
    max_iter: int = 300
    learning_rate: float = 0.05
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0
    monotone: bool = True


def load_settings() -> MLSettings:
    raw = yaml.safe_load(SETTINGS_FILE.read_text())
    return MLSettings(**{**raw, "features": tuple(raw["features"])})


def make_model(s: MLSettings) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=s.max_depth, max_iter=s.max_iter, learning_rate=s.learning_rate,
        min_samples_leaf=s.min_samples_leaf, l2_regularization=s.l2_regularization,
        monotonic_cst=[MONOTONE[f] if s.monotone else 0 for f in s.features],
        early_stopping=False, random_state=0)


def fit(panel: pd.DataFrame, days: pd.Index, s: MLSettings) -> HistGradientBoostingRegressor:
    rows = panel[panel.index.get_level_values("date").isin(days) & panel["target"].notna()]
    return make_model(s).fit(rows[list(s.features)], rows["target"])


def forecast(model: HistGradientBoostingRegressor, panel: pd.DataFrame, s: MLSettings) -> np.ndarray:
    n_decisions = panel.index.get_level_values("decision").nunique()
    return np.exp(model.predict(panel[list(s.features)])).reshape(-1, n_decisions)


def train_days(data) -> pd.Index:
    research = load_research()
    return data.daily.loc[research.train_start:research.train_end].index


def frozen_forecast(data, ks: np.ndarray, settings: MLSettings | None = None) -> np.ndarray:
    settings = settings or load_settings()
    panel = decision_panel(data, ks)
    model = fit(panel, train_days(data), settings)
    return forecast(model, panel, settings)


def walk_forward_forecast(data, ks: np.ndarray, folds: list[tuple[pd.Index, pd.Index]],
                          settings: MLSettings | None = None) -> np.ndarray:
    settings = settings or load_settings()
    panel = decision_panel(data, ks)
    out = np.full((len(data.daily), len(ks)), np.nan)
    row = pd.Series(np.arange(len(data.daily)), index=data.daily.index)
    for fit_days, val_days in folds:
        model = fit(panel, fit_days, settings)
        rows = row.loc[val_days].to_numpy()
        val = panel.loc[data.daily.index[rows]]
        out[rows] = forecast(model, val, settings)
    return out


def asset_settings(settings: MLSettings | None = None) -> MLSettings:
    settings = settings or load_settings()
    return replace(settings, features=tuple(f for f in settings.features if f not in SPX_ONLY))


def fit_pooled(panels: dict[str, pd.DataFrame], days: pd.Index, s: MLSettings) -> HistGradientBoostingRegressor:
    rows = pd.concat([p[p.index.get_level_values("date").isin(days) & p["target"].notna()] for p in panels.values()])
    return make_model(s).fit(rows[list(s.features)], rows["target"])


def walk_forward_pooled(panels: dict[str, pd.DataFrame], folds: list[tuple[pd.Index, pd.Index]],
                        s: MLSettings) -> dict[str, np.ndarray]:
    out = {sym: np.full((p.index.get_level_values("date").nunique(), p.index.get_level_values("decision").nunique()),
                        np.nan) for sym, p in panels.items()}
    for fit_days, val_days in folds:
        model = fit_pooled(panels, fit_days, s)
        for sym, panel in panels.items():
            dates = panel.index.get_level_values("date").unique()
            rows = np.flatnonzero(dates.isin(val_days))
            out[sym][rows] = forecast(model, panel.loc[dates[rows]], s)
    return out
