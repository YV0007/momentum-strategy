"""Own version B (challenger): size each trade by a shallow gradient-boosting forecast of how
volatile the rest of the day will be, relative to normal. (Phase 6)

The model predicts `target` of src/intraday.decision_panel: log(realized vol from the fill bar to
the close / its 14-day normal at the same time). It never sees trade direction or P&L: volatility
is forecastable, trade outcomes are not (docs/own_strategy_attempts.md). It trains on
every (day, decision time) of the training days, not just the ~1,500 trades.

Settings (features, hyper-parameters) live in config/ml_sizing.yaml, chosen on the train period
only by scripts/05a_tune_ml_sizing.py. For the test period the model is fitted ONCE on the whole
train period and then frozen: no refits.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import CONFIG_DIR, load_research
from src.intraday import PANEL_FEATURES, decision_panel

SETTINGS_FILE = CONFIG_DIR / "ml_sizing.yaml"

# Monotone constraints (used when settings.monotone is true), fixed from the economics, not fitted:
# +1 = more of this so far today means more volatility for the rest of the day; 0 = free.
MONOTONE = {"log_rel_vol_so_far": 1, "log_rel_vol_first30": 1, "log_rel_vol_last30": 1,
            "log_rel_opening_range": 1, "log_rel_volume_30m": 1, "log_rel_volume_so_far": 1,
            "vix_open": 0, "log_vix_vs_realized": 1, "log_rel_band_width": 0, "abs_gap_vol": 1,
            "decision_minute": 0}
assert set(MONOTONE) == set(PANEL_FEATURES)

# Inputs that describe the S&P 500 (VIX) rather than the traded asset; dropped when one model is
# pooled over many assets.
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
    """Deterministic (fixed seed, no early stopping), so the frozen model can be rebuilt exactly."""
    return HistGradientBoostingRegressor(
        max_depth=s.max_depth, max_iter=s.max_iter, learning_rate=s.learning_rate,
        min_samples_leaf=s.min_samples_leaf, l2_regularization=s.l2_regularization,
        monotonic_cst=[MONOTONE[f] if s.monotone else 0 for f in s.features],
        early_stopping=False, random_state=0)


def fit(panel: pd.DataFrame, days: pd.Index, s: MLSettings) -> HistGradientBoostingRegressor:
    """Fit on the rows of `days` that have a target (valid full days)."""
    rows = panel[panel.index.get_level_values("date").isin(days) & panel["target"].notna()]
    return make_model(s).fit(rows[list(s.features)], rows["target"])


def forecast(model: HistGradientBoostingRegressor, panel: pd.DataFrame, s: MLSettings) -> np.ndarray:
    """(days x decisions) forecast of rest-of-day vol / normal, as a ratio (1 = a normal day)."""
    n_decisions = panel.index.get_level_values("decision").nunique()
    return np.exp(model.predict(panel[list(s.features)])).reshape(-1, n_decisions)


def train_days(data) -> pd.Index:
    research = load_research()
    return data.daily.loc[research.train_start:research.train_end].index


def frozen_forecast(data, ks: np.ndarray, settings: MLSettings | None = None) -> np.ndarray:
    """Forecast for every day in data.daily from ONE model fitted on the whole train period.
    Rows after the train period are never used for fitting."""
    settings = settings or load_settings()
    panel = decision_panel(data, ks)
    model = fit(panel, train_days(data), settings)
    return forecast(model, panel, settings)


def walk_forward_forecast(data, ks: np.ndarray, folds: list[tuple[pd.Index, pd.Index]],
                          settings: MLSettings | None = None) -> np.ndarray:
    """Train-period evaluation: for each fold, fit on its training days only and forecast its
    validation block. NaN outside the validation blocks."""
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


# ---------------------------------------------------------------- one model pooled over many assets (Phase 9)

def asset_settings(settings: MLSettings | None = None) -> MLSettings:
    """The frozen settings, keeping only inputs that mean the same thing for every asset."""
    settings = settings or load_settings()
    return replace(settings, features=tuple(f for f in settings.features if f not in SPX_ONLY))


def fit_pooled(panels: dict[str, pd.DataFrame], days: pd.Index, s: MLSettings) -> HistGradientBoostingRegressor:
    """One model on the stacked rows of all assets for `days`. Inputs and target are relative to each
    asset's own normal level, so the assets share one scale."""
    rows = pd.concat([p[p.index.get_level_values("date").isin(days) & p["target"].notna()] for p in panels.values()])
    return make_model(s).fit(rows[list(s.features)], rows["target"])


def walk_forward_pooled(panels: dict[str, pd.DataFrame], folds: list[tuple[pd.Index, pd.Index]],
                        s: MLSettings) -> dict[str, np.ndarray]:
    """Per asset, (days x decisions) forecasts from models fitted on the days before each validation
    block, pooled over all assets. NaN outside the validation blocks."""
    out = {sym: np.full((p.index.get_level_values("date").nunique(), p.index.get_level_values("decision").nunique()),
                        np.nan) for sym, p in panels.items()}
    for fit_days, val_days in folds:
        model = fit_pooled(panels, fit_days, s)
        for sym, panel in panels.items():
            dates = panel.index.get_level_values("date").unique()
            rows = np.flatnonzero(dates.isin(val_days))
            out[sym][rows] = forecast(model, panel.loc[dates[rows]], s)
    return out
