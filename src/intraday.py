"""How today compares with normal at each decision time, and the ML feature panel."""

import numpy as np
import pandas as pd

LOOKBACK = 14
VOLUME_WINDOW = 30
OPENING_MINUTES = 30
BAND_NORMAL_DAYS = 250


def pivot(minute: pd.DataFrame, values: pd.Series, days: pd.Index) -> np.ndarray:
    table = pd.DataFrame({"date": minute["date"], "minute": minute["minute"], "v": values.values})
    return table.pivot(index="date", columns="minute", values="v").reindex(days).to_numpy()


def normal(stat: np.ndarray, days: pd.Index, valid: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(stat, index=days).where(pd.Series(valid, index=days), axis=0)
    return df.rolling(LOOKBACK, min_periods=LOOKBACK - 3).mean().shift(1).to_numpy()


def _padded_cumsum(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.zeros((len(x), 1)), np.nancumsum(x, axis=1)])


def _bars(data):
    days, daily = data.daily.index, data.daily
    close, high, low, volume = (pivot(data.minute, data.minute[c], days) for c in ("close", "high", "low", "volume"))
    day_open = daily["open"].to_numpy()[:, None]
    prev_close = np.hstack([day_open, close[:, :-1]])
    return days, daily["is_valid"].to_numpy(), close, high, low, volume, day_open, prev_close


def intraday_state(data, ks: np.ndarray) -> dict[str, np.ndarray]:
    days, valid, close, high, low, volume, day_open, prev_close = _bars(data)
    m = ks - 1

    cum_volume = _padded_cumsum(volume)
    recent_volume = cum_volume[:, m + 1] - cum_volume[:, m + 1 - VOLUME_WINDOW]

    sq_returns = np.log(close / prev_close) ** 2
    realized_var = np.nancumsum(sq_returns, axis=1)[:, m]
    path_length = np.nancumsum(np.abs(close - prev_close), axis=1)[:, m]
    with np.errstate(invalid="ignore", divide="ignore"):
        efficiency = np.abs(close[:, m] - day_open) / path_length

    opening_range = (np.nanmax(high[:, :OPENING_MINUTES], axis=1)
                     - np.nanmin(low[:, :OPENING_MINUTES], axis=1)) / day_open[:, 0]

    return {
        "rel_volume_30m": recent_volume / normal(recent_volume, days, valid),
        "rel_realized_vol": np.sqrt(realized_var / normal(realized_var, days, valid)),
        "efficiency": efficiency,
        "rel_opening_range": np.repeat((opening_range / normal(opening_range[:, None], days, valid)[:, 0])[:, None],
                                       len(ks), axis=1),
    }


def _log(x) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.asarray(x, dtype=float))
    return np.where(np.isfinite(out), out, np.nan)


PANEL_FEATURES = [
    "log_rel_vol_so_far",
    "log_rel_vol_first30",
    "log_rel_vol_last30",
    "log_rel_opening_range",
    "log_rel_volume_30m",
    "log_rel_volume_so_far",
    "vix_open",
    "log_vix_vs_realized",
    "log_rel_band_width",
    "abs_gap_vol",
    "decision_minute",
]


def decision_panel(data, ks: np.ndarray) -> pd.DataFrame:
    days, valid, close, high, low, volume, day_open, prev_close = _bars(data)
    n_days, n_dec = len(days), len(ks)
    m = ks - 1

    sq = np.log(close / prev_close) ** 2
    cum_sq, cum_volume = _padded_cumsum(sq), _padded_cumsum(volume)
    var_so_far = cum_sq[:, m + 1]
    var_first30 = cum_sq[:, OPENING_MINUTES]
    var_last30 = cum_sq[:, m + 1] - cum_sq[:, m + 1 - VOLUME_WINDOW]
    var_rest = cum_sq[:, -1:] - cum_sq[:, ks]
    volume_so_far = cum_volume[:, m + 1]
    volume_30m = cum_volume[:, m + 1] - cum_volume[:, m + 1 - VOLUME_WINDOW]
    opening_range = (np.nanmax(high[:, :OPENING_MINUTES], axis=1)
                     - np.nanmin(low[:, :OPENING_MINUTES], axis=1)) / day_open[:, 0]

    def rel(stat):
        stat = stat if stat.ndim == 2 else stat[:, None]
        return stat / normal(stat, days, valid)

    def per_decision(x):
        return np.repeat(np.asarray(x, dtype=float).reshape(-1, 1), n_dec, axis=1)

    feats = data.daily_feats.reindex(days)
    sigma = pivot(data.minute, data.minute_feats["sigma"], days)[:, m]
    sigma_norm = pd.DataFrame(sigma, index=days).rolling(BAND_NORMAL_DAYS, min_periods=BAND_NORMAL_DAYS // 2) \
        .mean().shift(1).to_numpy()
    vix_daily = feats["vix_open"].to_numpy() / 100 / np.sqrt(252)

    cols = {
        "log_rel_vol_so_far": 0.5 * _log(rel(var_so_far)),
        "log_rel_vol_first30": per_decision(0.5 * _log(rel(var_first30)[:, 0])),
        "log_rel_vol_last30": 0.5 * _log(rel(var_last30)),
        "log_rel_opening_range": per_decision(_log(rel(opening_range)[:, 0])),
        "log_rel_volume_30m": _log(rel(volume_30m)),
        "log_rel_volume_so_far": _log(rel(volume_so_far)),
        "vix_open": per_decision(feats["vix_open"]),
        "log_vix_vs_realized": per_decision(_log(vix_daily / feats["vol_daily"].to_numpy())),
        "log_rel_band_width": _log(sigma / sigma_norm),
        "abs_gap_vol": per_decision(np.abs(feats["gap"].to_numpy()) / feats["vol_daily"].to_numpy()),
        "decision_minute": np.tile(ks.astype(float), (n_days, 1)),
    }
    target = 0.5 * _log(rel(var_rest))
    trainable = valid & ~data.daily["is_half_day"].to_numpy().astype(bool)
    target[~trainable] = np.nan

    index = pd.MultiIndex.from_product([days, np.arange(n_dec)], names=["date", "decision"])
    panel = pd.DataFrame({name: values.ravel() for name, values in cols.items()}, index=index)
    panel["target"] = target.ravel()
    return panel
