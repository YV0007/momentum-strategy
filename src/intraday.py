"""How today looks at each decision point compared with normal, for every day and decision time.
(Phase 6)

Shared by the trade-level research (evaluation/trade_features.py), the turbulence sizing rule
(engine/sizing.py) and the ML volatility forecast (strategies/ml_sizing.py).

Every feature uses only information available at the decision time: minute rows up to the bar
that ends at the decision (index k-1) and daily features known at the open. "Normal" levels are
averages over the previous 14 valid days at the same time of day, like the noise area itself.
The one exception is the ML target in decision_panel, which is the FUTURE by design and is only
used to train.
"""

import numpy as np
import pandas as pd

LOOKBACK = 14            # days for the "normal" level of each intraday statistic
VOLUME_WINDOW = 30       # minutes of recent volume / volatility before a decision
OPENING_MINUTES = 30
BAND_NORMAL_DAYS = 250   # band width is compared with its own average over about a year


def pivot(minute: pd.DataFrame, values: pd.Series, days: pd.Index) -> np.ndarray:
    """(days x 390) array of a minute series; NaN after the close on half-days."""
    table = pd.DataFrame({"date": minute["date"], "minute": minute["minute"], "v": values.values})
    return table.pivot(index="date", columns="minute", values="v").reindex(days).to_numpy()


def normal(stat: np.ndarray, days: pd.Index, valid: np.ndarray) -> np.ndarray:
    """Average of a per-day statistic over the previous LOOKBACK valid days (today excluded)."""
    df = pd.DataFrame(stat, index=days).where(pd.Series(valid, index=days), axis=0)
    return df.rolling(LOOKBACK, min_periods=LOOKBACK - 3).mean().shift(1).to_numpy()


def _padded_cumsum(x: np.ndarray) -> np.ndarray:
    """cum[:, m] = sum of the first m bars (NaN bars count as 0), so a window of bars a..b-1 is
    cum[:, b] - cum[:, a]."""
    return np.hstack([np.zeros((len(x), 1)), np.nancumsum(x, axis=1)])


def _bars(data):
    days, daily = data.daily.index, data.daily
    close, high, low, volume = (pivot(data.minute, data.minute[c], days) for c in ("close", "high", "low", "volume"))
    day_open = daily["open"].to_numpy()[:, None]
    prev_close = np.hstack([day_open, close[:, :-1]])
    return days, daily["is_valid"].to_numpy(), close, high, low, volume, day_open, prev_close


def intraday_state(data, ks: np.ndarray) -> dict[str, np.ndarray]:
    """Per day and decision point (days x decisions): how today looks so far vs normal."""
    days, valid, close, high, low, volume, day_open, prev_close = _bars(data)
    m = ks - 1                                                     # bar ending at each decision

    # Recent volume: the last 30 minutes vs the same 30 minutes on normal days
    cum_volume = _padded_cumsum(volume)
    recent_volume = cum_volume[:, m + 1] - cum_volume[:, m + 1 - VOLUME_WINDOW]

    # Realized variance from the open, and how straight the path has been (net move / path length)
    sq_returns = np.log(close / prev_close) ** 2
    realized_var = np.nancumsum(sq_returns, axis=1)[:, m]
    path_length = np.nancumsum(np.abs(close - prev_close), axis=1)[:, m]
    efficiency = np.abs(close[:, m] - day_open) / path_length

    # Opening range: high-low of the first 30 minutes, known from 10:00 on
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


# Candidate inputs of the ML volatility forecast (the user's list + context). All scale-free.
PANEL_FEATURES = [
    "log_rel_vol_so_far",      # realized vol from the open to the decision vs normal
    "log_rel_vol_first30",     # realized vol of the first 30 minutes vs normal
    "log_rel_vol_last30",      # realized vol of the last 30 minutes before the decision vs normal
    "log_rel_opening_range",   # high-low of the first 30 minutes vs normal
    "log_rel_volume_30m",      # volume of the last 30 minutes vs the same 30 minutes normally
    "log_rel_volume_so_far",   # volume since the open vs normal at this time of day
    "vix_open",                # VIX at the open
    "log_vix_vs_realized",     # VIX (as a daily vol) vs the 14-day realized daily vol
    "log_rel_band_width",      # today's noise band (14-day sigma) vs its own ~1-year average
    "abs_gap_vol",             # |overnight gap| in units of daily vol
    "decision_minute",         # minutes after the open (30 = 10:00)
]


def decision_panel(data, ks: np.ndarray) -> pd.DataFrame:
    """One row per (day, decision time) for every day in data.daily: PANEL_FEATURES, all known at
    the decision, and `target` = log(realized vol from the fill bar to the close / its normal),
    which is NOT known then. The target is NaN on invalid days and half-days (never trained on)."""
    days, valid, close, high, low, volume, day_open, prev_close = _bars(data)
    n_days, n_dec = len(days), len(ks)
    m = ks - 1

    sq = np.log(close / prev_close) ** 2
    cum_sq, cum_volume = _padded_cumsum(sq), _padded_cumsum(volume)
    var_so_far = cum_sq[:, m + 1]
    var_first30 = cum_sq[:, OPENING_MINUTES]
    var_last30 = cum_sq[:, m + 1] - cum_sq[:, m + 1 - VOLUME_WINDOW]
    var_rest = cum_sq[:, -1:] - cum_sq[:, ks]                     # bars k..close: what the trade will live through
    volume_so_far = cum_volume[:, m + 1]
    volume_30m = cum_volume[:, m + 1] - cum_volume[:, m + 1 - VOLUME_WINDOW]
    opening_range = (np.nanmax(high[:, :OPENING_MINUTES], axis=1)
                     - np.nanmin(low[:, :OPENING_MINUTES], axis=1)) / day_open[:, 0]

    def rel(stat):                                                 # stat vs its normal at the same time
        stat = stat if stat.ndim == 2 else stat[:, None]
        return stat / normal(stat, days, valid)

    def per_decision(x):                                           # a daily value repeated on every decision
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
