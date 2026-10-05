"""Strategy features, all computed without look-ahead. (Phase 2)

Timing convention: a minute row with index m (0 = the 09:30 bar) describes the market at the
END of that bar, i.e. at 09:31 + m. Its close is the price observable at that moment, and
every minute feature on that row uses only bars 0..m of the day plus earlier days.

Daily features are known at the open of their day: they use only information up to the
previous close, plus that day's opening price and VIX open.
"""

import numpy as np
import pandas as pd

LOOKBACK = 14        # days for the noise-area sigma (paper: 14)
VOL_LOOKBACK = 14    # days for the daily volatility used in sizing (paper: 14)
RSI_PERIOD = 5       # paper's gamma-imbalance proxy


# ---------------------------------------------------------------- minute features

def move_from_open(minute: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """|close / day open - 1| as a days x minutes table. Invalid days are left blank."""
    day_open = minute["date"].map(daily["open"])
    move = (minute["close"] / day_open - 1).abs()
    table = pd.DataFrame({"date": minute["date"], "minute": minute["minute"], "move": move.values}) \
              .pivot(index="date", columns="minute", values="move")
    table.loc[~daily["is_valid"].reindex(table.index, fill_value=False)] = np.nan
    return table


def noise_sigma(minute: pd.DataFrame, daily: pd.DataFrame, lookback: int = LOOKBACK) -> pd.Series:
    """Paper eq. 2: average absolute move from the open at each minute over the previous
    `lookback` days (today excluded). Half-days and invalid days simply contribute no value;
    at most 3 missing days are tolerated in the window."""
    table = move_from_open(minute, daily)
    sigma = table.rolling(lookback, min_periods=lookback - 3).mean().shift(1)
    return _to_minute_rows(sigma, minute, "sigma")


def noise_bands(minute: pd.DataFrame, daily: pd.DataFrame, sigma: pd.Series,
                vm: float = 1.0, gap_adjust: bool = True) -> pd.DataFrame:
    """Paper eq. 3 with gap adjustment (and Section 4.4 volatility multiplier VM):
        upper = max(open, prev_close) * (1 + VM * sigma)
        lower = min(open, prev_close) * (1 - VM * sigma)
    prev_close is dividend-adjusted so the ex-dividend drop is not treated as a gap.
    Without gap adjustment (paper eq. 3, first version) both bands are built around the open."""
    day_open = daily["open"]
    prev_close = daily["prev_close_adj"] if gap_adjust else day_open
    ref_high = minute["date"].map(np.maximum(day_open, prev_close))
    ref_low = minute["date"].map(np.minimum(day_open, prev_close))
    return pd.DataFrame({"upper": ref_high * (1 + vm * sigma),
                         "lower": ref_low * (1 - vm * sigma)}, index=minute.index)


def vwap(minute: pd.DataFrame) -> pd.Series:
    """Session VWAP up to the end of each bar, reset every day (regular hours only).
    Uses each bar's own volume-weighted price, which is more exact than (H+L+C)/3."""
    pv = (minute["vwap_bar"] * minute["volume"]).groupby(minute["date"]).cumsum()
    vol = minute["volume"].groupby(minute["date"]).cumsum()
    # Before the first traded share of the day there is no VWAP; fall back to the price.
    return (pv / vol.replace(0, np.nan)).fillna(minute["close"]).rename("vwap")


def _to_minute_rows(table: pd.DataFrame, minute: pd.DataFrame, name: str) -> pd.Series:
    """Map a days x minutes table back onto the minute rows."""
    long = table.stack(future_stack=True).rename(name)
    keys = pd.MultiIndex.from_arrays([minute["date"], minute["minute"]])
    return pd.Series(long.reindex(keys).values, index=minute.index, name=name)


# ---------------------------------------------------------------- daily features

def daily_volatility(daily: pd.DataFrame, lookback: int = VOL_LOOKBACK) -> pd.Series:
    """Paper eq. 4: std of the previous `lookback` daily returns (sample std, 1/(n-1))."""
    return daily["ret_cc"].rolling(lookback).std().shift(1).rename("vol_daily")


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI on daily closes."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)


def narrow_range(daily: pd.DataFrame, n: int) -> pd.Series:
    """NRn: the day's high-low range is the smallest of the last n days."""
    day_range = daily["high"] - daily["low"]
    return day_range.eq(day_range.rolling(n).min())


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Everything known at the open of each day."""
    return pd.DataFrame({
        "vol_daily": daily_volatility(daily),
        "gap": daily["open"] / daily["prev_close_adj"] - 1,
        "vix_open": daily["vix_open"],
        "rsi5_prev": rsi(daily["close"]).shift(1),
        "nr4_prev": narrow_range(daily, 4).shift(1, fill_value=False),
        "nr7_prev": narrow_range(daily, 7).shift(1, fill_value=False),
    }, index=daily.index)


# ---------------------------------------------------------------- build

def build_features(minute: pd.DataFrame, daily: pd.DataFrame,
                   lookback: int = LOOKBACK, vm: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (minute features, daily features). Bands use the default VM; the engine can
    rebuild them for other VMs from sigma via noise_bands()."""
    sigma = noise_sigma(minute, daily, lookback)
    minute_feats = pd.concat([minute[["date", "minute"]], sigma,
                              noise_bands(minute, daily, sigma, vm), vwap(minute)], axis=1)
    return minute_feats, daily_features(daily)
