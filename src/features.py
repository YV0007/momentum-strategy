"""Noise bands, VWAP and daily features, computed without look-ahead."""

import numpy as np
import pandas as pd

LOOKBACK = 14
VOL_LOOKBACK = 14
RSI_PERIOD = 5


def move_from_open(minute: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    day_open = minute["date"].map(daily["open"])
    move = (minute["close"] / day_open - 1).abs()
    table = pd.DataFrame({"date": minute["date"], "minute": minute["minute"], "move": move.values}) \
              .pivot(index="date", columns="minute", values="move")
    table.loc[~daily["is_valid"].reindex(table.index, fill_value=False)] = np.nan
    return table


def noise_sigma(minute: pd.DataFrame, daily: pd.DataFrame, lookback: int = LOOKBACK) -> pd.Series:
    table = move_from_open(minute, daily)
    sigma = table.rolling(lookback, min_periods=lookback - 3).mean().shift(1)
    return _to_minute_rows(sigma, minute, "sigma")


def noise_bands(minute: pd.DataFrame, daily: pd.DataFrame, sigma: pd.Series,
                vm: float = 1.0, gap_adjust: bool = True) -> pd.DataFrame:
    day_open = daily["open"]
    prev_close = daily["prev_close_adj"] if gap_adjust else day_open
    ref_high = minute["date"].map(np.maximum(day_open, prev_close))
    ref_low = minute["date"].map(np.minimum(day_open, prev_close))
    return pd.DataFrame({"upper": ref_high * (1 + vm * sigma),
                         "lower": ref_low * (1 - vm * sigma)}, index=minute.index)


def vwap(minute: pd.DataFrame) -> pd.Series:
    pv = (minute["vwap_bar"] * minute["volume"]).groupby(minute["date"]).cumsum()
    vol = minute["volume"].groupby(minute["date"]).cumsum()
    return (pv / vol.replace(0, np.nan)).fillna(minute["close"]).rename("vwap")


def _to_minute_rows(table: pd.DataFrame, minute: pd.DataFrame, name: str) -> pd.Series:
    long = table.stack(future_stack=True).rename(name)
    keys = pd.MultiIndex.from_arrays([minute["date"], minute["minute"]])
    return pd.Series(long.reindex(keys).values, index=minute.index, name=name)


def daily_volatility(daily: pd.DataFrame, lookback: int = VOL_LOOKBACK) -> pd.Series:
    return daily["ret_cc"].rolling(lookback).std().shift(1).rename("vol_daily")


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)


def narrow_range(daily: pd.DataFrame, n: int) -> pd.Series:
    day_range = daily["high"] - daily["low"]
    return day_range.eq(day_range.rolling(n).min())


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "vol_daily": daily_volatility(daily),
        "gap": daily["open"] / daily["prev_close_adj"] - 1,
        "vix_open": daily["vix_open"],
        "rsi5_prev": rsi(daily["close"]).shift(1),
        "nr4_prev": narrow_range(daily, 4).shift(1, fill_value=False),
        "nr7_prev": narrow_range(daily, 7).shift(1, fill_value=False),
    }, index=daily.index)


def build_features(minute: pd.DataFrame, daily: pd.DataFrame,
                   lookback: int = LOOKBACK, vm: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    sigma = noise_sigma(minute, daily, lookback)
    minute_feats = pd.concat([minute[["date", "minute"]], sigma,
                              noise_bands(minute, daily, sigma, vm), vwap(minute)], axis=1)
    return minute_feats, daily_features(daily)
