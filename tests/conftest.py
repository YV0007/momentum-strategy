"""Shared helpers: a small synthetic market shaped exactly like the processed tables."""

import numpy as np
import pandas as pd
import pytest

from src.config import DAILY_FILE, MINUTE_FILE, NY_TZ
from src.data.clean import build_daily_table


def make_market(n_days: int = 30, seed: int = 0, dividend_days: dict | None = None,
                last_day: np.ndarray | None = None):
    """Random-walk minute bars for n_days full sessions + the matching daily table.

    last_day: optional 390 closes for the final day, as multiples of that day's open,
    to script an exact intraday path after a random warm-up.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    frames, price = [], 100.0
    for i, date in enumerate(dates):
        scripted = last_day is not None and i == n_days - 1
        if not scripted:
            price *= 1 + rng.normal(0, 0.005)  # overnight gap (none on the scripted day)
        ts = pd.date_range(f"{date:%Y-%m-%d} 09:30", periods=390, freq="min", tz=NY_TZ)
        if scripted:
            close = price * np.asarray(last_day, dtype=float)
        else:
            close = price * np.cumprod(1 + rng.normal(0, 0.0007, 390))
        open_ = np.r_[price, close[:-1]]
        frames.append(pd.DataFrame({
            "date": date, "minute": np.arange(390), "open": open_,
            "high": np.maximum(open_, close) * 1.0002, "low": np.minimum(open_, close) * 0.9998,
            "close": close, "volume": rng.integers(1_000, 50_000, 390), "trades": 10,
            "vwap_bar": (open_ + close) / 2, "is_filled": False}, index=ts))
        price = close[-1]
    minute = pd.concat(frames)
    minute.index.name = "timestamp"

    last = minute.groupby("date")["close"].last()
    official = pd.DataFrame({"open": minute.groupby("date")["open"].first(), "close": last})
    calendar = pd.DataFrame({"session_open": "09:30", "session_close": "16:00"}, index=dates)
    dividends = pd.DataFrame({"dividend": pd.Series(dividend_days or {}, dtype=float)})
    dividends.index = pd.to_datetime(dividends.index)
    vix = pd.DataFrame({"open": rng.uniform(12, 30, n_days), "close": rng.uniform(12, 30, n_days)},
                       index=dates)
    daily = build_daily_table(minute, official, calendar, dividends, vix)
    return minute, daily


@pytest.fixture(scope="session")
def real_market():
    """First ~60 trading days of the real processed data (skips if not built yet)."""
    if not MINUTE_FILE.exists():
        pytest.skip("processed data not built; run scripts/02_build_dataset.py")
    daily = pd.read_parquet(DAILY_FILE).iloc[:60]
    minute = pd.read_parquet(MINUTE_FILE)
    minute = minute[minute["date"] <= daily.index[-1]]
    return minute, daily
