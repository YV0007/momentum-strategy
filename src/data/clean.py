"""Builds the clean minute and daily tables from the raw downloads."""

import pandas as pd

from src.config import NY_TZ, RAW_DIR, SYMBOL

MAX_FILLED_SHARE = 0.05


def load_raw_minutes(symbol: str = SYMBOL) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"{symbol}_1min_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw minute files in {RAW_DIR}; run scripts/01_download_data.py first")
    df = pd.concat(pd.read_parquet(f) for f in files).sort_index()
    return df[~df.index.duplicated()]


def session_grid(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    minutes = []
    for date, row in calendar.iterrows():
        day = f"{date:%Y-%m-%d}"
        start = pd.Timestamp(f"{day} {row.session_open}", tz=NY_TZ)
        end = pd.Timestamp(f"{day} {row.session_close}", tz=NY_TZ) - pd.Timedelta(minutes=1)
        minutes.append(pd.date_range(start, end, freq="min"))
    return minutes[0].append(minutes[1:]).rename("timestamp")


def build_minute_table(raw: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    grid = session_grid(calendar)
    df = raw.reindex(grid)
    df["is_filled"] = df["close"].isna()
    df["date"] = df.index.tz_localize(None).normalize()

    by_day = df.groupby("date")
    last_close = by_day["close"].ffill()
    next_open = by_day["open"].bfill()
    fill_price = last_close.fillna(next_open)
    for col in ["open", "high", "low", "close", "vwap_bar"]:
        df[col] = df[col].fillna(fill_price)
    df[["volume", "trades"]] = df[["volume", "trades"]].fillna(0).astype("int64")

    df["minute"] = df.groupby("date").cumcount()
    return df[["date", "minute", "open", "high", "low", "close", "volume", "trades",
               "vwap_bar", "is_filled"]]


def build_daily_table(minute: pd.DataFrame, official: pd.DataFrame, calendar: pd.DataFrame,
                      dividends: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    by_day = minute.groupby("date")
    daily = pd.DataFrame({
        "open": by_day["open"].first(),
        "high": by_day["high"].max(),
        "low": by_day["low"].min(),
        "last_bar_close": by_day["close"].last(),
        "volume": by_day["volume"].sum(),
        "n_minutes": by_day.size(),
        "n_filled": by_day["is_filled"].sum(),
    })
    daily["official_open"] = official["open"].reindex(daily.index)
    daily["close"] = official["close"].reindex(daily.index).fillna(daily["last_bar_close"])
    daily = daily.join(calendar[["session_close"]])
    daily["is_half_day"] = daily["session_close"] != "16:00"

    daily["dividend"] = dividends["dividend"].reindex(daily.index).fillna(0.0)
    daily = add_returns(daily)

    daily["filled_share"] = daily["n_filled"] / daily["n_minutes"]
    daily["is_valid"] = (daily["filled_share"] <= MAX_FILLED_SHARE) & daily["close"].notna()

    vix_cols = vix[["open", "close"]].add_prefix("vix_")
    daily = daily.join(vix_cols)
    daily.index.name = "date"
    return daily


def add_returns(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["prev_close"] = daily["close"].shift(1)
    daily["prev_close_adj"] = daily["prev_close"] - daily["dividend"]
    daily["ret_cc"] = (daily["close"] + daily["dividend"]) / daily["prev_close"] - 1
    daily["ret_oc"] = daily["close"] / daily["open"] - 1
    return daily
