"""Turn raw downloads into the two tables every later phase uses. (Phase 1)

spy_minute.parquet  one row per regular-session minute (09:30 -> session close),
                    gaps filled so every day has a complete minute grid.
spy_daily.parquet   one row per trading day: official open/close, session info,
                    dividends, data-quality flags, VIX.

No features are computed here (noise bands, VWAP, ... live in src/features.py).
"""

import pandas as pd

from src.config import NY_TZ, RAW_DIR, SYMBOL

# A day is unusable if more than this share of its minutes had to be filled.
MAX_FILLED_SHARE = 0.05


def load_raw_minutes(symbol: str = SYMBOL) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"{symbol}_1min_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw minute files in {RAW_DIR}; run scripts/01_download_data.py first")
    df = pd.concat(pd.read_parquet(f) for f in files).sort_index()
    return df[~df.index.duplicated()]


def session_grid(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """Every regular-session minute of every trading day, as bar-start timestamps.

    A day closing at 16:00 has 390 bars (09:30 ... 15:59); a 13:00 half-day has 210.
    """
    minutes = []
    for date, row in calendar.iterrows():
        day = f"{date:%Y-%m-%d}"
        start = pd.Timestamp(f"{day} {row.session_open}", tz=NY_TZ)
        end = pd.Timestamp(f"{day} {row.session_close}", tz=NY_TZ) - pd.Timedelta(minutes=1)
        minutes.append(pd.date_range(start, end, freq="min"))
    return minutes[0].append(minutes[1:]).rename("timestamp")


def build_minute_table(raw: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Regular-session bars on a complete minute grid.

    Missing minutes (no trade printed) become flat bars at the last close with zero volume,
    and are flagged with is_filled. A missing first minute takes the next available open.
    """
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

    df["minute"] = df.groupby("date").cumcount()  # 0 = 09:30 bar
    return df[["date", "minute", "open", "high", "low", "close", "volume", "trades",
               "vwap_bar", "is_filled"]]


def build_daily_table(minute: pd.DataFrame, official: pd.DataFrame, calendar: pd.DataFrame,
                      dividends: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day.

    open/high/low come from the minute bars (open = first 09:30 bar, as in the paper).
    close is the official closing-auction price (Yahoo); the last minute bar can miss the
    auction by a lot on extreme days, so last_bar_close is kept separately. If the official
    close is missing for a day, the last bar close is used instead.
    prev_close_adj removes the dividend from the previous close on ex-dates, so the
    ex-dividend drop is not mistaken for an overnight gap.
    """
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
    """Previous close (raw and dividend-adjusted) and daily returns from open/close/dividend."""
    daily = daily.copy()
    daily["prev_close"] = daily["close"].shift(1)
    daily["prev_close_adj"] = daily["prev_close"] - daily["dividend"]
    # Total return close-to-close (dividend added back) and intraday open-to-close.
    daily["ret_cc"] = (daily["close"] + daily["dividend"]) / daily["prev_close"] - 1
    daily["ret_oc"] = daily["close"] / daily["open"] - 1
    return daily
