"""Build the processed minute and daily tables, the data-quality report and the strategy
features. (Phases 1-2)

    python -m scripts.02_build_dataset
"""

import pandas as pd

from src import features
from src.config import (DAILY_FILE, FEATURES_DAILY_FILE, FEATURES_MINUTE_FILE, MINUTE_FILE,
                        PROCESSED_DIR, RAW_DIR, SYMBOL)
from src.data import clean, quality


def main() -> None:
    raw = clean.load_raw_minutes()
    calendar = pd.read_parquet(RAW_DIR / "calendar.parquet")
    # Only sessions we actually have data for (the calendar runs to today).
    last_day = raw.index.max().tz_localize(None).normalize()
    calendar = calendar.loc[:last_day]

    minute = clean.build_minute_table(raw, calendar)
    daily = clean.build_daily_table(
        minute,
        official=pd.read_parquet(RAW_DIR / f"{SYMBOL}_daily_yahoo.parquet"),
        calendar=calendar,
        dividends=pd.read_parquet(RAW_DIR / f"{SYMBOL}_dividends_yahoo.parquet"),
        vix=pd.read_parquet(RAW_DIR / "vix_daily.parquet"),
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    minute.to_parquet(MINUTE_FILE)
    daily.to_parquet(DAILY_FILE)

    results = quality.check(minute, daily)
    quality.write_report(results)
    print(f"{results['n_days']:,} days, {results['n_minutes']:,} minutes, "
          f"{results['n_invalid']} invalid days -> {MINUTE_FILE.name}, {DAILY_FILE.name}, docs/data_quality.md")

    minute_feats, daily_feats = features.build_features(minute, daily)
    minute_feats.to_parquet(FEATURES_MINUTE_FILE)
    daily_feats.to_parquet(FEATURES_DAILY_FILE)
    print(f"features -> {FEATURES_MINUTE_FILE.name}, {FEATURES_DAILY_FILE.name}")


if __name__ == "__main__":
    main()
