"""Download all raw data into data/raw/. (Phase 1)

    python -m scripts.01_download_data            # 2016 -> today, skips finished years
    python -m scripts.01_download_data --force    # re-download everything
"""

import argparse
import time

import pandas as pd

from src.config import RAW_DIR, START_DATE, SYMBOL
from src.data import alpaca, vix, yahoo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download years already on disk")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    end = f"{today:%Y-%m-%d}"

    for year in range(pd.Timestamp(START_DATE).year, today.year + 1):
        path = RAW_DIR / f"{SYMBOL}_1min_{year}.parquet"
        # The current year is always refreshed since it is still growing.
        if path.exists() and not args.force and year != today.year:
            print(f"minute {year}: on disk, skipping")
            continue
        t0 = time.time()
        bars = alpaca.download_minute_bars(SYMBOL, year)
        bars.to_parquet(path)
        print(f"minute {year}: {len(bars):,} bars in {time.time() - t0:.0f}s")

    alpaca.download_calendar(START_DATE, end).to_parquet(RAW_DIR / "calendar.parquet")
    daily, dividends = yahoo.download_daily(SYMBOL, START_DATE)
    daily.to_parquet(RAW_DIR / f"{SYMBOL}_daily_yahoo.parquet")
    dividends.to_parquet(RAW_DIR / f"{SYMBOL}_dividends_yahoo.parquet")
    vix.download_vix().to_parquet(RAW_DIR / "vix_daily.parquet")
    print("calendar, official daily bars + dividends (Yahoo), VIX (CBOE): done")


if __name__ == "__main__":
    main()
