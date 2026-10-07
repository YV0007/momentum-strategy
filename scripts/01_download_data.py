"""Download all raw data into data/raw/. (Phase 1; multi-asset universe Phase 9)

    python -m scripts.01_download_data                    # every symbol in config/universe.yaml
    python -m scripts.01_download_data --symbols SPY      # selected symbols
    python -m scripts.01_download_data --force            # re-download years already on disk

Minute bars come from Alpaca, official daily bars and dividends from Yahoo, VIX from CBOE.
Finished years are skipped; the current year is always refreshed since it is still growing.
"""

import argparse
import time

import pandas as pd

from src.config import RAW_DIR, START_DATE, load_universe
from src.data import alpaca, vix, yahoo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(load_universe()))
    parser.add_argument("--force", action="store_true", help="re-download years already on disk")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    for symbol in args.symbols:
        daily, dividends = yahoo.download_daily(symbol, START_DATE)       # refuses symbols that split
        daily.to_parquet(RAW_DIR / f"{symbol}_daily_yahoo.parquet")
        dividends.to_parquet(RAW_DIR / f"{symbol}_dividends_yahoo.parquet")
        for year in range(pd.Timestamp(START_DATE).year, today.year + 1):
            path = RAW_DIR / f"{symbol}_1min_{year}.parquet"
            if path.exists() and not args.force and year != today.year:
                continue
            t0 = time.time()
            bars = alpaca.download_minute_bars(symbol, year)
            bars.to_parquet(path)
            print(f"{symbol} minute {year}: {len(bars):,} bars in {time.time() - t0:.0f}s", flush=True)

    alpaca.download_calendar(START_DATE, f"{today:%Y-%m-%d}").to_parquet(RAW_DIR / "calendar.parquet")
    vix.download_vix().to_parquet(RAW_DIR / "vix_daily.parquet")
    print("calendar and VIX: done")


if __name__ == "__main__":
    main()
