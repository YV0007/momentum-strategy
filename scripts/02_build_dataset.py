"""Build the processed minute and daily tables, the data-quality report and the strategy
features for each symbol. (Phases 1-2; multi-asset universe Phase 9)

    python -m scripts.02_build_dataset                    # every symbol in config/universe.yaml
    python -m scripts.02_build_dataset --symbols SPY

Writes data/processed/<SYMBOL>/ and the data-quality report results/data_quality/<SYMBOL>.md.
"""

import argparse

import pandas as pd

from src import features
from src.config import RAW_DIR, RESULTS_DIR, load_universe, processed_files
from src.data import clean, quality


def build(symbol: str) -> None:
    raw = clean.load_raw_minutes(symbol)
    calendar = pd.read_parquet(RAW_DIR / "calendar.parquet")
    calendar = calendar.loc[:raw.index.max().tz_localize(None).normalize()]   # sessions we have data for

    minute = clean.build_minute_table(raw, calendar)
    daily = clean.build_daily_table(
        minute,
        official=pd.read_parquet(RAW_DIR / f"{symbol}_daily_yahoo.parquet"),
        calendar=calendar,
        dividends=pd.read_parquet(RAW_DIR / f"{symbol}_dividends_yahoo.parquet"),
        vix=pd.read_parquet(RAW_DIR / "vix_daily.parquet"),
    )
    minute_feats, daily_feats = features.build_features(minute, daily)

    files = processed_files(symbol)
    files["minute"].parent.mkdir(parents=True, exist_ok=True)
    for name, table in [("minute", minute), ("daily", daily), ("features_minute", minute_feats),
                        ("features_daily", daily_feats)]:
        table.to_parquet(files[name])

    results = quality.check(minute, daily)
    report = RESULTS_DIR / "data_quality" / f"{symbol}.md"
    quality.write_report(results, report)
    print(f"{symbol}: {results['n_days']:,} days, {results['n_invalid']} invalid -> {files['minute'].parent}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(load_universe()))
    for symbol in parser.parse_args().symbols:
        build(symbol)


if __name__ == "__main__":
    main()
