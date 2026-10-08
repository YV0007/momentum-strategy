"""Downloads official daily prices and dividends from Yahoo."""

import pandas as pd
import requests

from src.config import NY_TZ

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def _to_dates(epoch_seconds) -> pd.DatetimeIndex:
    return (pd.to_datetime(epoch_seconds, unit="s", utc=True)
              .tz_convert(NY_TZ).normalize().tz_localize(None))


def download_daily(symbol: str, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
              "period2": int(pd.Timestamp.now(tz="UTC").timestamp()),
              "interval": "1d", "events": "div,split"}
    resp = requests.get(CHART_URL.format(symbol=symbol), params=params,
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]

    quote = result["indicators"]["quote"][0]
    bars = pd.DataFrame({k: quote[k] for k in ["open", "high", "low", "close", "volume"]},
                        index=_to_dates(result["timestamp"]))
    bars.index.name = "date"

    splits = result.get("events", {}).get("splits", {}).values()
    if splits:
        dates = ", ".join(f"{d:%Y-%m-%d}" for d in _to_dates([e["date"] for e in splits]))
        raise ValueError(f"{symbol} split on {dates}: unadjusted prices would break the pipeline")
    events = result.get("events", {}).get("dividends", {}).values()
    dividends = pd.DataFrame({"dividend": [e["amount"] for e in events]},
                             index=_to_dates([e["date"] for e in events]))
    dividends.index.name = "ex_date"
    return bars.dropna(subset=["close"]), dividends.sort_index()
