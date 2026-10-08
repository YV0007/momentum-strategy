"""Downloads 1-minute bars and the trading calendar from Alpaca."""

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import ENV_FILE, NY_TZ

DATA_URL = "https://data.alpaca.markets"
TRADING_URL = "https://paper-api.alpaca.markets"
MAX_RETRIES = 5

_BAR_COLUMNS = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
                "v": "volume", "n": "trades", "vw": "vwap_bar"}


def _session() -> requests.Session:
    load_dotenv(ENV_FILE)
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret or key.startswith("your_"):
        raise SystemExit(f"Alpaca keys missing: fill in ALPACA_API_KEY / ALPACA_SECRET_KEY in {ENV_FILE}")
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    return session


def _get(session: requests.Session, url: str, params: dict) -> dict | list:
    for attempt in range(MAX_RETRIES):
        resp = session.get(url, params=params, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in (401, 403):
            raise SystemExit(f"Alpaca rejected the request ({resp.status_code}): check the keys in .env")
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Alpaca request failed after {MAX_RETRIES} retries: {url}")


def fetch_bars(session: requests.Session, symbol: str, timeframe: str,
               start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    params = {"timeframe": timeframe, "start": start.isoformat(), "end": end.isoformat(),
              "limit": 10_000, "adjustment": "raw", "feed": "sip", "sort": "asc"}
    rows = []
    while True:
        payload = _get(session, f"{DATA_URL}/v2/stocks/{symbol}/bars", params)
        rows.extend(payload.get("bars") or [])
        if not payload.get("next_page_token"):
            break
        params["page_token"] = payload["next_page_token"]

    df = pd.DataFrame(rows, columns=list(_BAR_COLUMNS)).rename(columns=_BAR_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY_TZ)
    return df.set_index("timestamp")


def download_minute_bars(symbol: str, year: int) -> pd.DataFrame:
    session = _session()
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"),
              pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20))
    months = list(pd.date_range(start, end, freq="MS")) + [end]
    chunks = [fetch_bars(session, symbol, "1Min", a, b) for a, b in zip(months[:-1], months[1:]) if a < b]
    df = pd.concat(chunks).sort_index()
    return df[~df.index.duplicated()]


def download_calendar(start: str, end: str) -> pd.DataFrame:
    rows = _get(_session(), f"{TRADING_URL}/v2/calendar", {"start": start, "end": end})
    df = pd.DataFrame(rows)[["date", "open", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"open": "session_open", "close": "session_close"}).set_index("date")
