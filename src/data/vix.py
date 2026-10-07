"""Download CBOE VIX daily history (open/high/low/close). (Phase 1)

Alpaca does not carry indices, so VIX comes from CBOE's free daily CSV.
"""

from io import StringIO

import pandas as pd
import requests

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def download_vix() -> pd.DataFrame:
    # CBOE redirects to its API host and rejects requests without a browser-like user agent.
    resp = requests.get(VIX_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df.set_index("date").sort_index()
