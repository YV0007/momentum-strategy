"""Append every backtest run to results/experiment_log.csv. (Phase 0)

The log is the honest count of how many configurations were tried, which the Deflated
Sharpe Ratio (Phase 4) needs, and a record of what was looked at and when.
"""

import json
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from src.config import RESULTS_DIR, StrategyConfig

LOG_FILE = RESULTS_DIR / "experiment_log.csv"


def log_run(config: StrategyConfig, period: str, start: str, end: str, metrics: dict,
            note: str = "") -> None:
    params = asdict(config)
    if params["intraday_sizing"] == "none":      # paper configs keep the same params text as earlier rows
        for key in ("intraday_sizing", "size_floor", "size_cap"):
            params.pop(key)
    if params["lookback"] == 14:
        params.pop("lookback")
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "strategy": config.name,
           "period": period, "start": start, "end": end,
           "params": json.dumps(params, sort_keys=True), **metrics, "note": note}
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)
