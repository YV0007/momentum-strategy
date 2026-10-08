"""Benchmarks: buy & hold, open-to-close and the random-direction placebo."""

import numpy as np
import pandas as pd

from src.config import ResearchConfig
from src.engine.backtest import Prepared, simulate
from src.engine.sizing import shares


def buy_and_hold(daily: pd.DataFrame) -> pd.DataFrame:
    ret = daily["ret_cc"].fillna(0.0)
    return pd.DataFrame({"ret": ret, "trades": 0}, index=daily.index)


def open_to_close(daily: pd.DataFrame, research: ResearchConfig) -> pd.DataFrame:
    aum, rows = research.initial_aum, []
    for day, row in daily.iterrows():
        n = shares(aum, 1.0, row["open"])
        pnl = n * (row["close"] - row["open"]) - 2 * n * research.cost_per_share
        rows.append((day, pnl / aum, 1))
        aum += pnl
    return pd.DataFrame(rows, columns=["date", "ret", "trades"]).set_index("date")


def random_direction(position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    prev = np.hstack([np.zeros((position.shape[0], 1)), position[:, :-1]])
    starts = (position != 0) & (position != prev)
    trade_id = np.cumsum(starts.ravel()).reshape(position.shape)
    signs = rng.choice([-1.0, 1.0], size=trade_id.max() + 1)
    return np.where(position != 0, np.abs(position) * signs[trade_id], 0.0)


def placebo(prep: Prepared, research: ResearchConfig, n_runs: int = 200,
            seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    position = prep.positions()
    runs = {i: simulate(prep, random_direction(position, rng), research, with_trades=False).daily["ret"]
            for i in range(n_runs)}
    return pd.DataFrame(runs)
