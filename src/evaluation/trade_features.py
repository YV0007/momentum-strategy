"""Market conditions at each trade's entry."""

import numpy as np
import pandas as pd

from src.engine.backtest import BacktestResult, MarketData, Prepared, clock
from src.evaluation.diagnostics import trade_sequence, with_contribution
from src.intraday import intraday_state


def build(prep: Prepared, result: BacktestResult, data: MarketData) -> pd.DataFrame:
    trades = trade_sequence(with_contribution(result))
    state = intraday_state(data, prep.ks)
    day_pos = pd.Series(np.arange(len(data.daily)), index=data.daily.index)
    prep_pos = pd.Series(np.arange(len(prep.days)), index=prep.days)
    j = trades["entry_time"].map({t: i for i, t in enumerate(clock(prep.ks))}).to_numpy()
    d_all, d_prep = trades["date"].map(day_pos).to_numpy(), trades["date"].map(prep_pos).to_numpy()
    side = trades["side"].to_numpy()

    price, upper, lower, vwap = (getattr(prep, a)[d_prep, j] for a in ("price", "upper", "lower", "vwap"))
    daily = data.daily.loc[trades["date"]]
    feats = data.daily_feats.loc[trades["date"]]
    day_open = daily["open"].to_numpy()
    ref_high = np.maximum(day_open, daily["prev_close_adj"].to_numpy())
    ref_low = np.minimum(day_open, daily["prev_close_adj"].to_numpy())
    band_width = np.where(side > 0, upper - ref_high, ref_low - lower)

    out = trades[["date", "entry_time", "exit_time", "side", "side_name", "contribution",
                  "exit_type", "sequence", "reentry"]].copy()
    out["win"] = out["contribution"] > 0
    out["held_to_close"] = out["exit_type"] == "held to close"
    out["year"] = out["date"].dt.year
    out["weekday"] = out["date"].dt.day_name().str[:3]
    out["entry_minute"] = prep.ks[j]
    out["breakout_sigma"] = np.where(side > 0, price - upper, lower - price) / band_width
    out["vwap_distance_sigma"] = side * (price - vwap) / band_width
    for name, values in state.items():
        out[name] = values[d_all, j]
    out["gap_with_trade"] = side * feats["gap"].to_numpy()
    out["implied_vs_realized"] = (feats["vix_open"].to_numpy() / 100 / np.sqrt(252)) / feats["vol_daily"].to_numpy()
    out["vix_open"] = feats["vix_open"].to_numpy()
    out["leverage"] = result.daily.loc[out["date"], "leverage"].to_numpy()
    out["return_1x"] = out["contribution"] / out["leverage"]
    return out.reset_index(drop=True)
