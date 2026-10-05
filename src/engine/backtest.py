"""Backtest engine: half-hourly decisions, fills, sizing, costs, AUM. (Phase 3)

Timing, for a decision at time T (e.g. 10:00 = minute k = 30):
  observe  close of bar k-1 (the price at T) and the features on that row,
  fill     at the open of bar k, plus slippage and commission,
  exit     every position at the official close (market-on-close).
Shares are fixed at the open from yesterday's AUM, so within a day every trade of the paper
versions uses the same share count. Own versions scale each trade by an intraday multiplier
(engine/sizing.py) fixed at its entry; their positions are fractions of that share count,
rounded down to whole shares per trade.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import features
from src.config import (DAILY_FILE, FEATURES_DAILY_FILE, FEATURES_MINUTE_FILE, MINUTE_FILE,
                        ResearchConfig, StrategyConfig)
from src.engine import costs, rules, sizing


@dataclass
class MarketData:
    minute: pd.DataFrame
    daily: pd.DataFrame
    minute_feats: pd.DataFrame
    daily_feats: pd.DataFrame

    @classmethod
    def load(cls) -> "MarketData":
        return cls(*(pd.read_parquet(f) for f in
                     (MINUTE_FILE, DAILY_FILE, FEATURES_MINUTE_FILE, FEATURES_DAILY_FILE)))


@dataclass
class BacktestResult:
    config: StrategyConfig
    daily: pd.DataFrame    # one row per day: aum, ret, shares, leverage, pnl, costs, trades
    trades: pd.DataFrame   # one row per trade: entry/exit time and price, side, shares, pnl


def decision_minutes(config: StrategyConfig) -> np.ndarray:
    """Bar index of each decision time, e.g. 10:00, 10:30, ..., 15:30 -> 30, 60, ..., 360."""
    hh, mm = map(int, config.first_decision.split(":"))
    first = (hh * 60 + mm) - (9 * 60 + 30)
    return np.arange(first, 390, config.decision_every_min)


def clock(ks: np.ndarray) -> list[str]:
    """Bar indexes as clock times: 30 -> "10:00"."""
    return [f"{(570 + k) // 60:02d}:{(570 + k) % 60:02d}" for k in ks]


def _at_minutes(values: pd.Series, minute: pd.DataFrame, cols: np.ndarray, days: pd.Index) -> np.ndarray:
    """(days x len(cols)) array of `values` at the given bar indexes; NaN where a day has no
    such bar (half-days)."""
    mask = minute["minute"].isin(cols)
    table = pd.DataFrame({"date": minute.loc[mask, "date"], "minute": minute.loc[mask, "minute"],
                          "v": values[mask].values}).pivot(index="date", columns="minute", values="v")
    return table.reindex(index=days, columns=cols).to_numpy()


@dataclass
class Prepared:
    """Everything the simulation needs, as (days x decision points) arrays. Built once per
    config and period, so many position variants (e.g. the placebo) can reuse it."""
    config: StrategyConfig
    days: pd.Index
    ks: np.ndarray
    price: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    vwap: np.ndarray
    legs: np.ndarray          # fill prices per leg; last column is the official close
    in_session: np.ndarray
    tradable: np.ndarray
    leverage: pd.Series
    day_open: np.ndarray
    size: np.ndarray          # intraday size multiplier at each decision, already within the leverage cap

    def rule_positions(self) -> np.ndarray:
        """The rule's direction (+1/-1/0), flat outside the session and on non-tradable days."""
        position = rules.target_positions(self.config.stop, self.price, self.upper, self.lower, self.vwap)
        return self.mask(position)

    def positions(self) -> np.ndarray:
        """What the strategy holds: the rule's direction, each trade scaled by the intraday size
        multiplier at its entry (1 for the paper versions)."""
        return hold_from_entry(self.rule_positions(), self.size)

    def mask(self, position: np.ndarray) -> np.ndarray:
        position = position.copy()
        position[~self.in_session | ~self.tradable[:, None]] = 0.0
        return position


def hold_from_entry(position: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Scale each trade (a run of the same non-zero position within a day) by `size` at the
    decision where it started, and keep that size until the trade ends."""
    prev = np.hstack([np.zeros((len(position), 1)), position[:, :-1]])
    starts = (position != 0) & (position != prev)
    entry_col = np.maximum.accumulate(np.where(starts, np.arange(position.shape[1]), 0), axis=1)
    return position * np.take_along_axis(size, entry_col, axis=1)


def prepare(config: StrategyConfig, data: MarketData,
            start: str | None = None, end: str | None = None,
            multiplier: np.ndarray | None = None) -> Prepared:
    """multiplier: optional (all days in data.daily x decisions) intraday size multiplier that
    replaces the config's, e.g. walk-forward ML forecasts on the train period."""
    days = data.daily.loc[start:end].index
    daily = data.daily.loc[days]
    ks = decision_minutes(config)

    # ---- what is observed at each decision (end of bar k-1) and the fill price (open of bar k)
    feats = data.minute_feats
    if config.vm != 1.0 or not config.gap_adjust:
        feats = feats.assign(**features.noise_bands(data.minute, data.daily, feats["sigma"],
                                                    config.vm, config.gap_adjust))
    at_decision = {col: _at_minutes(feats[col], data.minute, ks - 1, days) for col in ("upper", "lower", "vwap")}
    price = _at_minutes(data.minute["close"], data.minute, ks - 1, days)
    fill = _at_minutes(data.minute["open"], data.minute, ks, days)

    # ---- which days may trade at all
    lev = sizing.leverage(config, data.daily_feats["vol_daily"]).reindex(days)
    warmed_up = ~np.isnan(at_decision["upper"]).all(axis=1)
    tradable = daily["is_valid"].to_numpy() & warmed_up & lev.notna().to_numpy()

    # ---- legs: hold position j from fill j to fill j+1; the last leg ends at the close
    in_session = ~np.isnan(fill)
    close = daily["close"].to_numpy()[:, None]
    legs = np.hstack([np.where(in_session, fill, close), close])  # after a half-day close, legs are flat

    # ---- intraday size multiplier (own versions), never above the leverage cap
    if multiplier is None:
        multiplier = sizing.intraday_multiplier(config, data, ks)
    size = pd.DataFrame(multiplier, index=data.daily.index).reindex(days).to_numpy()
    size = np.minimum(size, config.max_leverage / lev.to_numpy()[:, None])
    size = np.where(np.isfinite(size), size, 1.0)
    return Prepared(config, days, ks, price, at_decision["upper"], at_decision["lower"],
                    at_decision["vwap"], legs, in_session, tradable, lev, daily["open"].to_numpy(), size)


def simulate(prep: Prepared, position: np.ndarray, research: ResearchConfig,
             with_trades: bool = True) -> BacktestResult:
    """Account for a (days x decision points) position matrix: P&L, costs, compounding.
    Positions are in units of the day's share count; +1/-1/0 everywhere for the paper versions,
    fractions for the own versions (rounded down to whole shares)."""
    leg_moves = np.diff(prep.legs, axis=1)
    whole = np.isin(position, (-1.0, 0.0, 1.0)).all()
    prev = np.hstack([np.zeros((len(prep.days), 1)), position])
    if whole:   # one share count per day: P&L per share for all days at once
        pnl_per_share = (position * leg_moves).sum(axis=1)
        units_traded = np.abs(np.diff(prev, axis=1)).sum(axis=1) + np.abs(position[:, -1])
        in_market = np.abs(position).max(axis=1)
    else:
        abs_pos, sign = np.abs(position), np.sign(position)

    # ---- compound day by day: share count is fixed at the open from yesterday's AUM
    aum = research.initial_aum
    lev = prep.leverage.to_numpy()
    rows = []
    for i, day in enumerate(prep.days):
        n = sizing.shares(aum, lev[i], prep.day_open[i]) if prep.tradable[i] else 0
        if whole:
            gross, traded, peak = n * pnl_per_share[i], n * units_traded[i], n * in_market[i]
        else:
            held = np.floor(n * abs_pos[i] + 1e-9) * sign[i]           # whole shares held on each leg
            gross = held @ leg_moves[i]
            traded = np.abs(np.diff(held, prepend=0.0)).sum() + abs(held[-1])
            peak = np.abs(held).max()
        cost = costs.trading_cost(traded, research.cost_per_share)
        start_aum, aum = aum, aum + gross - cost
        rows.append((day, start_aum, aum, (aum - start_aum) / start_aum, n, peak,
                     lev[i] if prep.tradable[i] else 0.0, gross, cost))
    result = pd.DataFrame(rows, columns=["date", "aum_start", "aum", "ret", "shares", "peak_shares",
                                         "leverage", "pnl_gross", "costs"]).set_index("date")

    # A trade starts wherever the position becomes non-zero or changes sign.
    starts = (position != 0) & (position != prev[:, :-1])
    result["trades"] = starts.sum(axis=1)
    trades = (trade_log(position, prep.legs, prep.ks, prep.days, result["shares"].to_numpy(),
                        research.cost_per_share) if with_trades else pd.DataFrame())
    return BacktestResult(prep.config, result, trades)


def run(config: StrategyConfig, research: ResearchConfig, data: MarketData,
        start: str | None = None, end: str | None = None) -> BacktestResult:
    prep = prepare(config, data, start, end)
    return simulate(prep, prep.positions(), research)


def trade_log(position: np.ndarray, legs: np.ndarray, ks: np.ndarray, days: pd.Index,
              shares: np.ndarray, cost_per_share: float) -> pd.DataFrame:
    """One row per trade: a run of the same non-zero position within a day. `size` is the
    position in units of the day's share count (1 for the paper versions)."""
    decision_time = clock(ks) + ["close"]
    rows = []
    for d, j in zip(*np.nonzero(position)):
        value = position[d, j]
        if j > 0 and position[d, j - 1] == value:
            continue                                     # not the start of a trade
        exit_j = j + 1
        while exit_j < position.shape[1] and position[d, exit_j] == value:
            exit_j += 1
        entry, exit_ = legs[d, j], legs[d, exit_j]
        side, size = np.sign(value), abs(value)
        n = np.floor(shares[d] * size + 1e-9)            # whole shares, as in simulate
        rows.append((days[d], decision_time[j], decision_time[exit_j], int(side), size, n, entry, exit_,
                     n * side * (exit_ - entry) - 2 * n * cost_per_share))
    return pd.DataFrame(rows, columns=["date", "entry_time", "exit_time", "side", "size", "shares",
                                       "entry_price", "exit_price", "pnl"])
