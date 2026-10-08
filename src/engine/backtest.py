"""Backtest engine: decisions, fills, sizing, costs and account value."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import features
from src.config import SYMBOL, ResearchConfig, StrategyConfig, processed_files
from src.engine import costs, rules, sizing


@dataclass
class MarketData:
    minute: pd.DataFrame
    daily: pd.DataFrame
    minute_feats: pd.DataFrame
    daily_feats: pd.DataFrame

    @classmethod
    def load(cls, symbol: str = SYMBOL) -> "MarketData":
        files = processed_files(symbol)
        return cls(*(pd.read_parquet(files[name]) for name in ("minute", "daily", "features_minute", "features_daily")))


@dataclass
class BacktestResult:
    config: StrategyConfig
    daily: pd.DataFrame
    trades: pd.DataFrame


def decision_minutes(config: StrategyConfig) -> np.ndarray:
    hh, mm = map(int, config.first_decision.split(":"))
    first = (hh * 60 + mm) - (9 * 60 + 30)
    return np.arange(first, 390, config.decision_every_min)


def clock(ks: np.ndarray) -> list[str]:
    return [f"{(570 + k) // 60:02d}:{(570 + k) % 60:02d}" for k in ks]


def _at_minutes(values: pd.Series, minute: pd.DataFrame, cols: np.ndarray, days: pd.Index) -> np.ndarray:
    mask = minute["minute"].isin(cols)
    table = pd.DataFrame({"date": minute.loc[mask, "date"], "minute": minute.loc[mask, "minute"],
                          "v": values[mask].values}).pivot(index="date", columns="minute", values="v")
    return table.reindex(index=days, columns=cols).to_numpy()


@dataclass
class Prepared:
    config: StrategyConfig
    days: pd.Index
    ks: np.ndarray
    price: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    vwap: np.ndarray
    legs: np.ndarray
    in_session: np.ndarray
    tradable: np.ndarray
    leverage: pd.Series
    day_open: np.ndarray
    size: np.ndarray
    adv: np.ndarray
    vol_annual: np.ndarray

    def rule_positions(self) -> np.ndarray:
        position = rules.target_positions(self.config.stop, self.price, self.upper, self.lower, self.vwap)
        return self.mask(position)

    def positions(self) -> np.ndarray:
        return hold_from_entry(self.rule_positions(), self.size)

    def mask(self, position: np.ndarray) -> np.ndarray:
        position = position.copy()
        position[~self.in_session | ~self.tradable[:, None]] = 0.0
        return position


def hold_from_entry(position: np.ndarray, size: np.ndarray) -> np.ndarray:
    prev = np.hstack([np.zeros((len(position), 1)), position[:, :-1]])
    starts = (position != 0) & (position != prev)
    entry_col = np.maximum.accumulate(np.where(starts, np.arange(position.shape[1]), 0), axis=1)
    return position * np.take_along_axis(size, entry_col, axis=1)


def prepare(config: StrategyConfig, data: MarketData,
            start: str | None = None, end: str | None = None,
            multiplier: np.ndarray | None = None) -> Prepared:
    days = data.daily.loc[start:end].index
    daily = data.daily.loc[days]
    ks = decision_minutes(config)

    feats = data.minute_feats
    if config.vm != 1.0 or not config.gap_adjust or config.lookback != features.LOOKBACK:
        sigma = (feats["sigma"] if config.lookback == features.LOOKBACK
                 else features.noise_sigma(data.minute, data.daily, config.lookback))
        feats = feats.assign(**features.noise_bands(data.minute, data.daily, sigma, config.vm, config.gap_adjust))
    at_decision = {col: _at_minutes(feats[col], data.minute, ks - 1, days) for col in ("upper", "lower", "vwap")}
    price = _at_minutes(data.minute["close"], data.minute, ks - 1, days)
    fill = _at_minutes(data.minute["open"], data.minute, ks, days)

    lev = sizing.leverage(config, data.daily_feats["vol_daily"]).reindex(days)
    warmed_up = ~np.isnan(at_decision["upper"]).all(axis=1)
    tradable = daily["is_valid"].to_numpy() & warmed_up & lev.notna().to_numpy()

    in_session = ~np.isnan(fill)
    close = daily["close"].to_numpy()[:, None]
    legs = np.hstack([np.where(in_session, fill, close), close])

    if multiplier is None:
        multiplier = sizing.intraday_multiplier(config, data, ks)
    size = pd.DataFrame(multiplier, index=data.daily.index).reindex(days).to_numpy()
    size = np.minimum(size, config.max_leverage / lev.to_numpy()[:, None])
    size = np.where(np.isfinite(size), size, 1.0)

    window = data.daily[["volume", "ret_cc"]].rolling(costs.IMPACT_DAYS, min_periods=2)
    adv = window.mean()["volume"].shift(1).reindex(days).to_numpy()
    vol_annual = (window.std()["ret_cc"].shift(1) * np.sqrt(252)).reindex(days).to_numpy()
    return Prepared(config, days, ks, price, at_decision["upper"], at_decision["lower"], at_decision["vwap"],
                    legs, in_session, tradable, lev, daily["open"].to_numpy(), size, adv, vol_annual)


def simulate(prep: Prepared, position: np.ndarray, research: ResearchConfig,
             with_trades: bool = True) -> BacktestResult:
    leg_moves = np.diff(prep.legs, axis=1)
    flat_costs = research.slippage_model == "fixed" and not research.commission_tiered
    fast = flat_costs and np.isin(position, (-1.0, 0.0, 1.0)).all()
    prev = np.hstack([np.zeros((len(prep.days), 1)), position])
    if fast:
        pnl_per_share = (position * leg_moves).sum(axis=1)
        units_traded = np.abs(np.diff(prev, axis=1)).sum(axis=1) + np.abs(position[:, -1])
        in_market = np.abs(position).max(axis=1)
    else:
        abs_pos, sign = np.abs(position), np.sign(position)

    aum = research.initial_aum
    lev = prep.leverage.to_numpy()
    rows, traded_history = [], []
    for i, day in enumerate(prep.days):
        n = sizing.shares(aum, lev[i], prep.day_open[i]) if prep.tradable[i] else 0
        if fast:
            gross, traded, peak = n * pnl_per_share[i], n * units_traded[i], n * in_market[i]
            cost = costs.trading_cost(traded, research.cost_per_share)
        else:
            held = np.floor(n * abs_pos[i] + 1e-9) * sign[i]
            fills = np.abs(np.diff(held, prepend=0.0, append=0.0))
            gross, traded, peak = held @ leg_moves[i], fills.sum(), np.abs(held).max()
            rate = costs.commission_rate(research, sum(traded_history[-costs.TIER_DAYS:]))
            cost = rate * traded + costs.slippage(research, fills, prep.legs[i], prep.adv[i], prep.vol_annual[i])
        traded_history.append(traded)
        start_aum, aum = aum, aum + gross - cost
        rows.append((day, start_aum, aum, (aum - start_aum) / start_aum, n, peak,
                     lev[i] if prep.tradable[i] else 0.0, gross, cost, traded))
    result = pd.DataFrame(rows, columns=["date", "aum_start", "aum", "ret", "shares", "peak_shares",
                                         "leverage", "pnl_gross", "costs", "shares_traded"]).set_index("date")

    starts = (position != 0) & (position != prev[:, :-1])
    result["trades"] = starts.sum(axis=1)
    cost_per_share = (result["costs"] / result["shares_traded"]).fillna(0.0).to_numpy()
    trades = (trade_log(position, prep.legs, prep.ks, prep.days, result["shares"].to_numpy(), cost_per_share)
              if with_trades else pd.DataFrame())
    return BacktestResult(prep.config, result, trades)


def run(config: StrategyConfig, research: ResearchConfig, data: MarketData,
        start: str | None = None, end: str | None = None) -> BacktestResult:
    prep = prepare(config, data, start, end)
    return simulate(prep, prep.positions(), research)


def trade_log(position: np.ndarray, legs: np.ndarray, ks: np.ndarray, days: pd.Index,
              shares: np.ndarray, cost_per_share: np.ndarray) -> pd.DataFrame:
    decision_time = clock(ks) + ["close"]
    rows = []
    for d, j in zip(*np.nonzero(position)):
        value = position[d, j]
        if j > 0 and position[d, j - 1] == value:
            continue
        exit_j = j + 1
        while exit_j < position.shape[1] and position[d, exit_j] == value:
            exit_j += 1
        entry, exit_ = legs[d, j], legs[d, exit_j]
        side, size = np.sign(value), abs(value)
        n = np.floor(shares[d] * size + 1e-9)
        rows.append((days[d], decision_time[j], decision_time[exit_j], int(side), size, n, entry, exit_,
                     n * side * (exit_ - entry) - 2 * n * cost_per_share[d]))
    return pd.DataFrame(rows, columns=["date", "entry_time", "exit_time", "side", "size", "shares",
                                       "entry_price", "exit_price", "pnl"])
