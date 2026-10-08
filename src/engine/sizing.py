"""Position size: daily leverage and the per-trade multiplier."""

import numpy as np
import pandas as pd

from src.config import StrategyConfig
from src.intraday import intraday_state


def leverage(config: StrategyConfig, vol_daily: pd.Series) -> pd.Series:
    if config.sizing == "fixed":
        return pd.Series(config.leverage, index=vol_daily.index, name="leverage")
    return np.minimum(config.max_leverage, config.target_vol / vol_daily).rename("leverage")


def shares(aum: float, lev: float, open_price: float) -> int:
    return int(np.floor(aum * lev / open_price))


def clip_multiplier(raw: np.ndarray, config: StrategyConfig) -> np.ndarray:
    m = np.clip(raw, config.size_floor, config.size_cap)
    return np.where(np.isfinite(m), m, 1.0)


def intraday_multiplier(config: StrategyConfig, data, ks: np.ndarray) -> np.ndarray:
    if config.intraday_sizing == "none":
        return np.ones((len(data.daily), len(ks)))
    if config.intraday_sizing == "turbulence":
        vol_ratio = intraday_state(data, ks)["rel_realized_vol"]
    else:
        from src.strategies import ml_sizing
        vol_ratio = ml_sizing.frozen_forecast(data, ks)
    with np.errstate(divide="ignore", invalid="ignore"):
        return clip_multiplier(1 / vol_ratio, config)
