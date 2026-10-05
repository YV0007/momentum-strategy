"""Position size as leverage on start-of-day AUM, plus the own versions' per-trade multiplier.
(Phase 3; intraday multiplier Phase 6)

fixed       shares = AUM * leverage / open; 1x is paper eq. 5
vol_target  shares = AUM * min(cap, target / sigma) / open  (paper eq. 6)
            sigma = std of the previous 14 daily returns, so it is known at the open.

Intraday multiplier (own versions), applied on top at each trade's entry and held to its exit:
none        1
turbulence  1 / (realized vol so far today / its 14-day normal at the same time)
ml_vol      1 / ML forecast of (rest-of-day realized vol / its normal)   (strategies/ml_sizing.py)
Both are clipped to [size_floor, size_cap]; the engine also caps leverage x multiplier at max_leverage.
"""

import numpy as np
import pandas as pd

from src.config import StrategyConfig
from src.intraday import intraday_state


def leverage(config: StrategyConfig, vol_daily: pd.Series) -> pd.Series:
    """Leverage per day from the volatility known at the open. NaN (warm-up) = no trading."""
    if config.sizing == "fixed":
        return pd.Series(config.leverage, index=vol_daily.index, name="leverage")
    return np.minimum(config.max_leverage, config.target_vol / vol_daily).rename("leverage")


def shares(aum: float, lev: float, open_price: float) -> int:
    """Whole shares only (fractional shares are not used for SPY day trading)."""
    return int(np.floor(aum * lev / open_price))


def clip_multiplier(raw: np.ndarray, config: StrategyConfig) -> np.ndarray:
    """Clip to the configured bounds; missing values (warm-up) mean no adjustment (1)."""
    m = np.clip(raw, config.size_floor, config.size_cap)
    return np.where(np.isfinite(m), m, 1.0)


def intraday_multiplier(config: StrategyConfig, data, ks: np.ndarray) -> np.ndarray:
    """(all days in data.daily x decision points) size multiplier, before the leverage cap."""
    if config.intraday_sizing == "none":
        return np.ones((len(data.daily), len(ks)))
    if config.intraday_sizing == "turbulence":
        vol_ratio = intraday_state(data, ks)["rel_realized_vol"]
    else:
        from src.strategies import ml_sizing      # local import: ml_sizing builds on the engine's data
        vol_ratio = ml_sizing.frozen_forecast(data, ks)
    with np.errstate(divide="ignore", invalid="ignore"):
        return clip_multiplier(1 / vol_ratio, config)
