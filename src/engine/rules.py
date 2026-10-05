"""Entry and stop logic: target position at each decision point. (Phase 3)

All inputs are (days x decision points) arrays observed at the decision time.
Output values: +1 long, -1 short, 0 flat. Missing inputs (NaN) never trigger a trade.
"""

import numpy as np


def opposite_band(price, upper, lower) -> np.ndarray:
    """Paper base model: enter on a band break, hold until the price breaks the OPPOSITE band,
    then flip. Positions start flat each day and are only changed by a break."""
    signal = np.where(price > upper, 1.0, np.where(price < lower, -1.0, np.nan))
    # Carry the last signal forward within the day (rows are days, columns are time).
    position = np.zeros_like(signal)
    current = np.zeros(signal.shape[0])
    for j in range(signal.shape[1]):
        current = np.where(np.isnan(signal[:, j]), current, signal[:, j])
        position[:, j] = current
    return position


def band_vwap(price, upper, lower, vwap) -> np.ndarray:
    """Paper final model: long while price > max(upper band, VWAP), short while
    price < min(lower band, VWAP), flat otherwise.

    This is the paper's entry + trailing stop written as one rule: entering requires a band
    break, and the long trailing stop max(upper, VWAP) means a long can only be held while the
    price is above both. After a stop-out the position can re-enter at a later decision point."""
    long = price > np.fmax(upper, vwap)
    short = price < np.fmin(lower, vwap)
    # NaN comparisons are False, so missing features give a flat position.
    return np.where(long, 1.0, np.where(short, -1.0, 0.0))


def band_only(price, upper, lower) -> np.ndarray:
    """Paper Fig. 5a: the CURRENT band is the trailing stop. Long while price > upper band,
    short while price < lower band, flat inside the noise area."""
    return np.where(price > upper, 1.0, np.where(price < lower, -1.0, 0.0))


def vwap_only(price, upper, lower, vwap) -> np.ndarray:
    """Paper FAQ Q22: enter on a band break (on the right side of VWAP), then hold until price
    crosses VWAP, even if it falls back inside the noise area."""
    position = np.zeros_like(price)
    current = np.zeros(price.shape[0])
    for j in range(price.shape[1]):
        p, u, lo, v = price[:, j], upper[:, j], lower[:, j], vwap[:, j]
        stay_long = (current > 0) & (p > v)
        stay_short = (current < 0) & (p < v)
        go_long = stay_long | ((p > u) & (p > v))
        go_short = stay_short | ((p < lo) & (p < v))
        current = np.where(go_long, 1.0, np.where(go_short, -1.0, 0.0))
        position[:, j] = current
    return position


def target_positions(stop: str, price, upper, lower, vwap) -> np.ndarray:
    if stop == "opposite_band":
        return opposite_band(price, upper, lower)
    if stop == "band":
        return band_only(price, upper, lower)
    if stop == "vwap":
        return vwap_only(price, upper, lower, vwap)
    if stop == "band_vwap":
        return band_vwap(price, upper, lower, vwap)
    raise ValueError(f"unknown stop type: {stop}")
