"""Entry and stop rules: long, short or flat at each decision."""

import numpy as np


def opposite_band(price, upper, lower) -> np.ndarray:
    signal = np.where(price > upper, 1.0, np.where(price < lower, -1.0, np.nan))
    position = np.zeros_like(signal)
    current = np.zeros(signal.shape[0])
    for j in range(signal.shape[1]):
        current = np.where(np.isnan(signal[:, j]), current, signal[:, j])
        position[:, j] = current
    return position


def band_vwap(price, upper, lower, vwap) -> np.ndarray:
    long = price > np.fmax(upper, vwap)
    short = price < np.fmin(lower, vwap)
    return np.where(long, 1.0, np.where(short, -1.0, 0.0))


def band_only(price, upper, lower) -> np.ndarray:
    return np.where(price > upper, 1.0, np.where(price < lower, -1.0, 0.0))


def vwap_only(price, upper, lower, vwap) -> np.ndarray:
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
