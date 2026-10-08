"""Trading costs per fill."""

import numpy as np

TIER_COMMISSION = 0.002
TIER_SHARES = 300_000
TIER_DAYS = 21
ISTAR_A1, ISTAR_A2, ISTAR_A3 = 687.0, 0.70, 0.72
IMPACT_DAYS = 30


def trading_cost(shares_traded, cost_per_share: float):
    return shares_traded * cost_per_share


def commission_rate(research, shares_last_month: float) -> float:
    if research.commission_tiered and shares_last_month > TIER_SHARES:
        return TIER_COMMISSION
    return research.commission


def slippage(research, fills: np.ndarray, prices: np.ndarray, adv: float, vol_annual: float) -> float:
    if research.slippage_model == "fixed":
        return research.slippage * fills.sum()
    if not fills.any():
        return 0.0
    impact_bps = ISTAR_A1 * (fills / adv) ** ISTAR_A2 * vol_annual ** ISTAR_A3
    return float((impact_bps / 1e4 * prices * fills).sum())
