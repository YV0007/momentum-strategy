"""Transaction costs. (Phase 3; alternative cost models Phase 8)

Every fill pays commission + slippage per share (paper section 4.6: $0.0035 Interactive
Brokers entry-level commission + $0.001 slippage measured in live trading). A flip from long
to short is two fills (exit + entry), so it pays twice. This flat model is the default.

Alternatives from the paper, switched on in ResearchConfig for the robustness runs:
- commission_tiered (section 4.6): $0.002/share on days when the shares traded over the
  previous month exceed 300,000 (the paper benefits from this on ~20% of its days);
- slippage_model "istar" (FAQ Q15): Kissell's I-Star market impact with the large-cap
  parameters, impact in bps = 687 * (shares / ADV)^0.70 * sigma^0.72, ADV and sigma (annualized)
  over the previous 30 days.
The IB minimum ($0.35/order) and maximum (1% of value) per order are ignored.
"""

import numpy as np

TIER_COMMISSION = 0.002
TIER_SHARES = 300_000
TIER_DAYS = 21                       # one month of trading days
ISTAR_A1, ISTAR_A2, ISTAR_A3 = 687.0, 0.70, 0.72
IMPACT_DAYS = 30


def trading_cost(shares_traded, cost_per_share: float):
    """Cost of trading `shares_traded` shares (scalar or array) at a flat cost per share."""
    return shares_traded * cost_per_share


def commission_rate(research, shares_last_month: float) -> float:
    if research.commission_tiered and shares_last_month > TIER_SHARES:
        return TIER_COMMISSION
    return research.commission


def slippage(research, fills: np.ndarray, prices: np.ndarray, adv: float, vol_annual: float) -> float:
    """Dollar slippage of one day's fills (shares, unsigned) executed at `prices`."""
    if research.slippage_model == "fixed":
        return research.slippage * fills.sum()
    if not fills.any():
        return 0.0
    impact_bps = ISTAR_A1 * (fills / adv) ** ISTAR_A2 * vol_annual ** ISTAR_A3
    return float((impact_bps / 1e4 * prices * fills).sum())
