"""Transaction costs. (Phase 3)

Every fill pays commission + slippage per share (paper section 4.6: $0.0035 Interactive
Brokers entry-level commission + $0.001 slippage measured in live trading). A flip from long
to short is two fills (exit + entry), so it pays twice.

Simplifications, both slightly conservative or negligible for SPY at this size:
- the IB volume discount ($0.002 above 300k shares/month) is not applied;
- the IB minimum ($0.35/order) and maximum (1% of value) per order are ignored.
"""


def trading_cost(shares_traded, cost_per_share: float):
    """Cost of trading `shares_traded` shares (scalar or array)."""
    return shares_traded * cost_per_share
