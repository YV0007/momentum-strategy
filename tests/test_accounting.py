"""Bookkeeping invariants on a real-data backtest: money is neither created nor lost."""

import pytest

from src.config import load_research, load_strategies
from src.engine.backtest import MarketData, decision_minutes, run

R = load_research()


@pytest.fixture(scope="module", params=["base", "final"])
def result(request, real_market):
    try:
        data = MarketData.load()
    except FileNotFoundError:
        pytest.skip("processed data not built")
    return run(load_strategies()[request.param], R, data, "2016-01-01", "2019-12-31"), data


def test_aum_equals_initial_plus_all_pnl(result):
    result, _ = result
    d = result.daily
    assert d["aum"].iloc[-1] == pytest.approx(R.initial_aum + (d["pnl_gross"] - d["costs"]).sum())
    assert (d["aum_start"].iloc[1:].values == pytest.approx(d["aum"].iloc[:-1].values))


def test_trade_log_reconciles_with_daily_pnl(result):
    result, _ = result
    d = result.daily
    assert result.trades["pnl"].sum() == pytest.approx((d["pnl_gross"] - d["costs"]).sum())
    assert d["trades"].sum() == len(result.trades)


def test_trades_only_at_decision_times_and_flat_at_close(result):
    result, _ = result
    ks = decision_minutes(result.config)
    allowed = {f"{(570 + k) // 60:02d}:{(570 + k) % 60:02d}" for k in ks}
    assert set(result.trades["entry_time"]) <= allowed
    assert set(result.trades["exit_time"]) <= allowed | {"close"}
    assert result.trades["exit_time"].notna().all()


def test_notional_never_exceeds_leverage_cap(result):
    result, data = result
    d = result.daily
    notional = d["shares"] * data.daily.loc[d.index, "open"]
    assert (notional <= d["aum_start"] * result.config.max_leverage + 1e-6).all()
    assert d["leverage"].max() <= result.config.max_leverage


def test_no_trading_on_invalid_or_warmup_days(result):
    result, _ = result
    d = result.daily
    assert d.iloc[:11]["trades"].sum() == 0    # sigma needs >= 11 of the last 14 days
    if "2019-08-12" in d.index:
        assert d.loc["2019-08-12", "trades"] == 0
