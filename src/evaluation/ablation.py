"""Builds the final strategy up one decision at a time."""

import pandas as pd

from src.config import ResearchConfig, StrategyConfig
from src.engine.backtest import MarketData, run
from src.evaluation import baselines, metrics
from src.experiment_log import log_run

STOP_LABELS = {"opposite_band": "opposite band", "band": "current band",
               "vwap": "VWAP", "band_vwap": "current band + VWAP"}


def describe(config: StrategyConfig) -> dict:
    sizing = (f"vol target {config.target_vol:.0%}, cap {config.max_leverage:g}x"
              if config.sizing == "vol_target" else f"fixed {config.leverage:g}x")
    return {"band_reference": "open or prev. close" if config.gap_adjust else "open only",
            "stop": STOP_LABELS[config.stop], "sizing": sizing}


def run_ablation(configs: dict[str, StrategyConfig], research: ResearchConfig, data: MarketData,
                 period: str, start: str, end: str) -> pd.DataFrame:
    n, rf = research.trading_days, research.risk_free_rate
    daily = data.daily.loc[start:end]
    rows = {
        "SPY buy & hold": {"band_reference": "-", "stop": "-", "sizing": "1x, always invested",
                           **metrics.summary(baselines.buy_and_hold(daily), n, rf)},
        "SPY open-to-close": {"band_reference": "-", "stop": "close", "sizing": "1x, every day",
                              **metrics.summary(baselines.open_to_close(daily, research), n, rf)},
    }
    for name, config in configs.items():
        result = run(config, research, data, start, end)
        summary = metrics.summary(result.daily, n, rf)
        gross = result.daily["pnl_gross"] / result.daily["aum_start"]
        rows[name] = describe(config) | summary | {"annual_gross": metrics.annual_return(gross, n)}
        log_run(config, period, start, end, summary, note="ablation")
    return pd.DataFrame(rows).T
