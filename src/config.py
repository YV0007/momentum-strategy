"""Project paths and the YAML configs as dataclasses."""

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
ENV_FILE = ROOT / ".env"

SYMBOL = "SPY"
START_DATE = "2016-01-01"
NY_TZ = "America/New_York"

CONFIG_DIR = ROOT / "config"
BACKTEST_DIR = RESULTS_DIR / "backtests"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    stop: str = "band_vwap"
    sizing: str = "vol_target"
    leverage: float = 1.0
    gap_adjust: bool = True
    vm: float = 1.0
    lookback: int = 14
    target_vol: float = 0.02
    max_leverage: float = 4.0
    first_decision: str = "10:00"
    decision_every_min: int = 30
    intraday_sizing: str = "none"
    size_floor: float = 0.5
    size_cap: float = 1.5

    def __post_init__(self):
        assert self.stop in ("opposite_band", "band", "vwap", "band_vwap"), self.stop
        assert self.sizing in ("fixed", "vol_target"), self.sizing
        assert self.intraday_sizing in ("none", "turbulence", "ml_vol"), self.intraday_sizing
        assert 0 < self.size_floor <= 1 <= self.size_cap, (self.size_floor, self.size_cap)


@dataclass(frozen=True)
class ResearchConfig:
    train_start: str
    train_end: str
    test_start: str
    post_publication_start: str
    initial_aum: float
    commission: float
    slippage: float
    trading_days: int
    risk_free_rate: float
    commission_tiered: bool = False
    slippage_model: str = "fixed"

    def __post_init__(self):
        assert self.slippage_model in ("fixed", "istar"), self.slippage_model

    @property
    def cost_per_share(self) -> float:
        return self.commission + self.slippage


def processed_files(symbol: str = SYMBOL) -> dict[str, Path]:
    folder = PROCESSED_DIR / symbol
    return {name: folder / f"{name}.parquet" for name in ("minute", "daily", "features_minute", "features_daily")}


def load_universe() -> dict[str, str]:
    return yaml.safe_load((CONFIG_DIR / "universe.yaml").read_text())


def load_research() -> ResearchConfig:
    raw = yaml.safe_load((CONFIG_DIR / "research.yaml").read_text())
    return ResearchConfig(**raw["split"], initial_aum=raw["capital"]["initial_aum"],
                          **raw["costs"], trading_days=raw["metrics"]["trading_days_per_year"],
                          risk_free_rate=raw["metrics"]["risk_free_rate"])


def load_strategies(filename: str = "strategies.yaml") -> dict[str, StrategyConfig]:
    raw = yaml.safe_load((CONFIG_DIR / filename).read_text())
    return {name: StrategyConfig(name=name, **params) for name, params in raw.items()}
