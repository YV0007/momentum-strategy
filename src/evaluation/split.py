"""Train/test periods and time-series cross-validation folds. (Phase 5)"""

import numpy as np
import pandas as pd

from src.config import ResearchConfig


def periods(research: ResearchConfig) -> dict[str, tuple[str, str | None]]:
    """Named (start, end) date ranges; end None = up to the last day of data."""
    return {"train": (research.train_start, research.train_end),
            "test": (research.test_start, None),
            "post_publication": (research.post_publication_start, None)}


def walk_forward_folds(days: pd.DatetimeIndex, n_folds: int = 4,
                       min_train_years: int = 2) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window folds inside one period (used only within train).

    The days after the first `min_train_years` are cut into n_folds consecutive validation
    blocks; each fold trains on every day before its block. Validation always comes after
    training, so no fold sees its own future.
    """
    first_val = days[0] + pd.DateOffset(years=min_train_years)
    blocks = np.array_split(days[days >= first_val], n_folds)
    return [(days[days < block[0]], block) for block in blocks]
