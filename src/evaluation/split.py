"""Train/test periods and walk-forward folds."""

import numpy as np
import pandas as pd

from src.config import ResearchConfig


def periods(research: ResearchConfig) -> dict[str, tuple[str, str | None]]:
    return {"train": (research.train_start, research.train_end),
            "test": (research.test_start, None),
            "post_publication": (research.post_publication_start, None)}


def walk_forward_folds(days: pd.DatetimeIndex, n_folds: int = 4,
                       min_train_years: int = 2) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    first_val = days[0] + pd.DateOffset(years=min_train_years)
    blocks = np.array_split(days[days >= first_val], n_folds)
    return [(days[days < block[0]], block) for block in blocks]
