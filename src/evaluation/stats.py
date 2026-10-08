"""Statistical tests on Sharpe ratios."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

EULER_GAMMA = 0.5772156649


def _daily_sharpe(ret: np.ndarray) -> float:
    return ret.mean() / ret.std(ddof=1)


def sharpe_se(ret: pd.Series, n: int = 252) -> float:
    sr = _daily_sharpe(ret.to_numpy())
    skew, kurt = stats.skew(ret), stats.kurtosis(ret, fisher=False)
    var = (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) / 4 * sr**2) / len(ret)
    return np.sqrt(var) * np.sqrt(n)


def _block_indices(n_obs: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs - block + 1, size=(reps, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n_obs]


def _sharpes(samples: np.ndarray, n: int) -> np.ndarray:
    return samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(n)


def bootstrap_ci(ret: pd.Series, n: int = 252, block: int = 20, reps: int = 2000,
                 level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    x = ret.to_numpy()
    sharpes = _sharpes(x[_block_indices(len(x), block, reps, np.random.default_rng(seed))], n)
    tail = (1 - level) / 2
    return tuple(np.quantile(sharpes, [tail, 1 - tail]))


def sharpe_difference(a: pd.Series, b: pd.Series, n: int = 252, block: int = 20, reps: int = 2000,
                      level: float = 0.95, seed: int = 0) -> dict:
    df = pd.concat([a, b], axis=1).dropna().to_numpy()
    idx = _block_indices(len(df), block, reps, np.random.default_rng(seed))
    diff = _sharpes(df[:, 0][idx], n) - _sharpes(df[:, 1][idx], n)
    point = _sharpes(df[:, :1].T, n)[0] - _sharpes(df[:, 1:].T, n)[0]
    tail = (1 - level) / 2
    return {"difference": point, "ci_low": np.quantile(diff, tail), "ci_high": np.quantile(diff, 1 - tail),
            "share_of_resamples_positive": (diff > 0).mean()}


def deflated_sharpe(ret: pd.Series, trial_sharpes: list[float]) -> float:
    sr = _daily_sharpe(ret.to_numpy())
    n_trials = len(trial_sharpes)
    if n_trials > 1:
        sd = np.std(trial_sharpes, ddof=1)
        expected_max = sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                             + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    else:
        expected_max = 0.0
    skew, kurt = stats.skew(ret), stats.kurtosis(ret, fisher=False)
    z = (sr - expected_max) * np.sqrt(len(ret) - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return stats.norm.cdf(z)


def alpha_beta(ret: pd.Series, market: pd.Series, n: int = 252) -> dict:
    df = pd.concat([ret, market], axis=1, keys=["y", "x"]).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {"alpha": fit.params["const"] * n, "alpha_t": fit.tvalues["const"],
            "beta": fit.params["x"], "beta_t": fit.tvalues["x"]}


def empirical_p_value(value: float, null_values: np.ndarray) -> float:
    return (np.sum(null_values >= value) + 1) / (len(null_values) + 1)
