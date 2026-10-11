"""Predictive-uncertainty diagnostics and split-conformal rescaling for the scalar GPs.

Why: coverage alone can look "good" (99% at nominal 95%) while the predictive sd is simply too wide, and
rows of one run are correlated, so a coverage count is not a count of independent trials. We therefore
report CRPS (a proper score that rewards sharpness AND calibration) and a PIT histogram (uniform if
calibrated), and offer a split-conformal scale factor for the sd.

Protocol (no test leakage): a random 20% of the TRAIN *runs* (by run id) is held out as calibration set; a
GP fitted on the other 80% predicts it; the factor is the conformal quantile of |error| / sd divided by the
Gaussian 97.5% point, so that  factor * sd  gives intervals with the requested coverage on exchangeable
calibration runs. The factor is then applied to the GP fitted on all train runs. Test splits are never used.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def crps_gaussian(y, mu, sd) -> np.ndarray:
    """Closed-form CRPS of N(mu, sd^2) at y (lower is better; same unit as y)."""
    z = (np.asarray(y) - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def pit_values(y, mu, sd) -> np.ndarray:
    return norm.cdf((np.asarray(y) - mu) / sd)


def pit_histogram(y, mu, sd, bins: int = 10) -> list[int]:
    h, _ = np.histogram(pit_values(y, mu, sd), bins=bins, range=(0.0, 1.0))
    return h.tolist()


def conformal_scale(y, mu, sd, level: float = 0.95) -> float:
    """Factor c such that the intervals mu +- 1.96 * c * sd cover `level` of the calibration rows."""
    z = np.sort(np.abs(np.asarray(y) - mu) / sd)
    n = len(z)
    k = min(n, math.ceil((n + 1) * level))
    return float(z[k - 1] / norm.ppf(0.5 + level / 2))


def split_runs(groups: np.ndarray, frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean row masks (proper-train, calibration) from a random run-wise split."""
    runs = np.unique(groups)
    rng = np.random.default_rng(seed)
    cal_runs = rng.choice(runs, size=max(1, int(round(frac * len(runs)))), replace=False)
    cal = np.isin(groups, cal_runs)
    return ~cal, cal
