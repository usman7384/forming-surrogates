"""Model interfaces shared by Phase 2 (producer) and Phase 4 (consumer). CONTRACT: lead-owned.

Phase 4 develops against the stubs below; Phase 2 replaces them by implementing the same protocols in
`surrogates/gp_bo/api.py` and exposing `load(artifact_dir) -> (Surrogate, FrictionEstimator)`.
Check any implementation with:  python -m serving.check_model <module>:<load_function>
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

FEATURES = ["od_id", "h_id", "mf", "m", "reduction"]   # same order as common.data.FEATURES


@runtime_checkable
class Surrogate(Protocol):
    name: str
    version: str

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X: (n, 5) in FEATURES order -> (mean, std) of inner-diameter change in %, each shape (n,)."""
        ...


@runtime_checkable
class FrictionEstimator(Protocol):
    def update(self, ring: dict, reduction: float, measured_id_pct: float, sigma_pct: float) -> None:
        """Bayesian update from one measured inner-diameter change.
        ring: {"od_id", "h_id", "m"} of the measured part."""
        ...

    def mean_std(self) -> tuple[float, float]:
        """Current posterior mean and std of the friction factor m_f."""
        ...


# --------------------------------------------------------------------------- stubs for Phase 4 dev
class StubSurrogate:
    """RBF interpolation of the smoke dataset. Only for building the plant stack before Phase 2 lands."""
    name, version = "stub-rbf", "0"

    def __init__(self, dataset: str = "smoke"):
        from scipy.interpolate import RBFInterpolator
        from common.data import scalar_table
        X, Y, _, _ = scalar_table(dataset, reductions=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3))
        self._lo, self._hi = X.min(0), X.max(0)
        self._rbf = RBFInterpolator(self._norm(X), Y[:, 0], smoothing=1e-3, kernel="thin_plate_spline")
        self._std = float(np.std(Y[:, 0]) * 0.1)

    def _norm(self, X):
        return (np.asarray(X, float) - self._lo) / np.where(self._hi > self._lo, self._hi - self._lo, 1)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self._rbf(self._norm(X)), np.full(len(X), self._std)


class GridFrictionEstimator:
    """Exact grid posterior over m_f in [0, 1] using any Surrogate as the forward model.

    Why the error model matters: one estimator serves one lot (same ring, same rate exponent, reductions close
    together), so the surrogate's error at a given m_f is almost the same offset for every label in the lot, not fresh
    noise. With ``correlated_model_error=True`` (default) the surrogate error is one shared offset b ~ N(0, sd^2) per
    grid point, fully correlated across the lot's labels, plus independent measurement noise per label:
    y_i = mu_i(m_f) + sd_i * z + e_i, z ~ N(0, 1), e_i ~ N(0, sigma_i^2). The joint Gaussian likelihood
    (covariance D + s s^T) is evaluated exactly and still updated one label at a time from three running sums, via
    Sherman-Morrison: r'C^-1 r = A - B^2 / (1 + S), log|C| = sum log sigma_i^2 + log(1 + S), with
    A = sum r_i^2/sigma_i^2, B = sum s_i r_i/sigma_i^2, S = sum s_i^2/sigma_i^2.
    ``correlated_model_error=False`` restores the earlier model (independent error per label), which is
    overconfident when the surrogate error is large, e.g. on out-of-range geometry.
    """

    def __init__(self, surrogate: Surrogate, n: int = 101, prior=None, correlated_model_error: bool = True):
        self.s = surrogate
        self.grid = np.linspace(0.0, 1.0, n)
        self.logprior = np.log(prior if prior is not None else np.full(n, 1.0 / n))
        self.correlated = correlated_model_error
        self.logp = self.logprior.copy()
        self._A = np.zeros(n)      # running sums for the correlated model (see class docstring)
        self._B = np.zeros(n)
        self._S = np.zeros(n)

    def update(self, ring, reduction, measured_id_pct, sigma_pct):
        X = np.column_stack([np.full_like(self.grid, ring["od_id"]), np.full_like(self.grid, ring["h_id"]),
                             self.grid, np.full_like(self.grid, ring["m"]),
                             np.full_like(self.grid, reduction)])
        if not sigma_pct > 0:
            raise ValueError(f"sigma_pct must be positive (got {sigma_pct}); a measurement always has some noise")
        mu, sd = self.s.predict(X)
        r = measured_id_pct - mu
        if not self.correlated:
            var = sigma_pct ** 2 + sd ** 2
            self.logp += -0.5 * r ** 2 / var - 0.5 * np.log(var)
            self.logp -= self.logp.max()
            return
        w = 1.0 / sigma_pct ** 2
        self._A += w * r ** 2
        self._B += w * sd * r
        self._S += w * sd ** 2
        # the sum of log sigma_i^2 is the same at every grid point, so it cancels in the normalised posterior
        self.logp = self.logprior - 0.5 * (self._A - self._B ** 2 / (1.0 + self._S)) - 0.5 * np.log1p(self._S)
        self.logp -= self.logp.max()

    def posterior(self):
        p = np.exp(self.logp)
        return self.grid, p / p.sum()

    def mean_std(self):
        g, p = self.posterior()
        m = float((g * p).sum())
        return m, float(np.sqrt(((g - m) ** 2 * p).sum()))


def load_stub(_artifact_dir: str | None = None):
    s = StubSurrogate()
    return s, GridFrictionEstimator(s)
