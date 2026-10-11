"""Closed-form checks of serving.model_api.GridFrictionEstimator (the plant's friction posterior)."""
import numpy as np
from scipy.stats import multivariate_normal

from serving.model_api import GridFrictionEstimator


class _FakeSurrogate:
    """Deterministic forward model: mean and std depend on m_f and reduction only."""
    name, version = "fake", "0"

    def __init__(self, sd_scale=0.5):
        self.sd_scale = sd_scale

    def predict(self, X):
        mf, red = X[:, 2], X[:, 4]
        mu = 20.0 - 40.0 * mf * (red / 0.3)
        sd = self.sd_scale * (0.5 + mf) * (0.6 + red)      # varies per label through the reduction
        return mu, sd


RING = {"od_id": 2.0, "h_id": 0.667, "m": 0.15}


def _brute_force_posterior(sur, grid, labels):
    """Joint Gaussian likelihood per grid point with covariance diag(sigma^2) + s s^T (fully correlated error)."""
    logp = np.zeros(len(grid))
    for k, mf in enumerate(grid):
        X = np.array([[RING["od_id"], RING["h_id"], mf, RING["m"], red] for red, _, _ in labels])
        mu, sd = sur.predict(X)
        y = np.array([v for _, v, _ in labels])
        cov = np.diag([s ** 2 for _, _, s in labels]) + np.outer(sd, sd)
        logp[k] = multivariate_normal(mu, cov).logpdf(y)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def test_correlated_update_equals_joint_gaussian():
    sur = _FakeSurrogate(sd_scale=0.7)
    est = GridFrictionEstimator(sur, n=41)
    labels = [(0.28, 9.1, 0.3), (0.30, 8.2, 0.3), (0.31, 7.7, 0.5), (0.29, 8.9, 0.3)]
    for red, y, s in labels:
        est.update(RING, red, y, s)
    _, p = est.posterior()
    np.testing.assert_allclose(p, _brute_force_posterior(sur, est.grid, labels), atol=1e-10)


def test_zero_measurement_noise_is_rejected():
    import pytest
    est = GridFrictionEstimator(_FakeSurrogate(), n=11)
    with pytest.raises(ValueError):
        est.update(RING, 0.3, 8.0, 0.0)


def test_single_label_matches_independent_model():
    sur = _FakeSurrogate()
    a, b = GridFrictionEstimator(sur, n=51), GridFrictionEstimator(sur, n=51, correlated_model_error=False)
    for e in (a, b):
        e.update(RING, 0.3, 8.0, 0.3)
    np.testing.assert_allclose(a.posterior()[1], b.posterior()[1], atol=1e-12)


def test_repeated_labels_do_not_shrink_below_the_model_error():
    """With surrogate error >> measurement noise, many labels cannot beat the shared model error;
    the independent model wrongly keeps shrinking roughly like 1/sqrt(n)."""
    sur = _FakeSurrogate(sd_scale=3.0)
    corr, ind = GridFrictionEstimator(sur, n=401), GridFrictionEstimator(sur, n=401, correlated_model_error=False)
    for _ in range(10):
        for e in (corr, ind):
            e.update(RING, 0.3, 8.0, 0.3)
    s_corr, s_ind = corr.mean_std()[1], ind.mean_std()[1]
    assert s_corr > 2.5 * s_ind
