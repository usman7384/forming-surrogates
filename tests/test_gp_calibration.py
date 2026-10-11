"""Determinism of the GP fit, calibration tools and the data-support flag."""
import numpy as np

from surrogates.gp_bo.calibration import conformal_scale, crps_gaussian, pit_histogram, split_runs
from surrogates.gp_bo.gp import HI, LO, fit_gp
from surrogates.gp_bo.support import in_support


def test_fit_is_deterministic_for_fixed_seed_and_threads():
    rng = np.random.default_rng(1)
    X = LO + (HI - LO) * rng.random((20, 5))
    y = np.sin(3 * X[:, 2]) + X[:, 4]
    (g1, i1), (g2, i2) = (fit_gp(X, y, n_restarts=2, iters=25, seed=3, threads=2) for _ in range(2))
    np.testing.assert_allclose(g1.ls, g2.ls, rtol=0, atol=1e-8)
    assert abs(g1.noise - g2.noise) < 1e-8 and abs(g1.outputscale - g2.outputscale) < 1e-8
    Xq = LO + (HI - LO) * rng.random((5, 5))
    np.testing.assert_allclose(g1.predict(Xq)[0], g2.predict(Xq)[0], rtol=0, atol=1e-8)
    assert i1["env"]["torch_threads"] == 2 and {"torch", "gpytorch", "numpy"} <= set(i1["env"])


def test_crps_is_minimised_by_the_true_sd_and_zero_error_limit():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1.0, 20000)
    scores = {s: crps_gaussian(y, 0.0, s).mean() for s in (0.5, 1.0, 2.0)}
    assert scores[1.0] < scores[0.5] and scores[1.0] < scores[2.0]
    assert abs(crps_gaussian(0.0, 0.0, 1.0) - (2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))) < 1e-12


def test_conformal_scale_fixes_overdispersed_sd():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1.0, 4000)
    f = conformal_scale(y, 0.0, np.full_like(y, 2.0))          # sd claimed twice too wide
    assert 0.45 < f < 0.55
    cover = np.mean(np.abs(rng.normal(0, 1.0, 20000)) <= 1.96 * f * 2.0)
    assert abs(cover - 0.95) < 0.01


def test_pit_histogram_flat_when_calibrated():
    rng = np.random.default_rng(0)
    h = np.array(pit_histogram(rng.normal(0, 1, 20000), 0.0, 1.0))
    assert h.min() > 0.85 * h.mean() and h.max() < 1.15 * h.mean()


def test_split_runs_is_by_run_and_deterministic():
    g = np.repeat(np.arange(30), 3)
    p1, c1 = split_runs(g, 0.2, seed=0)
    p2, c2 = split_runs(g, 0.2, seed=0)
    assert (c1 == c2).all() and c1.sum() == 18 and not (p1 & c1).any()
    assert set(g[c1]).isdisjoint(set(g[p1]))


def test_support_flag_fires_only_in_censored_corner():
    # 10 thin high-friction runs that all stopped at 0.2; plenty of healthy runs elsewhere
    sup = dict(window=0.1, min_n=5, fraction=0.5,
               mf=[0.9] * 10 + [0.2] * 10, h_id=[0.45] * 10 + [0.8] * 10, final_reduction=[0.2] * 10 + [0.35] * 10)
    X = np.array([[2.0, 0.45, 0.9, 0.15, 0.3],       # censored corner, asked past where runs stopped
                  [2.0, 0.45, 0.9, 0.15, 0.1],       # same ring but earlier reduction: data exist
                  [2.0, 0.8, 0.2, 0.15, 0.3],        # healthy
                  [2.0, 0.45, 0.2, 0.15, 0.3]])      # no neighbours: rule does not fire
    assert in_support(X, sup).tolist() == [False, True, True, True]
