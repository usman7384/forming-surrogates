"""Tests for the GP package: numpy posterior == GPyTorch posterior, persistence, API contract, grid posterior."""
import numpy as np
import pytest

from surrogates.gp_bo.gp import HI, LO, NumpyGP, fit_gp


def _toy(n=24, seed=0):
    rng = np.random.default_rng(seed)
    X = LO + (HI - LO) * rng.random((n, 5))
    y = 3 * np.sin(2 * X[:, 2]) - 2 * X[:, 4] + 0.1 * X[:, 0]
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = _toy()
    gp, info = fit_gp(X, y, n_restarts=1, iters=40)
    return X, y, gp, info


def test_numpy_posterior_matches_gpytorch(fitted):
    import gpytorch
    import torch
    X, y, gp, info = fitted
    model, lik, Xu, yt, ymu, ysd = info["_torch"]
    Xq = _toy(7, seed=5)[0]
    from surrogates.gp_bo.gp import to_unit
    model.eval(); lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        pred = lik(model(torch.tensor(to_unit(Xq), dtype=torch.float64)))
        mu_t, sd_t = pred.mean.numpy() * ysd + ymu, pred.stddev.numpy() * ysd
    mu, sd = gp.predict(Xq)
    np.testing.assert_allclose(mu, mu_t, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sd, sd_t, rtol=1e-3, atol=1e-5)


def test_interpolates_training_points(fitted):
    X, y, gp, _ = fitted
    mu, sd = gp.predict(X)
    assert np.max(np.abs(mu - y)) < 0.05 * np.std(y)
    assert np.all(sd > 0)


def test_save_load_roundtrip(fitted, tmp_path):
    X, y, gp, _ = fitted
    gp.save(tmp_path / "g.npz")
    g2 = NumpyGP.load(tmp_path / "g.npz")
    Xq = _toy(5, seed=9)[0]
    np.testing.assert_allclose(gp.predict(Xq)[0], g2.predict(Xq)[0])
    np.testing.assert_allclose(gp.predict(Xq)[1], g2.predict(Xq)[1])


def test_api_satisfies_protocols(fitted, tmp_path):
    from serving.model_api import FrictionEstimator, Surrogate
    from surrogates.gp_bo.api import load
    X, y, gp, _ = fitted
    gp.save(tmp_path / "id_gp.npz")
    s, est = load(str(tmp_path))
    assert isinstance(s, Surrogate) and isinstance(est, FrictionEstimator)
    mu, sd = s.predict(X[:3])
    assert mu.shape == (3,) and sd.shape == (3,)
    with pytest.raises(ValueError):
        s.predict(X[:3, :4])
    est.update(dict(od_id=2.0, h_id=2 / 3, m=0.15), 0.2, float(mu[0]), 0.3)
    m, sdev = est.mean_std()
    assert 0.0 <= m <= 1.0 and sdev > 0


def test_baselines_fit_and_predict():
    from surrogates.baselines.scalar_gbt import GBTBaseline
    from surrogates.baselines.scalar_poly import PolyBaseline
    X, y = _toy(60)
    for B in (PolyBaseline(), GBTBaseline()):
        p = B.fit(X, y).predict(X)
        assert p.shape == y.shape and np.corrcoef(p, y)[0, 1] > 0.9


def test_grid_posterior_recovers_truth_and_id_collapses_ridge():
    """Synthetic forward model: log F = a*mf (weak), ID = -b*mf (strong). Force alone leaves a ridge with K."""
    from surrogates.gp_bo.identifiability import Forward, run_scenario
    nodes = np.linspace(0, 1, 11)
    s = np.linspace(0.05, 0.3, 4)
    logF = 0.15 * nodes[:, None] * (s / 0.3)[None, :]
    idc = 20 - 40 * nodes
    fw = Forward(nodes, logF, idc, np.linspace(0, 1, 201))
    rng = np.random.default_rng(0)
    force = run_scenario("force_only", fw, 0.4, np.log(1.05), rng, noise=False)
    both = run_scenario("force_plus_id", fw, 0.4, np.log(1.05), rng, noise=False)
    assert both["std_mf"] < force["std_mf"] / 5
    assert abs(both["mean_mf"] - 0.4) < 0.02
    assert force["corr"] < -0.5            # friction and strength trade off along the ridge
    assert np.isclose(force["post"].sum(), 1.0)


def test_discrepancy_study_bias_signs_on_synthetic_model():
    from surrogates.gp_bo.identifiability import Forward, discrepancy_study
    nodes = np.linspace(0, 1, 11)
    s = np.linspace(0.03, 0.3, 10)
    logF = 0.15 * nodes[:, None] * (s / 0.3)[None, :]
    idc = 20 - 40 * nodes
    fw = Forward(nodes, logF, idc, np.linspace(0, 1, 201))
    res = [dict(case=dict(mf=float(m)), curves=dict(reduction=[0.0, 0.3], id_min_change_pct=[0.0, 20 - 40 * m - 1.0]))
           for m in nodes]
    out = discrepancy_study(fw, res, np.log(1.05), truths=(0.4,))["true_mf=0.4"]
    assert abs(out["baseline_force_plus_id"]["bias_mf"]) < 1e-3
    # bore ID reads 1 point lower than mid-plane -> looks like more friction (ID falls with m_f)
    assert out["i_bore_id_instead_of_mid_plane"]["bias_mf"] > 0.01
    assert abs(out["ii_force_offset_5pct"]["force_plus_id"]["bias_mf"]) < 5e-3
    assert abs(out["ii_force_offset_5pct"]["force_only"]["bias_mf"]) > abs(out["ii_force_offset_1pct"]["force_only"]["bias_mf"])
