"""Design objective and BO loop on cheap synthetic forward models (no solver, no trained GP)."""
import numpy as np

from surrogates.gp_bo import design
from surrogates.gp_bo.inverse import invert_with_gp


class _FakeGP:
    """ID = 20 - slope(od) * mf: a bigger OD/ID makes the hole more sensitive to friction."""

    def predict(self, X):
        X = np.atleast_2d(X)
        mu = 20.0 - 25.0 * X[:, 0] * X[:, 2]
        return mu, np.full(len(X), 0.05)


def test_steeper_ring_has_smaller_expected_std():
    gp = _FakeGP()
    assert design.objective_gp(gp, 2.4, 0.7) < design.objective_gp(gp, 1.6, 0.7)


def test_flat_response_gives_prior_wide_std():
    class Flat:
        def predict(self, X):
            return np.zeros(len(X)), np.full(len(X), 0.01)
    # uniform prior on [0, 1] has std 0.289
    assert 0.25 < design.objective_gp(Flat(), 2.0, 0.7) < 0.3


def test_bo_improves_on_its_initial_design():
    f = lambda u: float((u[0] - 0.8) ** 2 + (u[1] - 0.3) ** 2)
    X, Y, best = design.run_bo(f, n_init=4, n_iter=4, seed=0)
    assert len(Y) == 8 and f(best) <= Y[:4].min() + 1e-12


def test_brentq_on_gp_mean_recovers_known_friction():
    mf_true = 0.37
    target = float(_FakeGP().predict(np.array([[2.0, 2 / 3, mf_true, 0.15, 0.3]]))[0][0])
    root = invert_with_gp(_FakeGP(), target)
    assert abs(root - mf_true) < 1e-5


def test_solver_objective_refuses_to_extrapolate_and_common_range_is_consistent():
    nodes = np.linspace(0, 0.9, 10)
    idc = 20 - 40 * nodes
    assert design.objective_solver(nodes, idc, 1.0) is None
    a = design.objective_solver(nodes, idc, 0.9)
    assert a is not None and 0 < a < 0.3
    assert design.objective_solver_extrapolated(nodes, idc) > 0
