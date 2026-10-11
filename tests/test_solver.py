"""Fast regression tests (about 10 s). Run: python -m pytest -q"""
import math

import numpy as np

from ringsim.solver import Numerics, RingCase, RingSolver, homogeneous_force

FAST = Numerics(nz=4, dh_frac=0.05)


def test_frictionless_matches_exact_solution():
    case = RingCase(mf=0.0, reduction=0.3)
    res = RingSolver(case, FAST).run()
    red = res.qoi["final_reduction"]
    assert abs(res.curves["force"][-1] / homogeneous_force(case, red) - 1) < 2e-3
    assert abs(res.qoi["id_change_pct"] - 100 * (math.sqrt(1 / (1 - red)) - 1)) < 0.1
    assert abs(res.qoi["max_strain"] + math.log(1 - red)) < 2e-3
    assert abs(res.qoi["volume_drift_pct"]) < 0.1


def test_shape_is_invariant_to_K_and_speed():
    base = RingSolver(RingCase(mf=0.3, reduction=0.15), FAST).run()
    hard = RingSolver(RingCase(mf=0.3, reduction=0.15, K=7.0), FAST).run()
    fast = RingSolver(RingCase(mf=0.3, reduction=0.15, V=5.0), FAST).run()
    assert abs(hard.qoi["id_change_pct"] - base.qoi["id_change_pct"]) < 1e-6
    assert abs(fast.qoi["id_change_pct"] - base.qoi["id_change_pct"]) < 1e-6
    assert np.isclose(hard.qoi["peak_force"] / base.qoi["peak_force"], 7.0, rtol=1e-6)
    assert np.isclose(fast.qoi["peak_force"] / base.qoi["peak_force"], 5.0 ** 0.15, rtol=1e-6)


def test_more_friction_closes_the_ring():
    low = RingSolver(RingCase(mf=0.05, reduction=0.2), FAST).run()
    high = RingSolver(RingCase(mf=0.6, reduction=0.2), FAST).run()
    assert low.qoi["id_change_pct"] > 0 > high.qoi["id_change_pct"]
    assert high.qoi["peak_force_norm"] > low.qoi["peak_force_norm"]


def test_unstructured_mesh_patch_test_and_boundaries():
    """Frictionless ring on a jittered Delaunay mesh reproduces the exact homogeneous solution."""
    from ringsim.solver import make_mesh
    m = make_mesh(1.0, 2.0, 2.0 / 3.0, 4, "unstructured", seed=3)
    assert np.isclose(m.p[0].min(), 1.0) and np.isclose(m.p[0].max(), 2.0)
    assert np.isclose(m.p[1].min(), 0.0) and np.isclose(m.p[1].max(), 2.0 / 3.0)
    assert len(m.boundary_facets()) == 2 * (6 + 4)     # 6 radial x 4 axial boundary edges, both sides
    case = RingCase(mf=0.0, reduction=0.3)
    res = RingSolver(case, Numerics(nz=4, dh_frac=0.05, mesh="unstructured", mesh_seed=3)).run()
    red = res.qoi["final_reduction"]
    assert abs(res.curves["force"][-1] / homogeneous_force(case, red) - 1) < 5e-4
    assert abs(res.qoi["volume_drift_pct"]) < 0.1
    assert res.numerics["mesh"] == "unstructured"


def test_default_numerics_record_is_unchanged():
    res = RingSolver(RingCase(mf=0.3, reduction=0.05), FAST).run()
    assert "mesh" not in res.numerics and "mesh_seed" not in res.numerics


def test_unstructured_campaign_plan():
    from ringsim.campaign import build_plan_unstructured
    plan = build_plan_unstructured(5, 0, 8)
    assert len(plan) == 5 and {p["split"] for p in plan} == {"test_unstructured"}
    assert len({p["mesh_seed"] for p in plan}) == 5


def test_verify_replot_redraws_without_solving(tmp_path):
    import shutil
    from pathlib import Path
    from ringsim.verify import main as verify
    src = Path(__file__).resolve().parents[1] / "results" / "verify" / "results.json"
    shutil.copy(src, tmp_path / "results.json")
    verify(["--out", str(tmp_path), "--replot"])
    assert (tmp_path / "calibration_chart.png").stat().st_size > 10_000
    assert not (tmp_path / "report.md").exists()


def test_censoring_table_on_smoke_and_empty_calibration_harness(tmp_path):
    import json
    from ringsim.dataset_facts import censoring
    from ringsim.calibration_compare import main as cal
    cells, n, early = censoring("smoke")
    assert n == sum(c["n_runs"] for c in cells) and 0 <= early <= n
    assert all(v is None or 0.0 <= v <= 1.0 for c in cells for v in c["fraction_reached"].values())
    cal(["--out", str(tmp_path)])
    res = json.load(open(tmp_path / "published_ring_comparison.json"))
    assert res["metrics"]["rows"] == [] and "no verified source data" in res["metrics"]["status"]
