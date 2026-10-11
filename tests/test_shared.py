"""Tests for the shared contracts (data loader, model API). Run: python -m pytest -q"""
import numpy as np

from common.data import FEATURES, iter_runs, list_runs, load_run, scalar_table


def test_smoke_dataset_loads():
    rows = list_runs("smoke")
    assert len(rows) >= 10
    run = load_run("smoke", rows[0]["run_id"])
    S = run.fields["p"].shape[0]
    assert run.fields["p"].shape[1] == 2 and run.fields["strain"].shape[0] == S
    assert len(run.curves["reduction"]) == len(run.curves["force"])


def test_values_beyond_stop_are_nan():
    for run in iter_runs("smoke"):
        v = run.at([run.max_reduction + 0.05])
        assert np.isnan(v[0])


def test_scalar_table_shapes_and_groups():
    X, Y, G, S = scalar_table("smoke", reductions=(0.1, 0.2, 0.3))
    assert X.shape[1] == len(FEATURES) and len(X) == len(Y) == len(G) == len(S)
    assert not np.isnan(Y).any()


def test_stub_model_contract():
    from serving.check_model import main
    main(["serving.model_api:load_stub"])
