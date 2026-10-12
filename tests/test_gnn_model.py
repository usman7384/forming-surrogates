"""N3/N5/N7 tests: model forward, short training, resume, overfit, MLP baseline, DDP (gloo, 2 CPU procs)."""
import json
import sys

import pytest
import torch
from torch_geometric.data import Batch

from surrogates.gnn.graph import Normaliser, load_graphs, make_samples
from surrogates.gnn.model import MeshGNN
from surrogates.gnn.train import Config, compute_losses, fit, launch_ddp


def _tiny_cfg(**kw):
    base = dict(name="t", latent=16, steps=3, epochs=2, batch_size=8, val_frac=0.2, threads=2, lr=2e-3)
    base.update(kw)
    return Config(**base)


def test_forward_shapes_batch_independence():
    gs = load_graphs("smoke", "train")[:3]
    nm = Normaliser.fit(gs)
    s = make_samples(gs, nm)
    model = MeshGNN(16, 3).eval()
    b = Batch.from_data_list([s[0], s[30]])
    d, e, f = model(b)
    assert d.shape == (b.num_nodes, 2) and e.shape == (b.num_nodes, 1) and f.shape == (2, 1)
    d0, _, f0 = model(Batch.from_data_list([s[0]]))
    assert torch.allclose(d[: s[0].num_nodes], d0, atol=1e-5) and torch.allclose(f[0], f0[0], atol=1e-5)


def test_train_smoke_resume_and_log(tmp_path):
    out = tmp_path / "run"
    fit(_tiny_cfg(epochs=2), "smoke", out)
    fit(_tiny_cfg(epochs=3), "smoke", out)                      # resumes from epoch 2
    rows = [json.loads(l) for l in open(out / "log.jsonl")]
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    assert all(torch.isfinite(torch.tensor(r["train_total"])) for r in rows)
    assert (out / "norm.json").exists() and (out / "best.pt").exists() and (out / "split.json").exists()


def test_overfit_two_graphs():
    """N3 DoD: the network must be able to memorise 2 graphs (loss -> near zero)."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(latent=32, steps=4, epochs=150, batch_size=2, val_frac=0.0, overfit_samples=2, lr=3e-3,
                        lr_final_frac=0.01)
        row = fit(cfg, "smoke", pathlib.Path(td))
    assert row["train_total"] < 0.02, row


def test_jloss_term_is_active():
    gs = load_graphs("smoke", "train")[:2]
    nm = Normaliser.fit(gs)
    b = Batch.from_data_list(make_samples(gs, nm)[:6])
    m = MeshGNN(16, 2)
    l0 = compute_losses(m, b, nm, Config(lambda_j=0.0))
    l1 = compute_losses(m, b, nm, Config(lambda_j=2.0))
    assert float(l1["total"] - l0["total"]) == pytest.approx(2.0 * float(l0["j"]), rel=1e-5)


def test_mlp_baseline_runs():
    from surrogates.baselines.mlp_scalar import run
    metrics, data = run("smoke", members=2, epochs=20)
    assert "test_random" in metrics and metrics["train"]["id_change_pct"]["rmse"] > 0


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed not available")
def test_ddp_gloo_two_cpu_processes(tmp_path):
    rows = launch_ddp(_tiny_cfg(epochs=2, val_frac=0.0, batch_size=4), "smoke", tmp_path / "ddp", world=2)
    assert [r[0] for r in rows] == [0, 1] and all(r[1]["world"] == 2 for r in rows)
    assert (tmp_path / "ddp" / "log.jsonl").exists()
    assert rows[0][2]["param_checksum"] == pytest.approx(rows[1][2]["param_checksum"], rel=1e-9), \
        "DDP replicas must hold identical weights after training"
    assert rows[0][2]["batches"] == rows[1][2]["batches"]


def test_id_loss_and_leakage_check(tmp_path):
    from surrogates.gnn.graph import TEST_SPLITS, case_overlap
    row = fit(_tiny_cfg(epochs=1, lambda_id=1.0), "smoke", tmp_path / "r")
    assert row["train_id"] > 0 and row["train_total"] > row["train_disp"]
    assert case_overlap("smoke", "train", "smoke", ("train",))          # a dataset overlaps itself: detector works
    assert not case_overlap("smoke", "train", "smoke", TEST_SPLITS)      # smoke train vs its own tests: disjoint


def test_evaluate_and_gallery_on_smoke(tmp_path):
    from surrogates.gnn.evaluate import evaluate
    run = tmp_path / "run"
    fit(_tiny_cfg(epochs=1), "smoke", run)
    res = evaluate(run, "smoke", tmp_path / "res")
    m = res["metrics"]
    assert "test_random" in m and "test_ood_geometry" in m and m["test_random"]["nrmse_dr"] > 0
    assert 1 <= len(m["failure_gallery"]) <= 10 and all("hypothesis (untested)" in g for g in m["failure_gallery"])
    assert (tmp_path / "res" / "figures" / "training_curves.png").exists()
