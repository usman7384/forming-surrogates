"""N1/N2 tests on smoke data: graph structure, normalisation, split by run, J on solver data."""
import numpy as np
import torch

from common.data import load_run
from surrogates.gnn.graph import (BOUNDARY, Normaliser, N_NODE_IN, boundary_flags, build_run_graph, id_change_pct,
                                  j_ratio, load_graphs, make_samples, split_run_ids, undirected_edges)
from surrogates.gnn.physics import j_torch


def test_graph_shapes_and_edges():
    run = load_run("smoke", "run_0000")
    g = build_run_graph(run)
    n, t = g.x.shape[0], g.tri.shape[1]
    assert g.x.shape == (n, N_NODE_IN) and n == 144 and t == 240
    ei = g.edge_index.numpy()
    pairs = set(map(tuple, ei.T))
    assert all((b, a) in pairs for a, b in pairs), "edges must exist in both directions"
    assert (ei[0] != ei[1]).all()
    for k in range(3):                                   # every triangle edge is in the graph
        a, b = g.tri[k].numpy(), g.tri[(k + 1) % 3].numpy()
        assert all((x, y) in pairs for x, y in zip(a, b))
    assert g.edge_attr.shape == (ei.shape[1], 3)
    assert torch.allclose(g.edge_attr[:, 2], g.edge_attr[:, :2].norm(dim=1), atol=1e-6)


def test_boundary_flags_and_probe():
    run = load_run("smoke", "run_0000")
    p0 = run.fields["p"][0]
    f = boundary_flags(p0)
    assert f.shape[1] == len(BOUNDARY)
    assert (f[:, 4] == (f[:, :4].sum(1) == 0)).all(), "interior = no other flag"
    assert f[:, 0].sum() > 0 and f[:, 2].sum() > 0 and f[:, 3].sum() > 0
    # the unit inner radius: ID = 2 so r_min = 1
    assert abs(p0[0].min() - 1.0) < 1e-9


def test_targets_use_initial_mesh_and_probe_matches_curve():
    run = load_run("smoke", "run_0001")                   # stopped early: fewer saved steps
    g = build_run_graph(run)
    S = len(run.fields["step"])
    assert g.disp.shape[0] == S and g.s.shape == (S,) and g.force.shape == (S,)
    assert float(g.disp[0].abs().max()) == 0.0, "displacement at s=0 is zero (Lagrangian)"
    assert 0.0 <= float(g.s.min()) and float(g.s.max()) <= 1.0 + 1e-6
    probe = id_change_pct(run.fields["p"][:, 0, g.inner_mid], run.fields["p"][0, 0, g.inner_mid])
    stored = run.curves["id_change_pct"][run.fields["step"].astype(int)]
    assert np.allclose(probe, stored, atol=1e-6)


def test_normaliser_roundtrip_and_save(tmp_path):
    gs = load_graphs("smoke", "train")[:4]
    nm = Normaliser.fit(gs)
    d = gs[0].disp
    assert torch.allclose(nm.denorm("disp", nm.norm("disp", d)), d, atol=1e-5)
    nm.save(tmp_path / "n.json")
    nm2 = Normaliser.load(tmp_path / "n.json")
    assert torch.allclose(nm2.norm("strain", gs[0].strain[:, :, None][0]), nm.norm("strain", gs[0].strain[:, :, None][0]))
    samples = make_samples(gs, nm)
    allv = torch.cat([s.y_disp for s in samples])
    assert allv.abs().max() < 10 and abs(float(allv.mean())) < 0.5      # roughly standardised


def test_split_by_run_no_leakage():
    tr, va = split_run_ids("smoke", 0.2, seed=0)
    assert set(tr).isdisjoint(va) and va
    gs = load_graphs("smoke", "train", tr)
    assert {g.split for g in gs} == {"train"}
    assert not ({g.run_id for g in gs} & set(va))


def test_j_on_solver_data_is_near_one_and_torch_matches_numpy():
    run = load_run("smoke", "run_0000")
    p, tri = run.fields["p"].astype(float), run.fields["t"].astype(int)
    J = j_ratio(p[1:], p[0], tri)
    assert np.mean(np.abs(J - 1)) < 0.05                   # measured ~0.02 on this run
    assert np.allclose(j_ratio(p[0], p[0], tri), 1.0)
    jt = j_torch(torch.tensor(p[-1].T), torch.tensor(p[0].T), torch.tensor(tri))
    assert np.allclose(jt.numpy(), J[-1], atol=1e-8)
