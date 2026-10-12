"""P4/P7: edge processing offline (no OPC-UA), end-to-end over OPC-UA, and repro bundles that really re-run."""
import asyncio
import json
import socket
import time

import numpy as np
import pytest

from plant.bundle import write_bundle
from plant.config import PlantConfig
from plant.opcua_server import PressServer
from plant.press_sim import PressSimulator, load_pool
from plant.rerun_bundle import rerun
from serving.edge.align import align, clean, features, find_contact
from serving.edge.service import EdgeService, run_edge
from plant.spec import load_spec


@pytest.fixture(scope="module")
def model():
    from serving.model_api import load_stub
    return load_stub()


@pytest.fixture(scope="module")
def pool():
    return load_pool(PlantConfig())


def _svc(model, **kw):
    return EdgeService(PlantConfig(**kw), model)


def test_clean_drops_bad_quality_and_sorts():
    t = np.array([0.3, 0.1, 0.2, 0.2, 0.4])
    v = np.array([3, 1, 2, 9, 1000.0])
    q = np.array([0, 0, 0, 0, 2.0])
    tt, vv, unc = clean(t, v, q)
    assert list(tt) == [0.1, 0.2, 0.3] and 1000.0 not in vv


def test_find_contact_requires_debounce():
    f = np.zeros(50)
    f[10] = 9.0                  # single spike: noise
    f[30:] = 9.0
    assert find_contact(f, 5.0, 3) == 30
    assert find_contact(np.zeros(20), 5.0, 3) is None


def test_features_recover_reduction_within_noise(model, pool):
    sim = PressSimulator(PlantConfig(scenario="nominal"), pool)
    proc = load_spec().process
    errs = []
    for i in range(0, 40, 4):
        p = sim.part(i)
        f = features(align(p.streams), p.ticket, proc, 5.0, 3)
        errs.append(f.reduction - p.truth["reduction"])
        assert not f.contact_anomaly
        assert abs(f.f_zero_kN) < 0.5
    assert np.abs(errs).max() < 0.01          # touch-off detection error well under one percent of height


def test_process_part_verdict_and_latency_fields(model, pool):
    cfg = PlantConfig(scenario="nominal")
    sim, svc = PressSimulator(cfg, pool), EdgeService(cfg, model)
    p = sim.part(0)
    rec = svc.process_part(p.ticket, p.streams, time.time_ns())
    assert rec["verdict"] in {"go", "check", "stop"}
    assert rec["latency_ms"] >= 0 and rec["edge_ms"] > 0
    assert np.isfinite(rec["pred_mean"]) and rec["pred_std"] > 0
    assert rec["verdict"] == "check"          # no label yet: friction unknown, so the +-2 sd interval is wide


def test_friction_posterior_is_per_lot_and_learns_from_labels(model, pool):
    cfg = PlantConfig(scenario="nominal", lot_len=20)
    sim, svc = PressSimulator(cfg, pool), EdgeService(cfg, model)
    for i in range(45):
        for lab in sim.labels_due(i):
            svc.on_label(lab)
        p = sim.part(i)
        svc.process_part(p.ticket, p.streams, time.time_ns())
    first_lot_start = svc.records[0]["mf_std"]
    assert first_lot_start > 0.25                           # flat prior at the start of lot 0
    assert svc.records[20]["mf_std"] > 0.25                 # lot 1 starts fresh, not with lot 0's posterior
    est0 = svc.est_by_lot[0].mean_std()
    assert est0[1] < 0.1                                    # learned from the delayed labels
    assert abs(est0[0] - sim.lot(0)["run"].case["mf"]) < 0.15
    assert svc._pristine.mean_std()[1] > 0.25               # the pristine estimator was never mutated


def test_range_guard_flags_extrapolation(model, pool):
    cfg = PlantConfig(scenario="drift", n_parts=40, drift_slope_kN_per_part=1.0, drift_start_frac=0.1)
    sim, svc = PressSimulator(cfg, pool), EdgeService(cfg, model)
    p = sim.part(30)                                        # offset 27 kN: contact "found" at the start of approach
    rec = svc.process_part(p.ticket, p.streams, time.time_ns())
    assert rec["reduction_est"] > 0.35 and rec["reason"] == "reduction_outside_model_range"


def test_verdict_logic_go_check_stop(model):
    svc = _svc(model)
    tk = dict(target_id_change_pct=0.0, tol_id_pct=2.0)
    assert svc.verdict(0.2, 0.3, tk, False) == ("go", "inside_spec")                 # +-0.6 inside +-2
    assert svc.verdict(0.2, 0.3, tk, True)[0] == "check"                              # data-quality guard
    assert svc.verdict(1.5, 0.8, tk, False)[0] == "check"                             # interval crosses the limit
    assert svc.verdict(6.0, 0.5, tk, False) == ("stop", "predicted_out_of_spec")      # interval entirely outside
    assert svc.verdict(float("nan"), float("nan"), tk, False)[0] == "check"


def test_deterministic_replay_gives_identical_predictions(model, pool):
    def run():
        cfg = PlantConfig(scenario="drift", n_parts=30, seed=5)
        sim, svc = PressSimulator(cfg, pool), EdgeService(cfg, model)
        for i in range(30):
            for lab in sim.labels_due(i):
                svc.on_label(lab)
            p = sim.part(i)
            svc.process_part(p.ticket, p.streams, time.time_ns())
        return np.array([r["pred_mean"] for r in svc.records])
    assert np.array_equal(run(), run())


def test_end_to_end_over_opcua(model, pool, tmp_path):
    cfg = PlantConfig(scenario="nominal", n_parts=12, lot_len=6)
    sim = PressSimulator(cfg, pool)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        ep = f"opc.tcp://127.0.0.1:{s.getsockname()[1]}/t/"

    async def go():
        srv = PressServer(ep)
        await srv.start()
        try:
            shift = asyncio.create_task(srv.run_shift(sim, pace_s=0.05))
            svc = await run_edge(ep, cfg, EdgeService(cfg, model), out_dir=tmp_path, idle_timeout_s=20)
            await shift
            return svc
        finally:
            await srv.stop()

    svc = asyncio.run(go())
    assert [r["part_id"] for r in svc.records] == list(range(12))
    assert not [e for e in svc.events if e["type"] == "part_lost"]
    # the OPC-UA path must give the same reduction estimate as the offline path for the same sensor data
    ref = EdgeService(cfg, model)
    p = sim.part(3)
    r_off = ref.process_part(p.ticket, p.streams, time.time_ns())
    assert svc.records[3]["reduction_est"] == pytest.approx(r_off["reduction_est"], abs=1e-9)
    assert (tmp_path / "parts.parquet").exists() and (tmp_path / "events.jsonl").exists()
    assert all(r["latency_ms"] < 6000 for r in svc.records)


def test_bundle_reruns_in_ringsim(model, pool, tmp_path):
    cfg = PlantConfig(scenario="nominal", bundle_nz=4)
    sim, svc = PressSimulator(cfg, pool), EdgeService(cfg, model)
    p = sim.part(0)
    rec = svc.process_part(p.ticket, p.streams, time.time_ns())
    label = dict(part_id=0, id_change_pct=rec["pred_mean"] + 5.0, sigma_pct=0.3)       # a forced miss
    d = write_bundle(tmp_path / "b0", rec=rec, ticket=p.ticket, streams=p.streams, label=label, cfg=cfg,
                     model_name=model[0].name, model_version=model[0].version, drift_alerts=[], reason="test")
    for f in ("bundle.json", "case.json", "signals.parquet", "plot.png"):
        assert (d / f).exists(), f
    info = json.loads((d / "bundle.json").read_text())
    assert info["config_hash"] == cfg.hash() and info["rerun_command"].startswith("python -m plant.rerun_bundle")
    out = rerun(d)                                                                       # runs ringsim on case.json
    assert np.isfinite(out["solver_id_change_pct"]) and not out["stopped_early"]
    # the inferred inputs reproduce the model's own prediction to within the surrogate's accuracy, not wildly off
    assert abs(out["solver_id_change_pct"] - rec["pred_mean"]) < 6.0
    assert out["case"]["reduction"] == pytest.approx(rec["reduction_est"])
