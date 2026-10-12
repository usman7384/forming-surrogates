"""Drift types, re-zeroing recovery, the data-support hook, label ordering at high speed, and the experiment harness."""
import asyncio
import socket
import time

import numpy as np
import pytest

from plant.config import PlantConfig, drift_type
from plant.experiments import aggregate, cluster_bootstrap, run_all
from plant.opcua_server import PressServer
from plant.press_sim import PressSimulator, load_pool
from serving.edge.monitoring import DriftMonitor
from serving.edge.service import EdgeService, run_edge


@pytest.fixture(scope="module")
def pool():
    return load_pool(PlantConfig())


@pytest.fixture(scope="module")
def model():
    from serving.model_api import load_stub
    return load_stub()


def _shift(cfg, model, pool, n=None):
    sim, svc = PressSimulator(cfg, pool), EdgeService(cfg, model)
    for i in range(n or cfg.n_parts):
        for lab in sim.labels_due(i):
            svc.on_label(lab)
        p = sim.part(i)
        svc.process_part(p.ticket, p.streams, time.time_ns())
    return sim, svc


def test_scenario_names_and_back_compat():
    assert drift_type("drift") == "offset" == drift_type("drift_offset")
    assert drift_type("drift_gain") == "gain" and drift_type("drift_friction") == "friction" and drift_type("nominal") is None
    with pytest.raises(ValueError):
        drift_type("drift_banana")


def test_each_drift_touches_only_its_own_channel(pool):
    kw = dict(n_parts=100, drift_start_frac=0.5, drift_slope_kN_per_part=0.1, drift_gain_per_part=0.01)
    off = PressSimulator(PlantConfig(scenario="drift_offset", **kw), pool)
    gain = PressSimulator(PlantConfig(scenario="drift_gain", **kw), pool)
    assert off.force_offset_kN(59) == pytest.approx(1.0) and off.force_gain(59) == 1.0
    assert gain.force_gain(49) == 1.0 and gain.force_gain(59) == pytest.approx(1.1) and gain.force_offset_kN(59) == 0.0
    # gain scales the loaded force, leaves the pre-contact zero alone
    from serving.edge.align import clean
    nom = clean(*PressSimulator(PlantConfig(scenario="nominal", **kw), pool).part(59).streams["stroke_force"])[1]
    g = clean(*gain.part(59).streams["stroke_force"])[1]
    assert g.max() > 1.05 * nom.max() and abs(np.median(g[:50])) < 1.5


def test_friction_drift_raises_true_friction_after_start_only(pool):
    kw = dict(n_parts=600, seed=3, lot_len=50)
    nom = PressSimulator(PlantConfig(scenario="nominal", **kw), pool)
    fr = PressSimulator(PlantConfig(scenario="drift_friction", **kw), pool)
    assert all(nom.lot(k)["run"].run_id == fr.lot(k)["run"].run_id for k in range(6))        # before the start: identical
    mf_n = np.mean([nom.lot(k)["run"].case["mf"] for k in range(6, 12)])
    mf_f = np.mean([fr.lot(k)["run"].case["mf"] for k in range(6, 12)])
    assert mf_f >= mf_n
    assert fr.part(310).truth["mf"] == fr.lot(6)["run"].case["mf"]


def test_gain_drift_is_seen_by_a_non_injected_feature_not_the_zero_feature(model, pool):
    cfg = PlantConfig(scenario="drift_gain", n_parts=500, drift_start_frac=0.6, drift_gain_per_part=0.004)
    sim, svc = _shift(cfg, model, pool)
    first = lambda f: svc.monitor.first_alert(f)                                     # noqa: E731
    pre = [a for a in svc.monitor.alerts if a["feature"] == "peak_rel_lot" and a["part_index"] < sim.drift_start]
    post = [a["part_index"] for a in svc.monitor.alerts if a["feature"] == "peak_rel_lot" and a["part_index"] >= sim.drift_start]
    assert not pre and post and post[0] - sim.drift_start < 60
    assert first("f_zero_kN") is None                                                # the zero reading does not see a gain error


def test_monitor_effect_size_gate():
    rng = np.random.default_rng(0)
    gated = DriftMonitor(features=("a",), primary="a", min_shift={"a": 0.5})
    free = DriftMonitor(features=("a",), primary="a")
    for i in range(200):
        v = rng.normal(0, 0.01) if i < 100 else rng.normal(0.1, 0.01)             # tiny but "significant" shift
        gated.add(i, {"a": v})
        free.add(i, {"a": v})
    assert free.alerts and not gated.alerts


def test_rezero_recovers_accuracy_after_zero_offset_alert(model, pool):
    base = dict(scenario="drift_offset", n_parts=500, drift_start_frac=0.4, drift_slope_kN_per_part=0.1, seed=2)
    _, none = _shift(PlantConfig(recovery="none", **base), model, pool)
    sim, rz = _shift(PlantConfig(recovery="rezero", **base), model, pool)
    a, b = none.acc.span(sim.drift_start, 500), rz.acc.span(sim.drift_start, 500)
    assert rz.rezero_started_at is not None and none.rezero_started_at is None
    assert b[0] / b[1] > a[0] / a[1] + 0.2                                           # clearly better with re-zeroing
    late = [r["offset_applied_kN"] for r in rz.records[-20:]]
    assert np.mean(late) > 5.0                                                       # it really was subtracting a big offset
    assert rz.monitor.first_alert("f_zero_kN") == none.monitor.first_alert("f_zero_kN")  # the alert itself is unchanged


def test_rezero_does_not_act_without_an_alert(model, pool):
    cfg = PlantConfig(scenario="nominal", n_parts=250, recovery="rezero", seed=1)
    _, svc = _shift(cfg, model, pool)
    assert svc.rezero_started_at is None and all(r["offset_applied_kN"] == 0.0 for r in svc.records)


def test_data_support_hook_is_feature_detected(model, pool, monkeypatch):
    import serving.model_api as api
    cfg = PlantConfig(scenario="nominal", n_parts=12)
    _, plain = _shift(cfg, model, pool)
    assert plain._support is None and not any(r["flag_outside_support"] for r in plain.records)   # no hook: silent
    monkeypatch.setattr(api, "in_data_support", lambda X: np.zeros(len(X), bool), raising=False)
    _, hooked = _shift(cfg, model, pool)
    assert hooked._support is not None
    assert all(r["reason"] == "outside_data_support" and r["verdict"] == "check" for r in hooked.records)
    monkeypatch.setattr(api, "in_data_support", lambda X: 1 / 0, raising=False)                   # a broken hook must not stop the line
    _, broken = _shift(cfg, model, pool)
    assert all(r["verdict"] in ("go", "check") for r in broken.records)


def test_labels_survive_a_fast_press_over_opcua(model, pool):
    """pace 0: the server writes everything at once, and OPC-UA may deliver stroke-ends before labels."""
    cfg = PlantConfig(scenario="nominal", n_parts=45, lot_len=15)
    sim = PressSimulator(cfg, pool)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        ep = f"opc.tcp://127.0.0.1:{s.getsockname()[1]}/t/"

    async def go():
        srv = PressServer(ep)
        await srv.start()
        try:
            task = asyncio.create_task(srv.run_shift(sim, pace_s=0.0))
            svc = await run_edge(ep, cfg, EdgeService(cfg, model), idle_timeout_s=30)
            await task
            return svc
        finally:
            await srv.stop()

    over = asyncio.run(go())
    _, off = _shift(cfg, model, pool)
    assert len(over.labels) == len(off.labels) > 5
    assert [l["part_id"] for l in over.labels] == [l["part_id"] for l in off.labels]
    assert [l["label_delay_parts"] for l in over.labels] == [l["label_delay_parts"] for l in off.labels]
    assert [r["pred_mean"] for r in over.records] == pytest.approx([r["pred_mean"] for r in off.records], abs=1e-9)


# ----------------------------------------------------------------------------------------- experiment harness
def _fake(sc, rc, seed, after, alerts=None, rz=None):
    z = dict(false_alarm_evaluations_before_drift=0, first_alert_part=None, detection_delay_parts=None)
    al = {f: dict(z) for f in ("f_zero_kN", "peak_rel_lot", "peak_force_norm")}
    for f, d in (alerts or {}).items():
        al[f] = dict(z, detection_delay_parts=d, first_alert_part=300 + d)
    return dict(scenario=sc, recovery=rc, seed=seed, n_parts=600, drift_start=300, alerts=al,
                acc=dict(before_drift=dict(accepted=50, measured=60), after_drift=dict(accepted=after, measured=60),
                         all=dict(accepted=50 + after, measured=120), after_first_primary_alert=None, drift_start_to_alert=None),
                rezero_started_at_part=rz, verdicts=dict(go=300, check=300, stop=0), misses_with_go=0, harness_wall_s=1.0,
                check_causes=dict(by_reason={"uncertain_or_borderline": 300}, flagged_reduction_outside_model_range=0,
                                  flagged_outside_data_support=0, data_support_hook_available=False),
                friction_by_lot=[dict(lot=k, mf_true=0.3 + (0.05 * k if k >= 6 else 0), mf_post_mean=0.3 + (0.04 * k if k >= 6 else 0),
                                      mf_post_std=0.01) for k in range(12)])


def test_aggregate_computes_rates_delays_and_recovery():
    runs = []
    for s in range(6):
        runs += [_fake("nominal", "none", s, 50), _fake("drift_offset", "none", s, 20, {"f_zero_kN": 15 + s}),
                 _fake("drift_offset", "rezero", s, 48, {"f_zero_kN": 15 + s}, rz=315),
                 _fake("drift_gain", "none", s, 50, {"peak_rel_lot": 30}), _fake("drift_friction", "none", s, 50)]
    a = aggregate(runs)
    d = a["detection"]
    assert d["drift_offset"]["f_zero_kN"]["detection_rate"] == 1.0 and d["drift_offset"]["f_zero_kN"]["delay_parts"]["median"] == 17.5
    assert d["drift_gain"]["f_zero_kN"]["detected_runs"] == 0 and d["drift_gain"]["peak_rel_lot"]["detected_runs"] == 6
    assert d["drift_friction"]["f_zero_kN"]["detection_rate"] == 0.0
    assert a["acceptance"]["drift_offset/none"]["after_drift"]["accepted"] == 120 and a["acceptance"]["drift_offset/none"]["after_drift"]["measured"] == 360
    r = a["recovery"]["by_scenario"]["drift_offset"]
    assert r["mean_rate_difference_rezero_minus_none"] == pytest.approx(28 / 60) and r["rezero_started_runs"] == 6
    assert a["false_alarms"]["by_feature"]["f_zero_kN"]["per_100_evaluated_parts"] == 0.0
    assert a["friction_trend"]["drift_friction"]["mean_true_friction_rise"] > 0


def test_cluster_bootstrap_widens_with_between_run_spread():
    same = [dict(accepted=50, measured=60)] * 5
    mixed = [dict(accepted=a, measured=60) for a in (20, 55, 30, 58, 25)]
    lo_s, hi_s = cluster_bootstrap(same)
    lo_m, hi_m = cluster_bootstrap(mixed)
    assert hi_s - lo_s < 1e-9 < hi_m - lo_m


def test_harness_runs_end_to_end_on_smoke(tmp_path):
    runs = run_all([0], tmp_path / "w", parts=300, model="serving.model_api:load_stub", artifact=None, data="smoke",
                   workers=2, scenarios=("nominal", "drift_offset"), recoveries=("none", "rezero"), log=lambda *a, **k: None)
    assert len(runs) == 4
    assert any(r["scenario"] == "drift_offset" and r["recovery"] == "rezero" and r["rezero_started_at_part"] for r in runs)
    again = run_all([0], tmp_path / "w", parts=300, model="serving.model_api:load_stub", artifact=None, data="smoke",
                    workers=2, scenarios=("nominal", "drift_offset"), recoveries=("none", "rezero"), log=lambda *a, **k: None)
    assert again == runs                                                              # resumed from the cache, identical
    assert aggregate(runs)["acceptance"]["nominal/none"]["all"]["measured"] > 0
