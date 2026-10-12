"""Drift demo (task P8):  python -m plant.demo --scenario drift --parts 600 --model <module>:<load_fn> --out results/plant/drift

Starts the simulated press (OPC-UA server in its own process), runs the edge service against it, then writes
the edge outputs (Parquet, events, repro bundles), figures, and a results JSON with the headline numbers.
Everything is deterministic given --seed except wall-clock latency.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from serving.edge.monitoring import wilson
from serving.edge.service import EdgeService, run_edge
from .bundle import _git_sha
from serving.edge.service import MONITORED
from .config import DRIFT_TYPES, PlantConfig
from .press_sim import PressSimulator


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cfg_to_yaml(cfg: PlantConfig, path: Path) -> None:
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}
    path.write_text(yaml.safe_dump(d, sort_keys=True), encoding="utf-8")


async def _run_against_server(cfg: PlantConfig, out: Path, endpoint: str) -> EdgeService:
    """Connect with retries while the server process is still starting."""
    last = None
    for _ in range(100):
        try:
            return await run_edge(endpoint, cfg, out_dir=out)
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:      # server not up yet
            last = e
            await asyncio.sleep(0.3)
    raise RuntimeError(f"could not reach OPC-UA server at {endpoint}: {last}")


def _interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return dict(accepted=int(k), measured=int(n), rate=None if n == 0 else p, wilson95=[lo, hi])


def compute_metrics(cfg: PlantConfig, svc: EdgeService, sim: PressSimulator, wall_s: float) -> dict:
    df = pd.DataFrame(svc.records)
    start = sim.drift_start
    n = len(df)
    lat = df["latency_ms"].to_numpy()
    edge = df["edge_ms"].to_numpy()
    m: dict = {"n_parts": n, "scenario": cfg.scenario, "wall_clock_s": wall_s}
    m["latency"] = dict(
        assumed_cycle_time_s=cfg.assumed_cycle_time_s, verdict_deadline_s=cfg.assumed_cycle_time_s,
        end_to_end_ms_p50=float(np.percentile(lat, 50)), end_to_end_ms_p99=float(np.percentile(lat, 99)),
        end_to_end_ms_max=float(lat.max()),
        edge_internal_ms_p50=float(np.percentile(edge, 50)), edge_internal_ms_p99=float(np.percentile(edge, 99)),
        transport_ms_p50=float(np.nanpercentile(df["transport_ms"], 50)),
        predict_ms_p50=float(np.percentile(df["predict_ms"], 50)),
        align_ms_p50=float(np.percentile(df["align_ms"], 50)),
        share_over_deadline=float(np.mean(lat > cfg.assumed_cycle_time_s * 1000)),
        definition="from the server's stroke-end event to the verdict, including OPC-UA delivery, buffer completion, "
                   "alignment, feature extraction, friction-posterior prediction; excludes PLC scan time and a real network")
    prim = svc.monitor.first_alert("f_zero_kN")
    pre_alerts = [a for a in svc.monitor.alerts if a["feature"] == "f_zero_kN" and start is not None and a["part_index"] < start]
    pk = [a for a in svc.monitor.alerts if a["feature"] == "peak_force_norm"]
    m["drift"] = dict(
        drift_start_part=start, primary_feature="f_zero_kN",
        first_alert_part=prim,
        detection_delay_parts=None if (prim is None or start is None) else int(prim - start),
        false_alarms_before_drift_primary=len(pre_alerts) if start is not None else len(
            [a for a in svc.monitor.alerts if a["feature"] == "f_zero_kN"]),
        peak_force_norm_alert_evaluations=len(pk),
        peak_force_norm_note="lot changes (a new ring and friction every lot_len parts) shift this feature, so it alerts "
                             "without drift; it is informational, not a drift signal",
        window_parts=cfg.drift_window_parts, reference_parts=cfg.reference_parts, stride=cfg.drift_stride,
        offset_at_end_kN=float(sim.force_offset_kN(n - 1)), gain_at_end=float(sim.force_gain(n - 1)),
        drift_type=sim.drift_type, recovery=cfg.recovery)
    by_feat = {}
    for feat in MONITORED:
        al_f = [a["part_index"] for a in svc.monitor.alerts if a["feature"] == feat]
        post = [x for x in al_f if start is not None and x >= start]
        by_feat[feat] = dict(first_alert_part=post[0] if post else None,
                             detection_delay_parts=(post[0] - start) if post else None,
                             false_alarm_evaluations_before_drift=len([x for x in al_f if start is None or x < start]))
    m["drift"]["alerts_by_feature"] = by_feat
    m["drift"]["alerts_by_feature_note"] = ("f_zero_kN watches the load-cell zero (injected channel for drift_offset); "
        "peak_rel_lot is the force at 20% reduction relative to the first parts of the same lot (responds to gain drift); "
        "peak_force_norm jumps at every lot change and is informational")
    rows = svc.acc.rows
    if start is not None:
        before = svc.acc.span(0, start)
        after = svc.acc.span(start, n)
        m["acceptance"] = dict(window_parts=cfg.window_parts, tolerance_pct=cfg.accept_tol_pct, before_drift=_interval(*before),
                               after_drift=_interval(*after), all=_interval(sum(r["accepted"] for r in rows), len(rows)))
    else:
        split_at = int(round(cfg.drift_start_frac * n))        # no drift: same split point, so runs are comparable
        m["acceptance"] = dict(window_parts=cfg.window_parts, tolerance_pct=cfg.accept_tol_pct,
                               split_at_part=split_at, before_drift=_interval(*svc.acc.span(0, split_at)),
                               after_drift=_interval(*svc.acc.span(split_at, n)),
                               all=_interval(sum(r["accepted"] for r in rows), len(rows)))
    # after the friction posterior has had a chance to learn (first label of each lot excluded)
    seen_lot: set = set()
    warm_k = warm_n = 0
    for r in rows:
        lot = int(df.loc[df.part_id == r["part_id"], "lot_id"].iloc[0])
        first = lot not in seen_lot
        seen_lot.add(lot)
        if first or (start is not None and r["part_id"] >= start):
            continue
        warm_n += 1
        warm_k += r["accepted"]
    if start is not None:
        m["acceptance"]["before_drift_excluding_first_label_of_each_lot"] = _interval(warm_k, warm_n)
    alert_part = svc.rezero_started_at if svc.rezero_started_at is not None else svc.monitor.first_alert("f_zero_kN")
    if start is not None and alert_part is not None:
        m["acceptance"]["after_first_primary_alert"] = _interval(*svc.acc.span(df.part_id.iloc[alert_part], n))
        m["acceptance"]["drift_start_to_alert"] = _interval(*svc.acc.span(start, df.part_id.iloc[alert_part]))
    m["recovery"] = dict(mode=cfg.recovery, rezero_started_at_part=svc.rezero_started_at,
                         max_offset_applied_kN=float(df["offset_applied_kN"].abs().max()))
    m["labels"] = dict(sampled_every=cfg.label_every, n_received=len(rows), n_orphan=svc.n_orphan_labels,
                       delay_parts_min=int(min((l["label_delay_parts"] for l in svc.labels), default=0)),
                       delay_parts_max=int(max((l["label_delay_parts"] for l in svc.labels), default=0)),
                       sigma_pct=cfg.label_sigma_pct)
    m["bundles"] = dict(written=svc.n_bundles, misses=svc.n_failures, misses_outside_predicted_interval=svc.n_surprising, misses_with_go_verdict=svc.n_go_missed,
                        cap=cfg.max_bundles)
    truth_in = np.array([sim.part(i).truth["in_spec"] for i in range(n)])
    m["verdicts"] = {v: int((df.verdict == v).sum()) for v in ("go", "check", "stop")}
    m["verdict_reasons"] = df["reason"].value_counts().to_dict()
    m["check_causes"] = dict(
        by_reason=df[df.verdict == "check"]["reason"].value_counts().to_dict(),
        flagged_reduction_outside_model_range=int(df.flag_reduction_range.sum()),
        flagged_outside_data_support=int(df.flag_outside_support.sum()),
        both_flags=int((df.flag_reduction_range & df.flag_outside_support).sum()),
        data_support_hook_available=svc._support is not None,
        note="by_reason is exclusive (reduction range wins over data support, which wins over uncertainty); the "
             "flagged_* counts are independent. data_support_hook_available=false means the model module has no in_data_support.")
    m["verdict_vs_truth_in_spec"] = {v: dict(in_spec=int(((df.verdict == v) & truth_in).sum()),
                                             out_of_spec=int(((df.verdict == v) & ~truth_in).sum()))
                                     for v in ("go", "check", "stop")}
    err = []
    for lot in sorted(svc.est_by_lot):
        mf_true = sim.lot(lot)["run"].case["mf"]
        mu, sd = svc.est_by_lot[lot].mean_std()
        err.append(dict(lot=int(lot), mf_true=float(mf_true), mf_post_mean=float(mu), mf_post_std=float(sd)))
    m["friction_by_lot"] = err
    m["friction_mean_abs_error"] = float(np.mean([abs(e["mf_true"] - e["mf_post_mean"]) for e in err]))
    m["model"] = dict(name=svc.surrogate.name, version=svc.surrogate.version, spec=cfg.model)
    return m


def make_figures(df: pd.DataFrame, svc: EdgeService, sim: PressSimulator, cfg: PlantConfig, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    start = sim.drift_start
    paths = []

    fig, ax = plt.subplots(3, 1, figsize=(8, 7), dpi=150, sharex=True)
    ax[0].plot(df.part_id, df.f_zero_kN, lw=0.8)
    ax[0].set_ylabel("load-cell zero reading (kN)")
    ax[1].plot(df.part_id, df.reduction_est, lw=0.8)
    ax[1].set_ylabel("estimated reduction (-)")
    ax[2].plot(df.part_id, df.peak_force_norm, lw=0.8)
    ax[2].set_ylabel("peak force, normalised (-)")
    ax[2].set_xlabel("part number")
    for a in ax:
        if start is not None:
            a.axvline(start, color="k", ls=":", label="drift starts")
        f = svc.monitor.first_alert("f_zero_kN")
        if f is not None:
            a.axvline(df.part_id.iloc[f], color="r", ls="--", label="first drift alert")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "signals_and_alert.png")
    plt.close(fig)
    paths.append("figures/signals_and_alert.png")

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    xs, ps, lo, hi = [], [], [], []
    for now in range(cfg.window_parts, len(df), 10):
        p, l, h = svc.acc.interval(now, cfg.window_parts)
        if np.isfinite(p):
            xs.append(now); ps.append(p); lo.append(l); hi.append(h)
    ax.plot(xs, ps, label=f"acceptance (last {cfg.window_parts} parts)")
    ax.fill_between(xs, lo, hi, alpha=0.25, label="Wilson 95%")
    if start is not None:
        ax.axvline(start, color="k", ls=":", label="drift starts")
    ax.set_xlabel("part number")
    ax.set_ylabel(f"share within {cfg.accept_tol_pct} pt of measurement")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "acceptance_trend.png")
    plt.close(fig)
    paths.append("figures/acceptance_trend.png")

    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    ax.hist(df.latency_ms, bins=40)
    ax.set_xlabel("stroke end to verdict (ms)")
    ax.set_ylabel("parts")
    ax.axvline(np.percentile(df.latency_ms, 50), color="g", label="p50")
    ax.axvline(np.percentile(df.latency_ms, 99), color="r", label="p99")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "latency.png")
    plt.close(fig)
    paths.append("figures/latency.png")
    return paths


def run_demo(cfg: PlantConfig, out: Path, command: str, figures: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cfg_to_yaml(cfg, out / "config.yaml")
    port = free_port()
    endpoint = f"opc.tcp://127.0.0.1:{port}/forming-surrogates/"
    server = subprocess.Popen([sys.executable, "-m", "plant.opcua_server", "--config", str(out / "config.yaml"),
                               "--endpoint", endpoint], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    try:
        svc = asyncio.run(_run_against_server(cfg, out, endpoint))
    finally:
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
    wall = time.time() - t0
    return finish_run(cfg, svc, out, wall, command, figures)


def finish_run(cfg: PlantConfig, svc: EdgeService, out: Path, wall: float, command: str, figures: bool = True) -> dict:
    """Metrics, figures and results JSON from a finished edge run (also used by `python -m serving.edge.service`)."""
    sim = PressSimulator(cfg)
    metrics = compute_metrics(cfg, svc, sim, wall)
    df = pd.DataFrame(svc.records)
    figs = make_figures(df, svc, sim, cfg, out) if figures else []
    result = {
        "task": "P8", "title": f"Plant demo: {cfg.scenario} scenario", "command": command, "git": _git_sha(),
        "data": {"dataset": cfg.dataset, "n_runs": len(sim.pool), "splits": {s: sum(r.split == s for r in sim.pool) for s in cfg.splits}},
        "metrics": metrics, "figures": [f"{out.as_posix()}/{f}" for f in figs],
        "config_hash": cfg.hash(),
        "notes": ("Simulated press replaying solver test runs; nothing here is validated against a real machine. "
                  "Model: " + f"{svc.surrogate.name} v{svc.surrogate.version}. "
                  + ("With the stub model the replay runs were part of the stub's own fit, so its accuracy is optimistic."
                     if "load_stub" in cfg.model else
                     "The model was trained on the train split only; the replayed runs come from the held-out test splits.")),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="drift", choices=sorted(DRIFT_TYPES))
    ap.add_argument("--parts", type=int, default=600)
    ap.add_argument("--model", default="serving.model_api:load_stub")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pace-s", type=float, default=0.25)
    ap.add_argument("--recovery", default="none", choices=["none", "rezero"],
                    help="rezero: after the zero-offset alert, subtract the estimated offset from the force")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--max-bundles", type=int, default=None)
    ap.add_argument("--out", default="results/plant/drift")
    a = ap.parse_args(argv)
    cfg = PlantConfig(scenario=a.scenario, n_parts=a.parts, model=a.model, artifact_dir=a.artifact, dataset=a.data,
                      seed=a.seed, pace_s=a.pace_s, recovery=a.recovery)
    if a.max_bundles is not None:
        cfg.max_bundles = a.max_bundles
    cmd = "python -m plant.demo " + " ".join(argv if argv is not None else sys.argv[1:])
    res = run_demo(cfg, Path(a.out), cmd, figures=not a.no_figures)
    m = res["metrics"]
    print(json.dumps({k: m[k] for k in ("latency", "drift", "acceptance", "bundles", "verdicts")}, indent=2, default=float))


if __name__ == "__main__":
    main()
