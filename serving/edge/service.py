"""Edge inference service (task P4).

Per stroke it: buffers the three signal chunks, aligns force and position, finds contact, estimates the achieved
reduction, predicts the inner-diameter change with an uncertainty that includes the friction posterior, issues a
go / check / stop verdict, and logs the end-to-end latency. When a delayed gauge measurement arrives it updates the
friction posterior of that part's lot, scores the old prediction, and (on a miss) writes a repro bundle.

Model access is ONLY through serving.model_api (Surrogate, FrictionEstimator), loaded from a "module:function" string
in the config, so the stub and the real GP are interchangeable.

Force and friction: the friction posterior is updated from measured inner diameters only. Phase 2 shows that force
barely constrains friction, and a drifting load cell would bias any force-based friction prior, so no force
likelihood is fed into the estimator ("force as a weak prior" is represented by the estimator's flat starting prior;
a force-informed prior is designed, not built).

Run against a live OPC-UA server:  python -m serving.edge.service --endpoint opc.tcp://127.0.0.1:4840/forming-surrogates/
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import sys
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from plant.bundle import write_bundle
from plant.chunks import decode_chunk
from plant.config import PlantConfig, load_model
from plant.spec import load_spec
from .align import align, features
from .monitoring import AcceptanceTracker, DriftMonitor

SQRT3 = float(np.sqrt(3.0))
GH_NODES = np.array([0.0, SQRT3, -SQRT3])
GH_WEIGHTS = np.array([2 / 3, 1 / 6, 1 / 6])           # 3-point Gauss-Hermite for a normal expectation
RED_MIN, RED_MAX = 0.02, 0.35                            # accepted reduction range; the GP was trained on 0.1-0.3, so 0.3-0.35 is extrapolated (flagged by its wider sd)
MONITORED = ("f_zero_kN", "peak_force_norm", "peak_rel_lot")
LOT_BASELINE_PARTS = 5                                  # first parts of a lot used as that lot's force baseline
RECENT_PARTS = 40                                       # raw signals kept for bundles until labels arrive


class EdgeService:
    def __init__(self, cfg: PlantConfig, model=None):
        self.cfg = cfg
        self.proc = load_spec().process
        self.surrogate, self._pristine = model if model is not None else load_model(cfg.model, cfg.artifact_dir)
        self.est_by_lot: dict[int, object] = {}
        self.monitor = DriftMonitor(features=MONITORED, reference=cfg.reference_parts, window=cfg.drift_window_parts,
                                    stride=cfg.drift_stride, psi_alert=cfg.psi_alert,
                                    min_shift={"peak_rel_lot": 0.01})   # 1% change of the lot-relative force
        self.acc = AcceptanceTracker(cfg.accept_tol_pct)
        self.records: list[dict] = []
        self.by_part: dict[int, dict] = {}
        self.recent: OrderedDict[int, dict] = OrderedDict()
        self.labels: list[dict] = []
        self.events: list[dict] = []
        self.signal_tables: list[pd.DataFrame] = []
        self.n_bundles = 0
        self.n_failures = 0
        self.n_go_missed = 0           # misses on parts that got a 'go'
        self.n_surprising = 0          # misses that the prediction's own uncertainty did not cover
        self.bundle_dir: Path | None = None
        self.n_orphan_labels = 0
        self.zero_hist: list[float] = []        # raw load-cell zero reading of every part (sensor health, before any re-zeroing)
        self.rezero_active = False
        self.rezero_started_at: int | None = None
        self.lot_peaks: dict[int, list[float]] = {}
        # optional model hook (not part of the frozen Surrogate contract): feature-detected, silently skipped if absent
        self._support = None
        try:
            fn = getattr(self.surrogate, "in_data_support", None)          # the loaded model's own artifact first
            if not callable(fn):
                fn = getattr(importlib.import_module(cfg.model.split(":")[0]), "in_data_support", None)
            self._support = fn if callable(fn) else None
        except Exception:                                    # noqa: BLE001  (a missing hook must never stop the line)
            self._support = None

    def in_support(self, ticket: dict, reduction: float, mf: float) -> bool | None:
        """True/False from the model's in_data_support hook for the posterior-mean input, None when no hook."""
        if self._support is None:
            return None
        try:
            X = np.array([[ticket["od_id"], ticket["h_id"], mf, ticket["m"], reduction]])
            return bool(np.all(np.asarray(self._support(X))))
        except Exception:                                    # noqa: BLE001
            return None

    # ------------------------------------------------------------------ friction
    def _estimator(self, lot_id: int):
        """Fresh friction posterior per lot: a new lot is a new lubricant batch. Deep copy with the surrogate shared."""
        if lot_id not in self.est_by_lot:
            self.est_by_lot[lot_id] = copy.deepcopy(self._pristine, memo={id(self.surrogate): self.surrogate})
        return self.est_by_lot[lot_id]

    def predict(self, ticket: dict, reduction: float, est) -> tuple[float, float, float, float]:
        """Predictive mean/std of ID change, integrating over the friction posterior (3-point quadrature)."""
        mf_m, mf_s = est.mean_std()
        mfs = np.clip(mf_m + mf_s * GH_NODES, 0.0, 1.0)
        X = np.column_stack([np.full(3, ticket["od_id"]), np.full(3, ticket["h_id"]), mfs,
                             np.full(3, ticket["m"]), np.full(3, reduction)])
        mu, sd = self.surrogate.predict(X)
        mean = float(GH_WEIGHTS @ mu)
        var = float(GH_WEIGHTS @ (np.asarray(sd) ** 2 + np.asarray(mu) ** 2) - mean ** 2)
        return mean, float(np.sqrt(max(var, 0.0))), float(mf_m), float(mf_s)

    def verdict(self, mean: float, std: float, ticket: dict, anomaly: bool) -> tuple[str, str]:
        """go: the whole +-k sd interval is inside the spec band; stop: it is entirely outside; otherwise check."""
        if not np.isfinite(mean):
            return "check", "no_contact_found"
        lo_s = ticket["target_id_change_pct"] - ticket["tol_id_pct"]
        hi_s = ticket["target_id_change_pct"] + ticket["tol_id_pct"]
        lo, hi = mean - self.cfg.interval_k * std, mean + self.cfg.interval_k * std
        if hi < lo_s or lo > hi_s:
            return "stop", "predicted_out_of_spec"
        if lo >= lo_s and hi <= hi_s:
            return ("check", "contact_position_anomaly") if anomaly else ("go", "inside_spec")
        return "check", "uncertain_or_borderline"

    # ------------------------------------------------------------------ one stroke
    def process_part(self, ticket: dict, streams: dict, emit_wall_ns: int, notify_wall_ns: int | None = None,
                     complete: bool = True, notify_perf_ns: int | None = None) -> dict:
        """Everything between 'stroke finished' and 'verdict'. emit_wall_ns starts the latency clock."""
        t_in = time.perf_counter_ns()
        wall_in = time.time_ns()
        pid, lot = int(ticket["part_id"]), int(ticket["lot_id"])
        idx = len(self.records)
        al = align(streams)
        t_al = time.perf_counter_ns()
        f = features(al, ticket, self.proc, self.cfg.contact_threshold_kN, self.cfg.contact_debounce)
        f_zero_raw = f.f_zero_kN
        self.zero_hist.append(f_zero_raw)
        offset_applied = 0.0
        if self.cfg.recovery == "rezero" and self.rezero_active:
            # tare: the load-cell zero (mean force with the die in the air) over the last few parts, minus the healthy
            # zero from the reference window, is subtracted from this stroke's force before touch-off detection
            recent = np.array(self.zero_hist[-self.cfg.rezero_window:])
            ref = np.array(self.zero_hist[:self.cfg.reference_parts])
            recent, ref = recent[np.isfinite(recent)], ref[np.isfinite(ref)]
            if len(recent) and len(ref):
                offset_applied = float(recent.mean() - np.median(ref))
                al = dict(al, force=al["force"] - offset_applied)
                f = features(al, ticket, self.proc, self.cfg.contact_threshold_kN, self.cfg.contact_debounce)
        base = self.lot_peaks.setdefault(lot, [])
        if len(base) < LOT_BASELINE_PARTS:
            base.append(f.force_ref_norm)
            peak_rel = float("nan")
        else:
            peak_rel = f.force_ref_norm / float(np.nanmedian(base)) if np.isfinite(f.force_ref_norm) else float("nan")
        t_ft = time.perf_counter_ns()
        est = self._estimator(lot)
        out_of_range = not (np.isfinite(f.reduction) and RED_MIN <= f.reduction <= RED_MAX)
        if np.isfinite(f.reduction):
            # outside 0.02-0.35: clip, predict, and flag; the GP's training data end at 0.3, so 0.3-0.35 is a mild extrapolation
            mean, std, mf_m, mf_s = self.predict(ticket, float(np.clip(f.reduction, RED_MIN, RED_MAX)), est)
        else:
            mean = std = float("nan")
            mf_m, mf_s = est.mean_std()
        t_pr = time.perf_counter_ns()
        verdict, reason = self.verdict(mean, std, ticket, f.contact_anomaly)
        support = self.in_support(ticket, float(np.clip(f.reduction, RED_MIN, RED_MAX)), mf_m) if np.isfinite(f.reduction) else None
        if out_of_range and np.isfinite(mean):
            verdict, reason = "check", "reduction_outside_model_range"
        elif support is False:
            verdict, reason = "check", "outside_data_support"
        if not complete:
            verdict, reason = "check", "incomplete_signal_buffer"
        wall_out = time.time_ns()
        perf_out = time.perf_counter_ns()
        rec = dict(
            index=idx, part_id=pid, lot_id=lot, emit_wall_ns=int(emit_wall_ns), verdict_wall_ns=wall_out,
            latency_ms=(wall_out - emit_wall_ns) / 1e6,
            edge_ms=(perf_out - (notify_perf_ns if notify_perf_ns is not None else t_in)) / 1e6,
            transport_ms=((notify_wall_ns - emit_wall_ns) / 1e6) if notify_wall_ns else float("nan"),
            buffer_wait_ms=(wall_in - (notify_wall_ns or wall_in)) / 1e6,
            align_ms=(t_al - t_in) / 1e6, feature_ms=(t_ft - t_al) / 1e6, predict_ms=(t_pr - t_ft) / 1e6,
            f_zero_raw_kN=f_zero_raw, offset_applied_kN=offset_applied, peak_rel_lot=peak_rel,
            flag_reduction_range=bool(out_of_range and np.isfinite(mean)), flag_outside_support=support is False,
            reduction_est=f.reduction, contact_pos_mm=f.contact_pos_mm, f_zero_kN=f_zero_raw, peak_kN=f.peak_kN,
            peak_force_norm=f.peak_force_norm, contact_anomaly=f.contact_anomaly, n_force=f.n_force, n_pos=f.n_pos,
            n_uncertain=al["n_uncertain"], pred_mean=mean, pred_std=std, mf_mean=mf_m, mf_std=mf_s,
            target=ticket["target_id_change_pct"], tol=ticket["tol_id_pct"], verdict=verdict, reason=reason,
            die_temp_mean_C=float(np.nanmean(streams["die_temperature"][1])),   # nuisance channel: logged, never used
            measured=float("nan"), residual=float("nan"), accepted=None)
        self.records.append(rec)
        self.by_part[pid] = rec
        self.recent[pid] = dict(ticket=ticket, streams=streams)
        while len(self.recent) > RECENT_PARTS:
            self.recent.popitem(last=False)
        step = max(1, len(al["t"]) // 120)
        self.signal_tables.append(pd.DataFrame(dict(part_id=pid, t=al["t"][::step], force_kN=al["force"][::step],
                                                    pos_mm=al["pos"][::step])))
        # the monitor sees the RAW zero reading: it watches the sensor, not the corrected signal
        for a in self.monitor.add(idx, dict(f_zero_kN=f_zero_raw, peak_force_norm=f.peak_force_norm, peak_rel_lot=peak_rel)):
            self.events.append(dict(type="drift_alert", **a))
            if a["primary"] and self.cfg.recovery == "rezero" and not self.rezero_active:
                self.rezero_active, self.rezero_started_at = True, idx
                self.events.append(dict(type="rezero_started", part_index=idx))
        return rec

    # ------------------------------------------------------------------ delayed label
    def on_label(self, label: dict) -> dict | None:
        pid = int(label["part_id"])
        rec = self.by_part.get(pid)
        if rec is None or pid not in self.recent:
            self.n_orphan_labels += 1
            return None
        seen_at = len(self.records) - 1
        rec.update(measured=float(label["id_change_pct"]))
        rec["residual"] = rec["pred_mean"] - rec["measured"]
        accepted = self.acc.add(pid, seen_at, rec["pred_mean"], rec["measured"])
        rec["accepted"] = accepted
        ticket = self.recent[pid]["ticket"]
        est = self._estimator(rec["lot_id"])
        red_for_update = float(label.get("reduction", rec["reduction_est"]))
        if np.isfinite(red_for_update):
            est.update({"od_id": ticket["od_id"], "h_id": ticket["h_id"], "m": ticket["m"]}, red_for_update,
                       float(label["id_change_pct"]), float(label["sigma_pct"]))
        delay = seen_at - rec["index"]
        ev = dict(type="label", part_index=seen_at, part_id=pid, label_delay_parts=delay, measured=rec["measured"],
                  predicted=rec["pred_mean"], residual=rec["residual"], accepted=accepted)
        self.events.append(ev)
        self.labels.append(dict(ev, sigma_pct=float(label["sigma_pct"])))
        surprising = not accepted and not (abs(rec["residual"]) <= self.cfg.interval_k * rec["pred_std"])
        if not accepted:
            self.n_failures += 1
        go_but_missed = (not accepted) and rec["verdict"] == "go"
        if surprising or go_but_missed:
            self.n_surprising += surprising
            self.n_go_missed += go_but_missed
            if self.bundle_dir is not None and self.n_bundles < self.cfg.max_bundles:
                d = write_bundle(self.bundle_dir / f"part_{pid:05d}", rec=rec, ticket=ticket,
                                 streams=self.recent[pid]["streams"], label=label, cfg=self.cfg,
                                 model_name=self.surrogate.name, model_version=self.surrogate.version,
                                 drift_alerts=self.monitor.alerts, reason=("go_verdict_but_missed" if go_but_missed else "miss_outside_predicted_interval"))
                self.n_bundles += 1
                self.events.append(dict(type="bundle", part_index=seen_at, part_id=pid, path=d.as_posix()))
        return ev

    # ------------------------------------------------------------------ output
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_parquet(out / "parts.parquet", index=False)
        pd.DataFrame(self.labels).to_parquet(out / "measurements.parquet", index=False)
        if self.signal_tables:
            pd.concat(self.signal_tables, ignore_index=True).to_parquet(out / "signals.parquet", index=False)
        pd.DataFrame(self.monitor.evaluations).to_parquet(out / "drift.parquet", index=False)
        with open(out / "events.jsonl", "w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(e, default=float) + "\n")


# ---------------------------------------------------------------------- OPC-UA client side
class _Handler:
    """Synchronous handler: asyncua calls it in arrival order, and it only buffers (no heavy work in the callback)."""

    def __init__(self, spec, loop_queue: asyncio.Queue):
        self.names = {s.browse_name: s.name for s in spec.signals.values()}
        self.q = loop_queue
        self.chunks: dict[int, dict[str, list]] = {}
        self.tickets: dict[int, dict] = {}
        self.n_chunks_seen: dict[int, dict[str, int]] = {}

    def datachange_notification(self, node, val, data):
        now = time.time_ns()
        name = self.names.get(str(node.nodeid.Identifier))
        if name in ("stroke_position", "stroke_force", "die_temperature"):
            if len(val) < 5:
                return                                   # initial placeholder value
            pid, seq, t, v, q = decode_chunk(val)
            self.chunks.setdefault(pid, {}).setdefault(name, []).append((t, v, q))
        elif name == "part_ticket" and val:
            tk = json.loads(val)
            self.tickets[int(tk["part_id"])] = tk
        elif name == "stroke_end" and val:
            self.q.put_nowait(("end", json.loads(val), now, time.perf_counter_ns()))
        elif name == "lab_measurement" and val:
            self.q.put_nowait(("label", json.loads(val), now, time.perf_counter_ns()))

    def take(self, pid: int, expected: dict) -> tuple[dict, bool]:
        got = self.chunks.pop(pid, {})
        complete = all(len(got.get(n, [])) >= k for n, k in expected.items())
        streams = {}
        for n in ("stroke_position", "stroke_force", "die_temperature"):
            parts = got.get(n, [])
            streams[n] = tuple(np.concatenate([p[i] for p in parts]) if parts else np.array([]) for i in range(3))
        return streams, complete

    def have_all(self, pid: int, expected: dict) -> bool:
        got = self.chunks.get(pid, {})
        return all(len(got.get(n, [])) >= k for n, k in expected.items()) and pid in self.tickets


async def run_edge(endpoint: str, cfg: PlantConfig, svc: EdgeService | None = None, out_dir: str | Path | None = None,
                   idle_timeout_s: float = 30.0) -> EdgeService:
    from asyncua import Client, ua
    spec = load_spec()
    svc = svc or EdgeService(cfg)
    if out_dir is not None:
        svc.bundle_dir = Path(out_dir) / "bundles"
    q: asyncio.Queue = asyncio.Queue()
    handler = _Handler(spec, q)
    async with Client(endpoint) as client:
        sub = await client.create_subscription(10, handler)
        nodes = [client.get_node(s.node_id) for n, s in spec.signals.items()
                 if n not in ("edge_ready", "stroke_state")]
        await sub.subscribe_data_change(nodes, queuesize=100000)
        await client.get_node(spec.signals["edge_ready"].node_id).write_value(ua.Variant(True, ua.VariantType.Boolean))
        done = 0
        pending: list[dict] = []                            # labels held until the server's stroke counter says they are due

        def flush():
            # Apply a label only once `due_index` parts have been processed, whatever the network timing was,
            # so label delay (and every number that depends on it) is reproducible run to run.
            for lab in [l for l in pending if int(l.get("due_index", 0)) <= len(svc.records)]:
                pending.remove(lab)
                svc.on_label(lab)

        backlog: list[tuple] = []                             # stroke-end events waiting for the labels published before them
        labels_seen = 0
        while done < cfg.n_parts:
            try:
                kind, msg, t_notify, p_notify = await asyncio.wait_for(q.get(), idle_timeout_s)
            except asyncio.TimeoutError:
                print("[edge] idle timeout; stopping early", flush=True)
                break
            if kind == "label":
                pending.append(msg)
                labels_seen += 1
                flush()
            else:
                backlog.append((msg, t_notify, p_notify))
            # OPC-UA delivers one monitored item's changes after another within a publish, so at high speed a stroke-end
            # can overtake a label the server wrote before it. The server stamps how many labels it had published at
            # stroke end; a part is processed only once that many labels have been received (deterministic order).
            while backlog and labels_seen >= int(backlog[0][0].get("labels_published", 0)) and done < cfg.n_parts:
                msg, t_notify, p_notify = backlog.pop(0)
                pid = int(msg["part_id"])
                for _ in range(2000):                       # wait up to 2 s for stragglers
                    if handler.have_all(pid, msg["n_chunks"]):
                        break
                    await asyncio.sleep(0.001)
                streams, complete = handler.take(pid, msg["n_chunks"])
                ticket = handler.tickets.pop(pid, None)
                if ticket is None or any(len(streams[n][0]) < 10 for n in ("stroke_force", "stroke_position")):
                    svc.events.append(dict(type="part_lost", part_id=pid))
                    done += 1
                    continue
                svc.process_part(ticket, streams, msg["emit_wall_ns"], t_notify, complete, p_notify)
                done += 1
                flush()
        await sub.delete()
    if out_dir is not None:
        svc.save(out_dir)
    return svc


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=load_spec().endpoint)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--parts", type=int, default=None)
    ap.add_argument("--scenario", default=None, help="must match the press (needed only for the end-of-shift report)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--recovery", default=None, choices=["none", "rezero"])
    ap.add_argument("--out", default="results/plant/edge")
    a = ap.parse_args(argv)
    cfg = PlantConfig.from_yaml(a.config) if a.config else PlantConfig()
    if a.model:
        cfg.model = a.model
    if a.artifact and a.artifact.lower() != "none":
        cfg.artifact_dir = a.artifact
    if a.parts:
        cfg.n_parts = a.parts
    if a.scenario:
        cfg.scenario = a.scenario
    if a.seed is not None:
        cfg.seed = a.seed
    if a.recovery:
        cfg.recovery = a.recovery
    t0 = time.time()
    svc = None
    for attempt in range(120):                         # the press container may still be starting
        try:
            svc = asyncio.run(run_edge(a.endpoint, cfg, out_dir=a.out))
            break
        except (ConnectionError, OSError):
            time.sleep(1.0)
    if svc is None:
        raise SystemExit(f"could not reach {a.endpoint}")
    from plant.demo import finish_run              # lazy: plant.demo imports this module
    finish_run(cfg, svc, Path(a.out), time.time() - t0, "python -m serving.edge.service " + " ".join(argv or sys.argv[1:]))
    print(f"[edge] processed {len(svc.records)} parts; output in {a.out}")


if __name__ == "__main__":
    main()
