"""Experiment harness (task P8 follow-up):  python -m plant.experiments --seeds 20 --out results/plant/experiments.json

Why: the single drift demo (seed 0, zero offset injected into the monitored channel) says little about the monitor.
This runs N seeds x {nominal, drift_offset, drift_gain, drift_friction} x {recovery none, rezero}, each as a full
OPC-UA demo (`python -m plant.demo`, its own port, its own server process), a few in parallel, and aggregates:

  - false-alarm rate per 100 evaluated parts (nominal runs, where every alert is false)
  - detection rate and detection delay (median, IQR) per drift type and per monitored feature
  - acceptance before / after the drift start, pooled over seeds with a Wilson interval AND a seed-level
    (cluster) bootstrap interval, because parts of one run are not independent
  - recovery: acceptance after the drift with and without re-zeroing, and the paired per-seed difference

Each run is cached in <work>/runs/<name>.json, so an interrupted experiment resumes where it stopped.
Nothing here is a statement about a real press: the drift is injected into a simulated sensor.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from serving.edge.monitoring import wilson
from .bundle import _git_sha

SCENARIOS = ("nominal", "drift_offset", "drift_gain", "drift_friction")
RECOVERIES = ("none", "rezero")
FEATURES = ("f_zero_kN", "peak_rel_lot", "peak_force_norm")
MAX_WORKERS = 3


# ---------------------------------------------------------------------------------------------- one run
def run_name(scenario: str, recovery: str, seed: int) -> str:
    return f"{scenario}__{recovery}__seed{seed:03d}"


def summarise(res: dict, scenario: str, recovery: str, seed: int) -> dict:
    """Keep only what the aggregation needs from one demo's metrics.json (the rest stays in the work folder)."""
    m = res["metrics"]
    acc = m["acceptance"]
    take = lambda d: None if d is None else dict(accepted=d["accepted"], measured=d["measured"])   # noqa: E731
    return dict(
        scenario=scenario, recovery=recovery, seed=seed, n_parts=m["n_parts"], drift_start=m["drift"]["drift_start_part"],
        alerts=m["drift"]["alerts_by_feature"],
        acc={k: take(acc.get(k)) for k in ("before_drift", "after_drift", "all", "after_first_primary_alert",
                                           "drift_start_to_alert")},
        rezero_started_at_part=m["recovery"]["rezero_started_at_part"],
        verdicts=m["verdicts"], check_causes=m["check_causes"], friction_by_lot=m["friction_by_lot"],
        misses_with_go=m["bundles"]["misses_with_go_verdict"], wall_clock_s=m["wall_clock_s"], model=m["model"])


def _one(job: dict) -> dict:
    name = run_name(job["scenario"], job["recovery"], job["seed"])
    cache = Path(job["work"]) / "runs" / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out = Path(job["work"]) / "demo" / name
    cmd = [sys.executable, "-m", "plant.demo", "--scenario", job["scenario"], "--recovery", job["recovery"],
           "--seed", str(job["seed"]), "--parts", str(job["parts"]), "--model", job["model"], "--data", job["data"],
           "--pace-s", "0", "--no-figures", "--max-bundles", "0", "--out", str(out)]
    if job["artifact"]:
        cmd += ["--artifact", job["artifact"]]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=os.getcwd())
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not (out / "metrics.json").exists():
        raise RuntimeError(f"{name} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    s = summarise(json.loads((out / "metrics.json").read_text(encoding="utf-8")), job["scenario"], job["recovery"], job["seed"])
    s["harness_wall_s"] = time.time() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(s, default=float), encoding="utf-8")
    for f in out.rglob("*"):                         # the per-run Parquet is not needed any more
        if f.is_file():
            f.unlink()
    return s


def run_all(seeds: list[int], work: Path, parts: int, model: str, artifact: str | None, data: str,
            workers: int, scenarios=SCENARIOS, recoveries=RECOVERIES, log=print) -> list[dict]:
    workers = max(1, min(workers, MAX_WORKERS))
    jobs = [dict(scenario=sc, recovery=rc, seed=s, work=str(work), parts=parts, model=model, artifact=artifact, data=data)
            for s in seeds for sc in scenarios for rc in recoveries]
    out, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(_one, jobs), 1):
            out.append(s)
            if i % 4 == 0 or i == len(jobs):
                log(f"[experiments] {i}/{len(jobs)} runs done, {time.time() - t0:.0f} s elapsed", flush=True)
    return out


# ---------------------------------------------------------------------------------------------- aggregation
def _pool(rows: list[dict | None]) -> dict:
    k = sum(r["accepted"] for r in rows if r)
    n = sum(r["measured"] for r in rows if r)
    p, lo, hi = wilson(k, n)
    return dict(accepted=int(k), measured=int(n), rate=None if n == 0 else p, wilson95=[lo, hi])


def cluster_bootstrap(rows: list[dict | None], reps: int = 4000, seed: int = 0) -> list[float] | None:
    """95% interval of the pooled rate when whole runs (seeds) are resampled: honest about within-run correlation."""
    rows = [r for r in rows if r and r["measured"] > 0]
    if len(rows) < 2:
        return None
    k = np.array([r["accepted"] for r in rows], float)
    n = np.array([r["measured"] for r in rows], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), (reps, len(rows)))
    est = k[idx].sum(1) / n[idx].sum(1)
    return [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def _delay_stats(delays: list[float]) -> dict:
    if not delays:
        return dict(median=None, q25=None, q75=None)
    q = np.quantile(delays, [0.25, 0.5, 0.75])
    return dict(median=float(q[1]), q25=float(q[0]), q75=float(q[2]))


def _rate(rows, key) -> list[float]:
    return [r["acc"][key]["accepted"] / r["acc"][key]["measured"] for r in rows
            if r["acc"].get(key) and r["acc"][key]["measured"] > 0]


def aggregate(runs: list[dict]) -> dict:
    by = lambda sc, rc: sorted((r for r in runs if r["scenario"] == sc and r["recovery"] == rc), key=lambda r: r["seed"])  # noqa: E731
    out: dict = {"n_runs": len(runs), "seeds": sorted({r["seed"] for r in runs})}

    # false alarms: nominal, no recovery -> every alert is a false alarm
    nom = by("nominal", "none")
    if nom:
        n_parts = nom[0]["n_parts"]
        evaluated = max(0, n_parts - 150 + 1)                 # monitoring starts once reference(100) + window(50) parts exist
        fa = {}
        for f in FEATURES:
            c = [r["alerts"][f]["false_alarm_evaluations_before_drift"] for r in nom]
            runs_any = sum(x > 0 for x in c)
            p, lo, hi = wilson(runs_any, len(c))
            fa[f] = dict(alert_evaluations=int(sum(c)), per_100_evaluated_parts=100.0 * sum(c) / (len(c) * evaluated),
                         runs_with_any_false_alarm=int(runs_any), runs=len(c), run_rate=p, run_rate_wilson95=[lo, hi])
        out["false_alarms"] = dict(basis="nominal runs, recovery none; evaluations are made every 10 parts, so one "
                                         "sustained alarm counts several evaluations", evaluated_parts_per_run=evaluated, by_feature=fa)

    # detection per drift type (no recovery: monitor behaviour does not depend on recovery before the alert)
    det = {}
    for sc in ("drift_offset", "drift_gain", "drift_friction"):
        rs = by(sc, "none")
        if not rs:
            continue
        d = {}
        for f in FEATURES:
            post = [r["alerts"][f]["detection_delay_parts"] for r in rs]
            ok = [x for x in post if x is not None]
            p, lo, hi = wilson(len(ok), len(rs))
            d[f] = dict(detected_runs=len(ok), runs=len(rs), detection_rate=p, detection_rate_wilson95=[lo, hi],
                        delay_parts=_delay_stats(ok),
                        false_alarm_evaluations_before_drift=int(sum(r["alerts"][f]["false_alarm_evaluations_before_drift"] for r in rs)))
        det[sc] = d
    out["detection"] = det

    # acceptance, pooled
    acc = {}
    for sc in SCENARIOS:
        for rc in RECOVERIES:
            rs = by(sc, rc)
            if not rs:
                continue
            e = {k: _pool([r["acc"].get(k) for r in rs]) for k in ("before_drift", "after_drift", "all")}
            e["after_drift"]["cluster_bootstrap95"] = cluster_bootstrap([r["acc"].get("after_drift") for r in rs])
            e["before_drift"]["cluster_bootstrap95"] = cluster_bootstrap([r["acc"].get("before_drift") for r in rs])
            if any(r["acc"].get("after_first_primary_alert") for r in rs):
                e["after_first_primary_alert"] = _pool([r["acc"].get("after_first_primary_alert") for r in rs])
                e["runs_with_primary_alert"] = sum(1 for r in rs if r["acc"].get("after_first_primary_alert"))
            acc[f"{sc}/{rc}"] = e
    out["acceptance"] = acc

    # recovery: paired per-seed difference of the post-drift rate, rezero minus none, and vs the nominal run of that seed
    rec = {}
    for sc in SCENARIOS:
        a, b = {r["seed"]: r for r in by(sc, "none")}, {r["seed"]: r for r in by(sc, "rezero")}
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        diff = np.array([_rate([b[s]], "after_drift")[0] - _rate([a[s]], "after_drift")[0] for s in seeds
                         if _rate([a[s]], "after_drift") and _rate([b[s]], "after_drift")])
        if len(diff) == 0:
            continue
        rng = np.random.default_rng(0)
        boot = diff[rng.integers(0, len(diff), (4000, len(diff)))].mean(1)
        rec[sc] = dict(seeds=len(diff), mean_rate_difference_rezero_minus_none=float(diff.mean()),
                       bootstrap95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                       rezero_started_runs=sum(r["rezero_started_at_part"] is not None for r in b.values()))
    out["recovery"] = dict(by_scenario=rec, note="rate = accepted/measured over labels of parts at or after the drift start "
                           "(for nominal: the same split point); rezero only acts after a zero-offset alert, so it can only "
                           "matter for drift_offset, and a nominal run with rezero_started_runs>0 is a false re-zero")

    # what produces `check`
    chk = {}
    for sc in SCENARIOS:
        rs = by(sc, "none")
        if rs:
            reasons: dict = {}
            for r in rs:
                for k, v in r["check_causes"]["by_reason"].items():
                    reasons[k] = reasons.get(k, 0) + v
            chk[sc] = dict(by_reason=reasons, flagged_reduction_outside_model_range=sum(r["check_causes"]["flagged_reduction_outside_model_range"] for r in rs),
                           flagged_outside_data_support=sum(r["check_causes"]["flagged_outside_data_support"] for r in rs),
                           data_support_hook_available=all(r["check_causes"]["data_support_hook_available"] for r in rs),
                           go=sum(r["verdicts"]["go"] for r in rs), check=sum(r["verdicts"]["check"] for r in rs),
                           stop=sum(r["verdicts"]["stop"] for r in rs), misses_with_go=sum(r["misses_with_go"] for r in rs))
    out["check_causes"] = chk

    # friction trend visible in the estimator (for drift_friction): posterior mean of lots after vs before the drift
    fr = {}
    for sc in ("nominal", "drift_friction"):
        rs = by(sc, "none")
        if not rs:
            continue
        d = []
        for r in rs:
            lots = r["friction_by_lot"]
            first_after = -(-int(r["n_parts"] * 0.5) // 50)
            pre = [x for x in lots if x["lot"] < first_after]
            post = [x for x in lots if x["lot"] >= first_after]
            if pre and post:
                d.append((np.mean([x["mf_post_mean"] for x in post]) - np.mean([x["mf_post_mean"] for x in pre]),
                          np.mean([x["mf_true"] for x in post]) - np.mean([x["mf_true"] for x in pre])))
        if d:
            d = np.array(d)
            fr[sc] = dict(runs=len(d), mean_estimated_friction_rise=float(d[:, 0].mean()), mean_true_friction_rise=float(d[:, 1].mean()))
    out["friction_trend"] = fr
    return out


def make_figure(agg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    labels, lo, hi, mid = [], [], [], []
    for k, v in agg["acceptance"].items():
        r = v["after_drift"]
        if r["rate"] is None:
            continue
        labels.append(k.replace("/", "\n"))
        mid.append(r["rate"])
        lo.append(r["rate"] - r["wilson95"][0])
        hi.append(r["wilson95"][1] - r["rate"])
    ax[0].errorbar(range(len(mid)), mid, yerr=[lo, hi], fmt="o", capsize=3)
    ax[0].set_xticks(range(len(mid)))
    ax[0].set_xticklabels(labels, fontsize=6)
    ax[0].set_ylabel("share within tolerance after drift start")
    ax[0].set_ylim(0, 1.02)
    names, med, e_lo, e_hi = [], [], [], []
    for sc, d in agg["detection"].items():
        for f, v in d.items():
            if v["delay_parts"]["median"] is not None and f != "peak_force_norm":
                names.append(f"{sc}\n{f}")
                med.append(v["delay_parts"]["median"])
                e_lo.append(v["delay_parts"]["median"] - v["delay_parts"]["q25"])
                e_hi.append(v["delay_parts"]["q75"] - v["delay_parts"]["median"])
    ax[1].errorbar(range(len(med)), med, yerr=[e_lo, e_hi], fmt="o", capsize=3)
    ax[1].set_xticks(range(len(med)))
    ax[1].set_xticklabels(names, fontsize=6)
    ax[1].set_ylabel("detection delay (parts), median and IQR")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--parts", type=int, default=600)
    ap.add_argument("--model", default="serving.model_api:load_stub")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--workers", type=int, default=3, help="parallel demos (capped at 3)")
    ap.add_argument("--work", default=None, help="cache folder for per-run results (default: <out stem>_work next to --out)")
    ap.add_argument("--out", default="results/plant/experiments.json")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--recoveries", default=",".join(RECOVERIES))
    a = ap.parse_args(argv)
    out = Path(a.out)
    work = Path(a.work) if a.work else out.with_name(out.stem + "_work")
    seeds = list(range(a.first_seed, a.first_seed + a.seeds))
    t0 = time.time()
    runs = run_all(seeds, work, a.parts, a.model, a.artifact, a.data, a.workers,
                   tuple(a.scenarios.split(",")), tuple(a.recoveries.split(",")))
    agg = aggregate(runs)
    figs = []
    if "acceptance" in agg and agg["detection"]:
        make_figure(agg, out.parent / "figures" / "experiments_summary.png")
        figs = [f"{(out.parent / 'figures' / 'experiments_summary.png').as_posix()}"]
    res = {"task": "P8", "title": f"Plant experiments: {len(seeds)} seeds x drift types x recovery",
           "command": "python -m plant.experiments " + " ".join(argv if argv is not None else sys.argv[1:]),
           "git": _git_sha(), "data": {"dataset": a.data, "model": a.model, "artifact": a.artifact},
           "metrics": agg, "figures": figs,
           "per_run_wall_s_median": float(np.median([r["harness_wall_s"] for r in runs if "harness_wall_s" in r] or [0])),
           "notes": ("Simulated press; drift is injected into simulated sensors/lots; truth and labels come from the same "
                     "solver the model was trained on. Parts within a run are correlated: use the cluster bootstrap "
                     "intervals, not only the Wilson ones. Nothing is validated against a real machine."),
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "wall_clock_s": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: agg[k] for k in ("false_alarms", "detection", "recovery")}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
