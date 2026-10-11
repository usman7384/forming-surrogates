"""Parameter campaign: sample dimensionless ring cases, run them in parallel, write a dataset.

Run (full):   python -m ringsim.campaign --out data --n-train 300 --n-ood 40 --n-fine 20 --workers 8
Run (smoke):  python -m ringsim.campaign --out data_smoke --n-train 6 --n-ood 2 --n-fine 2 --workers 2

Inputs are dimensionless because the solver proves (verification V2) that shape results do not
depend on K or the die speed V; force is stored normalised by K (V/h0)^m A0. K = V = 1 throughout.

Splits:
  train / test_random  : Latin hypercube sample inside the training box (test_random = every 10th point)
  test_ood_geometry    : rings outside the training box in outer-diameter or height ratio
  test_unstructured    : (only with --mesh unstructured) fresh Latin hypercube points inside the training
                         box, solved on jittered Delaunay meshes (a different mesh seed per run);
                         tests whether a surrogate learned the structured mesh instead of the physics
  test_fine_mesh       : copies of random test points re-run on a finer mesh
                         (tests discretisation dependence, not generalisation)

Every run writes runs/run_XXXX.npz (fields at saved steps + curves) and runs/run_XXXX.json
(inputs, outputs, solver version, git hash). index.csv lists all runs. Re-running the command skips
finished runs, so a crashed or interrupted campaign resumes where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import numpy as np
from scipy.stats import qmc

from .solver import SOLVER_VERSION, Numerics, RingCase, RingSolver, save_run

# training box (dimensionless)
BOX = dict(od_id=(1.6, 2.4), h_id=(0.4, 1.0), mf=(0.0, 1.0), m=(0.05, 0.3))
OOD = dict(od_id=(2.4, 2.8), h_id=(1.0, 1.25))
REDUCTION = 0.35          # target; sticking-friction runs may stop earlier (fold-over) and are flagged


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "no-git"


def build_plan(n_train, n_ood, n_fine, seed, nz):
    keys = list(BOX)
    lhs = qmc.LatinHypercube(d=len(keys), seed=seed)
    X = qmc.scale(lhs.random(n_train), [BOX[k][0] for k in keys], [BOX[k][1] for k in keys])
    plan = []
    for i, x in enumerate(X):
        split = "test_random" if i % 10 == 9 else "train"
        plan.append(dict(split=split, nz=nz, **{k: float(v) for k, v in zip(keys, x)}))
    rng = np.random.default_rng(seed + 1)
    for i in range(n_ood):
        row = {k: float(rng.uniform(*BOX[k])) for k in keys}
        # push one geometry ratio outside the box (alternate between the two)
        g = "od_id" if i % 2 == 0 else "h_id"
        row[g] = float(rng.uniform(*OOD[g]))
        plan.append(dict(split="test_ood_geometry", nz=nz, **row))
    rnd = [p for p in plan if p["split"] == "test_random"][:n_fine]
    for p in rnd:
        plan.append(dict(p, split="test_fine_mesh", nz=int(round(nz * 1.5))))
    for i, p in enumerate(plan):
        p["run_id"] = f"run_{i:04d}"
    return plan


def build_plan_unstructured(n, seed, nz):
    """n test points inside the training box (own seed, so they differ from the structured plan)."""
    keys = list(BOX)
    X = qmc.scale(qmc.LatinHypercube(d=len(keys), seed=seed + 100).random(n),
                  [BOX[k][0] for k in keys], [BOX[k][1] for k in keys])
    plan = []
    for i, x in enumerate(X):
        plan.append(dict(split="test_unstructured", nz=nz, run_id=f"run_{i:04d}", mesh_seed=1000 + i,
                         **{k: float(v) for k, v in zip(keys, x)}))
    return plan


def _one(job):
    out, p, num_kw = job
    stem = os.path.join(out, "runs", p["run_id"])
    if os.path.exists(stem + ".json"):
        return p["run_id"], "skipped", None
    try:
        case = RingCase(od_id=p["od_id"], h_id=p["h_id"], mf=p["mf"], m=p["m"],
                        reduction=REDUCTION)
        mesh_kw = dict(mesh="unstructured", mesh_seed=p["mesh_seed"]) if "mesh_seed" in p else {}
        res = RingSolver(case, Numerics(nz=p["nz"], **num_kw, **mesh_kw)).run()
        res.info.update(split=p["split"], run_id=p["run_id"], git=_git_hash(),
                        python=platform.python_version())
        save_run(res, stem)
        return p["run_id"], "ok", None
    except Exception:
        return p["run_id"], "failed", traceback.format_exc()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=300, help="Latin hypercube points (10%% become test_random)")
    ap.add_argument("--n-ood", type=int, default=40)
    ap.add_argument("--n-fine", type=int, default=20)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--dh-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mesh", choices=["structured", "unstructured"], default="structured",
                    help="'unstructured' writes only the test_unstructured split (--n-unstructured runs)")
    ap.add_argument("--n-unstructured", type=int, default=40)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)

    plan_path = os.path.join(a.out, "plan.json")
    if os.path.exists(plan_path):                      # resume with the exact same plan
        plan = json.load(open(plan_path))
    else:
        plan = (build_plan_unstructured(a.n_unstructured, a.seed, a.nz) if a.mesh == "unstructured"
                else build_plan(a.n_train, a.n_ood, a.n_fine, a.seed, a.nz))
        json.dump(plan, open(plan_path, "w"), indent=1)

    num_kw = dict(dh_frac=a.dh_frac)
    jobs = [(a.out, p, num_kw) for p in plan]
    print(f"{len(plan)} runs planned, {a.workers} workers")
    with ProcessPoolExecutor(max_workers=a.workers) as ex, \
            open(os.path.join(a.out, "failures.jsonl"), "a") as flog:
        futs = [ex.submit(_one, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            rid, status, tb = f.result()
            if status == "failed":
                flog.write(json.dumps(dict(run_id=rid, traceback=tb)) + "\n")
                flog.flush()
            print(f"  [{n}/{len(plan)}] {rid}: {status}", flush=True)

    write_index(a.out, plan)


def write_index(out, plan):
    rows = []
    for p in plan:
        path = os.path.join(out, "runs", p["run_id"] + ".json")
        if not os.path.exists(path):
            rows.append(dict(run_id=p["run_id"], split=p["split"], status="failed"))
            continue
        meta = json.load(open(path))
        c, q, info = meta["case"], meta["qoi"], meta["info"]
        rows.append(dict(run_id=p["run_id"], split=p["split"], status="ok",
                         od_id=c["od_id"], h_id=c["h_id"], mf=c["mf"], m=c["m"],
                         nz=meta["numerics"]["nz"], stopped_early=info["stopped_early"],
                         final_reduction=q["final_reduction"], id_change_pct=q["id_change_pct"],
                         id_min_change_pct=q["id_min_change_pct"],
                         newton_unconverged=info["newton_unconverged_solves"],
                         peak_force_norm=q["peak_force_norm"], barreling=q["barreling"],
                         max_strain=q["max_strain"], volume_drift_pct=q["volume_drift_pct"],
                         runtime_s=info["runtime_s"], solver_version=info["solver_version"]))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                  if k in rows[0] else 99)
    with open(os.path.join(out, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    card = [
        "# Dataset card", "",
        f"- Solver version: {SOLVER_VERSION}; physics: isothermal rigid-viscoplastic (Norton-Hoff), "
        "regularised Tresca friction, axisymmetric half ring, no remeshing.",
        f"- Training box: {BOX}; out-of-distribution geometry: {OOD}; target reduction {REDUCTION}.",
        f"- Runs: {len(rows)} planned, {len(ok)} ok, {len(rows) - len(ok)} failed, "
        f"{sum(1 for r in ok if r['stopped_early'])} stopped early by fold-over (kept, flagged).",
        "- Splits: " + ", ".join(f"{s}: {sum(1 for r in ok if r['split'] == s)}"
                                 for s in ["train", "test_random", "test_ood_geometry", "test_fine_mesh",
                                           "test_unstructured"]
                                 if s != "test_unstructured" or any(r["split"] == s for r in ok)),
        f"- Worst volume drift: {max((abs(r['volume_drift_pct']) for r in ok), default=0):.2f}%.",
        "- Known limits: no remeshing (sticking runs stop near 30% reduction); no temperature; "
        "flat dies; 2D axisymmetric only, so no ovality.",
        "- Force is normalised by K (V/h0)^m A0; shape results are independent of K and V (checked).",
        f"- Newton: {sum(int(r.get('newton_unconverged', 0)) for r in ok)} solves did not reach the 1e-8 "
        "tolerance within the iteration cap (see each run's warnings).",
        "- IMPORTANT: QoIs in this index are at each run's FINAL reduction, which differs for runs that "
        "stopped early. Build ML targets from the per-step curves at fixed reductions instead.",
        "- Inner diameter: id_change_pct is at mid-height; id_min_change_pct is the minimum over the inner "
        "surface (the two differ by a few points when the inner surface barrels).",
    ]
    if any(r["split"] == "test_unstructured" for r in rows):
        card.append("- Mesh: every run uses a jittered Delaunay triangulation (Numerics.mesh='unstructured', "
                    "per-run mesh_seed in plan.json and in each run's numerics), same nz as the structured "
                    "training runs. Points are drawn from the training box; this dataset is a test split only.")
    with open(os.path.join(out, "dataset_card.md"), "w") as fh:
        fh.write("\n".join(card) + "\n")
    print("\n".join(card))


if __name__ == "__main__":
    main()
