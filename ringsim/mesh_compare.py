"""Task S4 summary: how much do results change when the same cases run on unstructured meshes?

Run:  python -m ringsim.mesh_compare --unstructured $FS_DATA_ROOT/unstructured \
          --twin $FS_DATA_ROOT/unstructured_twin --out results/solver --workers 4

Re-solves every case of the unstructured dataset on the default structured mesh (the "twin", same nz,
same inputs; resumable) and writes results/solver/s4_unstructured_vs_structured.json (results-JSON schema
results schema) with the inner-diameter and normalised-force differences at fixed reductions.
This is a discretisation-sensitivity measurement, not an accuracy claim: neither mesh is the exact solution.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from .campaign import _one, write_index
from concurrent.futures import ProcessPoolExecutor

REDS = (0.1, 0.2, 0.3)


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "no-git"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--unstructured", required=True)
    ap.add_argument("--twin", required=True)
    ap.add_argument("--out", default="results/solver")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    plan = json.load(open(os.path.join(a.unstructured, "plan.json")))
    twin_plan = [{k: v for k, v in p.items() if k != "mesh_seed"} for p in plan]
    os.makedirs(os.path.join(a.twin, "runs"), exist_ok=True)
    json.dump(twin_plan, open(os.path.join(a.twin, "plan.json"), "w"), indent=1)
    jobs = [(a.twin, p, dict(dh_frac=0.01)) for p in twin_plan]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rid, status, tb in ex.map(_one, jobs):
            if status == "failed":
                print(rid, tb)
    write_index(a.twin, twin_plan)

    from common.data import load_run
    d_id = {r: [] for r in REDS}
    d_f = {r: [] for r in REDS}
    for p in plan:
        u, s = load_run(a.unstructured, p["run_id"]), load_run(a.twin, p["run_id"])
        for r in REDS:
            iu, is_ = u.at([r])[0], s.at([r])[0]
            fu, fs = u.at([r], "force_norm")[0], s.at([r], "force_norm")[0]
            if np.isfinite(iu) and np.isfinite(is_):
                d_id[r].append(iu - is_)
                d_f[r].append(100.0 * (fu - fs) / fs)
    metrics = {}
    for r in REDS:
        di, df = np.array(d_id[r]), np.array(d_f[r])
        metrics[f"reduction_{r}"] = dict(
            n_pairs=int(di.size), id_diff_points_mean=float(di.mean()), id_diff_points_rms=float(np.sqrt((di ** 2).mean())),
            id_diff_points_max_abs=float(np.abs(di).max()), force_diff_pct_mean=float(df.mean()),
            force_diff_pct_rms=float(np.sqrt((df ** 2).mean())), force_diff_pct_max_abs=float(np.abs(df).max()))
    n_ok = sum(1 for p in plan if os.path.exists(os.path.join(a.unstructured, "runs", p["run_id"] + ".json")))
    out = dict(task="S4", title="Unstructured vs structured mesh, same 40 cases",
               command="python -m ringsim.mesh_compare " + " ".join(sys.argv[1:] if argv is None else argv),
               git=_git(), data=dict(dataset="unstructured", n_runs=n_ok, splits={"test_unstructured": n_ok}),
               metrics=metrics, figures=[],
               notes="id_diff = unstructured minus structured, in percentage points of inner-diameter change; "
                     "force_diff in % of the structured normalised force. Pairs where either run stopped "
                     "before the reduction are skipped. Differences measure mesh sensitivity at nz=8, not error.",
               created=datetime.now(timezone.utc).isoformat())
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "s4_unstructured_vs_structured.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
