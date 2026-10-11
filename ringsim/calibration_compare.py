"""Compare ringsim with published ring-test calibration points (harness only).

Run:  python -m ringsim.calibration_compare --table results/solver/published_ring_calibration.yaml \
          --out results/solver --m 0.15 --workers 3

Reads the table, runs ringsim at each row's geometry / friction / reduction and writes
results/solver/published_ring_comparison.json. With no rows it reports
"harness built, no verified source data supplied" and runs nothing. Verification is not validation:
a published chart assumes its own material model, so only agreement in trend is expected.

Closed-form references: the only exact reference derived here is the frictionless homogeneous
compression (inner radius scales with sqrt(h0/h)), already used by check V1; it is not a friction
reference. No analytic reference for sticking friction or the no-slip bound was added.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import yaml

from .solver import Numerics, RingCase, RingSolver

REQUIRED = ("source", "figure", "od_id", "h_id", "mf", "reduction_pct", "id_change_pct", "uncertainty_pct")


def _run(job):
    case = RingCase(**job)
    res = RingSolver(case, Numerics()).run()
    return res.curves["reduction"].tolist(), res.curves["id_change_pct"].tolist()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/solver/published_ring_calibration.yaml")
    ap.add_argument("--out", default="results/solver")
    ap.add_argument("--m", type=float, default=0.15, help="rate exponent for sources without one")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    pts = yaml.safe_load(open(a.table)).get("points") or []
    for i, p in enumerate(pts):
        missing = [k for k in REQUIRED if k not in p]
        if missing:
            raise SystemExit(f"row {i} lacks {missing}; rows without a source are not accepted")
    jobs = [dict(od_id=p["od_id"], h_id=p["h_id"], mf=p["mf"], m=p.get("rate_exponent_m", a.m),
                 reduction=p["reduction_pct"] / 100.0 + 0.01) for p in pts]
    rows = []
    if jobs:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            outs = list(ex.map(_run, jobs))
        import numpy as np
        for p, (red, idc) in zip(pts, outs):
            r = p["reduction_pct"] / 100.0
            sim = float(np.interp(r, red, idc)) if r <= red[-1] + 1e-9 else None
            rows.append(dict(source=p["source"], figure=p["figure"], mf=p["mf"], reduction_pct=p["reduction_pct"],
                             published_pct=p["id_change_pct"], uncertainty_pct=p["uncertainty_pct"],
                             ringsim_pct=sim,
                             diff_points=None if sim is None else sim - p["id_change_pct"],
                             within_uncertainty=None if sim is None else
                             abs(sim - p["id_change_pct"]) <= p["uncertainty_pct"]))
    status = ("harness built, no verified source data supplied" if not pts
              else f"{len(rows)} published points compared")
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except Exception:
        git = "no-git"
    out = dict(task="S7", title="ringsim vs published ring calibration points", command=
               "python -m ringsim.calibration_compare " + " ".join(argv or []), git=git,
               data=dict(dataset="published_ring_calibration.yaml", n_runs=len(rows), splits={}),
               metrics=dict(status=status, rows=rows), figures=[],
               notes="Verification is not validation. " + status, created=datetime.now(timezone.utc).isoformat())
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "published_ring_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(status)


if __name__ == "__main__":
    main()
