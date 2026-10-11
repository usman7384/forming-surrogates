"""Facts about finished datasets, written to results JSON so docs can cite a file.

Run:  python -m ringsim.dataset_facts --full full --unstructured unstructured --out results/solver

* censoring  (results/solver/censoring.json): for every friction bin of the training box and every height
  reduction, the fraction of ok runs that reached that reduction before a fold-over stop. Runs stopped early
  are right-censored: their curves end, so a surrogate has no data beyond that point for them.
* unstructured facts (results/solver/unstructured_dataset_facts.json): counts of ok / early-stopped runs
  and the worst volume drift of the unstructured test dataset.
Datasets are read through common.data (names resolve under $FS_DATA_ROOT).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np

from common.data import dataset_dir, iter_runs, list_runs

REDUCTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
MF_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "no-git"


def censoring(dataset: str):
    mf, mred, split = [], [], []
    for run in iter_runs(dataset):
        mf.append(run.case["mf"])
        mred.append(run.max_reduction)
        split.append(run.split)
    mf, mred = np.array(mf), np.array(mred)
    cells = []
    for lo, hi in zip(MF_EDGES[:-1], MF_EDGES[1:]):
        sel = (mf >= lo) & (mf < hi)
        cells.append(dict(mf_lo=lo, mf_hi=min(hi, 1.0), n_runs=int(sel.sum()),
                          fraction_reached={f"{r:.2f}": (float((mred[sel] >= r - 1e-9).mean()) if sel.any() else None)
                                            for r in REDUCTIONS},
                          median_final_reduction=float(np.median(mred[sel])) if sel.any() else None))
    early = int((mred < max(REDUCTIONS) - 1e-9).sum())
    return cells, len(mred), early


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="full")
    ap.add_argument("--unstructured", default="unstructured")
    ap.add_argument("--out", default="results/solver")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cmd = "python -m ringsim.dataset_facts " + " ".join(argv or [])
    now = datetime.now(timezone.utc).isoformat()

    cells, n, early = censoring(a.full)
    splits = {}
    for r in list_runs(a.full):
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    out = dict(task="S8", title="Right-censoring of the dataset by fold-over stops", command=cmd, git=_git(),
               data=dict(dataset=a.full, n_runs=n, splits=splits),
               metrics=dict(reductions=REDUCTIONS, n_ok_runs=n, n_stopped_before_35pct=early,
                            fraction_stopped_before_35pct=early / n, cells=cells),
               figures=[], notes="All splits pooled, ok runs only. fraction_reached = share of runs in the "
                                 "friction bin whose curve extends to that reduction.", created=now)
    json.dump(out, open(os.path.join(a.out, "censoring.json"), "w"), indent=1)

    rows = list_runs(a.unstructured, ok_only=False)
    ok = [r for r in rows if r.get("status") == "ok"]
    vd = max(abs(float(r["volume_drift_pct"])) for r in ok)
    n_early = sum(1 for r in ok if str(r["stopped_early"]) == "True")
    out = dict(task="S4", title="Unstructured test dataset facts", command=cmd, git=_git(),
               data=dict(dataset=a.unstructured, n_runs=len(ok), splits={"test_unstructured": len(ok)}),
               metrics=dict(n_planned=len(rows), n_ok=len(ok), n_stopped_early=n_early, worst_volume_drift_pct=vd),
               figures=[], notes="From index.csv via common.data.list_runs.", created=now)
    json.dump(out, open(os.path.join(a.out, "unstructured_dataset_facts.json"), "w"), indent=1)
    print(json.dumps(out["metrics"]), f"\nfull: {n} runs, {early} stopped before 35%")


if __name__ == "__main__":
    main()
