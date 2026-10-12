"""Real-data leakage check for the mixmesh experiment: no extra-train case may equal a test case of the main dataset.

A 'case' is the four varied inputs (od_id, h_id, mf, m). Also reports the smallest normalised distance to any test case
(so a near-duplicate that is not an exact match is visible), and how the extra set's own test_random is NOT used.

    python -m surrogates.gnn.leakage_check --extra fine_train --main full --out results/gnn
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from common.data import list_runs
from surrogates.gnn.graph import TEST_SPLITS, case_overlap
from surrogates.gnn.results import write_result

KEYS = ("od_id", "h_id", "mf", "m")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="fine_train")
    ap.add_argument("--main", default="full")
    ap.add_argument("--out", default="results/gnn")
    a = ap.parse_args(argv)
    ex = [r for r in list_runs(a.extra, "train") if r.get("status", "ok") == "ok"]
    te = [r for sp in TEST_SPLITS for r in list_runs(a.main, sp)]
    E = np.array([[r[k] for k in KEYS] for r in ex]); T = np.array([[r[k] for k in KEYS] for r in te])
    allc = np.array([[r[k] for k in KEYS] for sp in ("train",) + TEST_SPLITS for r in list_runs(a.main, sp)])
    sc = allc.std(0)
    d = np.sqrt((((E[:, None, :] - T[None, :, :]) / sc) ** 2).sum(-1))
    exact = {sp: case_overlap(a.extra, "train", a.main, (sp,)) for sp in TEST_SPLITS}
    m = {"n_extra_train_runs": len(ex), "n_main_test_runs": len(te),
         "exact_overlaps_per_test_split": {sp: len(v) for sp, v in exact.items()},
         "min_normalised_distance_to_any_test_case": float(d.min()),
         "median_nearest_test_distance": float(np.median(d.min(1))),
         "extra_nz": sorted({int(r["nz"]) for r in ex}),
         "method": "exact match of (od_id, h_id, mf, m) within 1e-9, plus nearest-neighbour distance in units of the "
                   "per-input standard deviation over all main-dataset cases"}
    p = write_result(f"{a.out}/mixmesh_leakage_check.json", "N4b", "case-overlap check: extra train vs main test splits",
                     "python -m surrogates.gnn.leakage_check " + " ".join(sys.argv[1:]),
                     {"dataset": a.main, "n_runs": len(ex)}, m,
                     notes="Only the extra dataset's `train` split is used for training. Its own test_random is not used.")
    print(p, m)


if __name__ == "__main__":
    sys.exit(main())
