"""Task N2: measure J = (A/A0)(rbar/rbar0) on the SOLVER's own meshes before using it as a loss.

If the solver's J is far from 1 the physics loss would pull the network away from the data, so we
need this distribution first. Also checks that the inner-diameter probe node reproduces the stored
id_change_pct curve (used later to read the scalar answer off predicted nodes).

    python -m surrogates.gnn.j_stats --data smoke --out results/gnn
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from common.data import iter_runs, list_runs

from .graph import boundary_flags, id_change_pct, j_ratio
from .results import write_result


def summarise(x: np.ndarray) -> dict:
    q = np.quantile(x, [0.5, 0.9, 0.99])
    return {"mean": float(x.mean()), "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(x.max()), "n": int(x.size)}


def run(dataset: str) -> tuple[dict, dict]:
    per_run, allx, by_split = {}, [], {}
    id_err = []
    mean_J_dev = []
    for r in iter_runs(dataset):
        p, tri = r.fields["p"].astype(float), r.fields["t"].astype(int)
        J = j_ratio(p[1:], p[0], tri)                           # (S-1, T), skip the trivial step 0
        dev = np.abs(J - 1.0)
        per_run[r.run_id] = {"split": r.split, "stopped_early": bool(r.info["stopped_early"]),
                             "mean_abs_J_minus_1": float(dev.mean()), "max_abs_J_minus_1": float(dev.max()),
                             "worst_step_mean": float(dev.mean(1).max())}
        allx.append(dev.ravel())
        by_split.setdefault(r.split, []).append(dev.ravel())
        mean_J_dev.append(float((J - 1.0).mean()))
        # inner-diameter probe vs stored curve
        fl = boundary_flags(p[0])
        i = int(np.where((fl[:, 0] > 0) & (fl[:, 3] > 0))[0][0])
        probe = id_change_pct(p[:, 0, i], p[0, 0, i])
        stored = r.curves["id_change_pct"][r.fields["step"].astype(int)]
        id_err.append(float(np.abs(probe - stored).max()))
    metrics = {
        "abs_J_minus_1_all": summarise(np.concatenate(allx)),
        "abs_J_minus_1_by_split": {k: summarise(np.concatenate(v)) for k, v in by_split.items()},
        "signed_mean_J_minus_1_per_run_mean": float(np.mean(mean_J_dev)),
        "per_run": per_run,
        "id_probe_vs_stored_curve_max_abs_err_pct": float(np.max(id_err)),
    }
    return metrics, {"dataset": dataset, "n_runs": len(per_run)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--out", default="results/gnn")
    a = ap.parse_args(argv)
    metrics, data = run(a.data)
    data["splits"] = {s: int(sum(1 for v in list_runs(a.data, s))) for s in
                      ["train", "test_random", "test_ood_geometry", "test_fine_mesh"]}
    cmd = "python -m surrogates.gnn.j_stats " + " ".join(sys.argv[1:])
    path = write_result(f"{a.out}/n2_j_distribution_{a.data}.json", "N2",
                        "Per-triangle volume ratio J on solver data", cmd, data, metrics,
                        notes="J=(A/A0)*(rbar/rbar0) from saved node positions, steps 1.. of every ok run; "
                              "stopped-early runs included (flagged per run).")
    print(path, metrics["abs_J_minus_1_all"])


if __name__ == "__main__":
    main()
