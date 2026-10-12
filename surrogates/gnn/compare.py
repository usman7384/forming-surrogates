"""Side-by-side base (lambda_j=0) vs jloss (lambda_j=1) from the two evaluation files, and who wins per metric.

Why: the comparison must be honest, so the 'winner' is computed mechanically (lower is better for every metric
here) and ties/losses for the physics loss are reported as such.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from surrogates.gnn.results import write_result

KEYS = ["nrmse_dr", "nrmse_dz", "nrmse_strain", "force_rmse", "force_mean_rel_err", "id_rmse_pct_points",
        "mean_abs_J_minus_1_pred"]


def main():
    out = Path("results/gnn")
    base = json.load(open(out / "eval_base_full/metrics.json"))["metrics"]
    jl = json.load(open(out / "eval_jloss_full/metrics.json"))["metrics"]
    rows, wins = {}, {"jloss": 0, "base": 0}
    for split, mb in base.items():
        if not isinstance(mb, dict) or "n_runs" not in mb:
            continue
        pairs = {"all": (mb, jl[split])}
        pairs.update({g: (mb[g], jl[split][g]) for g in ("early_stopped_runs", "completed_runs") if g in mb})
        for grp, (a, b) in pairs.items():
            for k in KEYS:
                if k in a and k in b:
                    w = "jloss" if b[k] < a[k] else "base"
                    wins[w] += 1
                    rows[f"{split}/{grp}/{k}"] = {"base": a[k], "jloss": b[k], "lower_is_better_winner": w,
                                                  "rel_change_pct": 100 * (b[k] - a[k]) / abs(a[k]) if a[k] else None,
                                                  "n_runs": a["n_runs"]}
    cmd = "python -m surrogates.gnn.compare"
    p = write_result(out / "n4_base_vs_jloss_full.json", "N4", "base (lambda_j=0) vs jloss (lambda_j=1), 30 epochs each",
                     cmd, {"dataset": "full"}, {"rows": rows, "metric_wins": wins},
                     notes="30 epochs is a compute-budget cap, not a tuned value; single seed, so small differences are "
                           "within run-to-run noise that was not measured.")
    print(p, wins)


if __name__ == "__main__":
    sys.exit(main())
