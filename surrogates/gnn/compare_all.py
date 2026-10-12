"""All-variant table (GNN base/jloss/idloss/mixmesh, POD+GP field baseline, scalar MLP): field nRMSE and inner-diameter
RMSE per split, read only from the results files. Written to results/gnn/variants_comparison_full.json.

    python -m surrogates.gnn.compare_all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from surrogates.gnn.results import write_result

R = Path("results/gnn")
SPLITS = ("test_random", "test_ood_geometry", "test_fine_mesh")
KEYS = ("nrmse_dr", "nrmse_dz", "nrmse_strain", "id_rmse_pct_points", "force_mean_rel_err")


def pick(d: dict) -> dict:
    return {k: d[k] for k in KEYS if k in d}


def main():
    out: dict = {}
    for sp in SPLITS:
        out[sp] = {}
    for v in ("base", "jloss", "idloss", "mixmesh"):
        m = json.load(open(R / f"eval_{v}_full/metrics.json"))["metrics"]
        for sp in SPLITS:
            row = {"all": pick(m[sp])}
            for g in ("early_stopped_runs", "completed_runs"):
                if g in m[sp]:
                    row[g] = pick(m[sp][g]) | {"n_runs": m[sp][g]["n_runs"]}
            out[sp][f"gnn_{v}"] = row | {"n_runs": m[sp]["n_runs"]}
    pod = json.load(open(R / "pod_field_baseline_full.json"))["metrics"]
    mlp = json.load(open(R / "n5_mlp_baseline_full.json"))["metrics"]["by_split"]
    for sp in SPLITS:
        out[sp]["pod_gp_field"] = {"all": pick(pod[sp]["gp_pod"]),
                                   **{g: pick(pod[sp]["gp_pod"][g]) for g in ("early_stopped_runs", "completed_runs")
                                      if g in pod[sp]["gp_pod"]}}
        out[sp]["mlp_scalar"] = {"all": {"id_rmse_pct_points": mlp[sp]["id_change_pct"]["rmse"]}}
    # the two questions, mechanically
    q = {}
    for sp in SPLITS:
        q[sp] = {"id_rmse_jloss": out[sp]["gnn_jloss"]["all"]["id_rmse_pct_points"],
                 "id_rmse_idloss": out[sp]["gnn_idloss"]["all"]["id_rmse_pct_points"],
                 "id_rmse_mlp": out[sp]["mlp_scalar"]["all"]["id_rmse_pct_points"],
                 "id_rmse_pod_gp": out[sp]["pod_gp_field"]["all"]["id_rmse_pct_points"],
                 "nrmse_dr_jloss": out[sp]["gnn_jloss"]["all"]["nrmse_dr"],
                 "nrmse_dr_mixmesh": out[sp]["gnn_mixmesh"]["all"]["nrmse_dr"],
                 "nrmse_strain_jloss": out[sp]["gnn_jloss"]["all"]["nrmse_strain"],
                 "nrmse_strain_mixmesh": out[sp]["gnn_mixmesh"]["all"]["nrmse_strain"]}
    p = write_result(R / "variants_comparison_full.json", "N6b", "All GNN variants vs POD+GP and scalar MLP",
                     "python -m surrogates.gnn.compare_all", {"dataset": "full"}, {"by_split": out, "questions": q},
                     notes="Single seed per variant, 30 epochs (compute-budget cap). Seed-to-seed spread was NOT measured, "
                           "so differences of a few percent are not established. mixmesh also trains on 60 fine_train runs.")
    print(p)


if __name__ == "__main__":
    sys.exit(main())
