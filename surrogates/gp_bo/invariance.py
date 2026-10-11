"""G2: invariance demo. Material strength K must not change the shape (inner diameter), only the force.

    python -m surrogates.gp_bo.invariance --data smoke --out results/gp

1. Re-solve a few dataset cases with K = 0.6 and 1.6 (about 10 extra solver runs) and compare to the
   dataset run with K = 1: inner diameter should be identical, normalised force identical, raw force = K x.
2. Fit an ARD GP with log K as a sixth input. For the inner diameter its length scale should run to the
   upper bound (irrelevant input); for the raw log force it should stay finite (log F = log K + ...).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.data import iter_runs  # noqa: E402
from surrogates.gp_bo.gp import FEATURES, HI, LO, fit_gp, public_info  # noqa: E402
from surrogates.gp_bo.results_io import command_line, write_result  # noqa: E402
from surrogates.gp_bo.solver_calls import curve_at, run_cases  # noqa: E402

REDS = (0.1, 0.2, 0.3)
K_VALUES = (0.6, 1.6)
LO6 = np.append(LO, np.log(0.5))
HI6 = np.append(HI, np.log(2.0))
NAMES6 = tuple(FEATURES) + ("logK",)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--out", default="results/gp")
    ap.add_argument("--n-cases", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2, help="torch threads (pinned; low default: background job)")
    a = ap.parse_args(argv)
    cmd = command_line("surrogates.gp_bo.invariance", argv)
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    runs = [r for r in iter_runs(a.data, "train") if r.max_reduction >= REDS[-1] - 1e-9]
    idx = np.unique(np.linspace(0, len(runs) - 1, a.n_cases).round().astype(int))
    base = [runs[i] for i in idx]
    num = dict(nz=int(8))
    cases = []
    for r in base:
        c = r.case
        for K in K_VALUES:
            cases.append(dict(od_id=c["od_id"], h_id=c["h_id"], mf=c["mf"], m=c["m"], reduction=REDS[-1],
                              K=K, V=1.0))
    res = run_cases(cases, out / "k_solver_cache", numerics=num, workers=a.workers)

    # 1. exactness of the invariance
    d_id, d_fn, d_raw = [], [], []
    rows_id, rows_f = [], []           # K != 1 rows for the GP
    for i, r in enumerate(base):
        id0, fn0, f0 = (r.at(REDS, "id_change_pct"), r.at(REDS, "force_norm"), r.at(REDS, "force"))
        for j, K in enumerate(K_VALUES):
            s = res[i * len(K_VALUES) + j]
            d_id.append(np.max(np.abs(curve_at(s, REDS) - id0)))
            fn = curve_at(s, REDS, "force_norm")
            d_fn.append(np.max(np.abs(fn / fn0 - 1)))
            raw = curve_at(s, REDS, "force")
            d_raw.append(np.max(np.abs(raw / (K * f0) - 1)))
            c = r.case
            for k, red in enumerate(REDS):
                x = [c["od_id"], c["h_id"], c["mf"], c["m"], red, np.log(K)]
                rows_id.append((x, curve_at(s, REDS)[k]))
                rows_f.append((x, np.log(raw[k])))
    # K = 1 rows: every train run of the dataset
    for r in iter_runs(a.data, "train"):
        c = r.case
        idv, fv = r.at(REDS, "id_change_pct"), r.at(REDS, "force")
        for k, red in enumerate(REDS):
            if np.isnan(idv[k]):
                continue
            x = [c["od_id"], c["h_id"], c["mf"], c["m"], red, 0.0]
            rows_id.append((x, idv[k]))
            rows_f.append((x, np.log(fv[k])))

    X = np.array([x for x, _ in rows_id])
    out_gp = {}
    for tname, rows in (("id_change_pct", rows_id), ("log_force_raw", rows_f)):
        y = np.array([v for _, v in rows])
        gp6, info6 = fit_gp(X, y, lo=LO6, hi=HI6, names=NAMES6, iters=a.iters, n_restarts=2, threads=a.threads)
        gp5, info5 = fit_gp(X[:, :5], y, iters=a.iters, n_restarts=2, threads=a.threads)
        ls = info6["lengthscales"]
        others = [v for k, v in ls.items() if k != "logK"]
        out_gp[tname] = dict(with_K=public_info(info6), without_K_neg_mll_per_point=info5["neg_mll_per_point"],
                             K_lengthscale=ls["logK"], K_over_median_other=ls["logK"] / float(np.median(others)))

    metrics = dict(
        n_extra_solver_runs=len(cases), K_values=list(K_VALUES), n_base_cases=len(base),
        max_abs_id_difference_pct_points=float(max(d_id)),
        max_rel_force_norm_difference=float(max(d_fn)),
        max_rel_raw_force_vs_K_times_base=float(max(d_raw)),
        gp_with_K_input=out_gp,
        upper_bound_lengthscale_unit_box=1e3,
        K_range_for_unit_box=[0.5, 2.0])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.38
    for k, tname in enumerate(out_gp):
        ls = list(out_gp[tname]["with_K"]["lengthscales"].values())
        ax.bar(np.arange(len(NAMES6)) + (k - 0.5) * w, ls, w, label=tname)
    ax.set_xticks(range(len(NAMES6)), NAMES6)
    ax.set_yscale("log")
    ax.set_ylabel("ARD length scale (fraction of input range)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fp = out / "figures" / "ard_with_K.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    write_result(out / "invariance_K.json", task="G2", title="Invariance of shape to material strength K",
                 command=cmd, data=dict(dataset=a.data, n_runs=len(cases), splits={},
                                        gp_rows=int(len(X))), metrics=metrics, figures=[str(fp)],
                 notes="Extra runs re-solve dataset train cases with K = 0.6 and 1.6. GP: ARD over the five "
                       "features plus log K, fit on dataset train rows (K = 1) and the extra rows.")
    print({k: v for k, v in metrics.items() if k != "gp_with_K_input"})
    for t, g in out_gp.items():
        print(t, "K length scale", g["K_lengthscale"], "ratio", g["K_over_median_other"],
              {k: round(v, 3) for k, v in g["with_K"]["lengthscales"].items()})


if __name__ == "__main__":
    main()
