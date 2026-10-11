"""G1: scalar surrogates (GP + baselines) on the fixed-reduction table; writes metrics, figures, artifacts.

    python -m surrogates.gp_bo.run --data smoke --out results/gp

Targets: inner-diameter change (%) and log normalised force, at reductions 0.1/0.2/0.3 taken from the
per-step curves through `common.data.scalar_table` (never from index.csv). Fit on `train` only; every
other split is held out by run. The trained GP posteriors are written to `artifacts/gp/` for the API.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.data import FEATURES, list_runs, scalar_table  # noqa: E402
from surrogates.baselines.scalar_gbt import GBTBaseline  # noqa: E402
from surrogates.baselines.scalar_poly import PolyBaseline  # noqa: E402
from surrogates.gp_bo.calibration import (conformal_scale, crps_gaussian, pit_histogram, pit_values,  # noqa: E402
                                          split_runs)
from surrogates.gp_bo.gp import DEFAULT_THREADS, NumpyGP, fit_gp, public_info  # noqa: E402
from surrogates.gp_bo.support import build_support, in_support, save_support  # noqa: E402
from surrogates.gp_bo.results_io import command_line, git_sha, write_result  # noqa: E402

EVAL_SPLITS = ["test_random", "test_ood_geometry", "test_fine_mesh"]
TARGETS = {"id_change_pct": "inner-diameter change (%)", "log_force_norm": "log normalised force"}


def load_tables(dataset: str, reductions) -> dict[str, dict]:
    """splits -> {X, Y (id %, log force*), groups}; Y columns follow TARGETS."""
    out = {}
    for split in ["train", *EVAL_SPLITS]:
        X, Y, G, _ = scalar_table(dataset, reductions=reductions, split=split)
        if len(X) == 0:
            continue
        stopped = np.array([bool(r["stopped_early"]) for r in list_runs(dataset, split)])
        out[split] = dict(X=X, Y=np.column_stack([Y[:, 0], np.log(Y[:, 1])]), G=G, early=stopped[G],
                          n_runs=len(stopped), n_runs_early=int(stopped.sum()))
    return out


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def coverage_curve(y, mu, sd, levels):
    """Empirical coverage of central intervals at several nominal levels."""
    from scipy.stats import norm
    z = np.abs(np.asarray(y) - mu) / sd
    return [float(np.mean(z <= norm.ppf(0.5 + lv / 2))) for lv in levels]


class ScaledGP:
    """A GP whose predictive sd is multiplied by a (split-conformal) factor; the mean is unchanged."""

    def __init__(self, gp, factor: float):
        self.gp, self.factor = gp, float(factor)

    def predict(self, X):
        mu, sd = self.gp.predict(X)
        return mu, sd * self.factor


def evaluate(tables, models: dict) -> dict:
    """RMSE (all models); coverage, mean sd, CRPS and PIT histogram for probabilistic models ('gp*')."""
    m = {}
    for ti, tname in enumerate(TARGETS):
        for split, tab in tables.items():
            y = tab["Y"][:, ti]
            for mname, mod in models[tname].items():
                if mname.startswith("gp"):
                    mu, sd = mod.predict(tab["X"])
                    m.setdefault(f"rmse_{tname}", {}).setdefault(mname, {})[split] = rmse(y, mu)
                    m.setdefault(f"coverage95_{tname}", {}).setdefault(mname, {})[split] = coverage_curve(
                        y, mu, sd, [0.95])[0]
                    m.setdefault(f"mean_std_{tname}", {}).setdefault(mname, {})[split] = float(np.mean(sd))
                    m.setdefault(f"crps_{tname}", {}).setdefault(mname, {})[split] = float(
                        np.mean(crps_gaussian(y, mu, sd)))
                    m.setdefault(f"pit_hist10_{tname}", {}).setdefault(mname, {})[split] = pit_histogram(y, mu, sd)
                else:
                    m.setdefault(f"rmse_{tname}", {}).setdefault(mname, {})[split] = rmse(
                        y, mod.predict(tab["X"]))
    return m


def mlp_ensemble_metrics(path: str, tables) -> dict | None:
    """Pull the MLP-ensemble numbers (same rows) from the GNN engineer's results file; None if unavailable."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.load(open(p))
    by = d["metrics"]["by_split"]
    out = {"source": path, "command": d["command"], "members": d["metrics"].get("members")}
    for tname in TARGETS:
        for kind in ("rmse", "coverage95", "mean_std"):
            out.setdefault(f"{kind}_{tname}", {})
        for split, tab in tables.items():
            if split not in by or by[split][tname]["n_rows"] != len(tab["X"]):
                return None                  # not the same rows: refuse to compare
            for kind in ("rmse", "coverage95", "mean_std"):
                out[f"{kind}_{tname}"][split] = by[split][tname][kind]
    return out


def early_stop_report(tables, models, reductions) -> dict:
    """How runs that folded over early enter the metrics: rows exist only for reductions they reached.

    Reports per split: runs, early-stopped runs, rows kept per reduction, and RMSE / 95% coverage of the
    GP separately for rows from early-stopped runs and from runs that reached the end.
    """
    rep = {}
    for split, tab in tables.items():
        r = dict(n_runs=tab["n_runs"], n_runs_early_stopped=tab["n_runs_early"],
                 rows_per_reduction={str(red): int(np.sum(np.isclose(tab["X"][:, 4], red))) for red in reductions})
        for ti, tname in enumerate(TARGETS):
            mu, sd = models[tname]["gp"].predict(tab["X"])
            for lab, mask in (("early_stopped_runs", tab["early"]), ("full_length_runs", ~tab["early"])):
                if mask.sum() == 0:
                    continue
                z = np.abs(tab["Y"][mask, ti] - mu[mask]) / sd[mask]
                r.setdefault(tname, {})[lab] = dict(n_rows=int(mask.sum()), rmse=rmse(tab["Y"][mask, ti], mu[mask]),
                                                    coverage95=float(np.mean(z <= 1.96)))
        rep[split] = r
    return rep


def figures(tables, models, gp_info, out: Path) -> list[str]:
    fdir = out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    colors = {"train": "0.6", "test_random": "tab:blue", "test_ood_geometry": "tab:red",
              "test_fine_mesh": "tab:green"}
    paths = []

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for ti, (tname, label) in enumerate(TARGETS.items()):
        ax = axs[ti]
        for split, tab in tables.items():
            mu, sd = models[tname]["gp"].predict(tab["X"])
            ax.errorbar(tab["Y"][:, ti], mu, yerr=1.96 * sd, fmt="o", ms=3, lw=0.8, color=colors[split],
                        alpha=0.8, label=split)
        lim = ax.get_xlim()
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel(f"solver: {label}")
        ax.set_ylabel(f"GP prediction (95% interval)")
        ax.set_title(label)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    p = fdir / "parity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    levels = np.linspace(0.05, 0.99, 20)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for ti, (tname, label) in enumerate(TARGETS.items()):
        ax = axs[ti]
        for split, tab in tables.items():
            if split == "train":
                continue
            mu, sd = models[tname]["gp"].predict(tab["X"])
            ax.plot(levels, coverage_curve(tab["Y"][:, ti], mu, sd, levels), "o-", ms=3, color=colors[split],
                    label=split)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("nominal central-interval probability")
        ax.set_ylabel("empirical coverage")
        ax.set_title(label)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    p = fdir / "coverage.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ti, (tname, label) in enumerate(TARGETS.items()):
        for ci, mname in enumerate(("gp", "gp_conformal")):
            ax = axs[ti, ci]
            for split in ("test_random", "test_ood_geometry"):
                if split not in tables:
                    continue
                tab = tables[split]
                mu, sd = models[tname][mname].predict(tab["X"])
                ax.hist(pit_values(tab["Y"][:, ti], mu, sd), bins=10, range=(0, 1), histtype="step",
                        density=True, label=split, color=colors[split])
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
            ax.set_title(f"{label}: {mname}", fontsize=9)
            ax.set_xlabel("PIT value")
            ax.set_ylabel("density (flat = calibrated)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    p = fdir / "pit.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.38
    for k, tname in enumerate(TARGETS):
        ls = list(gp_info[tname]["lengthscales"].values())
        ax.bar(np.arange(len(FEATURES)) + (k - 0.5) * w, ls, w, label=TARGETS[tname])
    ax.set_xticks(range(len(FEATURES)), FEATURES)
    ax.set_yscale("log")
    ax.set_ylabel("ARD length scale (fraction of input range)")
    ax.set_title("short bar = influential input")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fdir / "ard.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--out", default="results/gp")
    ap.add_argument("--artifacts", default="artifacts/gp")
    ap.add_argument("--reductions", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="torch threads (pinned for determinism)")
    ap.add_argument("--cal-fraction", type=float, default=0.2, help="share of TRAIN runs held out to calibrate sd")
    ap.add_argument("--mlp-results", default="results/gnn/n5_mlp_baseline_full.json")
    a = ap.parse_args(argv)
    cmd = command_line("surrogates.gp_bo.run", argv)
    out, art = Path(a.out), Path(a.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    tables = load_tables(a.data, a.reductions)
    tr = tables["train"]
    # --- split-conformal calibration on a run-wise hold-out of TRAIN (never on test)
    proper, cal = split_runs(tr["G"], a.cal_fraction, a.seed)
    factors, cal_info = {}, {}
    for ti, tname in enumerate(TARGETS):
        t0 = time.time()
        gp_c, _ = fit_gp(tr["X"][proper], tr["Y"][proper, ti], seed=a.seed, iters=a.iters, threads=a.threads)
        mu, sd = gp_c.predict(tr["X"][cal])
        factors[tname] = conformal_scale(tr["Y"][cal, ti], mu, sd)
        cal_info[tname] = dict(factor=factors[tname], n_rows_cal=int(cal.sum()),
                               n_runs_cal=int(len(np.unique(tr["G"][cal]))),
                               n_rows_proper=int(proper.sum()), rmse_cal=rmse(tr["Y"][cal, ti], mu))
        print(f"[calibration fit {tname}] {time.time() - t0:.0f} s, factor {factors[tname]:.3f}", flush=True)
    # --- final fits on all train runs
    models, gp_info = {}, {}
    for ti, tname in enumerate(TARGETS):
        t0 = time.time()
        gp, info = fit_gp(tr["X"], tr["Y"][:, ti], seed=a.seed, iters=a.iters, threads=a.threads)
        print(f"[final fit {tname}] {time.time() - t0:.0f} s", flush=True)
        gp.save(art / f"{'id' if ti == 0 else 'force'}_gp.npz")
        models[tname] = dict(gp=gp, gp_conformal=ScaledGP(gp, factors[tname]),
                             poly=PolyBaseline().fit(tr["X"], tr["Y"][:, ti]),
                             gbt=GBTBaseline(a.seed).fit(tr["X"], tr["Y"][:, ti]))
        gp_info[tname] = public_info(info)
    metrics = evaluate(tables, models)
    metrics["gp_hyperparameters"] = gp_info
    metrics["early_stop_handling"] = early_stop_report(tables, models, a.reductions)
    metrics["calibration"] = dict(
        protocol="factor = conformal 95% quantile of |err|/sd on a run-wise 20% hold-out of train (GP fitted on "
                 "the other 80%), applied to the GP fitted on all train runs; test splits not used",
        **cal_info)
    mlp = mlp_ensemble_metrics(a.mlp_results, tables)
    metrics["mlp_ensemble_same_rows"] = mlp if mlp else "not available or rows differ"

    # --- data-support flag (censored corner)
    sup = build_support(a.data)
    save_support(art / "support.json", sup)
    sup_rep = {}
    for split, tab in tables.items():
        ok = in_support(tab["X"], sup)
        r = dict(n_rows=int(len(ok)), n_rows_flagged_out_of_support=int((~ok).sum()))
        for ti, tname in enumerate(TARGETS):
            mu, sd = models[tname]["gp"].predict(tab["X"])
            z = np.abs(tab["Y"][:, ti] - mu) / sd
            for lab, mask in (("in_support", ok), ("out_of_support", ~ok)):
                if mask.sum():
                    r.setdefault(tname, {})[lab] = dict(rmse=rmse(tab["Y"][mask, ti], mu[mask]),
                                                        coverage95=float(np.mean(z[mask] <= 1.96)), n_rows=int(mask.sum()))
        sup_rep[split] = r
    metrics["data_support_flag"] = dict(rule=sup["rule"], by_split=sup_rep)

    # --- which sd the plant sees: fixed rule, decided by the data
    cov_ood = metrics["coverage95_id_change_pct"]["gp_conformal"].get("test_ood_geometry", 0.0)
    cov_rnd = metrics["coverage95_id_change_pct"]["gp_conformal"].get("test_random", 0.0)
    served = "conformal" if (cov_ood >= 0.90 and cov_rnd >= 0.90) else "raw"
    metrics["served_sd"] = dict(choice=served, rule="conformal if ID coverage95 after scaling is >= 0.90 on both "
                                "test_random and test_ood_geometry, else raw GP sd",
                                id_scale=factors["id_change_pct"] if served == "conformal" else 1.0)
    figs = figures(tables, models, gp_info, out)

    runs = list_runs(a.data)
    splits = {s: sum(r["split"] == s for r in runs) for s in sorted({r["split"] for r in runs})}
    meta = dict(name="gp-matern52-ard", version=git_sha(), dataset=a.data, reductions=a.reductions,
                n_train_rows=int(len(tr["X"])), seed=a.seed, command=cmd, features=FEATURES,
                env=gp_info["id_change_pct"]["env"], served_sd=served,
                sd_scale=dict(id=factors["id_change_pct"], force=factors["log_force_norm"]))
    with open(art / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    data = dict(dataset=a.data, n_runs=len(runs), splits=splits, reductions=a.reductions,
                rows={s: int(len(t["X"])) for s, t in tables.items()})
    write_result(out / "gp_accuracy.json", task="G1", title="GP accuracy and coverage", command=cmd,
                 data=data, metrics=metrics, figures=figs,
                 notes="Fit on train only. RMSE of inner-diameter change is in percentage points; force target "
                       "is log(F*) so its RMSE is a relative error. Baselines: cubic ridge polynomial, "
                       "gradient-boosted trees. Coverage is for the GP predictive interval (incl. noise).")
    for k, v in metrics.items():
        if k not in ("gp_hyperparameters", "early_stop_handling", "data_support_flag", "mlp_ensemble_same_rows"):
            print(k, json.dumps(v, default=float))
    print("ARD:", {t: {k: round(x, 3) for k, x in g["lengthscales"].items()} for t, g in gp_info.items()})


if __name__ == "__main__":
    main()
