"""G5: which ring geometry makes one inner-diameter measurement most informative about friction?

    python -m surrogates.gp_bo.design --out results/design

Objective (to minimise) for a ring (OD/ID, H/ID) at fixed m = 0.15 and 30% reduction: the expected
posterior std of m_f after one inner-diameter measurement (sigma = 0.3 points), averaged over hidden
truths m_f in [0.05, 0.95] and computed on an exact m_f grid with the GP as forward model. The GP's own
predictive variance is added to the measurement variance, so geometries where the GP is unsure are
penalised. In plain words: find the ring whose hole changes most per unit of friction across the whole
friction range.

BoTorch is used here, and only here: the objective is a black box over two geometry ratios. Finding m_f
for a given diameter is root-finding (see inverse.py), not BO. The BO result is cross-checked by a dense
grid search, and the best and standard rings are checked with solver sweeps (GP-free).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from surrogates.gp_bo.gp import HI, LO, NumpyGP  # noqa: E402
from surrogates.gp_bo.results_io import command_line, write_result  # noqa: E402
from surrogates.gp_bo.solver_calls import curve_at, run_cases  # noqa: E402

M_FIXED, REDUCTION, SIGMA_ID = 0.15, 0.30, 0.3
THREADS = 4
MF_GRID = np.linspace(0, 1, 201)
TRUTHS = np.linspace(0.05, 0.95, 19)
STANDARD = (2.0, 2.0 / 3.0)
BOX = dict(od_id=(float(LO[0]), float(HI[0])), h_id=(float(LO[1]), float(HI[1])))


def _range(mf_max: float = 1.0):
    """m_f grid [0, mf_max] and the true values averaged over (0.05 .. mf_max - 0.05, never past the grid)."""
    grid = MF_GRID[MF_GRID <= mf_max + 1e-9]
    truths = TRUTHS[TRUTHS <= mf_max - 0.05 + 1e-9]
    return grid, truths


def _post_std(mu, extra_sd, y, grid, sigma=SIGMA_ID):
    var = sigma ** 2 + extra_sd ** 2
    ll = -0.5 * (y - mu) ** 2 / var - 0.5 * np.log(var)
    p = np.exp(ll - ll.max())
    p /= p.sum()
    m = (grid * p).sum()
    return float(np.sqrt(((grid - m) ** 2 * p).sum()))


def std_by_truth(mu, extra_sd, mf_max: float = 1.0):
    """Posterior std of m_f for each hidden truth (noise-free typical measurement), on [0, mf_max]."""
    grid, truths = _range(mf_max)
    mu, extra_sd = mu[:len(grid)], extra_sd[:len(grid)]
    ys = np.interp(truths, grid, mu)
    return np.array([_post_std(mu, extra_sd, y, grid) for y in ys])


def objective_gp(gp: NumpyGP, od: float, h: float, mf_max: float = 1.0) -> float:
    X = np.column_stack([np.full_like(MF_GRID, od), np.full_like(MF_GRID, h), MF_GRID,
                         np.full_like(MF_GRID, M_FIXED), np.full_like(MF_GRID, REDUCTION)])
    mu, sd = gp.predict(X)
    return float(np.mean(std_by_truth(mu, sd, mf_max)))


def objective_solver(nodes, idc, mf_max: float = 1.0) -> float | None:
    """Same objective from a solver sweep (PCHIP in m_f, no GP variance), on [0, mf_max].

    Returns None if the sweep does not cover [0, mf_max]: no silent extrapolation."""
    nodes = np.asarray(nodes)
    if nodes.max() < mf_max - 1e-9:
        return None
    mu = PchipInterpolator(nodes, idc)(MF_GRID)
    return float(np.mean(std_by_truth(mu, np.zeros_like(mu), mf_max)))


def objective_solver_extrapolated(nodes, idc) -> float:
    """The old full-range number for an incomplete sweep: PCHIP extrapolates past the last node."""
    mu = PchipInterpolator(nodes, idc, extrapolate=True)(MF_GRID)
    return float(np.mean(std_by_truth(mu, np.zeros_like(mu), 1.0)))


def _gp_dataset(artifact_dir) -> str:
    """Name of the dataset the forward-model GP was trained on (from artifacts/gp/meta.json)."""
    import json
    p = Path(artifact_dir) / "meta.json"
    return json.load(open(p)).get("dataset", "unknown") if p.exists() else "unknown"


def _data_block(artifact_dir, metrics) -> dict:
    """Runs behind the numbers: train runs of the forward-model GP plus the solver-check runs."""
    from common.data import list_runs
    ds = _gp_dataset(artifact_dir)
    n_gp = len(list_runs(ds, "train")) if ds != "unknown" else 0
    n_solver = 2 * len(metrics["solver_check"]["mf_nodes"]) if "solver_check" in metrics else 0
    return dict(dataset=ds, n_runs=n_gp + n_solver, splits={"gp_train_runs": n_gp, "solver_check_runs": n_solver})


def run_bo(f, n_init=8, n_iter=22, seed=0):
    """Minimise f over the unit square with BoTorch (log-EI on a GP of -f)."""
    import torch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    X = draw_sobol_samples(bounds, n=n_init, q=1, seed=seed).squeeze(1)
    Y = torch.tensor([[-f(x.numpy())] for x in X], dtype=torch.double)
    for _ in range(n_iter):
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acq = LogExpectedImprovement(model, best_f=Y.max())
        x_new, _ = optimize_acqf(acq, bounds, q=1, num_restarts=6, raw_samples=128)
        X = torch.cat([X, x_new])
        Y = torch.cat([Y, torch.tensor([[-f(x_new[0].numpy())]], dtype=torch.double)])
    i = int(Y.argmax())
    return X.numpy(), -Y.numpy().ravel(), X[i].numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/design")
    ap.add_argument("--artifacts", default="artifacts/gp")
    ap.add_argument("--cache", default="results/design/solver_cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-validate", action="store_true", help="skip the solver check of the best ring")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    cmd = command_line("surrogates.gp_bo.design", argv)
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    gp = NumpyGP.load(Path(a.artifacts) / "id_gp.npz")
    lo = np.array([BOX["od_id"][0], BOX["h_id"][0]])
    hi = np.array([BOX["od_id"][1], BOX["h_id"][1]])

    def f(u):
        od, h = lo + (hi - lo) * np.asarray(u)
        return objective_gp(gp, float(od), float(h))

    Xu, Yo, best_u = run_bo(f, seed=a.seed)
    best = lo + (hi - lo) * best_u
    # dense grid cross-check of the same objective
    ug = np.linspace(0, 1, 21)
    grid = np.array([[f([x, y]) for x in ug] for y in ug])            # rows: h, cols: od
    gi = np.unravel_index(np.argmin(grid), grid.shape)
    grid_best = (float(lo[0] + (hi[0] - lo[0]) * ug[gi[1]]), float(lo[1] + (hi[1] - lo[1]) * ug[gi[0]]))
    std_obj = objective_gp(gp, *STANDARD)
    best_obj = objective_gp(gp, float(best[0]), float(best[1]))
    metrics = dict(
        measurement_sigma_pct_points=SIGMA_ID, reduction=REDUCTION, m=M_FIXED, search_box=BOX,
        bo=dict(n_evaluations=int(len(Yo)), best_od_id=float(best[0]), best_h_id=float(best[1]),
                best_objective=best_obj),
        grid_search=dict(best_od_id=grid_best[0], best_h_id=grid_best[1], best_objective=float(grid.min()),
                         n_evaluations=int(grid.size)),
        standard_ring=dict(od_id=STANDARD[0], h_id=STANDARD[1], objective=std_obj),
        improvement_factor_gp=float(std_obj / best_obj),
        objective_definition="mean over true m_f in [0.05, 0.95] of the posterior std of m_f after one "
                             "ID measurement (GP forward model; GP variance added to measurement variance)")

    # GP-free check by solver sweeps: best ring vs standard ring
    if not a.no_validate:
        nodes = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        val = {}
        for name, (od, h) in (("standard", STANDARD), ("best", (float(best[0]), float(best[1])))):
            cases = [dict(od_id=od, h_id=h, mf=float(m), m=M_FIXED, reduction=REDUCTION, K=1.0, V=1.0)
                     for m in nodes]
            res = run_cases(cases, a.cache, workers=a.workers)
            idc = np.array([curve_at(r, [REDUCTION])[0] for r in res])
            early = [float(m) for m, r in zip(nodes, res) if r["stopped_early"]]
            ok = ~np.isnan(idc)
            val[name] = dict(od_id=od, h_id=h, stopped_early_at_mf=early, id_pct_at_30=idc.tolist(),
                             complete=bool(ok.all()), last_solved_mf=float(nodes[ok].max()),
                             _nodes=nodes[ok], _idc=idc[ok])
        # ONE common range: up to the last m_f both rings solved to 30% (no extrapolation in the headline)
        common = min(v["last_solved_mf"] for v in val.values())
        for name in val:
            n_, i_ = val[name].pop("_nodes"), val[name].pop("_idc")
            val[name]["objective_full_range_0_to_1"] = objective_solver(n_, i_, 1.0)
            val[name]["objective_full_range_extrapolated"] = (
                None if val[name]["complete"] else objective_solver_extrapolated(n_, i_))
            val[name]["objective_common_range"] = objective_solver(n_, i_, common)
            # robustness: the same objective on [0, 0.9], the range an earlier GP fit's best ring could cover
            val[name]["objective_0_to_0p9"] = objective_solver(n_, i_, 0.9)
            # which truths dominate the average? per-truth posterior std (full range, only if solved to m_f 1)
            if val[name]["complete"]:
                mu_ = PchipInterpolator(n_, i_)(MF_GRID)
                val[name]["std_by_true_mf"] = std_by_truth(mu_, np.zeros_like(mu_)).tolist()
        v_s, v_b = (val[k]["objective_common_range"] for k in ("standard", "best"))
        metrics["solver_check"] = dict(
            mf_nodes=nodes.tolist(), common_range_mf=[0.0, common], **val,
            improvement_factor_solver_common_range=v_s / v_b,
            improvement_factor_solver_0_to_0p9=(val["standard"]["objective_0_to_0p9"]
                                                / val["best"]["objective_0_to_0p9"]),
            improvement_factor_solver_full_range_extrapolated=(
                val["standard"]["objective_full_range_0_to_1"] / val["best"]["objective_full_range_extrapolated"]
                if val["best"]["objective_full_range_extrapolated"] else None),
            note="common range = [0, last m_f solved to 30% by both rings]; objectives there use no "
                 "extrapolation. The 'full range extrapolated' value is the earlier headline, kept for "
                 "comparison; it extrapolates ID(m_f) past the last solved node for an incomplete sweep.")
        if all("std_by_true_mf" in val[k] for k in ("standard", "best")):
            low = TRUTHS <= 0.6 + 1e-9
            s_s, s_b = (np.array(val[k]["std_by_true_mf"]) for k in ("standard", "best"))
            metrics["solver_check"]["true_mf_values"] = TRUTHS.tolist()
            metrics["solver_check"]["improvement_factor_solver_truths_le_0p6"] = float(s_s[low].mean() / s_b[low].mean())
            metrics["solver_check"]["improvement_factor_solver_truths_gt_0p6"] = float(s_s[~low].mean() / s_b[~low].mean())
            metrics["solver_check"]["share_of_average_from_truths_gt_0p6_standard"] = float(s_s[~low].sum() / s_s.sum())
        # same common range on the GP, so GP and solver numbers are comparable
        metrics["improvement_factor_gp_common_range"] = (
            objective_gp(gp, *STANDARD, mf_max=common) / objective_gp(gp, float(best[0]), float(best[1]), common))
        per = {}
        for name, (od, h) in (("standard", STANDARD), ("best", (float(best[0]), float(best[1])))):
            Xq = np.column_stack([np.full_like(MF_GRID, od), np.full_like(MF_GRID, h), MF_GRID,
                                  np.full_like(MF_GRID, M_FIXED), np.full_like(MF_GRID, REDUCTION)])
            per[name] = std_by_truth(*gp.predict(Xq))
        low = TRUTHS <= 0.6 + 1e-9
        metrics["gp_std_by_true_mf"] = dict(true_mf=TRUTHS.tolist(), **{k: v.tolist() for k, v in per.items()})
        metrics["improvement_factor_gp_truths_le_0p6"] = float(per["standard"][low].mean() / per["best"][low].mean())
        metrics["standard_ring"]["objective_common_range"] = objective_gp(gp, *STANDARD, mf_max=common)
        metrics["bo"]["objective_common_range"] = objective_gp(gp, float(best[0]), float(best[1]), common)

    # figures
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(grid, origin="lower", extent=[*BOX["od_id"], *BOX["h_id"]], aspect="auto", cmap="viridis_r")
    pts = lo + (hi - lo) * Xu
    ax.plot(pts[:, 0], pts[:, 1], "w.", ms=5, label="BO evaluations")
    ax.plot(*best, "r*", ms=14, label="BO best")
    ax.plot(*STANDARD, "ks", ms=8, label="standard 6:3:2")
    ax.set_xlabel("OD/ID")
    ax.set_ylabel("H/ID")
    fig.colorbar(im, label="expected posterior std of m_f")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    p1 = out / "figures" / "design_map.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name, (od, h) in (("standard 6:3:2", STANDARD), ("BO best", (best[0], best[1]))):
        X = np.column_stack([np.full_like(MF_GRID, od), np.full_like(MF_GRID, h), MF_GRID,
                             np.full_like(MF_GRID, M_FIXED), np.full_like(MF_GRID, REDUCTION)])
        mu, sd = gp.predict(X)
        ax.plot(TRUTHS, std_by_truth(mu, sd), "o-", ms=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("true friction factor m_f")
    ax.set_ylabel("posterior std of m_f (one ID measurement)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = out / "figures" / "std_vs_mf_best_vs_standard.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    write_result(out / "design.json", task="G5", title="Best ring geometry for sensing friction",
                 command=cmd, data=_data_block(a.artifacts, metrics),
                 metrics=metrics, figures=[str(p1), str(p2)],
                 notes="Forward model is the ID GP in artifacts/gp (trained on the dataset named in "
                       "artifacts/gp/meta.json); search box = training box, so no extrapolation. Rings are "
                       "ranked by sensitivity to friction only; manufacturability and force-capacity limits "
                       "are not considered.")
    print({k: v for k, v in metrics.items() if k not in ("solver_check",)})
    if "solver_check" in metrics:
        print({k: v for k, v in metrics["solver_check"].items() if k != "mf_nodes"})
        print("GP common-range factor", metrics["improvement_factor_gp_common_range"])


if __name__ == "__main__":
    main()
