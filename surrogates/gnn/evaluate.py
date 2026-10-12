"""Task N6: evaluate a trained GNN run directory; metrics JSON, figures, failure gallery.

    python -m surrogates.gnn.evaluate --run artifacts/gnn/base --data full --out results/gnn/base

Metrics (per split: test_random, test_ood_geometry, test_fine_mesh, test_unstructured if present)
  * field nRMSE  = RMSE / std(truth over the split), for displacement r, displacement z, nodal strain
  * inner-diameter change (%) read off the predicted inner mid-plane node at reductions 0.1/0.2/0.3,
    RMSE vs the solver; the MLP baseline's RMSE on the same rows is read from results/gnn/n5_*.json
  * force head: RMSE of F* and mean relative error over all saved steps
  * mean |J-1| of the predicted meshes next to the solver's own value
  * inference time per sample vs the solver's recorded runtime per run (teaching-scale solver; no
    claim about industrial solvers)
  * failure gallery: the 10 worst runs (worst stroke position of each) by normalised field error, each with a figure and an explicit,
    rule-derived hypothesis (a hypothesis, not a finding).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from common.data import list_runs  # noqa: E402

from .graph import S_MAX, Normaliser, RunGraph, id_change_pct, j_ratio, load_graphs, make_samples  # noqa: E402
from .model import MeshGNN  # noqa: E402
from .results import write_result  # noqa: E402
from .train import Config  # noqa: E402

SPLITS = ["test_random", "test_ood_geometry", "test_fine_mesh", "test_unstructured"]
ID_REDUCTIONS = (0.1, 0.2, 0.3)


def load_model(run_dir: Path, which: str = "best"):
    cfg = Config(**yaml.safe_load(open(run_dir / "config.yaml")))
    nm = Normaliser.load(run_dir / "norm.json")
    model = MeshGNN(cfg.latent, cfg.steps)
    model.load_state_dict(torch.load(run_dir / f"{which}.pt", weights_only=False)["model"])
    return model.eval(), nm, cfg


@torch.no_grad()
def predict_run(model, nm, g: RunGraph):
    """Physical-unit predictions for every saved step of one run: disp (S,N,2), strain (S,N), force (S,)."""
    samples = make_samples([g], nm)
    t0 = time.perf_counter()
    d, e, f = model(Batch.from_data_list(samples))
    dt = time.perf_counter() - t0
    S, N = g.s.shape[0], g.x.shape[0]
    return (nm.denorm("disp", d).reshape(S, N, 2), nm.denorm("strain", e).reshape(S, N),
            nm.denorm("force", f).reshape(S), dt)


def hypothesis(g: RunGraph, k: int, split: str, stopped: bool, jmax: float) -> str:
    """Rule-based guess from case attributes. It names the most plausible cause, to be tested, not proven."""
    c, s = g.case, float(g.s[k])
    why = []
    if split == "test_ood_geometry":
        why.append(f"extrapolation: ring shape (OD/ID {c['od_id']:.2f}, H/ID {c['h_id']:.2f}) lies outside the training box")
    if c["mf"] > 0.8:
        why.append(f"near-sticking friction (m_f {c['mf']:.2f}) gives a dead-metal zone and sharp strain gradients that 12 smooth message-passing steps blur")
    if stopped:
        why.append("run stopped early by fold-over: the last saved steps contain severely distorted triangles")
    if s > 0.8:
        why.append(f"late stroke (s={s:.2f}): largest displacements, so absolute error is largest")
    if jmax > 0.5:
        why.append("solver mesh itself has a triangle with |J-1| > 0.5 here, so targets are locally noisy")
    if split == "test_fine_mesh":
        why.append("finer mesh than training: node count and edge lengths differ from anything seen")
    return "; ".join(why) or "no rule fired: error is spread across the mesh; candidate cause is model capacity or too little data"


LEN = "(lengths / initial inner radius)"


def plot_gallery_item(g: RunGraph, k: int, pred_disp: np.ndarray, path: Path, title: str, vmax: float):
    """Left: solver (black) vs GNN (red) deformed meshes. Right: displacement error magnitude on the solver mesh,
    on a colour scale shared by every gallery figure of this evaluation (0..vmax)."""
    tri = g.tri.numpy().T
    p0 = g.p0.numpy().T
    true = p0 + g.disp[k].numpy()
    pred = p0 + pred_disp
    err = np.linalg.norm(pred_disp - g.disp[k].numpy(), axis=1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    a.triplot(true[:, 0], true[:, 1], tri, color="k", lw=0.5)
    a.triplot(pred[:, 0], pred[:, 1], tri, color="r", lw=0.5, alpha=0.7)
    a.plot([], [], "k", label="solver"); a.plot([], [], "r", label="GNN"); a.legend(fontsize=8, loc="upper right")
    a.set_title("deformed mesh", fontsize=9)
    tp = b.tripcolor(true[:, 0], true[:, 1], tri, err, shading="gouraud", cmap="magma", vmin=0.0, vmax=vmax)
    b.set_title("|displacement error| on solver mesh", fontsize=9)
    fig.colorbar(tp, ax=b, label=f"|u_GNN - u_solver| {LEN}")
    lo, hi = np.minimum(true.min(0), pred.min(0)), np.maximum(true.max(0), pred.max(0))
    pad = 0.03 * (hi - lo).max()
    for ax in (a, b):                              # identical axis limits on both panels
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_xlabel(f"r {LEN}"); ax.set_ylabel(f"z {LEN}"); ax.set_aspect("equal")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(run_dir: Path, path: Path):
    rows = [json.loads(l) for l in open(run_dir / "log.jsonl")]
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(ep, [r["train_total"] for r in rows], label="train total")
    if "val_total" in rows[0]:
        ax.semilogy(ep, [r["val_total"] for r in rows], label="validation total")
    ax.semilogy(ep, [max(r["train_abs_j_minus_1"], 1e-6) for r in rows], "--", label="train mean |J-1|")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (normalised units)"); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def _summarise(items) -> dict:
    """Field, inner-diameter, force and J metrics over a list of (graph, pred_disp, pred_strain, pred_force)."""
    err = {"dr": [], "dz": [], "strain": []}
    truth = {"dr": [], "dz": [], "strain": []}
    idp, idt, fe, ft, jp, js = [], [], [], [], [], []
    for g, pd_, ps, pf in items:
        td, ts = g.disp, g.strain
        err["dr"].append((pd_[..., 0] - td[..., 0]).numpy().ravel()); truth["dr"].append(td[..., 0].numpy().ravel())
        err["dz"].append((pd_[..., 1] - td[..., 1]).numpy().ravel()); truth["dz"].append(td[..., 1].numpy().ravel())
        err["strain"].append((ps - ts).numpy().ravel()); truth["strain"].append(ts.numpy().ravel())
        fe.append((pf - g.force).numpy()); ft.append(g.force.numpy())
        r0 = float(g.p0[0, g.inner_mid])
        red = g.s.numpy() * S_MAX
        for target in ID_REDUCTIONS:
            j = np.where(np.abs(red - target) < 1e-6)[0]
            if len(j):
                j = int(j[0])
                idp.append(float(id_change_pct(r0 + pd_[j, g.inner_mid, 0].item(), r0)))
                idt.append(float(id_change_pct(r0 + g.disp[j, g.inner_mid, 0].item(), r0)))
        p0 = g.p0.numpy(); tri = g.tri.numpy()
        P = np.transpose(p0.T[None] + pd_.numpy(), (0, 2, 1)); T = np.transpose(p0.T[None] + g.disp.numpy(), (0, 2, 1))
        jp.append(np.abs(j_ratio(P[1:], P[0], tri) - 1).mean()); js.append(np.abs(j_ratio(T[1:], T[0], tri) - 1).mean())
    cat = lambda d, key: np.concatenate(d[key])  # noqa: E731
    m = {f"nrmse_{k}": float(np.sqrt(np.mean(cat(err, k) ** 2)) / (cat(truth, k).std() + 1e-12)) for k in err}
    fe_, ft_ = np.concatenate(fe), np.concatenate(ft)
    m["force_rmse"] = float(np.sqrt(np.mean(fe_ ** 2)))
    m["force_mean_rel_err"] = float(np.mean(np.abs(fe_) / np.maximum(np.abs(ft_), 1e-9)))
    if idp:
        m["id_rmse_pct_points"] = float(np.sqrt(np.mean((np.array(idp) - np.array(idt)) ** 2)))
        m["id_n_rows"] = len(idp)
    m["mean_abs_J_minus_1_pred"] = float(np.mean(jp)); m["mean_abs_J_minus_1_solver"] = float(np.mean(js))
    m["n_runs"] = len(items)
    return m


def evaluate(run_dir: Path, dataset: str, out: Path, which: str = "best") -> dict:
    model, nm, cfg = load_model(run_dir, which)
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    metrics, all_cases = {}, []
    t_infer, n_samples, solver_s = 0.0, 0, []
    for split in SPLITS:
        ids = [r["run_id"] for r in list_runs(dataset, split)] if _has_split(dataset, split) else []
        if not ids:
            continue
        graphs = load_graphs(dataset, split, ids)
        stopped = {r["run_id"]: bool(r["stopped_early"]) for r in list_runs(dataset, split)}
        runtime = {r["run_id"]: float(r["runtime_s"]) for r in list_runs(dataset, split)}
        items = []                               # (graph, predictions) so metrics can be grouped afterwards
        for g in graphs:
            pd_, ps, pf, dt = predict_run(model, nm, g)
            t_infer += dt; n_samples += g.s.shape[0]; solver_s.append(runtime[g.run_id])
            items.append((g, pd_, ps, pf))
            p0 = g.p0.numpy(); tri = g.tri.numpy()
            T = np.transpose(p0.T[None] + g.disp.numpy(), (0, 2, 1))
            for k in range(1, g.s.shape[0]):     # per-sample ranking for the gallery
                sc = float(torch.sqrt(((pd_[k] - g.disp[k]) ** 2).mean()) / (g.disp.std() + 1e-9)
                           + torch.sqrt(((ps[k] - g.strain[k]) ** 2).mean()) / (g.strain.std() + 1e-9))
                jm = float(np.abs(j_ratio(T[k], T[0], tri) - 1).max())
                all_cases.append((sc, split, g, k, pd_[k].numpy(), stopped[g.run_id], jm))
        m = _summarise(items)
        for name, flag in (("early_stopped_runs", True), ("completed_runs", False)):
            sub = [it for it in items if stopped[it[0].run_id] == flag]
            if sub:
                m[name] = _summarise(sub)
        metrics[split] = m
    # MLP comparison from the N5 file, if it exists for this dataset
    n5 = Path("results/gnn") / f"n5_mlp_baseline_{dataset}.json"
    if n5.exists():
        by = json.load(open(n5))["metrics"]["by_split"]
        for sp, m in metrics.items():
            if sp in by:
                m["id_rmse_pct_points_mlp_baseline"] = by[sp]["id_change_pct"]["rmse"]
    metrics["inference"] = {"ms_per_sample_batched_run": 1e3 * t_infer / max(n_samples, 1),
                            "mean_solver_runtime_s_per_run": float(np.mean(solver_s)) if solver_s else None,
                            "note": "one GNN sample = one stroke position; a solver run covers the whole stroke on one CPU "
                                    "(teaching-scale solver, not an industrial one)"}
    # failure gallery
    all_cases.sort(key=lambda c: -c[0])
    seen, worst = set(), []                      # one entry per run: adjacent steps of a run are near-duplicates
    for c in all_cases:
        if c[2].run_id not in seen:
            seen.add(c[2].run_id); worst.append(c)
    gallery = []
    for old in fig_dir.glob("gallery_*.png"):     # names depend on the ranking: never leave stale figures behind
        old.unlink()
    # one colour scale for the whole gallery so the panels can be compared by eye
    vmax = max([float(np.linalg.norm(c[4] - c[2].disp[c[3]].numpy(), axis=1).max()) for c in worst[:10]] or [1.0])
    for rank, (sc, split, g, k, pd_k, stp, jm) in enumerate(worst[:10]):
        h = hypothesis(g, k, split, stp, jm)
        fn = fig_dir / f"gallery_{rank:02d}_{g.run_id}_s{int(100 * float(g.s[k]))}.png"
        plot_gallery_item(g, k, pd_k, fn, f"#{rank + 1} {g.run_id} | {split} | s = {float(g.s[k]):.2f} | "
                          f"m_f = {g.case['mf']:.2f} | error score = {sc:.2f} (disp nRMSE + strain nRMSE)", vmax)
        gallery.append({"rank": rank + 1, "run_id": g.run_id, "split": split, "s": float(g.s[k]), "score": sc,
                        "case": {k2: g.case[k2] for k2 in ("od_id", "h_id", "mf", "m")}, "figure": _rel(fn),
                        "hypothesis (untested)": h})
    metrics["failure_gallery"] = gallery
    plot_curves(run_dir, fig_dir / "training_curves.png")
    return {"metrics": metrics, "figures": [_rel(p) for p in sorted(fig_dir.glob("*.png"))],
            "cfg": cfg, "splits": {s: m["n_runs"] for s, m in metrics.items() if isinstance(m, dict) and "n_runs" in m}}


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(p)


def _has_split(dataset: str, split: str) -> bool:
    return any(True for _ in list_runs(dataset, split))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="best", choices=["best", "last"])
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    res =evaluate(Path(a.run), a.data, Path(a.out), a.which)
    cmd = "python -m surrogates.gnn.evaluate " + " ".join(sys.argv[1:])
    p = write_result(Path(a.out) / "metrics.json", "N6", f"GNN evaluation ({res['cfg'].name})", cmd,
                     {"dataset": a.data, "n_runs": sum(res["splits"].values()), "splits": res["splits"]},
                     res["metrics"], res["figures"],
                     notes=f"checkpoint {a.which}.pt of {a.run}; lambda_j={res['cfg'].lambda_j}; nRMSE = RMSE / std of the "
                           "solver truth over the split. Gallery hypotheses are rule-derived guesses to be tested.")
    print(p)


if __name__ == "__main__":
    main()
