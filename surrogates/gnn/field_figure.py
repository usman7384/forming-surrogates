"""Field comparison figure: solver vs best GNN variant vs POD+GP, displacement magnitude on the deformed cross-section.

No GNN training: the GNNs are loaded from the committed checkpoints (artifacts/gnn/<name>/best.pt). The POD+GP
baseline has no saved model, so it is refitted here with the same fixed seed and settings as
`surrogates.baselines.pod_field` (deterministic; a few minutes on CPU).

Representative runs: for each test split, the run whose per-run displacement error (of the GNN shown) is the
median of that split. Shown at the run's last saved stroke position.

    python -m surrogates.gnn.field_figure --data full --out results/gnn
"""
from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from surrogates.baselines.pod_field import PODGP, from_grid  # noqa: E402
from surrogates.gnn.evaluate import load_model, predict_run  # noqa: E402
from surrogates.gnn.graph import load_graphs  # noqa: E402
from surrogates.gnn.results import write_result  # noqa: E402

CHOICE = {"test_random": "jloss", "test_ood_geometry": "jloss", "test_fine_mesh": "mixmesh"}


def run_error(pd_, g) -> float:
    """Displacement RMSE over all saved steps and nodes, divided by the run's displacement std (dimensionless)."""
    return float(np.sqrt(((pd_.numpy() - g.disp.numpy()) ** 2).mean()) / (g.disp.numpy().std() + 1e-12))


def plot(rows, info, out: Path) -> None:
    """Rows = splits. Columns: solver |u|, GNN |u|, POD+GP |u| (one colour bar), then GNN and POD+GP |error|
    (second colour bar). Each prediction is drawn on its own predicted deformed shape; errors on the solver's."""
    n = len(rows)
    fig = plt.figure(figsize=(18, 3.1 * n + 0.8))
    gs = fig.add_gridspec(n, 8, width_ratios=[1, 1, 1, 0.05, 0.12, 1, 1, 0.05], wspace=0.25, hspace=0.55)
    unit = "= 1"
    for r, (split, v, g, k, u_gnn, u_pod) in enumerate(rows):
        tri = g.tri.numpy().T; p0 = g.p0.numpy().T; u_true = g.disp[k].numpy()
        mags = [np.linalg.norm(u, axis=1) for u in (u_true, u_gnn, u_pod)]
        errs = [np.linalg.norm(u - u_true, axis=1) for u in (u_gnn, u_pod)]
        fmax = max(m.max() for m in mags); emax = max(e.max() for e in errs)
        shapes = [p0 + u_true, p0 + u_gnn, p0 + u_pod, p0 + u_true, p0 + u_true]
        allp = np.concatenate(shapes[:3]); lo, hi = allp.min(0), allp.max(0); pad = 0.03 * (hi - lo).max()
        ir = info[split]
        titles = ["solver", f"GNN ({v})", "POD+GP",
                  f"|error| GNN ({v}), run error {ir['gnn_run_error']:.3f}",
                  f"|error| POD+GP, run error {ir['pod_gp_run_error']:.3f}"]
        cols = [0, 1, 2, 5, 6]
        for c, vals in enumerate(mags + errs):
            ax = fig.add_subplot(gs[r, cols[c]])
            is_err = c >= 3
            tp = ax.tripcolor(shapes[c][:, 0], shapes[c][:, 1], tri, vals, shading="gouraud",
                              cmap="magma" if is_err else "viridis", vmin=0, vmax=emax if is_err else fmax)
            ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad); ax.set_aspect("equal")
            ax.set_title(titles[c], fontsize=8.5)
            ax.set_xlabel(f"r / r_i0", fontsize=8); ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel(f"{split}\n{g.run_id}, s = {ir['s']:.2f}, m_f = {g.case['mf']:.2f}\nz / r_i0", fontsize=8)
            if c in (2, 4):
                cb = fig.colorbar(tp, cax=fig.add_subplot(gs[r, 3 if c == 2 else 7]))
                cb.set_label(("|u - u_solver| / r_i0" if is_err else "|u| / r_i0"), fontsize=8)
                cb.ax.tick_params(labelsize=7)
    fig.suptitle("Displacement magnitude on the deformed cross-section (lengths divided by the initial inner radius "
                 f"r_i0 {unit}). GNN = jloss for test_random / test_ood_geometry, mixmesh for test_fine_mesh.\n"
                 "Median-error test run per split, last saved stroke position; colour scales shared within each row. "
                 "Run error = displacement RMSE over all saved steps / that run's displacement std.", fontsize=9.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="full")
    ap.add_argument("--out", default="results/gnn")
    ap.add_argument("--artifacts", default="artifacts/gnn")
    ap.add_argument("--replot", action="store_true", help="redraw from the cached selection of the last full run")
    a = ap.parse_args(argv)
    if a.replot:
        rows, info = pickle.load(open(Path(tempfile.gettempdir()) / "fs_field_figure_rows.pkl", "rb"))
        plot(rows, info, Path(a.out) / "figures" / "field_comparison.png")
        return
    import torch
    torch.set_num_threads(2)
    pod =PODGP().fit(load_graphs(a.data, "train"))
    models = {v: load_model(Path(a.artifacts) / v) for v in set(CHOICE.values())}
    rows, info = [], {}
    for split, v in CHOICE.items():
        model, nm, _ = models[v]
        graphs = load_graphs(a.data, split)
        preds = [predict_run(model, nm, g)[0] for g in graphs]
        errs = np.array([run_error(p, g) for p, g in zip(preds, graphs)])
        i = int(np.argsort(errs)[len(errs) // 2])                 # median-error run of this split
        g, pd_ = graphs[i], preds[i]
        pdisp, _ = from_grid(pod.predict_grid(g), g)
        k = g.s.shape[0] - 1
        rows.append((split, v, g, k, pd_[k].numpy(), pdisp[k].numpy()))
        info[split] = {"run_id": g.run_id, "gnn_variant": v, "s": float(g.s[k]),
                       "gnn_run_error": float(errs[i]), "pod_gp_run_error": run_error(pdisp, g),
                       "split_median_gnn_run_error": float(np.median(errs)), "n_runs_in_split": len(graphs),
                       "case": {c: g.case[c] for c in ("od_id", "h_id", "mf", "m")}}
    out = Path(a.out) / "figures" / "field_comparison.png"
    with open(Path(tempfile.gettempdir()) / "fs_field_figure_rows.pkl", "wb") as fh:   # lets --replot skip the refit
        pickle.dump((rows, info), fh)
    plot(rows, info, out)
    p = write_result(Path(a.out) / "field_comparison.json", "N6c", "field comparison figure: selected runs and errors",
                     "python -m surrogates.gnn.field_figure " + " ".join(sys.argv[1:]),
                     {"dataset": a.data, "n_runs": len(rows)}, {"rows": info, "figure": out.as_posix()},
                     notes="GNNs from committed best.pt; POD+GP refitted with the fixed seed. Errors are per-run "
                           "displacement RMSE over all saved steps divided by that run's displacement std.")
    print(p, out)


if __name__ == "__main__":
    sys.exit(main())
