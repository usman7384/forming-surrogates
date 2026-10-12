"""Task N5: MLP baseline from the five inputs to the two scalar targets, same table as the GP.

Inputs  : FEATURES = (od_id, h_id, mf, m, reduction)         (common.data.scalar_table)
Targets : id_change_pct and log(force_norm) at reductions 0.1/0.2/0.3
Metrics : RMSE and 95% interval coverage per split, so rows line up with the GP results (G1).
An MLP has no native uncertainty: we use a deep ensemble (members differ in init and batch order) and
take the mean and std over members. Ensemble spread is usually too narrow, so coverage is reported
as measured, not tuned.

    python -m surrogates.baselines.mlp_scalar --data smoke --out results/gnn
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch import nn

from common.data import scalar_table
from surrogates.gnn.results import write_result

SPLITS = ["test_random", "test_ood_geometry", "test_fine_mesh"]
TARGETS = ("id_change_pct", "force_norm")
REDUCTIONS = (0.1, 0.2, 0.3)


def make_net(width: int = 64, depth: int = 3) -> nn.Module:
    layers, n = [], 5
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        n = width
    return nn.Sequential(*layers, nn.Linear(n, 2))


def transform_y(Y: np.ndarray) -> np.ndarray:
    return np.stack([Y[:, 0], np.log(Y[:, 1])], axis=1)


class MLPEnsemble:
    def __init__(self, members: int = 5, width: int = 64, depth: int = 3, epochs: int = 1500,
                 lr: float = 3e-3, weight_decay: float = 1e-5, seed: int = 0):
        self.members, self.width, self.depth = members, width, depth
        self.epochs, self.lr, self.wd, self.seed = epochs, lr, weight_decay, seed
        self.nets: list[nn.Module] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MLPEnsemble":
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        Yt = transform_y(Y)
        self.ym, self.ys = Yt.mean(0), Yt.std(0) + 1e-9
        Xt = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        Yn = torch.tensor((Yt - self.ym) / self.ys, dtype=torch.float32)
        self.nets = []
        for k in range(self.members):
            torch.manual_seed(self.seed + k)
            net = make_net(self.width, self.depth)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs, eta_min=self.lr * 0.01)
            g = torch.Generator().manual_seed(self.seed + k)
            for _ in range(self.epochs):
                perm = torch.randperm(len(Xt), generator=g)
                for i in range(0, len(Xt), 64):
                    idx = perm[i:i + 64]
                    loss = ((net(Xt[idx]) - Yn[idx]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
            self.nets.append(net.eval())
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray):
        """Returns (mean, std) in target space (id_change_pct, log force_norm), each (n, 2)."""
        Xt = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        P = torch.stack([n(Xt) for n in self.nets]).numpy() * self.ys + self.ym
        return P.mean(0), P.std(0, ddof=1) if len(self.nets) > 1 else np.zeros_like(P[0])


def metrics_for(model: MLPEnsemble, X, Y) -> dict:
    mu, sd = model.predict(X)
    Yt = transform_y(Y)
    out = {}
    for j, name in enumerate(["id_change_pct", "log_force_norm"]):
        err = mu[:, j] - Yt[:, j]
        cover = np.abs(err) <= 1.96 * np.maximum(sd[:, j], 1e-12)
        out[name] = {"rmse": float(np.sqrt(np.mean(err ** 2))), "coverage95": float(np.mean(cover)),
                     "mean_std": float(np.mean(sd[:, j])), "n_rows": int(len(err))}
    return out


def run(dataset: str, members: int, epochs: int):
    Xtr, Ytr, _, _ = scalar_table(dataset, REDUCTIONS, "train", TARGETS)
    model = MLPEnsemble(members=members, epochs=epochs).fit(Xtr, Ytr)
    per_split = {"train": metrics_for(model, Xtr, Ytr)}
    n_runs = {}
    for sp in SPLITS:
        X, Y, G, _ = scalar_table(dataset, REDUCTIONS, sp, TARGETS)
        if len(X):
            per_split[sp] = metrics_for(model, X, Y)
            n_runs[sp] = int(len(np.unique(G)))
    return per_split, {"dataset": dataset, "n_rows_train": int(len(Xtr)), "splits": n_runs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--out", default="results/gnn")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    metrics, data = run(a.data, a.members, a.epochs)
    cmd = "python -m surrogates.baselines.mlp_scalar " + " ".join(sys.argv[1:])
    p = write_result(f"{a.out}/n5_mlp_baseline_{a.data}.json", "N5", "MLP ensemble scalar baseline", cmd, data,
                     {"by_split": metrics, "reductions": list(REDUCTIONS), "members": a.members,
                      "epochs": a.epochs},
                     notes="Targets: id_change_pct and log(force_norm) at fixed reductions via scalar_table. "
                           "Std = ensemble spread; coverage reported as measured (not calibrated). "
                           "On smoke data the test splits hold 1-2 runs, so these numbers only prove the pipeline.")
    print(p)


if __name__ == "__main__":
    main()
