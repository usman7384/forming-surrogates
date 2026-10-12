"""Field-valued baseline: POD (PCA) of the displacement and strain fields + a GP from the five scalar inputs.

Why: the scalar MLP/GP cannot say anything about fields, so comparing them to the GNN on field error is not
like-for-like. This gives a field model with no graph and no message passing.

Meshes differ in node count between runs (the radial count follows the geometry, and the fine-mesh split has more
nodes). So every run's fields are first resampled, in the *initial* normalised coordinates (xi, eta) in [0,1]^2
(the solver is Lagrangian, so the initial position labels a material point), onto one fixed grid. POD is done on
that grid, the GP predicts the POD coefficients from (od_id, h_id, m_f, m, reduction), and the predicted grid
field is interpolated back to each test node. The resampling itself loses some accuracy; that "floor" is
reported by pushing the TRUE field through the same round trip.

Each target (dr, dz, strain) is standardised separately before the SVD so no channel dominates.

    python -m surrogates.baselines.pod_field --data full --out results/gnn
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # noqa: E402

from surrogates.gnn.evaluate import S_MAX, _summarise  # noqa: E402
from surrogates.gnn.graph import RunGraph, load_graphs  # noqa: E402
from surrogates.gnn.results import write_result  # noqa: E402

SPLITS = ["test_random", "test_ood_geometry", "test_fine_mesh"]
NXI, NETA = 21, 13
XI = np.linspace(0, 1, NXI)
ETA = np.linspace(0, 1, NETA)
GX, GY = np.meshgrid(XI, ETA, indexing="ij")
GRID = np.stack([GX.ravel(), GY.ravel()], axis=1)           # (G, 2)


def norm_coords(g: RunGraph) -> np.ndarray:
    r, z = g.p0.numpy()
    return np.stack([(r - r.min()) / (r.max() - r.min()), (z - z.min()) / (z.max() - z.min())], axis=1)


def to_grid(g: RunGraph) -> np.ndarray:
    """(S, 3, G): dr, dz, strain resampled from the run's nodes to the fixed grid."""
    c = norm_coords(g)
    out = np.zeros((g.s.shape[0], 3, len(GRID)))
    for k in range(g.s.shape[0]):
        for j, f in enumerate([g.disp[k, :, 0], g.disp[k, :, 1], g.strain[k]]):
            v = LinearNDInterpolator(c, f.numpy())(GRID)
            bad = np.isnan(v)
            if bad.any():
                v[bad] = NearestNDInterpolator(c, f.numpy())(GRID[bad])
            out[k, j] = v
    return out


def from_grid(field: np.ndarray, g: RunGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """(S, 3, G) grid fields -> per-node (S, N, 2) displacement and (S, N) strain for graph g."""
    c = norm_coords(g)
    f = field.reshape(field.shape[0], 3, NXI, NETA)
    disp = np.zeros((field.shape[0], len(c), 2)); strain = np.zeros((field.shape[0], len(c)))
    for k in range(field.shape[0]):
        for j in range(3):
            v = RegularGridInterpolator((XI, ETA), f[k, j], bounds_error=False, fill_value=None)(c)
            if j < 2:
                disp[k, :, j] = v
            else:
                strain[k] = v
    return torch.tensor(disp, dtype=torch.float32), torch.tensor(strain, dtype=torch.float32)


def inputs(g: RunGraph) -> np.ndarray:
    c = g.case
    return np.array([[c["od_id"], c["h_id"], c["mf"], c["m"], float(s) * S_MAX] for s in g.s])


class PODGP:
    def __init__(self, n_modes: int = 24, n_gp_rows: int = 2500, seed: int = 0):
        self.n_modes, self.n_gp_rows, self.seed = n_modes, n_gp_rows, seed

    def fit(self, graphs: list[RunGraph]) -> "PODGP":
        F = np.concatenate([to_grid(g) for g in graphs])            # (R, 3, G)
        X = np.concatenate([inputs(g) for g in graphs])
        self.cm, self.cs = F.mean((0, 2), keepdims=True), F.std((0, 2), keepdims=True) + 1e-12   # per-channel
        Z = ((F - self.cm) / self.cs).reshape(len(F), -1)
        self.mean = Z.mean(0)
        U, S, Vt = np.linalg.svd(Z - self.mean, full_matrices=False)
        self.V = Vt[: self.n_modes]
        self.energy = float((S[: self.n_modes] ** 2).sum() / (S ** 2).sum())
        C = (Z - self.mean) @ self.V.T                              # (R, K)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(self.n_gp_rows, len(X)), replace=False)
        self.cstd = C.std(0) + 1e-12
        k = ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=2.5, length_scale_bounds=(1e-2, 1e2)) \
            + WhiteKernel(1e-3, (1e-8, 1e-1))
        self.gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=0, random_state=self.seed)
        self.gp.fit((X[idx] - self.xm) / self.xs, C[idx] / self.cstd)
        return self

    def predict_grid(self, g: RunGraph) -> np.ndarray:
        C = self.gp.predict((inputs(g) - self.xm) / self.xs) * self.cstd
        Z = C @ self.V + self.mean
        return Z.reshape(len(C), 3, -1) * self.cs + self.cm

    def project_grid(self, F: np.ndarray) -> np.ndarray:
        """True field through POD truncation only (to isolate the GP's contribution)."""
        Z = ((F - self.cm) / self.cs).reshape(len(F), -1)
        return (((Z - self.mean) @ self.V.T) @ self.V + self.mean).reshape(len(F), 3, -1) * self.cs + self.cm


def run(dataset: str, n_modes: int, n_gp_rows: int):
    train = load_graphs(dataset, "train")
    model = PODGP(n_modes, n_gp_rows).fit(train)
    out = {"n_train_runs": len(train), "n_modes": n_modes, "pod_energy_captured": model.energy,
           "gp_rows": min(n_gp_rows, sum(g.s.shape[0] for g in train)), "gp_kernel": str(model.gp.kernel_)}
    drop = ("force_rmse", "force_mean_rel_err")
    from common.data import list_runs
    for sp in SPLITS:
        graphs = load_graphs(dataset, sp)
        if not graphs:
            continue
        stopped = {r["run_id"]: bool(r["stopped_early"]) for r in list_runs(dataset, sp)}
        res = {}
        for tag, mk in (("gp_pod", lambda g: from_grid(model.predict_grid(g), g)),
                        ("resampling_floor", lambda g: from_grid(to_grid(g), g)),
                        ("pod_truncation_floor", lambda g: from_grid(model.project_grid(to_grid(g)), g))):
            items = [(g, *mk(g), g.force) for g in graphs]
            m = _summarise(items)
            res[tag] = {k: v for k, v in m.items() if k not in drop}
            if tag == "gp_pod":
                for name, flag in (("early_stopped_runs", True), ("completed_runs", False)):
                    sub = [it for it in items if stopped[it[0].run_id] == flag]
                    if sub:
                        res[tag][name] = {k: v for k, v in _summarise(sub).items() if k not in drop}
        out[sp] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--out", default="results/gnn")
    ap.add_argument("--modes", type=int, default=24)
    ap.add_argument("--gp-rows", type=int, default=2500)
    a = ap.parse_args(argv)
    m = run(a.data, a.modes, a.gp_rows)
    cmd = "python -m surrogates.baselines.pod_field " + " ".join(sys.argv[1:])
    p = write_result(f"{a.out}/pod_field_baseline_{a.data}.json", "N5b", "POD + GP field baseline", cmd,
                     {"dataset": a.data, "n_runs": m["n_train_runs"]}, m,
                     notes="Fields resampled to a fixed 21x13 grid in initial normalised (xi, eta), POD per standardised "
                           "channel, GP (ARD Matern 5/2 + white noise) on a random subsample of training rows. "
                           "'resampling_floor' and 'pod_truncation_floor' push the TRUE fields through parts of the "
                           "pipeline and show what error is unavoidable. Force is not modelled. Same nRMSE definition "
                           "and ID rows as the GNN evaluation.")
    print(p)


if __name__ == "__main__":
    sys.exit(main())
