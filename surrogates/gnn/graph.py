"""Graph builder (task N1): one solver run -> one static graph + one target set per saved step.

Design (frozen by the lead): one-shot, stroke-conditioned, Lagrangian.
  * the graph is the run's INITIAL mesh (saved step 0), so topology never changes;
  * the stroke fraction s is an input; targets are displacement and nodal strain at s, plus the
    normalised die force at s (graph-level);
  * boundary type is an input only as it was at the start (side nodes roll onto the die later, so
    contact at s is something the network must predict, not read).

Boundary flags are multi-hot, not strictly one-hot: a corner node is both "inner" and "on die".
"Interior" is 1 only when no other flag is set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from common.data import Run, iter_runs, list_runs

S_MAX = 0.35                    # campaign target reduction; s = reduction / S_MAX lies in [0, 1]
BOUNDARY = ["inner", "outer", "die", "mid", "interior"]
TOL = 1e-9
N_NODE_IN = 2 + len(BOUNDARY)   # r, z, flags
N_GLOBAL = 3                    # m_f, m, s
N_EDGE_IN = 3                   # dr, dz, length
N_NODE_OUT = 3                  # dr, dz, strain


def boundary_flags(p0: np.ndarray) -> np.ndarray:
    """(N, 5) multi-hot flags from initial positions p0 (2, N). Geometry is a rectangle in (r, z)."""
    r, z = p0
    inner = np.abs(r - r.min()) < TOL
    outer = np.abs(r - r.max()) < TOL
    die = np.abs(z - z.max()) < TOL
    mid = np.abs(z - z.min()) < TOL
    interior = ~(inner | outer | die | mid)
    return np.stack([inner, outer, die, mid, interior], axis=1).astype(np.float32)


def undirected_edges(tri: np.ndarray) -> np.ndarray:
    """(2, 2*E_unique) directed edge index from triangles (3, T); each mesh edge appears both ways."""
    a = np.concatenate([tri[[0, 1]], tri[[1, 2]], tri[[2, 0]]], axis=1)
    a = np.sort(a, axis=0)
    a = np.unique(a, axis=1)
    return np.concatenate([a, a[::-1]], axis=1)


def node_strain(tri: np.ndarray, strain_el: np.ndarray, n_nodes: int) -> np.ndarray:
    """Average element strain (S, T) onto nodes (S, N): plain mean over the elements touching a node."""
    out = np.zeros((strain_el.shape[0], n_nodes))
    cnt = np.zeros(n_nodes)
    for k in range(3):
        np.add.at(cnt, tri[k], 1.0)
        for s in range(strain_el.shape[0]):
            np.add.at(out[s], tri[k], strain_el[s])
    return out / np.maximum(cnt, 1.0)


def triangle_area_rbar(p: np.ndarray, tri: np.ndarray):
    """Area and centroid radius per triangle. p: (..., 2, N) -> (..., T) each. Works for numpy."""
    r, z = p[..., 0, :], p[..., 1, :]
    r0, r1, r2 = (r[..., tri[k]] for k in range(3))
    z0, z1, z2 = (z[..., tri[k]] for k in range(3))
    area = 0.5 * np.abs((r1 - r0) * (z2 - z0) - (r2 - r0) * (z1 - z0))
    return area, (r0 + r1 + r2) / 3.0


def j_ratio(p: np.ndarray, p0: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """J = (A/A0) * (rbar/rbar0) per triangle. Axisymmetric volume is 2*pi*A*rbar exactly for a linear
    triangle, so J is the element's volume ratio and must be 1 for incompressible flow."""
    a, rb = triangle_area_rbar(p, tri)
    a0, rb0 = triangle_area_rbar(p0, tri)
    return (a / a0) * (rb / rb0)


def id_change_pct(p_inner_mid_r: np.ndarray, r_inner0: float) -> np.ndarray:
    """Inner-diameter change at mid-height in percent from the radius of the inner mid-plane node."""
    return (p_inner_mid_r / r_inner0 - 1.0) * 100.0


@dataclass
class RunGraph:
    """Everything about one run, as tensors. Samples are (run, saved-step) pairs sharing the static part."""
    run_id: str
    split: str
    case: dict
    x: torch.Tensor            # (N, 7) raw node features
    edge_index: torch.Tensor   # (2, 2E)
    edge_attr: torch.Tensor    # (2E, 3) raw
    glob: torch.Tensor         # (2,) m_f, m
    tri: torch.Tensor          # (3, T)
    p0: torch.Tensor           # (2, N)
    s: torch.Tensor            # (S,) stroke fraction per saved step
    disp: torch.Tensor         # (S, N, 2)
    strain: torch.Tensor       # (S, N)
    force: torch.Tensor        # (S,) normalised force
    inner_mid: int             # index of the node at (r_min, z=0): the inner-diameter probe


def build_run_graph(run: Run) -> RunGraph:
    p = run.fields["p"].astype(np.float64)                  # (S, 2, N)
    tri = run.fields["t"].astype(np.int64)
    p0 = p[0]
    n = p0.shape[1]
    flags = boundary_flags(p0)
    x = np.concatenate([p0.T, flags], axis=1).astype(np.float32)
    ei = undirected_edges(tri)
    d = p0[:, ei[1]] - p0[:, ei[0]]
    ea = np.stack([d[0], d[1], np.hypot(d[0], d[1])], axis=1).astype(np.float32)
    steps = run.fields["step"].astype(int)
    red = run.curves["reduction"][steps]
    force = run.curves["force_norm"][steps]
    disp = np.transpose(p - p0[None], (0, 2, 1))             # (S, N, 2)
    strain = node_strain(tri, run.fields["strain"], n)
    inner_mid = int(np.where((flags[:, 0] > 0) & (flags[:, 3] > 0))[0][0])
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return RunGraph(
        run_id=run.run_id, split=run.split, case=run.case, x=torch.from_numpy(x),
        edge_index=torch.from_numpy(ei), edge_attr=torch.from_numpy(ea),
        glob=f32([run.case["mf"], run.case["m"]]), tri=torch.from_numpy(tri), p0=f32(p0),
        s=f32(red / S_MAX), disp=f32(disp), strain=f32(strain), force=f32(force), inner_mid=inner_mid)


def load_graphs(dataset: str = "smoke", split: str | None = None, run_ids: list[str] | None = None) -> list[RunGraph]:
    out = []
    for run in iter_runs(dataset, split):
        if run_ids is None or run.run_id in run_ids:
            out.append(build_run_graph(run))
    return out


TEST_SPLITS = ("test_random", "test_ood_geometry", "test_fine_mesh")


def case_overlap(ds_a: str, split_a: str, ds_b: str, splits_b: tuple[str, ...], tol: float = 1e-9) -> list[tuple]:
    """Pairs (run in A, run in B) whose four varied inputs (od_id, h_id, mf, m) agree within tol.
    This is the leakage check when training data from dataset A is mixed in: no case may equal a test case of B."""
    key = lambda r: np.array([r["od_id"], r["h_id"], r["mf"], r["m"]])  # noqa: E731
    a = [(r["run_id"], key(r)) for r in list_runs(ds_a, split_a)]
    hits = []
    for sp in splits_b:
        for rb in list_runs(ds_b, sp):
            kb = key(rb)
            hits += [(ia, sp, rb["run_id"]) for ia, ka in a if np.max(np.abs(ka - kb)) < tol]
    return hits


def assert_no_case_overlap(ds_a: str, split_a: str, ds_b: str, splits_b: tuple[str, ...]) -> None:
    hits = case_overlap(ds_a, split_a, ds_b, splits_b)
    if hits:
        raise ValueError(f"leakage: {len(hits)} extra-train cases equal test cases, e.g. {hits[:3]}")


def split_run_ids(dataset: str, val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Train/validation run ids drawn from the `train` split only, by run (never by step)."""
    ids = sorted(r["run_id"] for r in list_runs(dataset, "train"))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_val = int(round(val_frac * len(ids))) if len(ids) > 2 else 0
    val = sorted(ids[i] for i in perm[:n_val])
    return [i for i in sorted(ids) if i not in val], val


# ----------------------------------------------------------------------------- normalisation
class Normaliser:
    """Per-quantity mean/std, fitted on training runs only and saved with the checkpoint.
    Each target gets its own statistics: displacement (2 comps), strain, force, and the inputs."""
    KEYS = ["x", "edge", "glob", "disp", "strain", "force"]

    def __init__(self, stats: dict | None = None):
        self.stats = stats or {}

    @classmethod
    def fit(cls, graphs: list[RunGraph]) -> "Normaliser":
        def ms(t: torch.Tensor):
            t = t.reshape(-1, t.shape[-1]) if t.dim() > 1 else t.reshape(-1, 1)
            return {"mean": t.mean(0).tolist(), "std": t.std(0).clamp_min(1e-8).tolist()}
        def cat(f):
            return torch.cat([f(g).reshape(-1, f(g).shape[-1]) for g in graphs])
        st = {
            "x": ms(cat(lambda g: g.x[:, :2])),
            "edge": ms(cat(lambda g: g.edge_attr)),
            "glob": ms(torch.stack([g.glob for g in graphs])),
            "disp": ms(cat(lambda g: g.disp.reshape(-1, 2))),
            "strain": ms(cat(lambda g: g.strain.reshape(-1, 1))),
            "force": ms(cat(lambda g: g.force.reshape(-1, 1))),
        }
        return cls(st)

    def _t(self, key: str, what: str) -> torch.Tensor:
        return torch.tensor(self.stats[key][what], dtype=torch.float32)

    def norm(self, key, t):
        return (t - self._t(key, "mean")) / self._t(key, "std")

    def denorm(self, key, t):
        return t * self._t(key, "std") + self._t(key, "mean")

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.stats, indent=1))

    @classmethod
    def load(cls, path) -> "Normaliser":
        return cls(json.loads(Path(path).read_text()))


def make_sample(g: RunGraph, k: int, nm: Normaliser) -> Data:
    """One training graph: run g at saved step k, all inputs and targets normalised."""
    n = g.x.shape[0]
    x = g.x.clone()
    x[:, :2] = nm.norm("x", g.x[:, :2])
    glob = torch.cat([nm.norm("glob", g.glob[None])[0], g.s[k:k + 1]])        # (3,) m_f, m, s
    probe = torch.zeros(n); probe[g.inner_mid] = 1.0         # inner-diameter probe node (mask)
    return Data(
        probe=probe, r0n=g.p0[0, g.inner_mid].expand(n).contiguous(),
        x=x, edge_index=g.edge_index, edge_attr=nm.norm("edge", g.edge_attr),
        glob=glob[None], y_disp=nm.norm("disp", g.disp[k]), y_strain=nm.norm("strain", g.strain[k][:, None]),
        y_force=nm.norm("force", g.force[k:k + 1][:, None]),
        tri_index=g.tri, p0=g.p0.T.contiguous(), num_nodes=n, run_idx=torch.tensor([0]), step_k=torch.tensor([k]))


def make_samples(graphs: list[RunGraph], nm: Normaliser) -> list[Data]:
    out = []
    for ri, g in enumerate(graphs):
        for k in range(g.s.shape[0]):
            d = make_sample(g, k, nm)
            d.run_idx = torch.tensor([ri])
            out.append(d)
    return out
