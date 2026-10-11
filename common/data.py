"""Shared dataset access for every phase. CONTRACT: do not change signatures without the lead.

A dataset directory is what `python -m ringsim.campaign --out <dir>` writes:
    <dir>/plan.json, <dir>/index.csv, <dir>/dataset_card.md, <dir>/runs/run_XXXX.{npz,json}

Where data lives:
    data/smoke                       13 runs committed in the repo (for tests and fast development)
    $FS_DATA_ROOT/full               the full campaign (300-500 runs), NOT in git
    $FS_DATA_ROOT/unstructured       optional unstructured-mesh test runs (task S4)
FS_DATA_ROOT defaults to ~/forming-data. It is outside the repo on purpose, so every git worktree
(every agent) reads the same data by absolute path.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "data" / "smoke"
FEATURES = ["od_id", "h_id", "mf", "m", "reduction"]      # scalar-model input order (CONTRACT)


def data_root() -> Path:
    return Path(os.environ.get("FS_DATA_ROOT", "~/forming-data")).expanduser()


def dataset_dir(name: str) -> Path:
    """'smoke' -> repo data/smoke; anything else -> $FS_DATA_ROOT/<name>; an existing path is used as is."""
    if name == "smoke":
        return SMOKE
    p = Path(name).expanduser()
    return p if p.exists() else data_root() / name


@dataclass
class Run:
    run_id: str
    split: str
    case: dict          # inputs: od_id, h_id, mf, m, reduction (target), K, V, ...
    qoi: dict           # values at the run's FINAL reduction (may be early for stopped runs!)
    info: dict          # solver version, git, newton stats, warnings, stopped_early, runtime
    fields: dict        # t (3,E), p (S,2,N), vel (S,2,N), press (S,N), strain (S,E), rate (S,E), step (S,)
    curves: dict        # per step: reduction, stroke, force, force_norm, id_change_pct, id_min_change_pct

    @property
    def max_reduction(self) -> float:
        return float(self.curves["reduction"][-1])

    def at(self, reductions, key: str = "id_change_pct") -> np.ndarray:
        """Curve value at fixed reductions; NaN where the run did not reach that reduction."""
        r = np.asarray(reductions, dtype=float)
        x, y = self.curves["reduction"], self.curves[key]
        out = np.interp(r, x, y)
        out[r > x[-1] + 1e-9] = np.nan
        return out


def list_runs(dataset: str = "smoke", split: str | None = None, ok_only: bool = True) -> list[dict]:
    """Rows of index.csv (strings converted to numbers where possible)."""
    d = dataset_dir(dataset)
    with open(d / "index.csv") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if ok_only and r.get("status") != "ok":
            continue
        if split and r["split"] != split:
            continue
        out.append({k: _num(v) for k, v in r.items()})
    return out


def load_run(dataset: str, run_id: str) -> Run:
    d = dataset_dir(dataset) / "runs"
    meta = json.load(open(d / f"{run_id}.json"))
    z = np.load(d / f"{run_id}.npz")
    fields = {k[len("field_"):]: z[k] for k in z.files if k.startswith("field_")}
    curves = {k[len("curve_"):]: z[k] for k in z.files if k.startswith("curve_")}
    # stroke has one more entry than per-step curves when a run ends normally; align on per-step length
    n = len(curves["force"])
    for k in ("stroke", "reduction"):
        curves[k] = curves[k][:n]
    return Run(run_id=run_id, split=meta["info"].get("split", "?"), case=meta["case"], qoi=meta["qoi"],
               info=meta["info"], fields=fields, curves=curves)


def iter_runs(dataset: str = "smoke", split: str | None = None):
    for r in list_runs(dataset, split):
        yield load_run(dataset, r["run_id"])


def scalar_table(dataset: str = "smoke", reductions=(0.1, 0.2, 0.3), split: str | None = None,
                 targets=("id_change_pct", "force_norm")):
    """Long table for scalar surrogates: one row per (run, reduction actually reached).

    Returns (X, Y, groups, splits) as numpy arrays: X columns = FEATURES, Y columns = targets,
    groups = run index (use it for group-wise CV; never split rows of one run across train/test).
    """
    X, Y, G, S = [], [], [], []
    for gi, run in enumerate(iter_runs(dataset, split)):
        vals = {t: run.at(reductions, t) for t in targets}
        for j, red in enumerate(reductions):
            y = [vals[t][j] for t in targets]
            if np.any(np.isnan(y)):
                continue
            c = run.case
            X.append([c["od_id"], c["h_id"], c["mf"], c["m"], red])
            Y.append(y)
            G.append(gi)
            S.append(run.split)
    return np.array(X), np.array(Y), np.array(G), np.array(S)


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except (TypeError, ValueError):
        return {"True": True, "False": False}.get(v, v)
