"""Data-support flag: where the training data are censored by early fold-over.

The solver stops a run when an element collapses (no remeshing). Those stops are concentrated at high
friction and thin rings, so at (high m_f, small H/ID, high reduction) the GP's training rows exist only for
the minority of runs that survived: the data there are a censored, survivor-biased sample. A GP posterior
sd does not know this, so the plant asks for an explicit flag.

Rule (documented, derived from the data, not tuned): a query (od_id, h_id, m_f, m, reduction) is OUT of
support iff, among the training runs with |m_f - m_f_q| <= WINDOW and |H/ID - h_q| <= WINDOW, at least
MIN_N exist and more than HALF of them ended (final reduction) before the requested reduction.
With no or too few neighbours the rule does not fire (outside the box is a different question).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WINDOW = 0.10
MIN_N = 5
FRACTION = 0.5


def build_support(dataset: str, split: str = "train") -> dict:
    """Training-run descriptors the rule needs: m_f, H/ID and final reduction of every ok run."""
    from common.data import list_runs
    rows = list_runs(dataset, split)
    return dict(rule="out of support iff >50% of >=5 training runs within +-0.10 in m_f and H/ID ended "
                     "before the requested reduction", window=WINDOW, min_n=MIN_N, fraction=FRACTION,
                dataset=dataset, split=split, n_runs=len(rows),
                mf=[float(r["mf"]) for r in rows], h_id=[float(r["h_id"]) for r in rows],
                final_reduction=[float(r["final_reduction"]) for r in rows])


def save_support(path: str | Path, sup: dict) -> None:
    with open(path, "w") as fh:
        json.dump(sup, fh)


def load_support(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def in_support(X: np.ndarray, sup: dict) -> np.ndarray:
    """Boolean array, True where the query is inside the data support (FEATURES order)."""
    X = np.atleast_2d(np.asarray(X, float))
    mf, h, fr = (np.asarray(sup[k]) for k in ("mf", "h_id", "final_reduction"))
    out = np.ones(len(X), bool)
    for i, x in enumerate(X):
        near = (np.abs(mf - x[2]) <= sup["window"]) & (np.abs(h - x[1]) <= sup["window"])
        n = int(near.sum())
        if n >= sup["min_n"] and np.mean(fr[near] < x[4] - 1e-9) > sup["fraction"]:
            out[i] = False
    return out
