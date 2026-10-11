"""Cached, parallel calls to the ringsim solver for the GP studies (G2-G5).

Why a cache: one solver run costs 10-25 s, and several studies (identifiability, inverse friction,
design validation) ask for the same case. A run is keyed by a hash of (case, numerics); the cached JSON
holds only the per-step curves (a few kB), never the fields. ringsim is used read-only.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MAX_WORKERS = 4          # lead's instruction: the full campaign is using the rest of the CPU


def _key(case: dict, num: dict) -> str:
    s = json.dumps([case, num], sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _solve(job):
    from ringsim.solver import Numerics, RingCase, RingSolver
    case, num = job
    res = RingSolver(RingCase(**case), Numerics(**num)).run()
    return dict(case=case, numerics=num, key=_key(case, num),
                curves={k: np.asarray(v).tolist() for k, v in res.curves.items()},
                stopped_early=bool(res.info["stopped_early"]), runtime_s=res.info["runtime_s"],
                final_reduction=res.qoi["final_reduction"],
                qoi={k: float(v) for k, v in res.qoi.items()})


def run_cases(cases: list[dict], cache_dir: str | Path, numerics: dict | None = None,
              workers: int = MAX_WORKERS, log=print) -> list[dict]:
    """Solve each case dict (RingCase fields) unless cached; returns result dicts in input order.

    Each result: case, curves (lists), stopped_early, runtime_s, final_reduction, cached (bool).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    num = dict(numerics or {})
    out: list[dict | None] = [None] * len(cases)
    todo = []
    for i, c in enumerate(cases):
        f = cache / f"{_key(c, num)}.json"
        if f.exists():
            with open(f) as fh:
                out[i] = dict(json.load(fh), cached=True)
        else:
            todo.append(i)
    if todo:
        jobs = [(cases[i], num) for i in todo]
        w = max(1, min(workers, MAX_WORKERS, len(jobs)))
        log(f"solver: {len(jobs)} runs on {w} workers ({len(cases) - len(jobs)} cached)")
        with ProcessPoolExecutor(max_workers=w) as ex:
            for i, r in zip(todo, ex.map(_solve, jobs)):
                with open(cache / f"{r['key']}.json", "w") as fh:
                    json.dump(r, fh)
                out[i] = dict(r, cached=False)
    return out  # type: ignore[return-value]


def run_one(case: dict, cache_dir: str | Path, numerics: dict | None = None) -> dict:
    """One solver call in the current process (cached). Used where calls are sequential (brentq)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    num = dict(numerics or {})
    f = cache / f"{_key(case, num)}.json"
    if f.exists():
        with open(f) as fh:
            return dict(json.load(fh), cached=True)
    r = _solve((case, num))
    with open(f, "w") as fh:
        json.dump(r, fh)
    return dict(r, cached=False)


def curve_at(res: dict, reductions, key: str = "id_change_pct") -> np.ndarray:
    """Interpolate a cached curve at fixed reductions (NaN beyond the last reached reduction)."""
    x, y = np.asarray(res["curves"]["reduction"]), np.asarray(res["curves"][key])
    n = min(len(x), len(y))
    r = np.asarray(reductions, float)
    v = np.interp(r, x[:n], y[:n])
    v[r > x[n - 1] + 1e-9] = np.nan
    return v


def default_case() -> dict:
    """Standard 6:3:2 ring, m = 0.15, K = V = 1."""
    return dict(od_id=2.0, h_id=2.0 / 3.0, mf=0.3, m=0.15, reduction=0.30, K=1.0, V=1.0)
