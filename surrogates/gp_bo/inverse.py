"""G3: inverse friction. Find m_f that gives a target inner-diameter change, by root-finding.

    python -m surrogates.gp_bo.inverse --out results/gp

The inner-diameter change falls monotonically with m_f, so "which m_f hits this diameter?" is a 1D
bracketed root problem: scipy.optimize.brentq on the solver itself (about 8-10 solver calls per target).
Bayesian optimisation would be the wrong tool here. As a cheap comparison the same root is found on the
GP posterior mean (zero solver calls) and checked with one solver call.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from surrogates.gp_bo.results_io import command_line, write_result
from surrogates.gp_bo.solver_calls import MAX_WORKERS, curve_at, default_case, run_one

TARGETS_ID_PCT = [10.0, -5.0, -15.0, -20.0]       # the last one sits in the flat high-friction region
REDUCTION = 0.30


def solver_id(mf: float, cache: str) -> float:
    r = run_one(dict(default_case(), mf=float(mf)), cache)
    return float(curve_at(r, [REDUCTION])[0])


def invert_with_solver(job):
    target, cache, xtol = job
    calls = []

    def f(mf):
        t0 = time.time()
        v = solver_id(mf, cache)
        calls.append((float(mf), v, time.time() - t0))
        return v - target

    root, info = brentq(f, 0.0, 1.0, xtol=xtol, full_output=True)
    return dict(target_id_pct=target, mf_root=float(root), solver_calls=len(calls),
                residual_pct=float(solver_id(root, cache) - target),    # one extra check call, not counted
                iterations=int(info.iterations), converged=bool(info.converged),
                calls=[dict(mf=a, id_pct=b) for a, b, _ in calls], seconds=float(sum(c[2] for c in calls)))


def invert_with_gp(gp, target, ring=None, reduction=REDUCTION):
    """Same root on the GP posterior mean: no solver calls."""
    c = default_case()
    ring = ring or dict(od_id=c["od_id"], h_id=c["h_id"], m=c["m"])

    def f(mf):
        mu, _ = gp.predict(np.array([[ring["od_id"], ring["h_id"], mf, ring["m"], reduction]]))
        return float(mu[0]) - target

    if f(0.0) * f(1.0) > 0:
        return None
    return float(brentq(f, 0.0, 1.0, xtol=1e-6))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gp")
    ap.add_argument("--cache", default="results/identifiability/solver_cache")
    ap.add_argument("--artifacts", default="artifacts/gp")
    ap.add_argument("--xtol", type=float, default=1e-3)
    a = ap.parse_args(argv)
    cmd = command_line("surrogates.gp_bo.inverse", argv)
    jobs = [(t, a.cache, a.xtol) for t in TARGETS_ID_PCT]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as ex:
        results = list(ex.map(invert_with_solver, jobs))

    gp = None
    p = Path(a.artifacts) / "id_gp.npz"
    if p.exists():
        from surrogates.gp_bo.gp import NumpyGP
        gp = NumpyGP.load(p)
    for r in results:
        r["gp_root_mf"] = invert_with_gp(gp, r["target_id_pct"]) if gp is not None else None
        if r["gp_root_mf"] is not None:
            r["solver_id_at_gp_root_pct"] = solver_id(r["gp_root_mf"], a.cache)
            r["gp_root_error_mf"] = r["gp_root_mf"] - r["mf_root"]
    write_result(Path(a.out) / "inverse_friction.json", task="G3", title="Inverse friction by brentq",
                 command=cmd, data=dict(dataset="solver calls (standard ring 6:3:2, m=0.15, 30% reduction)",
                                        n_runs=sum(r["solver_calls"] for r in results), splits={}),
                 metrics=dict(xtol=a.xtol, results=results,
                              total_solver_calls=sum(r["solver_calls"] for r in results)),
                 notes="solver_calls counts brentq function evaluations; results may have been served from "
                       "the solver cache (deterministic solver, so the count is what a cold start needs). "
                       "The 3 first targets are in the sensitive range; the last is in the flat high-friction "
                       "range, where the root is found but is poorly determined by a noisy measurement.")
    for r in results:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "calls"})


if __name__ == "__main__":
    main()
