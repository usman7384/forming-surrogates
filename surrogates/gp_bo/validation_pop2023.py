"""First contact with experiment: invert published AA6068 ring tests (Pop et al. 2023, Table 2) for m_f.

    python -m surrogates.gp_bo.validation_pop2023 --out results/validation

Data: data/external/pop2023_table2.yaml (verbatim; CC BY; doi:10.3390/ma16052045). Standard 6:3:2 ring
(15/30/10 mm), cold AA6068, three lubrication conditions x reductions 0.2/0.3/0.4.

What is done, and why:
 1. measured inner-diameter change ID% = 100 (d1i / d0i - 1) per row;
 2. the solver (standard ring, rate exponent m = 0.05, the dataset's lowest; m = 0.02 as sensitivity) gives
    ID%(m_f; reduction); the Tresca factor m_f reproducing each row is found by brentq on the solver (as in G3),
    and also by inverting a 26-node solver sweep (PCHIP) for the uncertainty and sensitivity variants;
 3. consistency: one m_f per lubricant should explain all three reductions (spread reported);
 4. loose comparison with the authors' Coulomb mu via m_f ~ sqrt(3) mu (assumes contact pressure ~ flow
    stress and von Mises; their mu also comes from a different law read off a chart);
 5. secondary: outer diameter predicted at the inferred m_f vs measured;
 6. reading resolution +-0.05 mm on d1i propagated into an m_f interval.
This is a consistency check of the friction model against one published dataset, not a validation.
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from surrogates.gp_bo.results_io import command_line, write_result  # noqa: E402
from surrogates.gp_bo.solver_calls import curve_at, run_cases, run_one  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "external" / "pop2023_table2.yaml"
NODES = np.round(np.concatenate([np.arange(0.0, 0.5001, 0.025), [0.6, 0.7, 0.8, 0.9, 1.0]]), 4)
RED_MAX = 0.40
RES_MM = 0.05                     # half of the 0.1 mm reading resolution


def ring_case(mf: float, m: float, reduction: float = RED_MAX) -> dict:
    return dict(od_id=2.0, h_id=2.0 / 3.0, mf=float(mf), m=float(m), reduction=float(reduction), K=1.0, V=1.0)


def load_rows(path=DATA) -> tuple[dict, list[dict]]:
    doc = yaml.safe_load(open(path))
    cols = doc["columns"]
    rows = [dict(zip(cols, r)) for r in doc["rows"]]
    g = doc["ring_initial_mm"]
    for r in rows:
        r["id_change_pct"] = 100 * (r["d1i_mm"] / g["d0i"] - 1)
        r["id_change_pct_lo"] = 100 * ((r["d1i_mm"] - RES_MM) / g["d0i"] - 1)
        r["id_change_pct_hi"] = 100 * ((r["d1i_mm"] + RES_MM) / g["d0i"] - 1)
        r["od_change_pct"] = 100 * (r["d1e_mm"] / g["d0e"] - 1)
        r["od_change_pct_res"] = 100 * RES_MM / g["d0e"]
        r["height_reduction_from_h1"] = 1 - r["h1_mm"] / g["h0"]
    return doc, rows


def sweep(m: float, cache: str, workers: int):
    res = run_cases([ring_case(f, m) for f in NODES], cache, workers=workers)
    final = np.array([r["final_reduction"] for r in res])
    return res, final


def root_from_sweep(res, final, eps: float, target: float, key="id_change_pct"):
    """m_f with ID(eps) = target by inverting the PCHIP of the sweep (only nodes that reached eps)."""
    ok = final >= eps - 1e-9
    nodes = NODES[ok]
    vals = np.array([curve_at(r, [eps], key)[0] for r, k in zip(res, ok) if k])
    f = PchipInterpolator(nodes, vals)
    g = lambda x: float(f(x)) - target
    if g(nodes[0]) * g(nodes[-1]) > 0:
        return None, float(nodes[-1])
    return float(brentq(g, nodes[0], nodes[-1], xtol=1e-6)), float(nodes[-1])


def _brentq_job(job):
    """brentq on the solver itself (each call is a solver run to 40% reduction, cached)."""
    eps, target, m, hi, cache = job
    calls = []

    def f(mf):
        r = run_one(ring_case(mf, m), cache)
        v = float(curve_at(r, [eps])[0])
        calls.append(float(mf))
        return v - target

    if f(0.0) * f(hi) > 0:
        return dict(mf=None, solver_calls=len(calls))
    root = brentq(f, 0.0, hi, xtol=1e-3)
    return dict(mf=float(root), solver_calls=len(calls))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validation")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    cmd = command_line("surrogates.gp_bo.validation_pop2023", argv)
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cache = a.cache or str(out / "solver_cache")
    doc, rows = load_rows()

    sweeps = {}
    for m in (0.05, 0.02):
        try:
            sweeps[m] = sweep(m, cache, a.workers)
        except Exception as e:                     # report plainly if the solver refuses m = 0.02
            sweeps[m] = e
    res5, fin5 = sweeps[0.05]
    fold = {f"m={m}": ({f"{n}": float(fr) for n, fr in zip(NODES, s[1])} if not isinstance(s, Exception) else repr(s))
            for m, s in sweeps.items()}

    # brentq on the solver (nominal, m = 0.05), parallel over rows
    jobs = []
    for r in rows:
        hi = float(NODES[fin5 >= r["eps"] - 1e-9].max())
        jobs.append((r["eps"], r["id_change_pct"], 0.05, hi, cache))
    with ProcessPoolExecutor(max_workers=min(4, a.workers)) as ex:
        bq = list(ex.map(_brentq_job, jobs))

    table = []
    for r, b in zip(rows, bq):
        e = r["eps"]
        nom, hi_mf = root_from_sweep(res5, fin5, e, r["id_change_pct"])
        lo_b, _ = root_from_sweep(res5, fin5, e, r["id_change_pct_hi"])      # larger hole -> lower friction
        hi_b, _ = root_from_sweep(res5, fin5, e, r["id_change_pct_lo"])
        bore, _ = root_from_sweep(res5, fin5, e, r["id_change_pct"], key="id_min_change_pct")
        m2 = None
        if not isinstance(sweeps[0.02], Exception):
            m2, _ = root_from_sweep(*sweeps[0.02], e, r["id_change_pct"])
        row = dict(condition=r["condition"], eps=e, d1i_mm=r["d1i_mm"], d1e_mm=r["d1e_mm"], h1_mm=r["h1_mm"],
                   mu_authors=r["mu"], id_change_pct_measured=r["id_change_pct"],
                   mf_brentq_solver=b["mf"], solver_calls=b["solver_calls"],
                   mf_sweep_inversion=nom, mf_interval_reading_resolution=[lo_b, hi_b],
                   mf_if_id_is_bore_minimum=bore, mf_with_m_0p02=m2,
                   largest_mf_reaching_this_reduction_m0p05=hi_mf,
                   sqrt3_mu=math.sqrt(3) * r["mu"])
        # outer diameter at the inferred m_f: one solver run to exactly this reduction
        if b["mf"] is not None:
            q = run_one(ring_case(b["mf"], 0.05, e), cache)
            row["od_change_pct_measured"] = r["od_change_pct"]
            row["od_change_pct_measured_resolution"] = r["od_change_pct_res"]
            row["od_change_pct_solver_at_mf"] = float(q["qoi"]["od_change_pct"])
            row["id_change_pct_solver_at_mf_check"] = float(curve_at(q, [e])[0])
            row["solver_reached_reduction"] = bool(q["final_reduction"] >= e - 1e-9)
        table.append(row)

    by = {}
    for cond in dict.fromkeys(r["condition"] for r in rows):
        mfs = [t["mf_brentq_solver"] for t in table if t["condition"] == cond and t["mf_brentq_solver"] is not None]
        mus = [t["mu_authors"] for t in table if t["condition"] == cond]
        by[cond] = dict(mf_per_reduction=mfs, mf_mean=float(np.mean(mfs)) if mfs else None,
                        mf_spread_max_minus_min=float(np.ptp(mfs)) if mfs else None,
                        mf_sd=float(np.std(mfs, ddof=1)) if len(mfs) > 1 else None,
                        mu_authors_mean=float(np.mean(mus)), sqrt3_mu_mean=float(math.sqrt(3) * np.mean(mus)),
                        ratio_mf_to_sqrt3_mu=(float(np.mean(mfs)) / (math.sqrt(3) * np.mean(mus))) if mfs else None)
    order = sorted(by, key=lambda c: -(by[c]["mf_mean"] or 0))

    # figure: measured ID vs reduction, solver curves at the inferred mean m_f per lubricant
    colors = {"Dry": "tab:red", "Mineral oil": "tab:blue", "Graphite in oil": "tab:green"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cond, d in by.items():
        rr = [t for t in table if t["condition"] == cond]
        x = [t["eps"] for t in rr]
        y = [t["id_change_pct_measured"] for t in rr]
        ax.errorbar(x, y, yerr=100 * RES_MM / 15, fmt="o", color=colors.get(cond), capsize=3,
                    label=f"{cond}: measured")
        if d["mf_mean"] is not None:
            c = run_one(ring_case(d["mf_mean"], 0.05), cache)
            n = min(len(c["curves"]["reduction"]), len(c["curves"]["id_change_pct"]))
            ax.plot(c["curves"]["reduction"][:n], c["curves"]["id_change_pct"][:n], "-", color=colors.get(cond),
                    lw=1, label=f"solver, m_f = {d['mf_mean']:.3f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("height reduction (-)")
    ax.set_ylabel("inner-diameter change (%)")
    ax.set_title("AA6068 ring tests (Pop et al. 2023, Table 2) vs ringsim, m = 0.05", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fp = out / "figures" / "ring_test_pop2023.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)

    n_runs = len(NODES) * len([s for s in sweeps.values() if not isinstance(s, Exception)])
    write_result(out / "ring_test_pop2023.json", task="V-ext1",
                 title="First contact with experiment: AA6068 ring tests (Pop et al. 2023) inverted for m_f",
                 command=cmd, data=dict(dataset="data/external/pop2023_table2.yaml", n_runs=n_runs, splits={},
                                        source=doc["source"]),
                 metrics=dict(rows=table, by_lubricant=by, ordering_by_mf=order,
                              ordering_by_authors_mu=sorted(by, key=lambda c: -by[c]["mu_authors_mean"]),
                              sweep_final_reduction=fold, rate_exponent_m=0.05, sensitivity_m=0.02,
                              reading_resolution_mm=RES_MM),
                 figures=[str(fp)],
                 notes="Consistency check of the Tresca friction model against one published dataset, not a "
                       "validation: the authors' mu is Coulomb, read off a Male-Cockcroft chart; m_f ~ sqrt(3) mu "
                       "assumes contact pressure ~ flow stress (von Mises); m and the material curve of AA6068 are "
                       "not known here (no hardening in the solver); one lab, one alloy. ID is the solver's "
                       "mid-plane inner diameter; the bore-minimum variant is reported.")
    for t in table:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()})
    for c, d in by.items():
        print(c, {k: (np.round(v, 4).tolist() if v is not None and not isinstance(v, str) else v) for k, v in d.items()})
    print("fold-over:", fold)


if __name__ == "__main__":
    main()
