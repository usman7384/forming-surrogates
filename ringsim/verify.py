"""Verification suite for the ring solver.

Run:  python -m ringsim.verify --out verify_out --workers 4

Checks (each prints PASS/FAIL/INFO and is written to verify_out/report.md + results.json):
  V1  frictionless ring = homogeneous compression: force, radii, strain and volume vs the exact solution
  V2  scale invariance: shape results do not depend on K or on the die speed V; force scales as K and V**m
  V3  friction trend: inner-diameter change falls monotonically with friction and changes sign
  V4  mesh convergence of the inner-diameter change
  V5  sticking bound: Tresca m=1 vs no-slip at the same reduction
  V6  solver health: Newton convergence and volume drift on every run
Extended checks (flag --extended; default output is unchanged):
  V1b patch test on unstructured meshes (3 seeds)
  V4b mesh convergence at m=0.7, 25% reduction (nz = 6/8/12)
  V5b friction-smoothing sensitivity: delta_rel 1e-2 vs 1e-3, Tresca m=1 vs no-slip
  V6b solver health on the extended runs
  V1-label what V1/V1b do and do not test (honest relabelling of the "patch test")
  V3b rate-exponent sensitivity of the V3 friction curve, m = 0.05 / 0.15 / 0.30
It also draws the ring-test calibration chart (inner-diameter change vs height reduction per friction factor).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from .solver import Numerics, RingCase, RingSolver, homogeneous_force

FRICTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
RATES = [0.05, 0.15, 0.3]       # rate exponents spanning the dataset range
BASE = RingCase(od_id=2.0, h_id=2.0 / 3.0, m=0.15, reduction=0.4)   # standard 6:3:2 ring
NUM = Numerics(nz=8, dh_frac=0.01)


def _run(args):
    name, case, num = args
    res = RingSolver(case, num).run()
    return name, dict(case=res.case, qoi=res.qoi, info=res.info,
                      curves={k: v.tolist() for k, v in res.curves.items()})


def _at(curves, red):
    """Inner-diameter change interpolated at a given height reduction (None if not reached)."""
    r = np.array(curves["reduction"])
    if red > r[-1] + 1e-12:
        return None
    return float(np.interp(red, r, curves["id_change_pct"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--extended", action="store_true",
                    help="also run V1b, V4b, V5b, V6b (about 12 more runs)")
    ap.add_argument("--replot", action="store_true",
                    help="redraw calibration_chart.png from <out>/results.json without solving")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.replot:
        _plots(json.load(open(os.path.join(a.out, "results.json")))["runs"], a.out)
        return

    jobs = []
    jobs.append(("homog", replace(BASE, mf=0.0, reduction=0.4), NUM))
    jobs.append(("inv_base", replace(BASE, mf=0.3, reduction=0.3), NUM))
    jobs.append(("inv_K", replace(BASE, mf=0.3, reduction=0.3, K=10.0), NUM))
    jobs.append(("inv_V", replace(BASE, mf=0.3, reduction=0.3, V=10.0), NUM))
    for mf in FRICTIONS:
        jobs.append((f"fric_{mf}", replace(BASE, mf=mf), NUM))
    for nz in (6, 12):
        jobs.append((f"mesh_{nz}", replace(BASE, mf=0.3, reduction=0.3), replace(NUM, nz=nz)))
    jobs.append(("noslip", replace(BASE, friction="noslip", mf=1.0), NUM))

    base_names = [j[0] for j in jobs]
    if a.extended:
        jobs += _extended_jobs()

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        R_all = dict(ex.map(_run, jobs))
    R = {k: R_all[k] for k in base_names}          # V6 and the plots stay on the original runs

    checks = []

    def add(cid, title, status, detail):
        checks.append(dict(id=cid, title=title, status=status, detail=detail))
        print(f"[{status:4s}] {cid} {title}: {detail}")

    # V1 homogeneous compression
    h = R["homog"]
    red = h["qoi"]["final_reduction"]
    f_ex = homogeneous_force(BASE, red)
    f_err = abs(h["curves"]["force"][-1] - f_ex) / f_ex * 100
    id_ex = 100 * (math.sqrt(1 / (1 - red)) - 1)
    id_err = abs(h["qoi"]["id_change_pct"] - id_ex) / id_ex * 100
    st_ex = -math.log(1 - red)
    st_err = abs(h["qoi"]["max_strain"] - st_ex) / st_ex * 100
    vol = abs(h["qoi"]["volume_drift_pct"])
    ok = f_err < 0.1 and id_err < 0.1 and st_err < 0.1 and vol < 0.05
    add("V1", "Patch test: frictionless ring vs exact homogeneous solution (exact solution lies in the FE space)", "PASS" if ok else "FAIL",
        f"at {100*red:.0f}% reduction: force error {f_err:.3f}%, inner-diameter error {id_err:.3f}%, "
        f"strain error {st_err:.3f}%, volume drift {vol:.4f}%")

    # V2 invariance
    b, k, v = R["inv_base"], R["inv_K"], R["inv_V"]
    d_k = abs(k["qoi"]["id_change_pct"] - b["qoi"]["id_change_pct"])
    d_v = abs(v["qoi"]["id_change_pct"] - b["qoi"]["id_change_pct"])
    rk = k["qoi"]["peak_force"] / b["qoi"]["peak_force"]
    rv = v["qoi"]["peak_force"] / b["qoi"]["peak_force"]
    rv_ex = 10 ** BASE.m
    ok = d_k < 1e-4 and d_v < 1e-4 and abs(rk - 10) < 1e-4 and abs(rv - rv_ex) / rv_ex < 1e-4
    add("V2", "Scaling in K and die speed implemented and checked", "PASS" if ok else "FAIL",
        f"inner-diameter change differs by {d_k:.1e} (K x10) and {d_v:.1e} (V x10) percentage points; "
        f"force ratio {rk:.6f} (expected 10) and {rv:.6f} (expected 10^m = {rv_ex:.6f})")

    # V3 friction trend at 30 % reduction
    red3 = 0.3
    ids = [_at(R[f"fric_{mf}"]["curves"], red3) for mf in FRICTIONS]
    pairs = [(mf, x) for mf, x in zip(FRICTIONS, ids) if x is not None]
    vals = [x for _, x in pairs]
    mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    sign = vals[0] > 0 and vals[-1] < 0
    add("V3", "Friction trend of the inner diameter (30% reduction)", "PASS" if mono and sign else "FAIL",
        "; ".join(f"m={mf}: {x:+.1f}%" for mf, x in pairs))

    # V4 mesh convergence
    i6, i8, i12 = (R["mesh_6"]["qoi"]["id_change_pct"], R["inv_base"]["qoi"]["id_change_pct"],
                   R["mesh_12"]["qoi"]["id_change_pct"])
    ok = abs(i8 - i12) < 1.0 and abs(i8 - i12) < abs(i6 - i12) + 1e-12
    add("V4", "Mesh convergence (m=0.3, 30% reduction)", "PASS" if ok else "FAIL",
        f"inner-diameter change {i6:+.2f}% (nz=6), {i8:+.2f}% (nz=8), {i12:+.2f}% (nz=12)")

    # V5 sticking bound
    t1, ns = R["fric_1.0"], R["noslip"]
    red5 = min(t1["qoi"]["final_reduction"], ns["qoi"]["final_reduction"])
    a5, b5 = _at(t1["curves"], red5), _at(ns["curves"], red5)
    add("V5", "Sticking bound: Tresca m=1 vs no-slip", "INFO",
        f"at {100*red5:.0f}% reduction: Tresca m=1 {a5:+.2f}%, no-slip {b5:+.2f}% "
        f"(difference {a5-b5:+.2f} points; regularised Tresca always allows some slip)")

    # V6 health
    worst_vol = max(abs(r["qoi"]["volume_drift_pct"]) for r in R.values())
    max_it = max(r["info"]["newton_iters_max"] for r in R.values())
    warns = [f"{n}: {w}" for n, r in R.items() for w in r["info"]["warnings"]]
    stops = [n for n, r in R.items() if r["info"]["stopped_early"]]
    nonconv = [w for w in warns if "not converged" in w]
    err_max = max(r["info"]["newton_err_max"] for r in R.values())
    add("V6", "Solver health on all runs", "PASS" if (not nonconv and worst_vol < 1.0) else "FAIL",
        f"{len(R)} runs, worst volume drift {worst_vol:.2f}%, max Newton iterations {max_it}, "
        f"solves not reaching the 1e-8 tolerance {len(nonconv)} (worst relative update {err_max:.1e}), "
        f"stopped early by fold-over: {', '.join(stops) or 'none'}")

    if a.extended:
        _extended_checks(R_all, add, a.out)

    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(dict(checks=checks, runs=R_all), fh, indent=1)
    _plots(R, a.out)
    _report(checks, R, a.out)


def _extended_jobs():
    jobs = []
    for seed in (0, 1, 2):
        jobs.append((f"unstr_patch_{seed}", replace(BASE, mf=0.0, reduction=0.4),
                     replace(NUM, mesh="unstructured", mesh_seed=seed)))
    for nz in (6, 8, 12):
        jobs.append((f"m07_nz{nz}", replace(BASE, mf=0.7, reduction=0.25), replace(NUM, nz=nz)))
    for d in (1e-2, 1e-3):
        jobs.append((f"stick_d{d:g}", replace(BASE, mf=1.0), replace(NUM, delta_rel=d)))
    for m in RATES:
        if m != BASE.m:                       # m = 0.15 is the V3 run set
            for mf in FRICTIONS:
                jobs.append((f"rate_{m}_{mf}", replace(BASE, mf=mf, m=m), NUM))
    return jobs


def _extended_checks(R, add, out=None):
    add("V1-label", "What V1 and V1b test (relabelled: frictionless homogeneous-field test)", "INFO",
        "V1 is not a patch test in the strict finite-element sense and not evidence about friction or contact. "
        "With zero friction the exact solution is a homogeneous field that lies in the P2/P1 space, so V1 only shows "
        "that the discretisation reproduces an exact homogeneous solution, and that assembly, the power-balance "
        "force and the time loop are consistent for it. It would pass with a wrong friction law or wrong contact "
        "handling. The friction physics is covered only by V3 (trend), V4/V4b (convergence) and V5/V5b (sticking bound).")
    # V1b patch test on unstructured meshes: same exact solution, now with arbitrary element shapes
    worst = dict(f=0.0, i=0.0, s=0.0, v=0.0)
    for seed in (0, 1, 2):
        h = R[f"unstr_patch_{seed}"]
        red = h["qoi"]["final_reduction"]
        f_ex = homogeneous_force(BASE, red)
        id_ex = 100 * (math.sqrt(1 / (1 - red)) - 1)
        st_ex = -math.log(1 - red)
        worst["f"] = max(worst["f"], abs(h["curves"]["force"][-1] - f_ex) / f_ex * 100)
        worst["i"] = max(worst["i"], abs(h["qoi"]["id_change_pct"] - id_ex) / id_ex * 100)
        worst["s"] = max(worst["s"], abs(h["qoi"]["max_strain"] - st_ex) / st_ex * 100)
        worst["v"] = max(worst["v"], abs(h["qoi"]["volume_drift_pct"]))
    ok = worst["f"] < 0.05 and worst["i"] < 0.1 and worst["s"] < 0.1 and worst["v"] < 0.05
    add("V1b", "Patch test on unstructured meshes (3 random Delaunay meshes, 40% reduction)",
        "PASS" if ok else "FAIL",
        f"worst over seeds: force error {worst['f']:.3f}%, inner-diameter error {worst['i']:.3f}%, "
        f"strain error {worst['s']:.3f}%, volume drift {worst['v']:.4f}%")

    # V4b mesh convergence at high friction, where the strain gradients are steepest
    red4 = 0.25
    v = {nz: _at(R[f"m07_nz{nz}"]["curves"], red4) for nz in (6, 8, 12)}
    if any(x is None for x in v.values()):
        add("V4b", "Mesh convergence (m=0.7, 25% reduction)", "FAIL",
            "a run stopped before 25% reduction: " + str({k: x for k, x in v.items()}))
    else:
        e6, e8 = abs(v[6] - v[12]), abs(v[8] - v[12])
        fn = {nz: float(np.interp(red4, R[f"m07_nz{nz}"]["curves"]["reduction"],
                                  R[f"m07_nz{nz}"]["curves"]["force_norm"])) for nz in (6, 8, 12)}
        ok = e8 < 0.25                # gating: inner diameter at nz=8 within 0.25 points of nz=12
        mono = "monotone" if e8 < e6 else "NOT monotone (the nz=6 mesh agrees with nz=12 at least as well as nz=8)"
        add("V4b", "Mesh convergence of the inner diameter (m=0.7, 25% reduction)", "PASS" if ok else "FAIL",
            f"inner-diameter change {v[6]:+.2f}% (nz=6), {v[8]:+.2f}% (nz=8), {v[12]:+.2f}% (nz=12); "
            f"distance to nz=12: {e6:.2f} points (nz=6), {e8:.2f} points (nz=8), {mono}")
        add("V4b-F", "Mesh sensitivity of the normalised force (m=0.7, 25% reduction), not gating", "INFO",
            f"normalised force {fn[6]:.4f} / {fn[8]:.4f} / {fn[12]:.4f} for nz=6/8/12; nz=8 differs from "
            f"nz=12 by {100*abs(fn[8]-fn[12])/fn[12]:.2f}%, nz=6 by {100*abs(fn[6]-fn[12])/fn[12]:.2f}%")

    # V5b smoothing sensitivity of the regularised Tresca law
    a2, a3, ns = R["stick_d0.01"], R["stick_d0.001"], R["noslip"]
    red5 = min(a2["qoi"]["final_reduction"], a3["qoi"]["final_reduction"], ns["qoi"]["final_reduction"])
    x2, x3, xn = (_at(r["curves"], red5) for r in (a2, a3, ns))
    add("V5b", "Friction-smoothing sensitivity: Tresca m=1 with delta_rel 1e-2 vs 1e-3, vs no-slip", "INFO",
        f"at {100*red5:.0f}% reduction: delta_rel=1e-2 {x2:+.2f}%, delta_rel=1e-3 {x3:+.2f}%, "
        f"no-slip {xn:+.2f}%; gap to no-slip {x2-xn:+.2f} -> {x3-xn:+.2f} points; "
        f"Newton iterations max {a2['info']['newton_iters_max']} -> {a3['info']['newton_iters_max']}; "
        f"final reduction reached {100*a2['qoi']['final_reduction']:.0f}% -> {100*a3['qoi']['final_reduction']:.0f}%")

    # V3b rate-exponent sensitivity at 30% reduction
    red3, table = 0.3, {}
    for m in RATES:
        key = (lambda mf: f"fric_{mf}") if m == BASE.m else (lambda mf, m=m: f"rate_{m}_{mf}")
        table[m] = [_at(R[key(mf)]["curves"], red3) for mf in FRICTIONS]
    ref = table[BASE.m]
    shifts = {m: [None if (x is None or r is None) else x - r for x, r in zip(table[m], ref)]
              for m in RATES if m != BASE.m}
    def mx(v):
        v = [abs(x) for x in v if x is not None]
        return max(v) if v else float("nan")
    rng = {m: [x for x in table[m] if x is not None] for m in RATES}
    spread = {m: max(v) - min(v) for m, v in rng.items()}
    txt = "; ".join(f"m_rate={m}: " + ", ".join("n/a" if x is None else f"{x:+.1f}" for x in table[m])
                    for m in RATES)
    add("V3b", "Rate-exponent sensitivity of the friction curve (30% reduction; m_f = "
        + ", ".join(str(f) for f in FRICTIONS) + ")", "INFO",
        f"inner-diameter change (%): {txt}; largest shift vs m_rate=0.15: "
        + ", ".join(f"{mx(v):.2f} points (m_rate={m})" for m, v in shifts.items())
        + "; total span over m_f: " + ", ".join(f"{spread[m]:.1f} points (m_rate={m})" for m in RATES)
        + ". K-independence is exact by construction (V2); dependence on the rate exponent is not.")
    if out:
        with open(os.path.join(out, "rate_sensitivity.json"), "w") as fh:
            json.dump(dict(task="S6", title="Rate-exponent sensitivity of the ring friction curve",
                           command="python -m ringsim.verify --out " + out + " --extended",
                           data=dict(dataset="verify", n_runs=len(table) * len(FRICTIONS), splits={}),
                           metrics=dict(frictions=FRICTIONS, reduction=red3,
                                        id_change_pct={str(m): table[m] for m in RATES},
                                        shift_vs_m015_points={str(m): shifts[m] for m in shifts},
                                        span_points={str(m): spread[m] for m in RATES}),
                           figures=[], notes="None = run stopped before 30% reduction.",
                           created=__import__("datetime").datetime.now().isoformat()), fh, indent=1)

    ext = [n for n in R if n.startswith(("unstr_patch", "m07_", "stick_d"))]
    nonconv = [w for n in ext for w in R[n]["info"]["warnings"] if "not converged" in w]
    wv = max(abs(R[n]["qoi"]["volume_drift_pct"]) for n in ext)
    stops = [n for n in ext if R[n]["info"]["stopped_early"]]
    add("V6b", "Solver health on the extended runs", "PASS" if (not nonconv and wv < 1.0) else "FAIL",
        f"{len(ext)} runs, worst volume drift {wv:.2f}%, max Newton iterations "
        f"{max(R[n]['info']['newton_iters_max'] for n in ext)}, solves not converged {len(nonconv)}, "
        f"stopped early by fold-over: {', '.join(stops) or 'none'}")


def _plots(R, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for mf in FRICTIONS:
        c = R[f"fric_{mf}"]["curves"]
        ax[0].plot(100 * np.array(c["reduction"]), c["id_change_pct"], label=f"m_f = {mf}")
        ax[1].plot(100 * np.array(c["reduction"]), c["force_norm"], label=f"m_f = {mf}")
    c = R["noslip"]["curves"]
    ax[0].plot(100 * np.array(c["reduction"]), c["id_change_pct"], "k--", label="no-slip")
    ax[1].plot(100 * np.array(c["reduction"]), c["force_norm"], "k--", label="no-slip")
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set(xlabel="Height reduction (%)", ylabel="Inner-diameter change at mid-height (%)",
              title=f"Ring calibration chart, 6:3:2 ring, rate exponent m = {BASE.m}")
    ax[1].set(xlabel="Height reduction (%)",
              ylabel="Normalised die force F / (K (V/h0)^m A0) (-)",
              title="Normalised force-stroke curves")
    # the step in the m_f = 1 force curve: an outer side-wall node is clipped onto the die and joins the
    # contact set in one step (contact is never smoothed or released) - a contact-treatment artefact
    c = R["fric_1.0"]["curves"]
    f, r = np.array(c["force_norm"]), 100 * np.array(c["reduction"])[: len(c["force_norm"])]
    if len(f) > 2:
        k = int(np.argmax(np.diff(f))) + 1
        ax[1].annotate("step: side-wall node joins die contact\n(contact-treatment artefact, m_f = 1)",
                       xy=(r[k], f[k]), xytext=(r[k] - 20, f[k] + 0.35), fontsize=8,
                       arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0].legend(fontsize=8, ncol=2, title="Tresca friction factor", title_fontsize=8)
    ax[1].legend(fontsize=8, ncol=2)
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.text(0.5, 0.005, "Curves of runs stopped early by fold-over end before 40% reduction (no remeshing).",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "calibration_chart.png"), dpi=150)


def _report(checks, R, out):
    lines = ["# Ring solver verification", "",
             "| Check | Result | Details |", "| --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c['id']} {c['title']} | {c['status']} | {c['detail']} |")
    lines += ["", "![calibration chart](calibration_chart.png)", ""]
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
