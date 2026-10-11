"""Where to measure: posterior std of m_f for ID measured at different strokes, with unknown rate sensitivity m.

    python -m surrogates.gp_bo.measurement_design --out results/identifiability

Extends the identifiability study (identifiability.py) in two ways asked for by the expert review:
 1. WHICH inner-diameter measurement is most informative: one ID at 10%, 20% or 30% reduction, two or three
    measurements at different reductions, or a repeated measurement at 30% (independent gauge noise 0.3 pt
    each).
 2. A latent, unknown strain-rate sensitivity m (uniform on the dataset range 0.05-0.30) marginalised in the
    posterior, instead of the study's earlier assumption that m = 0.15 is known. m changes both force and ID.

Forward model: standard 6:3:2 ring; solver sweep on m_f in {0, 0.1, ..., 1} x m in {0.05, 0.10, 0.15, 0.225,
0.30} (the m = 0.15 runs are the identifiability sweep's cached runs); PCHIP interpolation in m_f, then in m.
Force model: F = K (V/h0)^m A0 F*(m_f, m; s) at 10 strokes. With m unknown the speed factor (V/h0)^m is
m-dependent and must be kept: for the solver's standard ring V = 1 and the initial half height h0 = 2/3
(in units of the inner radius), so V/h0 = 1.5 and log F = log K + m log 1.5 + log F* (+ a constant A0 term that
cancels). (A bug fixed after the correctness review: an earlier version took V/h0 = 1 and dropped this term.)
On a real press the size of this term depends on V/h0 in the units K is quoted in; it is a K-like confound.
Same prior on K and noise as identifiability.py. Data are noise-free (typical-case widths; the posterior std
hardly depends on the noise draw, see identifiability.json).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from surrogates.gp_bo.results_io import command_line, write_result  # noqa: E402
from surrogates.gp_bo.solver_calls import curve_at, default_case, run_cases  # noqa: E402

MF_NODES = np.round(np.linspace(0, 1, 11), 4)
M_NODES = np.array([0.05, 0.10, 0.15, 0.225, 0.30])
M_GRID = np.round(np.linspace(0.05, 0.30, 11), 4)
MF_GRID = np.linspace(0, 1, 201)
S_FORCE = np.linspace(0.03, 0.30, 10)
R_ID = (0.1, 0.2, 0.3)
SIG_F, SIG_ID, SIG_K = 0.01, 0.3, 0.075
V_OVER_H0 = 1.0 / (2.0 / 3.0)                 # die speed 1 over initial half height h_id = 2/3 (solver units)
DESIGNS = {                                  # name -> reductions at which ID is measured
    "force_only": (),
    "force+ID@10%": (0.1,), "force+ID@20%": (0.2,), "force+ID@30%": (0.3,),
    "force+ID@10%+30%": (0.1, 0.3), "force+ID@20%+30%": (0.2, 0.3), "force+ID@10%+20%": (0.1, 0.2),
    "force+ID@10%+20%+30%": (0.1, 0.2, 0.3), "force+ID@30% twice": (0.3, 0.3),
}


def build_forward(cache: str, workers: int = 4):
    """Arrays LF[m, mf, stroke] (log F*) and ID[m, mf, reduction] on the fine (m, m_f) grid."""
    base = default_case()
    cases = [dict(base, mf=float(f), m=float(m)) for m in M_NODES for f in MF_NODES]
    res = run_cases(cases, cache, workers=workers)
    stopped = [(r["case"]["m"], r["case"]["mf"]) for r in res if r["final_reduction"] < 0.3 - 1e-9]
    assert not stopped, f"runs stopped before 30%: {stopped}"
    LF = np.array([np.log(curve_at(r, S_FORCE, "force_norm")) for r in res]).reshape(len(M_NODES), len(MF_NODES), -1)
    ID = np.array([curve_at(r, R_ID) for r in res]).reshape(len(M_NODES), len(MF_NODES), -1)

    def interp(A):                                    # PCHIP in m_f, then in m
        A = PchipInterpolator(MF_NODES, A, axis=1)(MF_GRID)
        return PchipInterpolator(M_NODES, A, axis=0)(M_GRID)

    LFi = interp(LF) + M_GRID[:, None, None] * np.log(V_OVER_H0)        # speed factor (V/h0)^m, m-dependent
    return res, LFi, interp(ID), LF, ID


def posterior_std(LF, ID, y_force, y_ids, r_idx, m_index=None, sig_k=SIG_K):
    """Exact grid posterior over (m, m_f, log K); returns (std of m_f, mean of m_f, std of m).

    m_index: fix m to that grid index (m known) or None to marginalise it (uniform prior)."""
    ms = slice(None) if m_index is None else slice(m_index, m_index + 1)
    lf, idv = LF[ms], ID[ms]
    lk = np.linspace(-5 * sig_k, 5 * sig_k, 161)
    r = y_force[None, None, :] - lf                              # (m, mf, s)
    n = r.shape[-1]
    S1, S2 = r.sum(-1)[..., None], (r ** 2).sum(-1)[..., None]
    ll = -0.5 * (S2 - 2 * lk * S1 + n * lk ** 2) / SIG_F ** 2 - 0.5 * lk ** 2 / sig_k ** 2
    for y, j in zip(y_ids, r_idx):
        ll = ll + (-0.5 * (y - idv[:, :, j]) ** 2 / SIG_ID ** 2)[..., None]
    p = np.exp(ll - ll.max())
    p /= p.sum()
    pf = p.sum((0, 2))
    mean_f = float((MF_GRID * pf).sum())
    std_f = float(np.sqrt(((MF_GRID - mean_f) ** 2 * pf).sum()))
    mg = M_GRID[ms]
    pm = p.sum((1, 2))
    mean_m = float((mg * pm).sum())
    std_m = float(np.sqrt(((mg - mean_m) ** 2 * pm).sum()))
    return std_f, mean_f, std_m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/identifiability")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k-true", type=float, default=1.07)
    a = ap.parse_args(argv)
    cmd = command_line("surrogates.gp_bo.measurement_design", argv)
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    res, LF, ID, LF_nodes, ID_nodes = build_forward(a.cache or str(out / "solver_cache"), a.workers)
    logk = float(np.log(a.k_true))
    im_known = int(np.argmin(np.abs(M_GRID - 0.15)))

    # check of the interpolation in m: leave the m = 0.15 node out, predict it from the other four
    keep = [0, 1, 3, 4]
    f_hat = PchipInterpolator(M_NODES[keep], LF_nodes[keep], axis=0)(0.15)
    i_hat = PchipInterpolator(M_NODES[keep], ID_nodes[keep], axis=0)(0.15)
    low = MF_NODES <= 0.5 + 1e-9                       # above ~0.55 the force has contact-artefact jumps
    interp_check = dict(max_force_err_pct_leaving_out_m_0p15=float(100 * np.max(np.abs(np.exp(f_hat - LF_nodes[2]) - 1))),
                        max_force_err_pct_leaving_out_m_0p15_mf_le_0p5=float(
                            100 * np.max(np.abs(np.exp(f_hat[low] - LF_nodes[2][low]) - 1))),
                        max_id_err_pts_leaving_out_m_0p15=float(np.max(np.abs(i_hat - ID_nodes[2]))))

    table = {}
    truths = [(0.1, 0.15), (0.25, 0.15), (0.5, 0.15), (0.25, 0.05), (0.25, 0.30)]
    for mf_t, m_t in truths:
        i_f = int(np.argmin(np.abs(MF_GRID - mf_t)))
        i_m = int(np.argmin(np.abs(M_GRID - m_t)))
        y_force = logk + LF[i_m, i_f]
        row = {}
        for name, reds in DESIGNS.items():
            r_idx = [R_ID.index(x) for x in reds]
            y_ids = [ID[i_m, i_f, j] for j in r_idx]
            s_unk, mean_unk, std_m = posterior_std(LF, ID, y_force, y_ids, r_idx, None)
            entry = dict(std_mf_m_unknown=s_unk, mean_mf_m_unknown=mean_unk, std_m_posterior=std_m)
            if abs(m_t - 0.15) < 1e-9:
                entry["std_mf_m_known"] = posterior_std(LF, ID, y_force, y_ids, r_idx, im_known)[0]
            row[name] = entry
        table[f"true_mf={mf_t},true_m={m_t}"] = row

    # figure: truth m_f 0.25, m = 0.15
    key = "true_mf=0.25,true_m=0.15"
    names = list(DESIGNS)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [table[key][n]["std_mf_m_known"] for n in names], 0.4, label="m known (0.15)")
    ax.bar(x + 0.2, [table[key][n]["std_mf_m_unknown"] for n in names], 0.4, label="m unknown (0.05-0.30)")
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("posterior std of m_f (true m_f 0.25)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fp = out / "figures" / "where_to_measure.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)

    write_result(out / "where_to_measure.json", task="G4b", title="Where to measure: ID at which reduction, and unknown m",
                 command=cmd, data=dict(dataset="solver sweep (standard ring)", n_runs=len(res), splits={}),
                 metrics=dict(designs={k: list(v) for k, v in DESIGNS.items()}, truths_mf_m=truths,
                              force_noise=SIG_F, id_sigma_pct_points=SIG_ID, prior_logk_sigma=SIG_K,
                              m_prior="uniform 0.05-0.30", mf_nodes=MF_NODES.tolist(), m_nodes=M_NODES.tolist(),
                              interpolation_check=interp_check, std_mf=table),
                 figures=[str(fp)],
                 notes="Noise-free data (typical widths). Force includes the m-dependent speed factor (V/h0)^m with "
                       "V/h0 = 1.5 (solver units, standard ring); an earlier version wrongly used V/h0 = 1. "
                       "Standard ring only.")
    for k, row in table.items():
        print(k)
        for n, e in row.items():
            print("  ", n, {kk: round(v, 4) for kk, v in e.items()})
    print(interp_check)


if __name__ == "__main__":
    main()
