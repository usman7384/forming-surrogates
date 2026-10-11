"""G4: can the press force identify friction? Exact grid posterior over (m_f, K).

    python -m surrogates.gp_bo.identifiability --out results/identifiability

Model. The measured force is F(s) = K * F*(m_f; s) (K absorbs material strength and speed; F* is the
solver's normalised force). The inner-diameter change ID(m_f; s) does not depend on K. So after a 1D
solver sweep over m_f for the ring in question, the (m_f, K) posterior is a 2D grid computation: K is
analytic and costs no solver runs. Observations (all synthetic, from a hidden truth):
  force   : log F_i = log K + log F*(m_f; s_i) + 1% Gaussian noise at 10 strokes between 3% and 30%
  ID      : one measurement at 30% reduction with sigma = 0.3 percentage points
  K known : an independent K measurement with 2% error (instead of the ID measurement)
Prior: m_f uniform on [0, 1]; log K ~ N(0, 0.075^2) (about +-15% at two sigma).
Caveat repeated in the docs: this is a self-consistency study on a verified solver, not validation.
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

MF_NODES = np.round(np.concatenate([np.arange(0.0, 0.5001, 0.025), [0.6, 0.7, 0.8, 0.9, 1.0]]), 4)
S_OBS = np.linspace(0.03, 0.30, 10)
S_ID = 0.30
STEP_PAIRS = [(0.0, 0.1), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]


class Forward:
    """Smooth forward model m_f -> (log F*(s_i), ID(30%)) built from a solver sweep (PCHIP in m_f)."""

    def __init__(self, nodes, logF, idc, mf_fine):
        self.nodes, self.logF_nodes, self.id_nodes = nodes, logF, idc
        self.mf = mf_fine
        self.logF = PchipInterpolator(nodes, logF, axis=0)(mf_fine)     # (Nm, Ns)
        self.id = PchipInterpolator(nodes, idc)(mf_fine)                 # (Nm,)

    def subset(self, sl) -> "Forward":
        """Same model restricted to some force strokes (columns)."""
        f = Forward.__new__(Forward)
        f.nodes, f.mf, f.id, f.id_nodes = self.nodes, self.mf, self.id, self.id_nodes
        f.logF, f.logF_nodes = self.logF[:, sl], self.logF_nodes[:, sl]
        return f

    def at(self, mf0: float):
        i = int(np.argmin(np.abs(self.mf - mf0)))
        return self.logF[i], self.id[i]


def build_sweep(cache: str, s_obs=S_OBS, mfs=MF_NODES, case=None, workers=4):
    base = dict(default_case() if case is None else case)
    cases = [dict(base, mf=float(m)) for m in mfs]
    res = run_cases(cases, cache, workers=workers)
    for r in res:
        assert r["final_reduction"] >= S_ID - 1e-9, f"run mf={r['case']['mf']} stopped early"
    logF = np.array([np.log(curve_at(r, s_obs, "force_norm")) for r in res])
    idc = np.array([curve_at(r, [S_ID], "id_change_pct")[0] for r in res])
    return res, logF, idc


def posterior(fw: Forward, y_force, *, sig_f=0.01, y_id=None, sig_id=0.3, sig_k=0.075,
              k_meas=None, sig_kmeas=0.02, logk_grid=None):
    """Exact grid posterior over (m_f, log K). Returns dict with the 2D array and the marginals."""
    lk = np.linspace(-5 * sig_k, 5 * sig_k, 401) if logk_grid is None else logk_grid
    r = y_force[None, :] - fw.logF                                   # (Nm, Ns)
    n = r.shape[1]
    S1, S2 = r.sum(1)[:, None], (r ** 2).sum(1)[:, None]
    ll = -0.5 * (S2 - 2 * lk[None, :] * S1 + n * lk[None, :] ** 2) / sig_f ** 2
    ll = ll + (-0.5 * lk ** 2 / sig_k ** 2)[None, :]
    if y_id is not None:
        ll = ll + (-0.5 * (y_id - fw.id) ** 2 / sig_id ** 2)[:, None]
    if k_meas is not None:
        ll = ll + (-0.5 * (k_meas - lk) ** 2 / sig_kmeas ** 2)[None, :]
    p = np.exp(ll - ll.max())
    p /= p.sum()
    pm, pk = p.sum(1), p.sum(0)
    mean_m = float((fw.mf * pm).sum())
    std_m = float(np.sqrt(((fw.mf - mean_m) ** 2 * pm).sum()))
    mean_k = float((lk * pk).sum())
    std_k = float(np.sqrt(((lk - mean_k) ** 2 * pk).sum()))
    cov = float(((fw.mf[:, None] - mean_m) * (lk[None, :] - mean_k) * p).sum())
    c = np.cumsum(pm)
    lo, hi = float(np.interp(0.05, c, fw.mf)), float(np.interp(0.95, c, fw.mf))
    return dict(post=p, lk=lk, pm=pm, pk=pk, mean_mf=mean_m, std_mf=std_m, mean_logk=mean_k,
                std_logk=std_k, corr=cov / (std_m * std_k + 1e-30), ci90=(lo, hi))


SCENARIOS = ("force_only", "force_plus_id", "force_known_k")


def run_scenario(name, fw, mf_true, logk_true, rng, *, noise=True, sig_f=0.01, sig_id=0.3, sig_k=0.075,
                 sig_kmeas=0.02, s_idx=None):
    """Simulate data from the hidden truth and return the posterior for one scenario."""
    logF_t, id_t = fw.at(mf_true)
    sl = slice(None) if s_idx is None else s_idx
    z = (lambda sd: rng.normal(0, sd)) if noise else (lambda sd: 0.0)
    y_force = logk_true + logF_t[sl] + (rng.normal(0, sig_f, size=logF_t[sl].shape) if noise else 0.0)
    fw_s = fw if s_idx is None else fw.subset(sl)
    kw = dict(sig_f=sig_f, sig_id=sig_id, sig_k=sig_k, sig_kmeas=sig_kmeas)
    if name == "force_only":
        return posterior(fw_s, y_force, **kw)
    if name == "force_plus_id":
        return posterior(fw_s, y_force, y_id=id_t + z(sig_id), **kw)
    return posterior(fw_s, y_force, k_meas=logk_true + z(sig_kmeas), **kw)


def expected_std(name, fw, mf_true, logk_true, n=100, seed=0, **kw):
    rng = np.random.default_rng(seed)
    s = [run_scenario(name, fw, mf_true, logk_true, rng, **kw)["std_mf"] for _ in range(n)]
    return float(np.mean(s)), float(np.std(s))


def friction_steps(res, s_grid=S_OBS):
    """Friction-step table recomputed from solver curves (standard ring)."""
    by = {round(r["case"]["mf"], 4): curve_at(r, s_grid, "force_norm") for r in res}
    rows = []
    for a, b in STEP_PAIRS:
        ra, rb = by[round(a, 4)], by[round(b, 4)]
        q = rb / ra
        lift = float(np.mean(q) - 1) * 100
        dev = q / np.mean(q) - 1                       # shape left after removing a uniform scale
        rows.append(dict(step=f"{a} -> {b}", mean_force_lift_pct=lift,
                         shape_change_max_pct=float(np.max(np.abs(dev)) * 100),
                         shape_change_rms_pct=float(np.sqrt(np.mean(dev ** 2)) * 100)))
    return rows


def _bias_row(p: dict, truth: float) -> dict:
    return dict(mean_mf=p["mean_mf"], std_mf=p["std_mf"], bias_mf=p["mean_mf"] - truth,
                bias_over_std=(p["mean_mf"] - truth) / p["std_mf"])


def discrepancy_study(fw: Forward, res, logk_true: float, truths=(0.25, 0.1), offsets=(0.01, 0.02, 0.05),
                      sig_f=0.01, sig_id=0.3) -> dict:
    """What happens when the data are NOT generated by the model that is inverted (noise-free data, so
    every number is a pure bias, not noise).

    (i) 'bore' ID: the measurement is the minimum inner radius change (id_min_change_pct, what a bore gauge
        sees) while the likelihood assumes the mid-plane value id_change_pct. The two differ by barrelling.
    (ii) additive force offset: F_obs = K F* + delta, delta = x% of the mean force over the strokes, constant;
        the model has only a multiplicative K.
    """
    mf_nodes = np.array([r["case"]["mf"] for r in res])
    idmin_nodes = np.array([curve_at(r, [S_ID], "id_min_change_pct")[0] for r in res])
    id_min = PchipInterpolator(mf_nodes, idmin_nodes)(fw.mf)
    out = {"definition": "noise-free synthetic data; bias = posterior mean m_f minus truth; ratio = bias / "
                         "posterior std (a ratio well above 1 means confidently wrong)"}
    for t in truths:
        logF_t, id_t = fw.at(t)
        y_force = logk_true + logF_t
        i = int(np.argmin(np.abs(fw.mf - t)))
        row = {"id_mid_plane_pct": float(id_t), "id_bore_min_pct": float(id_min[i]),
               "bore_minus_mid_pts": float(id_min[i] - id_t)}
        row["baseline_force_plus_id"] = _bias_row(posterior(fw, y_force, y_id=id_t, sig_f=sig_f, sig_id=sig_id), t)
        row["i_bore_id_instead_of_mid_plane"] = _bias_row(
            posterior(fw, y_force, y_id=float(id_min[i]), sig_f=sig_f, sig_id=sig_id), t)
        for off in offsets:
            F = np.exp(y_force)
            y_off = np.log(F + off * F.mean())
            row[f"ii_force_offset_{int(off * 100)}pct"] = dict(
                force_only=_bias_row(posterior(fw, y_off, sig_f=sig_f), t),
                force_plus_id=_bias_row(posterior(fw, y_off, y_id=id_t, sig_f=sig_f, sig_id=sig_id), t))
        out[f"true_mf={t}"] = row
    return out


def plot_posterior(res_post, fw, truth, title, path, k_truth):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    p = res_post["post"]
    K = np.exp(res_post["lk"])
    ax.contourf(fw.mf, K, p.T, levels=20, cmap="viridis")
    ax.plot(truth, k_truth, "r*", ms=12, label="hidden truth")
    ax.set_ylim(0.8, 1.3)
    ax.set_xlabel("friction factor m_f")
    ax.set_ylabel("strength multiplier K")
    ax.set_title(f"{title}\nposterior std m_f = {res_post['std_mf']:.3f}", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/identifiability")
    ap.add_argument("--cache", default=None, help="solver cache dir (default <out>/solver_cache)")
    ap.add_argument("--mf-true", type=float, default=0.25)
    ap.add_argument("--k-true", type=float, default=1.07)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cmd = command_line("surrogates.gp_bo.identifiability", argv)
    out = Path(a.out)
    fdir = out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    cache = a.cache or str(out / "solver_cache")

    res, logF, idc = build_sweep(cache, workers=a.workers)
    mf_fine = np.linspace(0, 1, 401)
    fw = Forward(MF_NODES, logF, idc, mf_fine)
    logk_true = float(np.log(a.k_true))

    # interpolation check: leave one node out and predict it (interior nodes 0.1 .. 0.5)
    loo_f, loo_i = [], []
    for j in range(len(MF_NODES)):
        if not (0.1 <= MF_NODES[j] <= 0.5):
            continue
        keep = np.arange(len(MF_NODES)) != j
        f_l = PchipInterpolator(MF_NODES[keep], logF[keep], axis=0)(MF_NODES[j])
        i_l = PchipInterpolator(MF_NODES[keep], idc[keep])(MF_NODES[j])
        loo_f.append(float(np.max(np.abs(np.exp(f_l - logF[j]) - 1))) * 100)
        loo_i.append(float(abs(i_l - idc[j])))

    # main scenario: one seeded realisation + expectation over noise draws
    rng = np.random.default_rng(a.seed)
    metrics = {"truth": dict(mf=a.mf_true, K=a.k_true, force_noise=0.01, id_sigma_pct=0.3,
                             k_known_sigma=0.02, prior_logk_sigma=0.075, force_strokes=S_OBS.tolist()),
               "posterior": {}, "expected_std_mf_over_100_noise_draws": {}}
    figs = []
    posts = {}
    for name, title in zip(SCENARIOS, ["Force only", "Force + one ID measurement (0.3 pt)",
                                       "Force + K known (2%)"]):
        p = run_scenario(name, fw, a.mf_true, logk_true, rng)
        posts[name] = p
        m, s = expected_std(name, fw, a.mf_true, logk_true, seed=a.seed + 1)
        metrics["posterior"][name] = dict(
            mean_mf=p["mean_mf"], std_mf=p["std_mf"], ci90_mf=p["ci90"], mean_K=float(np.exp(p["mean_logk"])),
            std_K_pct=100 * p["std_logk"], corr_mf_logk=p["corr"])
        metrics["expected_std_mf_over_100_noise_draws"][name] = dict(mean=m, sd=s)
        pth = fdir / f"posterior_{name}.png"
        plot_posterior(p, fw, a.mf_true, title, pth, a.k_true)
        figs.append(str(pth))
    ratio = (metrics["posterior"]["force_only"]["std_mf"] / metrics["posterior"]["force_plus_id"]["std_mf"])
    metrics["std_reduction_factor_id_vs_force_only"] = float(ratio)
    e = metrics["expected_std_mf_over_100_noise_draws"]
    metrics["std_reduction_factor_over_100_draws"] = e["force_only"]["mean"] / e["force_plus_id"]["mean"]
    metrics["prior_std_mf"] = float(np.sqrt(1.0 / 12.0))          # uniform prior on [0, 1]

    # marginals
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name in SCENARIOS:
        ax.plot(fw.mf, posts[name]["pm"] / (fw.mf[1] - fw.mf[0]), label=name)
    ax.axvline(a.mf_true, color="k", ls="--", lw=0.8)
    ax.set_xlabel("friction factor m_f")
    ax.set_ylabel("posterior density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    pth = fdir / "marginals_mf.png"
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    figs.append(str(pth))

    # where is force informative? expected posterior std vs hidden m_f (noise-free data = typical case)
    truths = np.linspace(0.05, 0.95, 19)
    curve = {n: [] for n in SCENARIOS}
    for t in truths:
        for n in SCENARIOS:
            curve[n].append(run_scenario(n, fw, t, logk_true, np.random.default_rng(0), noise=False)["std_mf"])
    metrics["std_mf_vs_true_mf"] = dict(true_mf=truths.tolist(), **{k: v for k, v in curve.items()})
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for n in SCENARIOS:
        ax.plot(truths, curve[n], "o-", ms=3, label=n)
    ax.set_yscale("log")
    ax.axvspan(0.45, 0.55, color="0.95", zorder=0)
    ax.axvspan(0.55, 1.0, color="0.9", zorder=0)
    ax.text(0.5, ax.get_ylim()[1] * 0.6, "force not\nmonotone\nin m_f", fontsize=6, ha="center", va="top")
    ax.text(0.77, ax.get_ylim()[1] * 0.6, "force steps from contact\nartefact: force-only\nstd unreliable here",
            fontsize=7, ha="center", va="top")
    ax.set_xlabel("true friction factor m_f")
    ax.set_ylabel("posterior std of m_f")
    ax.legend(fontsize=8)
    fig.tight_layout()
    pth = fdir / "std_vs_true_mf.png"
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    figs.append(str(pth))

    # sensitivity to analyst choices
    sens = {}
    for sk in (0.03, 0.075, 0.15):
        sens[f"prior_logK_sigma={sk}"] = {n: expected_std(n, fw, a.mf_true, logk_true, n=50, sig_k=sk)[0]
                                           for n in ("force_only", "force_plus_id")}
    for sid in (0.1, 0.3, 1.0):
        sens[f"id_sigma={sid}"] = {"force_plus_id": expected_std("force_plus_id", fw, a.mf_true, logk_true,
                                                                  n=50, sig_id=sid)[0]}
    sens["force_strokes_only_up_to_20pct"] = {
        n: expected_std(n, fw, a.mf_true, logk_true, n=50, s_idx=slice(0, int(np.sum(S_OBS <= 0.2 + 1e-9))))[0]
        for n in ("force_only", "force_plus_id")}
    for sf in (0.0025, 0.005, 0.01, 0.02):
        sens[f"force_noise={sf}"] = {n: expected_std(n, fw, a.mf_true, logk_true, n=50, sig_f=sf)[0]
                                     for n in ("force_only", "force_plus_id")}
    metrics["sensitivity_expected_std_mf"] = sens
    metrics["model_discrepancy"] = discrepancy_study(fw, res, logk_true)
    metrics["interpolation_check"] = dict(
        max_force_err_pct_leave_one_out=float(max(loo_f)), max_id_err_pts_leave_one_out=float(max(loo_i)),
        n_solver_runs=len(MF_NODES), mf_nodes=MF_NODES.tolist())
    metrics["friction_step_table"] = friction_steps(res)
    f30 = np.exp(logF[:, -1])
    hi = MF_NODES >= 0.5
    metrics["contact_artefact_force_jump_pct_at_30"] = dict(
        mf_from_to=[[float(x), float(y)] for x, y in zip(MF_NODES[hi][:-1], MF_NODES[hi][1:])],
        jump_pct=(100 * (f30[hi][1:] / f30[hi][:-1] - 1)).tolist())
    peak = int(np.argmax(f30[MF_NODES <= 0.55]))
    metrics["force_at_30_not_monotone"] = dict(
        peak_mf=float(MF_NODES[peak]), peak_force_norm=float(f30[peak]),
        force_norm_at_0p5=float(f30[np.argmin(np.abs(MF_NODES - 0.5))]),
        note="F*(30%) rises with m_f up to the peak and then falls before the contact-artefact band, so the "
             "force-only likelihood is flat or two-humped over about m_f 0.4-0.5")
    metrics["sweep"] = dict(mf=MF_NODES.tolist(), id_change_pct_at_30=idc.tolist(),
                            force_norm_at_30=np.exp(logF[:, -1]).tolist(),
                            runtime_s_total=float(sum(r["runtime_s"] for r in res)),
                            stopped_early=[bool(r["stopped_early"]) for r in res])

    # sweep figure
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for r in res[::2]:
        x = r["curves"]["reduction"]
        axs[0].plot(x, r["curves"]["force_norm"][:len(x)], lw=0.9, label=f"m_f={r['case']['mf']}")
    axs[0].set_xlabel("height reduction")
    axs[0].set_ylabel("normalised force F*")
    axs[0].legend(fontsize=5, ncol=2)
    axs[1].plot(MF_NODES, idc, "o-", ms=3)
    axs[1].set_xlabel("friction factor m_f")
    axs[1].set_ylabel("inner-diameter change at 30% (%)")
    fig.tight_layout()
    pth = fdir / "sweep.png"
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    figs.append(str(pth))

    write_result(out / "identifiability.json", task="G4", title="Friction identifiability: force vs force + ID",
                 command=cmd, data=dict(dataset="solver sweep (standard ring 6:3:2, m=0.15)", n_runs=len(MF_NODES),
                                        splits={}), metrics=metrics, figures=figs,
                 notes="Synthetic data from a hidden truth on a verified solver: a self-consistency study, "
                       "not validation against experiments. High-m_f force curves contain steps from nodes "
                       "landing on the die (contact artefact); see docs/identifiability.md.")
    for n in SCENARIOS:
        print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics["posterior"][n].items()},
              metrics["expected_std_mf_over_100_noise_draws"][n])
    print("step table:", metrics["friction_step_table"])


if __name__ == "__main__":
    main()
