"""One 'hero' figure: posterior over (m_f, K) from force only vs force + one inner-diameter measurement.

    python -m surrogates.gp_bo.hero_figure --out results/identifiability

Uses the cached identifiability sweep (no new solver runs) and the same seeded realisation as
`surrogates.gp_bo.identifiability`; the posterior std values printed in the titles are checked against
results/identifiability/identifiability.json so the figure cannot drift from the results file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from surrogates.gp_bo.identifiability import MF_NODES, Forward, build_sweep, run_scenario  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/identifiability")
    a = ap.parse_args(argv)
    out = Path(a.out)
    res_file = json.load(open(out / "identifiability.json"))
    truth = res_file["metrics"]["truth"]
    mf_t, k_t = truth["mf"], truth["K"]
    _, logF, idc = build_sweep(str(out / "solver_cache"), workers=1)
    fw = Forward(MF_NODES, logF, idc, np.linspace(0, 1, 401))
    rng = np.random.default_rng(0)                       # same order of draws as identifiability.main
    posts = {n: run_scenario(n, fw, mf_t, float(np.log(k_t)), rng) for n in ("force_only", "force_plus_id")}
    for n, p in posts.items():
        ref = res_file["metrics"]["posterior"][n]["std_mf"]
        assert abs(p["std_mf"] - ref) < 1e-9, f"{n}: figure std {p['std_mf']} != results file {ref}"

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    titles = {"force_only": "Force curve only (1% noise)",
              "force_plus_id": "Force + one inner-diameter measurement (0.3 pt)"}
    for ax, (n, p) in zip(axs, posts.items()):
        K = np.exp(p["lk"])
        ax.contourf(fw.mf, K, p["post"].T, levels=30, cmap="viridis")
        ax.plot(mf_t, k_t, "r*", ms=12, label="hidden truth")
        ax.set_title(f"{titles[n]}\nposterior std of m_f = {p['std_mf']:.4f}", fontsize=10)
        ax.set_xlabel("friction factor m_f (-)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.8, 1.3)
        ax.legend(loc="upper right", fontsize=8)
    axs[0].set_ylabel("strength multiplier K (-, relative to nominal)")
    ins = axs[1].inset_axes([0.55, 0.08, 0.4, 0.4])
    p = posts["force_plus_id"]
    ins.contourf(fw.mf, np.exp(p["lk"]), p["post"].T, levels=30, cmap="viridis")
    ins.plot(mf_t, k_t, "r*", ms=8)
    ins.set_xlim(mf_t - 0.03, mf_t + 0.03)
    ins.set_ylim(k_t - 0.03, k_t + 0.03)
    ins.tick_params(labelsize=6, colors="w")  # labels sit on the dark main panel
    ins.set_title("zoom", fontsize=7, color="w")
    fig.suptitle("Standard 6:3:2 ring, hidden truth m_f = 0.25, K = 1.07 (synthetic data, same solver)", fontsize=9)
    fig.tight_layout()
    fp = out / "figures" / "hero_posterior.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(fp, {n: p["std_mf"] for n, p in posts.items()})


if __name__ == "__main__":
    main()
