"""Plot one run: deformed half-ring coloured by cumulated strain, and its force curve.

Run: python -m ringsim.viz data/runs/run_0000 --out run_0000.png
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def plot_run(stem: str, out: str | None = None, step: int = -1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    d = np.load(stem + ".npz")
    meta = json.load(open(stem + ".json"))
    p, t, strain = d["field_p"][step], d["field_t"], d["field_strain"][step]
    p0 = d["field_p"][0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    tri = Triangulation(p[0], p[1], t.T)
    tc = ax[0].tripcolor(tri, facecolors=strain, edgecolors="k", linewidth=0.2, cmap="viridis")
    ax[0].triplot(Triangulation(p0[0], p0[1], t.T), color="0.7", lw=0.3)
    ax[0].set_aspect("equal")
    ax[0].set(xlabel="r", ylabel="z (mid-plane at 0)")
    fig.colorbar(tc, ax=ax[0], label="cumulated equivalent strain")
    c, q = meta["case"], meta["qoi"]
    ax[0].set_title(f"OD/ID={c['od_id']:.2f}  H/ID={c['h_id']:.2f}  m={c['mf']:.2f}  "
                    f"rate sens.={c['m']:.2f}  ID change {q['id_change_pct']:+.1f}%", fontsize=9)
    ax[1].plot(100 * d["curve_reduction"], d["curve_force_norm"])
    ax[1].set(xlabel="Height reduction (%)", ylabel="Normalised die force")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out or stem + ".png", dpi=140)
    return out or stem + ".png"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="path without extension, e.g. data/runs/run_0000")
    ap.add_argument("--out")
    ap.add_argument("--step", type=int, default=-1)
    a = ap.parse_args()
    print(plot_run(a.stem, a.out, a.step))
