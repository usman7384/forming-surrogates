"""Animate two real ring-compression runs (low and high friction) in 3D, with their cross-sections and curves.

    python tools/animate_ring.py              # solves both rings (about a minute each), then renders
    python tools/animate_ring.py --replot     # reuse the cached solver output

Writes docs/media/ring_compression.mp4 (needs imageio-ffmpeg) and docs/media/ring_compression.gif.
Everything drawn comes from ringsim's own output: node positions, strain per element, force and hole curves.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so worker processes find ringsim

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation, cm, colors, font_manager  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

OUT = Path("docs/media")
CACHE = OUT / "_ring_runs.npz"
CASES = {"low": 0.05, "high": 0.5}            # Tresca friction factor m_f
REDUCTION = 0.35
BLACK, GREY, LINE = "#111111", "#5c6873", "#d6dce2"
C_LOW, C_HIGH = "#2f6fab", "#c4572a"
STEEL = "#b9c4ce"
CMAP = plt.get_cmap("YlOrRd")
for fam in ("Segoe UI", "Helvetica Neue", "Arial"):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break


# ----------------------------------------------------------------------------- data
def _solve(mf):
    from ringsim.solver import Numerics, RingCase, RingSolver
    res = RingSolver(RingCase(od_id=2.0, h_id=2.0 / 3.0, mf=mf, reduction=REDUCTION, m=0.15),
                     Numerics(save_every=1)).run()
    c = res.curves
    return dict(p=res.fields["p"], strain=res.fields["strain"], t=res.fields["t"], step=res.fields["step"],
                red=np.asarray(c["reduction"]), force=np.asarray(c["force_norm"]),
                idc=np.asarray(c["id_change_pct"]), stopped=res.info["stopped_early"])


def load_runs(replot: bool):
    if replot and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["runs"].item()
    with ProcessPoolExecutor(2) as ex:
        out = dict(zip(CASES, ex.map(_solve, CASES.values())))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, runs=np.array(out, dtype=object))
    return out


# ----------------------------------------------------------------------------- geometry helpers
def boundary_loops(p0, t):
    """Indices of the inner and outer face nodes (fixed by the initial mesh), sorted bottom to top."""
    r0, z0 = p0
    inner = np.where(np.isclose(r0, r0.min()))[0]
    outer = np.where(np.isclose(r0, r0.max()))[0]
    top = np.where(np.isclose(z0, z0.max()))[0]
    return inner[np.argsort(z0[inner])], outer[np.argsort(z0[outer])], top[np.argsort(r0[top])]


def revolve(ax, r, z, th, color, alpha=1.0, ls=None):
    TH, R = np.meshgrid(th, r)
    _, Z = np.meshgrid(th, z)
    ax.plot_surface(R * np.cos(TH), R * np.sin(TH), Z, color=color, linewidth=0, antialiased=True, shade=True,
                    lightsource=ls, alpha=alpha, rstride=1, cstride=1)


def draw_ring3d(ax, run, i, smax, title, accent):
    ax.cla()
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    p = run["p"][i]
    r, z = p
    inner, outer, top = run["_loops"]
    ls = LightSource(azdeg=225, altdeg=35)
    th = np.radians(np.linspace(-15, 255, 64))      # three-quarter ring; the cut faces the camera
    # full height: mirror the computed upper half about the mid-plane
    zi = np.r_[-z[inner][::-1], z[inner]]
    ri = np.r_[r[inner][::-1], r[inner]]
    zo = np.r_[-z[outer][::-1], z[outer]]
    ro = np.r_[r[outer][::-1], r[outer]]
    revolve(ax, ro, zo, th, STEEL, ls=ls)
    revolve(ax, ri, zi, th, "#d5dde4", ls=ls)
    # top and bottom faces (in contact with the dies)
    rt = r[top]
    for sgn in (1, -1):
        ann_r = np.linspace(rt.min(), rt.max(), 6)
        revolve(ax, ann_r, np.full_like(ann_r, sgn * z.max()), th, "#cfd8e0", ls=ls)
    # cut faces: the real mesh, coloured by strain, mirrored for the lower half
    tri = run["t"].T
    fc = CMAP(colors.Normalize(0, smax)(run["strain"][i]))
    for ang in (th[0], th[-1]):
        ca, sa = np.cos(ang), np.sin(ang)
        polys = []
        for sgn in (1, -1):
            for e in tri:
                polys.append([(r[v] * ca, r[v] * sa, sgn * z[v]) for v in e])
        pc = Poly3DCollection(polys, facecolors=np.vstack([fc, fc]), edgecolors=(0, 0, 0, 0.18), linewidths=0.25)
        ax.add_collection3d(pc)
    # dies: flat discs that follow the top face
    zd = z.max() + 0.01
    for sgn in (1, -1):
        dr = np.linspace(0, 2.45, 12)
        dth = np.radians(np.linspace(-15, 255, 64))
        TH, R = np.meshgrid(dth, dr)
        ax.plot_surface(R * np.cos(TH), R * np.sin(TH), np.full_like(R, sgn * (zd + 0.06)), color="#6f7a84",
                        alpha=0.42, linewidth=0, shade=False)
    lim = 2.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.95, 0.95)
    ax.set_box_aspect((1, 1, 0.4), zoom=1.32)
    ax.view_init(elev=26, azim=-60)
    ax.text2D(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top", fontsize=15, fontweight="bold",
              color=accent)


def draw_section(ax, run, i, smax, accent):
    ax.cla()
    r, z = run["p"][i]
    tri = run["t"].T
    tp = ax.tripcolor(r, z, tri, facecolors=run["strain"][i], cmap=CMAP, vmin=0, vmax=smax, edgecolors="#00000026",
                      linewidth=0.3)
    zt = z.max()
    ax.add_patch(plt.Rectangle((0.6, zt), 2.1, 0.06, color="#8a949d"))
    ax.axhline(0, color=accent, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlim(0.7, 2.55)
    ax.set_ylim(-0.06, 0.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.text(0.62, 0.77, "computed slice", fontsize=10, color=GREY, va="top")
    return tp


def draw_curves(ax, runs, frac, key, ylabel, title):
    ax.cla()
    for name, col in (("low", C_LOW), ("high", C_HIGH)):
        run = runs[name]
        red, y = run["red"], run[key]
        n = min(len(red), len(y))
        red, y = red[:n], y[:n]
        k = max(1, int(round(frac * (n - 1))) + 1)
        ax.plot(red * 100, y, color=col, lw=1.0, alpha=0.18)
        ax.plot(red[:k] * 100, y[:k], color=col, lw=2.6)
        ax.plot(red[k - 1] * 100, y[k - 1], "o", color=col, ms=7)
    ax.set_xlim(0, REDUCTION * 100 + 1)
    ax.set_xlabel("height reduction (%)", fontsize=10.5, color=GREY)
    ax.set_ylabel(ylabel, fontsize=10.5, color=GREY)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=BLACK, loc="left")
    ax.tick_params(colors=GREY, labelsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.grid(color="#eef1f4", lw=0.8)
    if key == "idc":
        ax.axhline(0, color=GREY, lw=0.8)


# ----------------------------------------------------------------------------- animation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()
    runs = load_runs(a.replot)
    for run in runs.values():
        run["_loops"] = boundary_loops(run["p"][0], run["t"])
    nsave = min(len(runs["low"]["p"]), len(runs["high"]["p"]))
    smax = float(max(np.percentile(runs[k]["strain"][nsave - 1], 99) for k in runs))
    # timeline: hold, press, hold
    press = np.linspace(0, nsave - 1, 3 * nsave)
    timeline = np.r_[np.zeros(int(0.8 * a.fps)), press, np.full(int(2.2 * a.fps), nsave - 1)]

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    top = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.85, bottom=0.33, wspace=0.02)
    bot = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.25, 1, 1], left=0.04, right=0.975, top=0.36, bottom=0.08,
                           wspace=0.30)
    ax3l = fig.add_subplot(top[0, 0], projection="3d")
    ax3h = fig.add_subplot(top[0, 1], projection="3d")
    axsl, axsh = fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 1])
    axid, axf = fig.add_subplot(bot[0, 2]), fig.add_subplot(bot[0, 3])
    fig.text(0.04, 0.955, "Ring compression, simulated", fontsize=24, fontweight="bold", color=BLACK, va="center")
    fig.text(0.04, 0.915, "Standard 6:3:2 ring pressed to 35%. Real output of the finite-element solver ringsim: "
             "deforming mesh, strain, die force and hole size.", fontsize=12.5, color=GREY, va="center")
    cax = fig.add_axes([0.80, 0.905, 0.17, 0.016])
    cb = fig.colorbar(cm.ScalarMappable(colors.Normalize(0, smax), CMAP), cax=cax, orientation="horizontal")
    cb.set_label("equivalent plastic strain (-)", fontsize=10, color=GREY)
    cb.ax.tick_params(labelsize=9, colors=GREY)
    cb.outline.set_edgecolor(LINE)
    status = fig.text(0.04, 0.868, "", fontsize=13, color=BLACK, fontweight="bold", va="center")

    def frame(j):
        x = timeline[j]
        i = int(np.floor(x))
        frac_i = i / (nsave - 1)
        idl = runs["low"]["idc"][min(i, len(runs["low"]["idc"]) - 1)]
        idh = runs["high"]["idc"][min(i, len(runs["high"]["idc"]) - 1)]
        draw_ring3d(ax3l, runs["low"], i, smax, f"Low friction  m_f = {CASES['low']}   ·   hole {idl:+.1f}%", C_LOW)
        draw_ring3d(ax3h, runs["high"], i, smax, f"High friction  m_f = {CASES['high']}   ·   hole {idh:+.1f}%", C_HIGH)
        draw_section(axsl, runs["low"], i, smax, C_LOW)
        draw_section(axsh, runs["high"], i, smax, C_HIGH)
        draw_curves(axid, runs, frac_i, "idc", "inner-diameter change (%)", "The hole: grows or shrinks")
        draw_curves(axf, runs, frac_i, "force", "normalised die force (-)", "The force: rises with friction")
        status.set_text(f"Height reduction: {runs['low']['red'][min(i, len(runs['low']['red']) - 1)] * 100:4.1f}%")
        return []

    anim = animation.FuncAnimation(fig, frame, frames=len(timeline), blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        anim.save(OUT / "ring_compression.mp4", writer=animation.FFMpegWriter(fps=a.fps, bitrate=6000,
                                                                              extra_args=["-pix_fmt", "yuv420p"]))
        print("wrote", OUT / "ring_compression.mp4")
    except Exception as e:  # noqa: BLE001
        print("mp4 skipped:", e)
    # a lighter GIF for the README: half frame rate, smaller
    fig.set_size_inches(12, 6.75)
    fig.set_dpi(80)
    gif_tl = timeline[::2]
    anim2 = animation.FuncAnimation(fig, lambda j: frame(2 * j), frames=len(gif_tl), blit=False)
    anim2.save(OUT / "ring_compression.gif", writer=animation.PillowWriter(fps=a.fps // 2))
    print("wrote", OUT / "ring_compression.gif")
    frame(len(timeline) - 1)
    fig.set_size_inches(16, 9)
    fig.savefig(OUT / "ring_compression_final.png", dpi=120, facecolor="white")
    print("wrote", OUT / "ring_compression_final.png")


if __name__ == "__main__":
    main()
