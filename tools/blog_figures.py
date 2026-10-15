"""Draw the figures for the blog post (docs/blog/*.png) from the project's own data.

    python tools/blog_figures.py

Needs the committed results and artifacts, plus docs/media/_ring_runs.npz (run tools/animate_ring.py once).
White background, black headings, blue for low friction / healthy, orange for high friction / drifted.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "blog"
BLACK, GREY, LIGHT, LINE = "#111111", "#5c6873", "#f4f6f8", "#d6dce2"
BLUE, ORANGE, GREEN, RED, AMBER = "#2f6fab", "#c4572a", "#2e8b57", "#c0392b", "#d39e00"
for fam in ("Segoe UI", "Helvetica Neue", "Arial"):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams.update({"axes.edgecolor": LINE, "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.titlecolor": BLACK, "axes.titleweight": "bold", "axes.titlesize": 13})


def canvas(w, h, title=None, sub=None):
    fig = plt.figure(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor("white")
    if title:
        fig.text(0.03, 1 - 0.38 / h, title, fontsize=19, fontweight="bold", color=BLACK, va="center")
    if sub:
        fig.text(0.03, 1 - 0.78 / h, sub, fontsize=11.5, color=GREY, va="center")
    return fig


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(color="#eef1f4", lw=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def card(ax, x, y, w, h, fc="white", ec=LINE, lw=1.2, r=0.018):
    ax.add_patch(FancyBboxPatch((x + 0.003, y - 0.012), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc="#000000", ec="none", alpha=0.06, transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                transform=ax.transAxes))


def arrow(ax, p, q, color=GREY, lw=2, rad=0.0, ms=16):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", transform=ax.transAxes))


# ----------------------------------------------------------------------------- data
def part494():
    import pandas as pd
    d = pd.read_parquet(ROOT / "results/plant/drift/parts.parquet")
    n = pd.read_parquet(ROOT / "results/plant/nominal/parts.parquet")
    b = json.loads((ROOT / "results/plant/drift/bundles/part_00494/bundle.json").read_text())
    rr = json.loads((ROOT / "results/plant/drift/bundles/part_00494/rerun.json").read_text())
    return d, n, b, rr


def ring_runs():
    z = np.load(ROOT / "docs/media/_ring_runs.npz", allow_pickle=True)
    return z["runs"].item()


# ----------------------------------------------------------------------------- 1. the journey of part 494
def fig_journey():
    d, n, b, rr = part494()
    p = d[d.part_id == 494].iloc[0]
    fig = canvas(17, 4.9, "One forged ring, part 494",
                 f"Its hole shrank {abs(p.measured):.1f}%, inside a spec of {abs(p.target):.1f}% ± {p.tol:.0f}. "
                 "Here is what the pipeline saw, step by step.")
    ax = fig.add_axes([0, 0, 1, 0.84])
    ax.axis("off")
    steps = [
        ("THE PRESS", "Load-cell zero has\ncrept up since part 300", f"+{p.f_zero_kN:.1f} kN", ORANGE),
        ("TOUCH-OFF", "Force passes 5 kN before\nthe die reaches the ring", "too early", ORANGE),
        ("FEATURE", "Estimated press depth\n(true depth about 25%)", f"{p.reduction_est * 100:.0f}%", ORANGE),
        ("SURROGATE", "GP at the largest depth\nit accepts (35%)", f"{p.pred_mean:.1f}%\n± {p.pred_std:.1f}", GREY),
        ("VERDICT", "Rule alone says stop;\nguard: depth out of range", "CHECK", BLUE),
        ("RERUN", f"Solver {rr['solver_id_change_pct']:.1f}%, gauge "
                  f"{rr['measured_pct']:.1f}%:\nthe model was faithful", "input fault", BLUE),
    ]
    w, gap, x0, y0, h = 0.142, 0.022, 0.03, 0.36, 0.56
    for k, (lab, txt, big, col) in enumerate(steps):
        x = x0 + k * (w + gap)
        card(ax, x, y0, w, h)
        ax.add_patch(Rectangle((x, y0 + h - 0.012), w, 0.012, color=col, transform=ax.transAxes))
        ax.text(x + 0.012, y0 + h - 0.06, f"{k + 1}  {lab}", fontsize=10.5, color=GREY, fontweight="bold",
                transform=ax.transAxes, va="top", family="monospace")
        ax.text(x + 0.012, y0 + h - 0.17, big, fontsize=19 if "\n" not in big else 16, fontweight="bold", color=col,
                transform=ax.transAxes, va="top")
        ax.text(x + 0.012, y0 + 0.16, txt, fontsize=10.6, color=GREY, transform=ax.transAxes, va="top")
        if k < len(steps) - 1:
            arrow(ax, (x + w + 0.002, y0 + h / 2), (x + w + gap - 0.002, y0 + h / 2), color=LINE, lw=2.2)
    # brackets
    for (a, bb, col, txt) in ((0, 3, ORANGE, "a drifting sensor corrupted the input"),
                              (4, 6, BLUE, "the guard rail and the rerun caught it")):
        xa = x0 + a * (w + gap)
        xb = x0 + (bb - 1) * (w + gap) + w
        ax.plot([xa, xa, xb, xb], [0.25, 0.21, 0.21, 0.25], color=col, lw=1.8, transform=ax.transAxes)
        ax.text((xa + xb) / 2, 0.15, txt, ha="center", va="top", fontsize=12, color=col, fontweight="bold",
                transform=ax.transAxes)
    save(fig, "01_part494_journey.png")


# ----------------------------------------------------------------------------- 2. the ring test, before and after
def outline(p, p0):
    """Closed outline of the computed upper-half section, walked along the boundary nodes of the initial mesh."""
    r0, z0 = p0
    inner = np.where(np.isclose(r0, r0.min()))[0]
    outer = np.where(np.isclose(r0, r0.max()))[0]
    top = np.where(np.isclose(z0, z0.max()))[0]
    bot = np.where(np.isclose(z0, 0))[0]
    idx = np.r_[inner[np.argsort(z0[inner])], top[np.argsort(r0[top])], outer[np.argsort(-z0[outer])],
                bot[np.argsort(-r0[bot])]]
    return p[0][idx], p[1][idx]


def mid_node(p0, face):
    sel = np.where(np.isclose(p0[0], face))[0]
    return sel[np.argmin(np.abs(p0[1][sel]))]


def fig_ring_test():
    from matplotlib.patches import Polygon
    runs = ring_runs()
    fig = canvas(16, 4.9, "The ring compression test",
                 "Squeeze a flat ring between two dies to 35% of its height. Friction decides whether the hole opens "
                 "up or closes in. Real solver output.")
    specs = [("low", "Low friction: metal slides outward", BLUE), ("high", "High friction: metal flows into the hole", ORANGE)]
    for k, (key, label, col) in enumerate(specs):
        run = runs[key]
        p0, p1 = run["p"][0], run["p"][-1]
        ax = fig.add_axes([0.03 + k * 0.49, 0.12, 0.45, 0.58])
        r1, z1 = outline(p1, p0)
        ri0, ro0, h0 = p0[0].min(), p0[0].max(), p0[1].max()
        for sx in (1, -1):
            for sz in (1, -1):
                ax.add_patch(Polygon(np.c_[sx * r1, sz * z1], closed=True, fc=col, ec=col, lw=0.8, alpha=0.9))
            ax.add_patch(Rectangle((min(sx * ri0, sx * ro0), -h0), ro0 - ri0, 2 * h0, fc="none", ec=GREY, lw=1.1,
                                   ls=(0, (3, 2)), zorder=4))
        zt = p1[1].max()
        for sz in (1, -1):
            ax.add_patch(Rectangle((-2.8, zt if sz > 0 else -zt - 0.11), 5.6, 0.11, color="#8a949d"))
        # movement of each face at mid-height, from the original face to where the solver put it
        for face in (p0[0].min(), p0[0].max()):
            j = mid_node(p0, face)
            for sx in (1, -1):
                ax.annotate("", (sx * p1[0][j], 0.0), (sx * p0[0][j], 0.0),
                            arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.8, shrinkA=0, shrinkB=0))
        rin = p1[0][mid_node(p0, p0[0].min())]
        ax.annotate("", (-rin, -zt - 0.30), (rin, -zt - 0.30), arrowprops=dict(arrowstyle="<->", color=BLACK, lw=1.3))
        ax.plot([-rin, -rin], [-zt - 0.36, -0.02], color=BLACK, lw=0.6, ls=":")
        ax.plot([rin, rin], [-zt - 0.36, -0.02], color=BLACK, lw=0.6, ls=":")
        ax.text(0, -zt - 0.40, f"hole {run['idc'][-1]:+.0f}%", ha="center", va="top", fontsize=13, fontweight="bold",
                color=BLACK)
        ax.text(-2.75, 0.90, label, fontsize=13, fontweight="bold", color=col, va="top")
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-1.0, 0.92)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.5, 0.04, "A cut through both walls of the ring. Dashed: the ring before. Filled: after. "
             "Arrows: how far each face moved at mid-height. Grey bars: the dies.", ha="center", fontsize=10.5, color=GREY)
    save(fig, "02_ring_test.png")


# ----------------------------------------------------------------------------- 3. symmetry: ring -> half -> slice
def revolve(ax, r, z, th, color, alpha=1.0):
    TH, R = np.meshgrid(th, r)
    _, Z = np.meshgrid(th, z)
    ax.plot_surface(R * np.cos(TH), R * np.sin(TH), Z, color=color, linewidth=0, shade=True, alpha=alpha,
                    lightsource=LightSource(azdeg=225, altdeg=35))


def fig_symmetry():
    runs = ring_runs()
    p0, tri = runs["low"]["p"][0], runs["low"]["t"].T
    ri, ro, h = p0[0].min(), p0[0].max(), p0[1].max()
    fig = canvas(16, 5.8, "Why the solver only computes a small rectangle",
                 "A ring is the same all the way round, and its top half mirrors its bottom half. "
                 "So one slice of the wall carries all the physics.")
    titles = ["1  the whole ring", "2  cut it: every angle is the same", "3  the slice the solver meshes"]
    for k in range(3):
        if k < 2:
            ax = fig.add_axes([0.02 + k * 0.33, 0.06, 0.30, 0.72], projection="3d")
            th = np.radians(np.linspace(0, 360, 90)) if k == 0 else np.radians(np.linspace(-15, 255, 70))
            zz = np.array([-h, h])
            revolve(ax, np.full(2, ro), zz, th, "#b9c4ce")
            revolve(ax, np.full(2, ri), zz, th, "#d5dde4")
            for sz in (1, -1):
                revolve(ax, np.linspace(ri, ro, 4), np.full(4, sz * h), th, "#cfd8e0")
            if k == 1:
                for ang in (th[0], th[-1]):
                    xs = np.array([ri, ro, ro, ri, ri]) * np.cos(ang)
                    ys = np.array([ri, ro, ro, ri, ri]) * np.sin(ang)
                    ax.plot(xs, ys, [-h, -h, h, h, -h], color=BLACK, lw=1)
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                ang = th[0]
                quad = [(ri * np.cos(ang), ri * np.sin(ang), 0), (ro * np.cos(ang), ro * np.sin(ang), 0),
                        (ro * np.cos(ang), ro * np.sin(ang), h), (ri * np.cos(ang), ri * np.sin(ang), h)]
                ax.add_collection3d(Poly3DCollection([quad], facecolors=BLUE, alpha=0.8, edgecolors=BLACK))
            ax.set_xlim(-2.1, 2.1)
            ax.set_ylim(-2.1, 2.1)
            ax.set_zlim(-1, 1)
            ax.set_box_aspect((1, 1, 0.45), zoom=1.25)
            ax.view_init(24, -60)
            ax.set_axis_off()
            ax.set_facecolor((1, 1, 1, 0))
            ax.text2D(0.08, 0.95, titles[k], transform=ax.transAxes, fontsize=13, fontweight="bold", color=BLACK)
        else:
            ax = fig.add_axes([0.70, 0.12, 0.28, 0.58])
            ax.triplot(p0[0], p0[1], tri, color=BLUE, lw=0.6)
            ax.fill_between([ri, ro], 0, h, color=BLUE, alpha=0.10)
            ax.add_patch(Rectangle((ri - 0.1, h), ro - ri + 0.2, 0.06, color="#8a949d"))
            ax.text((ri + ro) / 2, h + 0.1, "die: friction $m_f$", ha="center", fontsize=10.5, color=BLACK)
            ax.plot([ri - 0.1, ro + 0.1], [0, 0], color=ORANGE, lw=2.4)
            ax.text((ri + ro) / 2, -0.1, "mid-plane: mirror", ha="center", va="top", fontsize=10.5, color=ORANGE)
            ax.text(ri - 0.05, h / 2, "hole", ha="right", va="center", fontsize=10.5, color=GREY)
            ax.text(ro + 0.05, h / 2, "outside", ha="left", va="center", fontsize=10.5, color=GREY)
            ax.set_xlim(ri - 0.45, ro + 0.55)
            ax.set_ylim(-0.3, h + 0.3)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.text(0.70, 0.745, titles[2], fontsize=13, fontweight="bold", color=BLACK, va="center")
            fig.text(0.70, 0.16, f"{p0.shape[1]} nodes, {len(tri)} triangles: about a minute per ring",
                     fontsize=10.5, color=GREY)
    for xa in (0.33, 0.66):
        fig.text(xa, 0.42, "→", fontsize=30, color=LINE, ha="center", va="center")
    save(fig, "03_symmetry.png")


# ----------------------------------------------------------------------------- 4. confounding: same force, different hole
def fig_confounding():
    cache = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "results/identifiability/solver_cache/*.json"))]
    std = [c for c in cache if abs(c["case"]["od_id"] - 2) < 1e-9 and abs(c["case"]["h_id"] - 2 / 3) < 1e-6
           and abs(c["case"]["m"] - 0.15) < 1e-9 and abs(c["case"]["K"] - 1) < 1e-9 and c["final_reduction"] >= 0.299]

    def pick(mf):
        return min(std, key=lambda c: abs(c["case"]["mf"] - mf))
    a, bcase = pick(0.25), pick(0.40)
    ra, fa, ia = (np.asarray(a["curves"][k]) for k in ("reduction", "force_norm", "id_change_pct"))
    rb, fb, ib = (np.asarray(bcase["curves"][k]) for k in ("reduction", "force_norm", "id_change_pct"))
    na, nb = min(len(ra), len(fa), len(ia)), min(len(rb), len(fb), len(ib))
    ra, fa, ia, rb, fb, ib = ra[:na], fa[:na], ia[:na], rb[:nb], fb[:nb], ib[:nb]
    Ka = 1.07
    Kb = Ka * np.interp(0.3, ra, fa) / np.interp(0.3, rb, fb)
    grid = np.linspace(0.02, 0.3, 60)
    FA, FB = Ka * np.interp(grid, ra, fa), Kb * np.interp(grid, rb, fb)
    dev = float(np.max(np.abs(FA / FB - 1)) * 100)
    fig = canvas(15, 6.0, "Why the press cannot feel friction",
                 "More friction with softer metal pushes back almost exactly like less friction with harder metal. "
                 "Only the hole tells them apart.")
    lab_a = f"friction {a['case']['mf']:.2f}, metal {(Ka - 1) * 100:+.0f}% stronger"
    lab_b = f"friction {bcase['case']['mf']:.2f}, metal {(Kb - 1) * 100:+.0f}% stronger"
    ax1 = fig.add_axes([0.05, 0.12, 0.42, 0.62])
    ax1.plot(grid * 100, FA, color=BLUE, lw=3.2, label=lab_a)
    ax1.plot(grid * 100, FB, color=ORANGE, lw=2.2, ls=(0, (5, 3)), label=lab_b)
    ax1.set_title("What the press measures: force", loc="left")
    ax1.set_xlabel("height reduction (%)")
    ax1.set_ylabel("die force (normalised)")
    ax1.text(0.03, 0.80, f"the two curves differ by at most {dev:.1f}%\n(the force sensor's noise is about 1%)",
             transform=ax1.transAxes, fontsize=11, color=BLACK)
    ax1.legend(frameon=False, fontsize=10.5, loc="lower right")
    tidy(ax1)
    ax2 = fig.add_axes([0.55, 0.12, 0.42, 0.62])
    ax2.plot(ra * 100, ia, color=BLUE, lw=3.2, label=lab_a)
    ax2.plot(rb * 100, ib, color=ORANGE, lw=2.2, ls=(0, (5, 3)), label=lab_b)
    ax2.axhline(0, color=GREY, lw=0.8)
    ax2.set_xlim(0, 32)
    ax2.set_title("What a gauge measures: the hole", loc="left")
    ax2.set_xlabel("height reduction (%)")
    ax2.set_ylabel("inner-diameter change (%)")
    ya, yb = ia[-1], float(np.interp(ra[-1], rb, ib))
    xe = ra[-1] * 100
    ax2.annotate("", (xe + 0.8, ya), (xe + 0.8, yb), arrowprops=dict(arrowstyle="<->", color=BLACK, lw=1.4,
                                                                    shrinkA=0, shrinkB=0))
    ax2.text(xe + 0.4, (ya + yb) / 2, f"{abs(ya - yb):.0f} points\napart", ha="right", va="center", fontsize=11.5,
             color=BLACK, fontweight="bold")
    ax2.legend(frameon=False, fontsize=10.5, loc="lower left")
    tidy(ax2)
    save(fig, "04_same_force_different_hole.png")


# ----------------------------------------------------------------------------- 5. the GP's view of part 494
def edge_prediction(sur, tk, mf_m, mf_s, reds):
    nodes, w = np.array([0.0, np.sqrt(3), -np.sqrt(3)]), np.array([2 / 3, 1 / 6, 1 / 6])
    out = []
    for red in reds:
        mfs = np.clip(mf_m + mf_s * nodes, 0, 1)
        X = np.column_stack([np.full(3, tk["od_id"]), np.full(3, tk["h_id"]), mfs, np.full(3, tk["m"]), np.full(3, red)])
        mu, sd = sur.predict(X)
        mean = float(w @ mu)
        var = float(w @ (np.asarray(sd) ** 2 + np.asarray(mu) ** 2) - mean ** 2)
        out.append((mean, np.sqrt(max(var, 0))))
    return np.array(out)


def fig_gp_view():
    from surrogates.gp_bo.api import load
    d, n, b, rr = part494()
    sur, _ = load(str(ROOT / "artifacts/gp"))
    tk, post = b["ticket"], b["friction_posterior_at_prediction"]
    reds = np.linspace(0.05, 0.45, 81)
    P = edge_prediction(sur, tk, post["mean"], post["std"], reds)
    at35 = edge_prediction(sur, tk, post["mean"], post["std"], [0.35])[0]
    assert abs(at35[0] - b["prediction"]["mean_id_change_pct"]) < 1e-6, (at35, b["prediction"])
    pn = n[n.part_id == 494].iloc[0]
    fig = canvas(15, 6.6, "What the model was actually asked",
                 "The GP's prediction for part 494's ring, as a function of press depth. "
                 "The model was right; the depth it was given was wrong.")
    ax = fig.add_axes([0.06, 0.12, 0.66, 0.64])
    ax.axvspan(10, 30, color=GREEN, alpha=0.07)
    ax.axvspan(35, 46, color=RED, alpha=0.06)
    ax.text(20, 0.96, "trained here", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, color=GREEN)
    ax.text(40.5, 0.96, "never seen", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, color=RED)
    lo, hi = tk["target_id_change_pct"] - tk["tol_id_pct"], tk["target_id_change_pct"] + tk["tol_id_pct"]
    ax.axhspan(lo, hi, color=BLUE, alpha=0.10)
    ax.text(5.6, hi - 0.3, "spec band", fontsize=10.5, color=BLUE, va="top")
    ax.fill_between(reds * 100, P[:, 0] - 2 * P[:, 1], P[:, 0] + 2 * P[:, 1], color=GREY, alpha=0.22, lw=0)
    ax.plot(reds * 100, P[:, 0], color=BLACK, lw=2.2, label="GP mean ± 2 sd")
    ax.plot(pn.reduction_est * 100, pn.pred_mean, "o", color=GREEN, ms=10, zorder=5)
    ax.annotate(f"without drift: depth {pn.reduction_est * 100:.0f}%,\npredicted {pn.pred_mean:.1f}% → go",
                (pn.reduction_est * 100, pn.pred_mean), (8, pn.pred_mean - 9), fontsize=11, color=GREEN,
                arrowprops=dict(arrowstyle="-|>", color=GREEN))
    ax.plot(pn.reduction_est * 100, b["measurement"]["id_change_pct"], "D", color=BLUE, ms=8, zorder=5)
    ax.annotate(f"gauge: {b['measurement']['id_change_pct']:.1f}%", (pn.reduction_est * 100, b["measurement"]["id_change_pct"]),
                (27, -12.5), fontsize=11, color=BLUE, arrowprops=dict(arrowstyle="-|>", color=BLUE))
    est = b["features"]["reduction_est"] * 100
    ax.axvline(est, color=ORANGE, lw=1.6, ls=(0, (4, 3)))
    ax.text(est + 0.4, -8, f"edge's\nestimate\n{est:.0f}%", color=ORANGE, fontsize=10.5, va="top")
    ax.plot(35, at35[0], "o", color=ORANGE, ms=10, zorder=5)
    ax.annotate(f"clipped to 35%:\n{at35[0]:.1f}% ± {at35[1]:.1f}", (35, at35[0]), (37.5, -24), fontsize=11,
                color=ORANGE, arrowprops=dict(arrowstyle="-|>", color=ORANGE))
    ax.set_xlim(5, 45)
    ax.set_xlabel("press depth given to the model (%)")
    ax.set_ylabel("predicted inner-diameter change (%)")
    tidy(ax)
    side = fig.add_axes([0.75, 0.12, 0.23, 0.64])
    side.axis("off")
    side.text(0, 0.98, "Reading the plot", fontsize=13, fontweight="bold", color=BLACK, va="top")
    side.text(0, 0.86, "The grey band is the model's own\nuncertainty. It is narrow inside\nthe green zone and widens "
              "where\nthe model has no data.\n\nThe drift moved part 494 from\nthe green dot to the orange one.\n"
              "Same ring, same model: only\nthe depth it was given changed.", fontsize=11, color=GREY, va="top",
              linespacing=1.45)
    save(fig, "05_what_the_model_was_asked.png")


# ----------------------------------------------------------------------------- 6. the model-machine contract
def fig_contract():
    fig = canvas(15, 5.4, "A small contract between two worlds",
                 "The model side and the machine side only meet at two functions. Either side can change "
                 "without the other noticing.")
    ax = fig.add_axes([0, 0, 1, 0.82])
    ax.axis("off")
    card(ax, 0.03, 0.12, 0.27, 0.72, fc=LIGHT)
    ax.text(0.165, 0.78, "MODEL SIDE", ha="center", fontsize=11, color=GREY, fontweight="bold", family="monospace")
    for k, (t, s, col) in enumerate((("Stub", "built first, for testing", GREY), ("Gaussian process", "the real model", BLUE))):
        card(ax, 0.06, 0.52 - k * 0.25, 0.21, 0.18)
        ax.text(0.08, 0.65 - k * 0.25, t, fontsize=13, fontweight="bold", color=col, va="center")
        ax.text(0.08, 0.57 - k * 0.25, s, fontsize=10.5, color=GREY, va="center")
    ax.text(0.165, 0.17, "swapping them: one line of config", ha="center", fontsize=10.5, color=BLUE)
    card(ax, 0.36, 0.20, 0.28, 0.56, fc="white", ec=BLUE, lw=2)
    ax.text(0.50, 0.70, "THE CONTRACT", ha="center", fontsize=11, color=BLUE, fontweight="bold", family="monospace")
    ax.text(0.38, 0.58, "predict(ring, depth)\n  → mean, std of hole change", fontsize=11.5, color=BLACK,
            family="monospace", va="top")
    ax.text(0.38, 0.40, "friction.update(measurement)\nfriction.mean_std()", fontsize=11.5, color=BLACK,
            family="monospace", va="top")
    card(ax, 0.70, 0.12, 0.27, 0.72, fc=LIGHT)
    ax.text(0.835, 0.78, "MACHINE SIDE", ha="center", fontsize=11, color=GREY, fontweight="bold", family="monospace")
    for k, t in enumerate(("Edge service", "Verdicts and monitoring", "Lab measurements")):
        card(ax, 0.73, 0.58 - k * 0.17, 0.21, 0.12)
        ax.text(0.745, 0.64 - k * 0.17, t, fontsize=12, color=BLACK, va="center")
    arrow(ax, (0.30, 0.48), (0.36, 0.48), color=BLUE, lw=2.4)
    arrow(ax, (0.70, 0.48), (0.64, 0.48), color=BLUE, lw=2.4)
    save(fig, "06_contract.png")


# ----------------------------------------------------------------------------- 7. shared versus independent error
def fig_shared_error():
    from serving.model_api import GridFrictionEstimator
    from surrogates.gp_bo.api import load
    sur, _ = load(str(ROOT / "artifacts/gp"))
    ring = {"od_id": 2.6, "h_id": 1.15, "m": 0.1}     # an out-of-range ring, where the GP is genuinely unsure
    true_mf, red, sigma = 0.45, 0.25, 0.3
    mu, sd = sur.predict(np.array([[ring["od_id"], ring["h_id"], true_mf, ring["m"], red]]))
    bias = float(sd[0])                                # a shared model error of one GP standard deviation
    rng = np.random.default_rng(3)
    N = 15
    labels = float(mu[0]) + bias + rng.normal(0, sigma, N)
    res = {}
    for corr in (False, True):
        est = GridFrictionEstimator(sur, n=801, correlated_model_error=corr)
        rows = []
        for y in labels:
            est.update(ring, red, float(y), sigma)
            rows.append(est.mean_std())
        res[corr] = np.array(rows)
    fig = canvas(15, 5.8, "When every measurement shares the same error",
                 "Synthetic lab measurements for one lot of an unfamiliar ring, all sharing the same model error. "
                 "Same data, combined two ways.")
    nn = np.arange(1, N + 1)
    for k, (corr, col, name) in enumerate(((False, ORANGE, "Treat the model error as fresh each time"),
                                           (True, BLUE, "Treat it as one error shared by the lot"))):
        ax = fig.add_axes([0.05 + k * 0.49, 0.12, 0.43, 0.62])
        m_, s_ = res[corr][:, 0], res[corr][:, 1]
        ax.fill_between(nn, m_ - 2 * s_, m_ + 2 * s_, color=col, alpha=0.18, lw=0)
        ax.plot(nn, m_, "o-", color=col, lw=2, ms=5)
        ax.axhline(true_mf, color=BLACK, lw=1.4, ls=(0, (4, 3)))
        ax.text(N + 0.4, true_mf + 0.003, "true\nfriction", va="bottom", fontsize=10.5, color=BLACK)
        ax.set_title(name, loc="left", color=col)
        ax.set_xlabel("number of lab measurements")
        ax.set_ylabel("friction estimate (mean ± 2 sd)")
        ax.set_xlim(0.5, N + 1.8)
        ax.set_ylim(0.26, 0.52)
        tidy(ax)
        verdict = ("the band keeps shrinking, and the truth falls\noutside it: confident and wrong" if not corr
                   else "the band stops shrinking at the size of the\nmodel error, and the truth stays inside")
        ax.text(0.22, 0.04, verdict, transform=ax.transAxes, fontsize=11, color=col, fontweight="bold")
    save(fig, "07_shared_error.png")


# ----------------------------------------------------------------------------- 8. touch-off: how the drift broke the feature
def fig_touchoff():
    import pandas as pd
    d, n, b, _ = part494()
    sd = pd.read_parquet(ROOT / "results/plant/drift/signals.parquet")
    sn = pd.read_parquet(ROOT / "results/plant/nominal/signals.parquet")
    a, c = sn[sn.part_id == 494].sort_values("t"), sd[sd.part_id == 494].sort_values("t")
    a = a.iloc[: int(np.argmax(a.pos_mm.values)) + 1]      # loading stroke only
    c = c.iloc[: int(np.argmax(c.pos_mm.values)) + 1]
    pn, pdr = n[n.part_id == 494].iloc[0], d[d.part_id == 494].iloc[0]
    fig = canvas(15, 6.0, "How a sensor offset becomes a wrong press depth",
                 "The edge decides the die has touched the ring when the force first passes 5 kN. "
                 "A drifted zero gets there before the die does.")
    ax = fig.add_axes([0.06, 0.12, 0.60, 0.62])
    ax.plot(a.pos_mm, a.force_kN, color=BLUE, lw=2.2, label="part 494 without drift")
    ax.plot(c.pos_mm, c.force_kN, color=ORANGE, lw=2.2, label=f"part 494 with the drift (zero at {pdr.f_zero_kN:.1f} kN)")
    ax.axhline(5, color=BLACK, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.9, 4.2, "5 kN touch-off threshold", va="top", fontsize=10.5, color=BLACK)
    ax.set_xlim(-0.1, 2.6)
    ax.set_ylim(-4, 46)
    ia = np.argmax(a.force_kN.values > 5)
    ic = np.argmax(c.force_kN.values > 5)
    xa, xc = a.pos_mm.values[ia], c.pos_mm.values[ic]
    for (pos, col, txt) in ((xa, BLUE, "real touch-off"), (xc, ORANGE, "detected touch-off")):
        ax.axvline(pos, color=col, lw=1.4, alpha=0.6)
        ax.text(pos + 0.03, 11, txt, color=col, fontsize=10.5, rotation=90, va="bottom")
    ax.annotate("", (xa, 30), (xc, 30), arrowprops=dict(arrowstyle="<->", color=BLACK, lw=1.3, shrinkA=0, shrinkB=0))
    ax.text((xa + xc) / 2, 31, f"depth counted from {xa - xc:.1f} mm too early", ha="center", va="bottom", fontsize=11.5,
            color=BLACK, fontweight="bold")
    ax.set_xlabel("ram position (mm)")
    ax.set_ylabel("force (kN), start of the stroke")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    tidy(ax)
    side = fig.add_axes([0.70, 0.12, 0.28, 0.62])
    side.axis("off")
    side.text(0, 0.98, "The consequence", fontsize=13, fontweight="bold", color=BLACK, va="top")
    side.text(0, 0.84, "The press depth is measured from\nthe touch-off point. Start counting\ntoo early "
              "and the ring looks\npressed much further than it was.", fontsize=11, color=GREY, va="top",
              linespacing=1.45)
    for k, (lab, v, col) in enumerate((("without drift", pn.reduction_est, BLUE), ("with drift", pdr.reduction_est, ORANGE))):
        side.text(0, 0.42 - k * 0.2, f"{v * 100:.0f}%", fontsize=26, fontweight="bold", color=col, va="top")
        side.text(0.40, 0.385 - k * 0.2, f"estimated depth\n{lab}", fontsize=10.5, color=GREY, va="top")
    save(fig, "08_touchoff.png")


# ----------------------------------------------------------------------------- 9. the verdict rule
def fig_verdict():
    _, _, b, _ = part494()
    tk, pr = b["ticket"], b["prediction"]
    lo_s, hi_s = tk["target_id_change_pct"] - tk["tol_id_pct"], tk["target_id_change_pct"] + tk["tol_id_pct"]
    fig = canvas(15, 5.4, "Decide with the uncertainty, not with the mean",
                 "Each prediction is an interval (mean ± 2 sd). Compare the whole interval with the spec band.")
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.64])
    ax.axvspan(lo_s, hi_s, color=GREEN, alpha=0.12)
    ax.text((lo_s + hi_s) / 2, 4.75, "spec band", ha="center", fontsize=11, color=GREEN, fontweight="bold")
    rows = [("go", "interval inside the band", tk["target_id_change_pct"] + 0.3, 0.5, GREEN),
            ("check", "interval crosses the edge", hi_s - 0.2, 0.8, AMBER),
            ("stop", "interval outside the band", tk["target_id_change_pct"] + 7.5, 0.6, RED),
            ("part 494", "rule says stop; guard says check", pr["mean_id_change_pct"], pr["std_id_change_pct"], ORANGE)]
    for k, (name, why, m, s, col) in enumerate(rows):
        y = 3.8 - k * 1.05
        ax.plot([m - 2 * s, m + 2 * s], [y, y], color=col, lw=7, solid_capstyle="round")
        ax.plot(m, y, "o", color="white", ms=7, mec=col, mew=2)
        ax.text(-33.5, y, name.upper(), fontsize=13, fontweight="bold", color=col, va="center")
        ax.text(m + 2 * s + 0.4, y, why, fontsize=11, color=GREY, va="center")
    ax.set_xlim(-34, -4)
    ax.set_ylim(0.3, 5.1)
    ax.set_yticks([])
    ax.set_xlabel("predicted inner-diameter change (%)")
    tidy(ax, grid=False)
    ax.spines["left"].set_visible(False)
    save(fig, "09_verdict_rule.png")


# ----------------------------------------------------------------------------- 10. the drift, part by part
def fig_drift():
    d, n, _, _ = part494()
    fig = canvas(15, 5.6, "The drift, part by part",
                 "The load cell's zero reading over the shift. The monitor flags it 19 parts after it starts; "
                 "from part 400 it also breaks touch-off detection.")
    ax = fig.add_axes([0.06, 0.12, 0.90, 0.62])
    ax.plot(n.part_id, n.f_zero_kN, color=BLUE, lw=1.2, alpha=0.8, label="shift without drift")
    ax.plot(d.part_id, d.f_zero_kN, color=ORANGE, lw=1.6, label="shift with drift")
    ax.axhline(5, color=BLACK, lw=1, ls=(0, (4, 3)))
    ax.text(5, 5.4, "5 kN touch-off threshold", fontsize=10.5, color=BLACK)
    for x, txt, col, yy, ha in ((300, "drift starts (part 300)", GREY, 13.3, "right"), (319, "alert (part 319)", RED, 11.9, "left"),
                                (400, "zero passes 5 kN (part 400)", BLACK, 13.3, "left"), (494, "part 494", ORANGE, 1.6, "left")):
        ax.axvline(x, color=col, lw=1.2, alpha=0.7)
        ax.text(x + (4 if ha == "left" else -4), yy, txt, fontsize=10.5, color=col, va="top", ha=ha)
    ax.set_xlim(0, 600)
    ax.set_ylim(-2, 14)
    ax.set_xlabel("part number")
    ax.set_ylabel("load-cell zero reading (kN)")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    tidy(ax)
    save(fig, "10_drift.png")


# ----------------------------------------------------------------------------- 11. which monitor sees which drift
def fig_coverage():
    e = json.loads((ROOT / "results/plant/experiments.json").read_text())["metrics"]
    det = e["detection"]
    rows = [("Load-cell zero offset", "drift_offset"), ("Load-cell gain error", "drift_gain"), ("True friction change", "drift_friction")]
    cols = [("Zero reading", "f_zero_kN"), ("Force relative to lot", "peak_rel_lot")]
    fig = canvas(14, 5.0, "Which monitor can see which drift",
                 "Twenty simulated shifts for each kind of drift. A monitor only helps for the failures it can see.")
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.70])
    ax.axis("off")
    x0, cw, rh = 0.25, 0.22, 0.22
    for j, (cn, _) in enumerate(cols + [("Friction estimate", None)]):
        ax.text(x0 + j * cw + cw / 2, 0.92, cn, ha="center", fontsize=12, fontweight="bold", color=BLACK)
    for i, (rn, key) in enumerate(rows):
        y = 0.68 - i * rh
        ax.text(0.0, y + rh / 2 - 0.02, rn, fontsize=12.5, color=BLACK, va="center")
        for j, (_, f) in enumerate(cols + [(None, None)]):
            x = x0 + j * cw
            if f is None:
                if key == "drift_friction":
                    fc, tc, txt = "#fff4e5", ORANGE, "tracks it, but nothing\nwatches it"
                else:
                    fc, tc, txt = LIGHT, GREY, "—"
            else:
                dd = det[key][f]
                if dd["detected_runs"] == dd["runs"]:
                    fc, tc = "#e8f5ee", GREEN
                    txt = f"caught 20/20\nafter {dd['delay_parts']['median']:.0f} parts"
                elif dd["detected_runs"] == 0:
                    fc, tc, txt = "#fdecea", RED, "missed 0/20"
                else:
                    fc, tc = "#fff8e1", AMBER
                    txt = f"caught {dd['detected_runs']}/{dd['runs']}\nafter {dd['delay_parts']['median']:.0f} parts"
            ax.add_patch(FancyBboxPatch((x + 0.008, y + 0.01), cw - 0.016, rh - 0.03, boxstyle="round,pad=0,rounding_size=0.02",
                                        fc=fc, ec="none", transform=ax.transAxes))
            ax.text(x + cw / 2, y + rh / 2 - 0.01, txt, ha="center", va="center", fontsize=11, color=tc,
                    fontweight="bold" if tc in (GREEN, RED, ORANGE) else None, transform=ax.transAxes)
    ax.text(0.0, 0.0, "The friction change is the one that matters most to the parts, and no sensor monitor sees it: "
            "the model explains it as normal variation.", fontsize=10.5, color=GREY)
    save(fig, "11_monitor_coverage.png")


# ----------------------------------------------------------------------------- 12. the rerun
def fig_rerun():
    _, _, b, rr = part494()
    tk = b["ticket"]
    lo_s, hi_s = tk["target_id_change_pct"] - tk["tol_id_pct"], tk["target_id_change_pct"] + tk["tol_id_pct"]
    fig = canvas(14, 4.8, "The rerun: was it the model or the input?",
                 "The repro bundle stores the inputs the edge inferred. The real solver, rerun with them, sides with "
                 "the model: the fault was the drifted input.")
    ax = fig.add_axes([0.05, 0.14, 0.90, 0.58])
    ax.axvspan(lo_s, hi_s, color=GREEN, alpha=0.12)
    ax.text((lo_s + hi_s) / 2, 3.55, "spec band", ha="center", fontsize=11, color=GREEN, fontweight="bold")
    vals = [("Solver, at the inferred inputs", rr["solver_id_change_pct"], GREY),
            ("Model prediction", rr["model_prediction_pct"], ORANGE),
            ("Lab gauge", rr["measured_pct"], BLUE)]
    for k, (name, v, col) in enumerate(vals):
        y = 2.8 - k * 0.95
        ax.plot([v], [y], "o", color=col, ms=15)
        ax.text(v + 0.7, y, f"{v:.1f}%", va="center", fontsize=12.5, fontweight="bold", color=col)
        ax.text(-41.5, y, name, va="center", fontsize=12, color=BLACK)
    xs, xm = rr["solver_id_change_pct"], rr["model_prediction_pct"]
    xl = min(xs, xm) - 1.1
    ax.plot([xs - 0.6, xl, xl, xm - 0.6], [2.8, 2.8, 1.85, 1.85], color=BLACK, lw=1)
    ax.text(xl - 0.3, 2.33, "these two agree:\nthe model is faithful", ha="right", va="center", fontsize=10.5, color=BLACK)
    ax.text(rr["measured_pct"], 0.62, "in spec: a good part", ha="center", fontsize=10.5, color=BLUE)
    ax.set_xlim(-42, -10)
    ax.set_ylim(0.4, 3.8)
    ax.set_yticks([])
    ax.set_xlabel("inner-diameter change (%)")
    tidy(ax, grid=False)
    ax.spines["left"].set_visible(False)
    save(fig, "12_rerun.png")


# ----------------------------------------------------------------------------- 13. the calibration chart
def standard_ring_cache():
    cache = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "results/identifiability/solver_cache/*.json"))]
    std = [c for c in cache if abs(c["case"]["od_id"] - 2) < 1e-9 and abs(c["case"]["h_id"] - 2 / 3) < 1e-6
           and abs(c["case"]["m"] - 0.15) < 1e-9 and abs(c["case"]["K"] - 1) < 1e-9 and c["final_reduction"] >= 0.299]
    out = []
    for c in std:
        r, i = (np.asarray(c["curves"][k]) for k in ("reduction", "id_change_pct"))
        n = min(len(r), len(i))
        out.append((c["case"]["mf"], r[:n], i[:n]))
    return sorted(out, key=lambda t: t[0])


def fig_calibration():
    runs = standard_ring_cache()
    fig = canvas(15, 6.4, "The chart forging engineers read friction from",
                 "Hole change against height reduction, one curve per friction value, for the standard 6:3:2 ring. "
                 "All curves from this project's solver.")
    ax = fig.add_axes([0.06, 0.12, 0.62, 0.64])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "fric", ["#1f4e8c", "#4f86c6", "#8a7fb8", "#d0754a", "#9c3412"])
    shown = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
    for target in shown:
        mf, r, i = min(runs, key=lambda t: abs(t[0] - target))
        ax.plot(r * 100, i, color=cmap(mf), lw=2.2)
        ax.text(r[-1] * 100 + 0.4, i[-1], f"$m_f$ = {mf:.2f}", va="center", fontsize=10, color=cmap(mf))
    # one measured ring: read its friction off the chart
    mf_t, r_t, i_t = min(runs, key=lambda t: abs(t[0] - 0.3))
    x, y = 25, float(np.interp(0.25, r_t, i_t))
    ax.plot(x, y, "o", color=BLACK, ms=9, zorder=5)
    ax.annotate("a measured ring: 25% down, hole " f"{y:+.1f}%\nlies on the $m_f$ = 0.30 curve", (x, y), (5.5, -15.5),
                fontsize=11, color=BLACK, arrowprops=dict(arrowstyle="-|>", color=BLACK))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlim(0, 34)
    ax.set_xlabel("height reduction (%)")
    ax.set_ylabel("inner-diameter change (%)")
    tidy(ax)
    side = fig.add_axes([0.73, 0.12, 0.25, 0.64])
    side.axis("off")
    side.text(0, 0.98, "How it is used", fontsize=13, fontweight="bold", color=BLACK, va="top")
    side.text(0, 0.86, "Press a ring, measure its height\nand its hole, and see which curve\nthe point lands on. "
              "That curve's\nfriction is the answer.\n\nThe chart is a lookup table built\nfrom simulations. A surrogate "
              "is\nthe same idea, with more inputs\nand an error bar.", fontsize=11, color=GREY, va="top",
              linespacing=1.45)
    save(fig, "13_calibration_chart.png")


# ----------------------------------------------------------------------------- 14. a GP in one dimension
def fig_gp_explainer():
    runs = standard_ring_cache()
    mf = np.array([t[0] for t in runs])
    y = np.array([float(np.interp(0.3, t[1], t[2])) for t in runs])
    train = np.array([np.argmin(np.abs(mf - v)) for v in (0.0, 0.1, 0.25, 0.4, 1.0)])
    test = np.setdiff1d(np.arange(len(mf)), train)
    xt, yt = mf[train], y[train]
    ym, ys = yt.mean(), yt.std()

    def k(a, b, ell, s2=1.0):              # Matérn 5/2, the project's kernel family
        d = np.abs(a[:, None] - b[None, :]) / ell
        return s2 * (1 + np.sqrt(5) * d + 5 * d ** 2 / 3) * np.exp(-np.sqrt(5) * d)

    def nll(ell, noise=1e-4):
        K = k(xt, xt, ell) + noise * np.eye(len(xt))
        L = np.linalg.cholesky(K)
        a = np.linalg.solve(L.T, np.linalg.solve(L, (yt - ym) / ys))
        return 0.5 * ((yt - ym) / ys) @ a + np.log(np.diag(L)).sum()
    ell = min(np.linspace(0.1, 2.0, 191), key=nll)
    xs = np.linspace(0, 1, 300)
    K = k(xt, xt, ell) + 1e-4 * np.eye(len(xt))
    Ks = k(xs, xt, ell)
    mu = ym + ys * Ks @ np.linalg.solve(K, (yt - ym) / ys)
    var = np.clip(1 - np.einsum("ij,ji->i", Ks, np.linalg.solve(K, Ks.T)), 0, None)
    sd = ys * np.sqrt(var)
    fig = canvas(15, 6.2, "What a Gaussian process does, in one dimension",
                 "Hole change at 30% reduction against friction, standard ring. The GP sees five solver runs and "
                 "is asked about every friction in between.")
    ax = fig.add_axes([0.06, 0.12, 0.62, 0.64])
    ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.16, lw=0, label="GP mean ± 2 sd")
    ax.plot(xs, mu, color=BLUE, lw=2.4)
    ax.plot(mf[test], y[test], "o", color=GREY, ms=4.5, alpha=0.75, label=f"{len(test)} solver runs it never saw")
    ax.plot(xt, yt, "o", color=ORANGE, ms=11, mec="white", mew=1.5, zorder=5, label="the 5 runs it learned from")
    gap = np.argmax(sd)
    ax.annotate("between data points the band\nopens: the model says how\nunsure it is", (xs[gap], mu[gap] - 2 * sd[gap]),
                (0.66, -2.5), fontsize=11, color=BLUE, arrowprops=dict(arrowstyle="-|>", color=BLUE))
    ax.set_xlabel("friction factor $m_f$")
    ax.set_ylabel("inner-diameter change at 30% (%)")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    tidy(ax)
    side = fig.add_axes([0.73, 0.12, 0.25, 0.64])
    side.axis("off")
    side.text(0, 0.98, "Two things to notice", fontsize=13, fontweight="bold", color=BLACK, va="top")
    side.text(0, 0.86, "The grey dots it never saw fall\ninside the band. That is what a\ncalibrated error bar means.\n\n"
              "The real surrogate does this in\nfive dimensions (ring shape,\nfriction, rate sensitivity, depth)\n"
              "learning from 360 solver runs\nand tested on 100 more.", fontsize=11, color=GREY, va="top",
              linespacing=1.45)
    save(fig, "14_gp_explainer.png")


# ----------------------------------------------------------------------------- 15. friction learned lot by lot
def fig_friction_learning():
    import pandas as pd
    n = pd.read_parquet(ROOT / "results/plant/nominal/parts.parquet").sort_values("part_id")
    lots = json.loads((ROOT / "results/plant/nominal/metrics.json").read_text())["metrics"]["friction_by_lot"]
    meas = pd.read_parquet(ROOT / "results/plant/nominal/measurements.parquet")
    fig = canvas(15, 5.6, "Friction, learned one lab measurement at a time",
                 "Every 50 parts a new lot arrives with unknown friction. One part in five is measured, a few parts late; "
                 "each measurement updates the estimate.")
    ax = fig.add_axes([0.06, 0.13, 0.90, 0.62])
    x = n.part_id.values
    ax.fill_between(x, n.mf_mean - 2 * n.mf_std, n.mf_mean + 2 * n.mf_std, color=BLUE, alpha=0.16, lw=0, step="post")
    ax.step(x, n.mf_mean, where="post", color=BLUE, lw=1.8, label="estimate (mean ± 2 sd)")
    for k, L in enumerate(lots):
        ax.plot([L["lot"] * 50, L["lot"] * 50 + 50], [L["mf_true"]] * 2, color=BLACK, lw=2,
                label="true friction of the lot" if k == 0 else None)
        ax.axvline(L["lot"] * 50, color=LINE, lw=0.8)
    ids = meas[meas["type"] == "label"].part_index.values if "type" in meas else meas.part_index.values
    ax.plot(ids, np.full(len(ids), -0.06), "|", color=ORANGE, ms=8, mew=1.4, label="a lab measurement arrives")
    ax.set_xlim(0, 600)
    ax.set_ylim(-0.1, 1.08)
    ax.set_xlabel("part number")
    ax.set_ylabel("friction factor $m_f$")
    ax.legend(frameon=False, fontsize=10.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    ax.annotate("each lot starts knowing nothing\n(prior: anywhere from 0 to 1)", (52, 0.5), (75, 0.95), fontsize=10.5,
                color=GREY, arrowprops=dict(arrowstyle="-|>", color=GREY))
    tidy(ax, grid=False)
    save(fig, "15_friction_learning.png")


# ----------------------------------------------------------------------------- 16. PSI, explained on the real alert
def fig_psi():
    import pandas as pd
    from serving.edge.monitoring import psi
    d = pd.read_parquet(ROOT / "results/plant/drift/parts.parquet").sort_values("part_id")
    dr = pd.read_parquet(ROOT / "results/plant/drift/drift.parquet")
    f = d.f_zero_kN.values
    ref, cur = f[:100], f[270:320]
    edges = np.quantile(ref, [0.2, 0.4, 0.6, 0.8])
    p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=5) / len(ref)
    q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=5) / len(cur)
    val = psi(ref, cur)
    fig = canvas(15, 5.6, "How the drift monitor noticed",
                 "Split the reference parts into five equal bins, then ask how the latest 50 parts fill those bins. "
                 "Shown at part 319, the first alert.")
    ax = fig.add_axes([0.05, 0.13, 0.42, 0.60])
    xb = np.arange(5)
    ax.bar(xb - 0.2, p * 100, 0.38, color=BLUE, label="reference: parts 0 to 99")
    ax.bar(xb + 0.2, q * 100, 0.38, color=ORANGE, label="window: parts 270 to 319")
    ax.set_xticks(xb, ["lowest\n20%", "", "middle", "", "highest\n20%"])
    ax.set_ylabel("share of parts in bin (%)")
    ax.set_xlabel("zero reading, binned by the reference")
    ax.set_title(f"PSI = Σ (window − ref) · ln(window / ref) = {val:.2f}", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    tidy(ax)
    ax2 = fig.add_axes([0.56, 0.13, 0.40, 0.60])
    ax2.plot(dr.part_index, dr.psi_f_zero_kN, "o-", color=ORANGE, ms=4, lw=1.6)
    ax2.axhline(0.25, color=BLACK, lw=1.1, ls=(0, (4, 3)))
    ax2.text(598, 0.22, "alert line, 0.25\n(the KS test must agree too)", ha="right", va="top", fontsize=10.5,
             color=BLACK)
    ax2.axvline(300, color=GREY, lw=1)
    ax2.text(296, 3.0, "drift\nstarts", ha="right", fontsize=10.5, color=GREY)
    ax2.plot([319], [val], "o", color=RED, ms=10, zorder=5)
    ax2.annotate("first alert, part 319", (319, val), (170, 0.9), fontsize=11, color=RED,
                 arrowprops=dict(arrowstyle="-|>", color=RED))
    ax2.set_yscale("log")
    ax2.set_ylim(0.03, 12)
    ax2.set_xlim(140, 600)
    ax2.set_title("PSI of the zero reading, every 10 parts", loc="left", fontsize=12)
    ax2.set_xlabel("part number")
    tidy(ax2)
    save(fig, "16_psi.png")


# ----------------------------------------------------------------------------- 17. anatomy of one decision
def fig_edge_anatomy():
    import pandas as pd
    d = pd.read_parquet(ROOT / "results/plant/drift/parts.parquet")
    med = d[["latency_ms", "transport_ms", "align_ms", "feature_ms", "predict_ms"]].median()
    fig = canvas(17, 6.6, "Anatomy of one decision on the edge",
                 f"From the press finishing a stroke to a verdict: a median of {med.latency_ms:.0f} ms, "
                 "on a 6-second press cycle. Times are this project's measured medians.")
    ax = fig.add_axes([0, 0, 1, 0.86])
    ax.axis("off")
    main = [("PRESS", "force 1 kHz, ram\nposition 250 Hz, over\nOPC-UA", None, GREY),
            ("DELIVERY", "OPC-UA delivers the\nstroke; a counter ties\nevents to strokes", f"{med.transport_ms:.1f} ms", GREY),
            ("FEATURES", "touch-off at 5 kN,\npress depth, peak\nforce, zero reading", f"{med.align_ms + med.feature_ms:.1f} ms", GREY),
            ("GUARD", "is the depth inside\nthe model's range?\nis the ring familiar?", "flags", RED),
            ("PREDICT", "GP at 3 friction\nvalues, averaged over\nthe lot's estimate", f"{med.predict_ms:.1f} ms", BLUE),
            ("VERDICT", "go, check or stop,\nfrom mean ± 2 sd\nagainst the spec", "go / check / stop", GREEN)]
    w, gap, x0, y0, h = 0.135, 0.026, 0.03, 0.50, 0.38
    for k, (lab, txt, chip, col) in enumerate(main):
        x = x0 + k * (w + gap)
        card(ax, x, y0, w, h)
        ax.add_patch(Rectangle((x, y0 + h - 0.014), w, 0.014, color=col, transform=ax.transAxes))
        ax.text(x + 0.012, y0 + h - 0.06, lab, fontsize=11, color=col, fontweight="bold", family="monospace",
                transform=ax.transAxes, va="top")
        ax.text(x + 0.012, y0 + h - 0.13, txt, fontsize=10.5, color=BLACK, transform=ax.transAxes, va="top",
                linespacing=1.35)
        if chip:
            ax.text(x + 0.012, y0 + 0.035, chip, fontsize=10.5, color=col, fontweight="bold", transform=ax.transAxes)
        if k < len(main) - 1:
            arrow(ax, (x + w + 0.003, y0 + h / 2), (x + w + gap - 0.003, y0 + h / 2), color=GREY, lw=2)
    side = [("MONITOR", "PSI and KS on the last\n50 parts, every 10 parts", 2, AMBER),
            ("LEARN", "a lab label arrives late:\nupdate the lot's friction", 4, BLUE),
            ("BUNDLE", "a miss writes signals,\nversions and a rerun file", 5, ORANGE)]
    for (lab, txt, k, col) in side:
        x = x0 + k * (w + gap)
        card(ax, x, 0.07, w, 0.25, fc=LIGHT)
        ax.text(x + 0.012, 0.27, lab, fontsize=11, color=col, fontweight="bold", family="monospace",
                transform=ax.transAxes, va="top")
        ax.text(x + 0.012, 0.20, txt, fontsize=10.5, color=BLACK, transform=ax.transAxes, va="top", linespacing=1.35)
        arrow(ax, (x + w / 2, y0 - 0.005), (x + w / 2, 0.325), color=LINE, lw=1.8, ms=13)
    ax.text(x0, 0.20, "Side paths that run\nafter, or alongside,\nthe verdict", fontsize=11, color=GREY, va="top",
            transform=ax.transAxes, linespacing=1.4)
    save(fig, "17_edge_anatomy.png")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for f in (fig_journey, fig_ring_test, fig_symmetry, fig_confounding, fig_gp_view, fig_contract, fig_shared_error,
              fig_touchoff, fig_verdict, fig_drift, fig_coverage, fig_rerun, fig_calibration, fig_gp_explainer,
              fig_friction_learning, fig_psi, fig_edge_anatomy):
        if not only or f.__name__ in only:
            f()
