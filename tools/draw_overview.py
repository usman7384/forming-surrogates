"""Draw the one-page overview diagram of the sandbox (docs/diagrams/sandbox_overview.png and .svg).

    python tools/draw_overview.py

White background, black headings, four softly tinted stage bands, one blue accent for the flow.
Numbers quoted here come from the results files named in README.md; keep them in step.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

BLACK, GREY, LINE = "#111111", "#5c6873", "#d6dce2"
ACCENT = "#2f6fab"
TINTS = {"physics": "#eef3f9", "data": "#f1f3f5", "models": "#eef7f1", "plant": "#fbf4ea"}
TINT_EDGE = {"physics": "#c7d7ea", "data": "#cfd5db", "models": "#c3e0cd", "plant": "#ecd9bb"}
OUT = Path("docs/diagrams")

# prefer a clean UI font when the machine has one; fall back to matplotlib's default
for fam in ("Segoe UI", "Helvetica Neue", "Arial"):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break


# ----------------------------------------------------------------------------- drawing helpers
def band(ax, x, w, label, key, y=4.7, h=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=TINTS[key], ec="none"))
    ax.text(x + 0.35, y + h - 0.38, label, fontsize=11.5, fontweight="bold", color=GREY, va="top",
            ha="left", family="monospace")


def card(ax, x, y, w, h, title, lines=(), title_size=14.5, body_size=10.8):
    ax.add_patch(FancyBboxPatch((x + 0.07, y - 0.07), w, h, boxstyle="round,pad=0,rounding_size=0.22",
                                fc="#000000", ec="none", alpha=0.07))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.22",
                                fc="white", ec=LINE, lw=1.1))
    ax.text(x + w / 2, y + 1.05, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=BLACK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + 0.55 - i * 0.36, ln, ha="center", va="top", fontsize=body_size, color=GREY)


def arrow(ax, p, q, rad=0.0, color=ACCENT, lw=2.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=22, lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3))


# ----------------------------------------------------------------------------- icons (drawn, not images)
def icon_ring(ax, cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx - 0.9 * s, cy + 0.45 * s), 1.8 * s, 0.14 * s, fc="#7f8a94", ec="none"))
    ax.add_patch(Rectangle((cx - 0.9 * s, cy - 0.59 * s), 1.8 * s, 0.14 * s, fc="#7f8a94", ec="none"))
    for sign in (-1, 1):
        ax.add_patch(Rectangle((cx + (0.25 if sign > 0 else -0.7) * s, cy - 0.45 * s), 0.45 * s, 0.9 * s,
                               fc=ACCENT, ec="none", alpha=0.85))
    ax.annotate("", (cx, cy + 0.08 * s), (cx, cy + 0.42 * s), arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.4))


def icon_mesh(ax, cx, cy, s=1.0):
    x0, x1, y0, y1 = cx - 0.9 * s, cx + 0.9 * s, cy - 0.45 * s, cy + 0.45 * s
    nx, ny = 7, 3
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        ax.plot([x, x], [y0, y1], color=ACCENT, lw=0.9, alpha=0.9)
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        ax.plot([x0, x1], [y, y], color=ACCENT, lw=0.9, alpha=0.9)
    for i in range(nx):
        for j in range(ny):
            ax.plot([x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx],
                    [y0 + (y1 - y0) * (j + 1) / ny, y0 + (y1 - y0) * j / ny], color=ACCENT, lw=0.6, alpha=0.7)


def icon_dots(ax, cx, cy, s=1.0):
    cols = [ACCENT] * 18 + ["#6fbf8f"] * 2 + ["#e0b45b"] * 2 + ["#e08a8a"] * 2
    k = 0
    for j in range(3):
        for i in range(8):
            ax.add_patch(Circle((cx - 0.8 * s + i * 0.23 * s, cy + 0.3 * s - j * 0.3 * s), 0.07 * s,
                                fc=cols[k], ec="none"))
            k += 1


def icon_posterior(ax, cx, cy, s=1.0):
    t = np.linspace(-1, 1, 60)
    ax.plot(cx - 0.9 * s + 0.9 * s * (t + 1), cy + 0.25 * s - 0.35 * s * t ** 2 * 0.6 - 0.1 * s * t,
            color="#9fb8d3", lw=9, solid_capstyle="round", alpha=0.9)
    ax.add_patch(Circle((cx + 0.35 * s, cy - 0.05 * s), 0.11 * s, fc=ACCENT, ec="none"))
    ax.text(cx - 0.75 * s, cy - 0.45 * s, "force", fontsize=8, color=GREY)
    ax.text(cx + 0.2 * s, cy - 0.45 * s, "+ hole", fontsize=8, color=ACCENT)


def icon_gp(ax, cx, cy, s=1.0):
    x = np.linspace(-0.9, 0.9, 80)
    y = 0.22 * np.sin(2.2 * x) + 0.05 * x
    band_w = 0.07 + 0.12 * (np.abs(x) > 0.5)
    ax.fill_between(cx + x * s, cy + (y - band_w) * s, cy + (y + band_w) * s, color=ACCENT, alpha=0.18, lw=0)
    ax.plot(cx + x * s, cy + y * s, color=ACCENT, lw=1.8)


def icon_graph(ax, cx, cy, s=1.0):
    pts = np.array([[-0.8, 0.3], [-0.3, 0.45], [0.2, 0.3], [0.75, 0.4], [-0.6, -0.3], [-0.05, -0.4], [0.5, -0.3]])
    edges = [(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (4, 5), (5, 6)]
    for a, b in edges:
        ax.plot([cx + pts[a, 0] * s, cx + pts[b, 0] * s], [cy + pts[a, 1] * s, cy + pts[b, 1] * s], color="#9fb8d3", lw=1.3)
    for p in pts:
        ax.add_patch(Circle((cx + p[0] * s, cy + p[1] * s), 0.085 * s, fc=ACCENT, ec="white", lw=0.8))


def icon_press(ax, cx, cy, s=1.0):
    icon_ring(ax, cx - 0.45 * s, cy, s=0.75)
    x = np.linspace(0, 1, 60)
    y = 0.3 * np.sqrt(x) + 0.03 * np.sin(18 * x)
    ax.plot(cx + 0.3 * s + x * 0.65 * s, cy - 0.35 * s + y * s, color=ACCENT, lw=1.6)
    ax.text(cx + 0.62 * s, cy - 0.52 * s, "force", fontsize=8, color=GREY, ha="center")


def icon_edge(ax, cx, cy, s=1.0):
    for i, (c, lab) in enumerate((("#6fbf8f", "go"), ("#e0b45b", "check"), ("#e08a8a", "stop"))):
        ax.add_patch(Circle((cx - 0.55 * s + i * 0.55 * s, cy + 0.18 * s), 0.14 * s, fc=c, ec="none"))
        ax.text(cx - 0.55 * s + i * 0.55 * s, cy - 0.2 * s, lab, fontsize=8, color=GREY, ha="center")


# ----------------------------------------------------------------------------- page
def main():
    fig = plt.figure(figsize=(26, 15), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 15)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # header
    ax.text(0.9, 14.05, "forming-surrogates", fontsize=32, fontweight="bold", color=BLACK, va="center")
    ax.text(0.9, 13.3, "From a verified metal-forming simulation to surrogate models to a simulated press, in one sandbox.",
            fontsize=14.5, color=GREY, va="center")
    for i, tag in enumerate(("verified, not validated", "everything simulated", "one published dataset", "every number traces to a file")):
        x = 25.1 - (3 - i) * 4.3
        ax.add_patch(FancyBboxPatch((x - 3.9, 13.72), 3.9, 0.62, boxstyle="round,pad=0,rounding_size=0.31",
                                    fc="white", ec=LINE, lw=1.1))
        ax.text(x - 1.95, 14.03, tag, fontsize=10.5, color=GREY, ha="center", va="center")

    # stage bands
    band(ax, 0.6, 8.3, "1  PHYSICS", "physics")
    band(ax, 9.15, 3.9, "2  DATA", "data")
    band(ax, 13.3, 5.0, "3  MODELS", "models")
    band(ax, 18.55, 6.85, "4  PLANT", "plant")

    # cards: physics
    cy, ch = 7.4, 3.9
    card(ax, 0.95, cy, 3.7, ch, "Ring compression test", ["squeeze a ring: the hole grows", "at low friction, shrinks at high"])
    icon_ring(ax, 2.8, cy + 2.45)
    card(ax, 4.95, cy, 3.7, ch, "FE solver  ·  ringsim", ["one axisymmetric slice, viscoplastic,", "Tresca friction; verified to 0.002%"])
    icon_mesh(ax, 6.8, cy + 2.45)
    # data
    card(ax, 9.5, cy, 3.2, ch, "Dataset", ["460 simulated rings", "train 360 · test 100 in 3 splits"])
    icon_dots(ax, 11.1, cy + 2.45)
    # models (three stacked)
    mx, mw, mh = 13.65, 4.3, 1.95
    for yy, title, sub, ic in (
            (9.85, "Identifiability", "exact Bayes over friction and strength", icon_posterior),
            (7.75, "GP surrogate", "calibrated uncertainty, ring design", icon_gp),
            (5.65, "Mesh GNN + baselines", "whole fields; MLP, POD+GP, trees", icon_graph)):
        ax.add_patch(FancyBboxPatch((mx + 0.07, yy - 0.07), mw, mh, boxstyle="round,pad=0,rounding_size=0.22",
                                    fc="#000000", ec="none", alpha=0.07))
        ax.add_patch(FancyBboxPatch((mx, yy), mw, mh, boxstyle="round,pad=0,rounding_size=0.22", fc="white", ec=LINE, lw=1.1))
        ic(ax, mx + 0.95, yy + mh / 2, s=0.62)
        ax.text(mx + 1.85, yy + mh - 0.55, title, fontsize=13.5, fontweight="bold", color=BLACK, va="top")
        ax.text(mx + 1.85, yy + mh - 1.1, sub, fontsize=10.2, color=GREY, va="top")
    # plant
    card(ax, 18.9, cy, 3.1, ch, "Simulated press", ["force 1 kHz, position, delayed", "lab labels, over OPC-UA"])
    icon_press(ax, 20.45, cy + 2.45)
    card(ax, 22.3, cy, 2.85, ch, "Edge service", ["go / check / stop, drift", "monitor, friction update"])
    icon_edge(ax, 23.72, cy + 2.45)

    # flow arrows (row centre)
    ym = cy + ch / 2
    arrow(ax, (4.65, ym), (4.95, ym))
    arrow(ax, (8.65, ym), (9.5, ym))
    for yy in (10.8, 8.7, 6.6):
        arrow(ax, (12.7, ym), (mx, yy))
    arrow(ax, (mx + mw, 8.7), (18.9, ym))
    arrow(ax, (22.0, ym), (22.3, ym))
    # solver -> identifiability (direct, grey, over the top)
    arrow(ax, (6.8, cy + ch), (mx + 1.9, 11.82), rad=-0.3, color="#9aa6b1", lw=1.6)
    ax.text(10.6, 12.78, "solver runs feed the Bayesian study directly", fontsize=10.5, color=GREY, ha="center")
    # feedback loop under the row
    yl = 5.3
    ax.plot([23.72, 23.72], [cy, yl], color=ACCENT, lw=2.2, ls=(0, (6, 3)))
    ax.plot([23.72, 6.8], [yl, yl], color=ACCENT, lw=2.2, ls=(0, (6, 3)))
    arrow(ax, (6.8, yl), (6.8, cy), color=ACCENT, lw=2.2)
    ax.text(15.3, yl - 0.4, "a mispredicted part becomes a solver case R&D can rerun", fontsize=12, color=ACCENT,
            ha="center", va="top", fontweight="bold")
    # published data in
    ax.add_patch(FancyBboxPatch((0.95, 5.15), 3.7, 1.55, boxstyle="round,pad=0,rounding_size=0.22", fc="white", ec=TINT_EDGE["physics"], lw=1.4))
    ax.text(2.8, 6.3, "Published ring tests", fontsize=12.5, fontweight="bold", color=BLACK, ha="center", va="top")
    ax.text(2.8, 5.8, "AA6068 · 3 lubricants · 3 reductions", fontsize=10, color=GREY, ha="center", va="top")
    arrow(ax, (4.65, 5.95), (5.6, 7.4), rad=-0.25, color="#9aa6b1", lw=1.6)
    ax.text(5.35, 6.0, "reality check", fontsize=10, color=GREY, ha="left", va="center")

    # results strip
    ax.plot([0.6, 25.4], [4.2, 4.2], color=LINE, lw=1.2)
    ax.text(0.9, 3.75, "What it found", fontsize=15, fontweight="bold", color=BLACK, va="center")
    cards = [
        ("0.10 → 0.0046", "uncertainty on the friction factor:\nforce only, then force + one hole measurement"),
        ("1.4 to 4.9 σ", "bias if the hole is measured at the\nbore minimum instead of mid-height"),
        ("0.09 pts", "GP error on held-out rings;\na plain MLP ensemble ties it"),
        ("1.6 to 3×", "a POD + GP field model beats\nthe mesh GNN in distribution"),
        ("19 parts", "to catch a load-cell drift (20/20 seeds);\na friction drift is not caught"),
    ]
    cw, gap = 4.6, 0.37
    for i, (big, small) in enumerate(cards):
        x0 = 0.9 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x0, 1.05), cw, 2.3, boxstyle="round,pad=0,rounding_size=0.22", fc="#f7f9fb", ec="none"))
        ax.plot([x0 + 0.35, x0 + 1.35], [2.95, 2.95], color=ACCENT, lw=3, solid_capstyle="round")
        ax.text(x0 + 0.35, 2.45, big, fontsize=22, fontweight="bold", color=BLACK, va="center")
        ax.text(x0 + 0.35, 1.65, small, fontsize=10.3, color=GREY, va="center")
    ax.text(25.4, 0.5, "92 tests · independent correctness reviews · a fresh clone reproduces the results · "
            "teaching-scale solver, simulated plant, single-seed GNN", fontsize=10.5, color=GREY, ha="right", va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "sandbox_overview.png", dpi=100, facecolor="white")
    fig.savefig(OUT / "sandbox_overview.svg", facecolor="white")
    print("wrote", OUT / "sandbox_overview.png", OUT / "sandbox_overview.svg")


if __name__ == "__main__":
    main()
