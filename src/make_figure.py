"""Final figures from results/summary/overview.json (written by overview.py).

    python src/make_figure.py
        figures/main_qini.png      pooled Qini with 95% bootstrap intervals, five methods, eight tables
        figures/new_vs_others.png  estimate-as-a-column (all chunks) minus the default pipeline / minus the weak learner

All methods of a table are compared on the same folds (those that every listed method has); a
panel says so when that is fewer than five.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OV = json.loads((ROOT / "results" / "summary" / "overview.json").read_text())

TABLES = [
    ("x5_sub50k", "X5, 50K rows"),
    ("x5_full", "X5, full (200K)"),
    ("megafon_sub50k", "Megafon, 50K rows"),
    ("megafon_full", "Megafon, full (600K)"),
    ("hill1_full", "Hillstrom, women's (43K)"),
    ("hill2_full", "Hillstrom, men's (43K)"),
    ("lenta_sub50k", "Lenta, 50K rows"),
    ("lenta_full", "Lenta, full (687K)"),
]
GREEN, BLUE, ORANGE, YELLOW = "#1baf7a", "#2a78d6", "#eb6834", "#eda100"
# (method key, label, colour, filled marker). Order and colours are fixed across panels. The two
# estimate-as-a-column variants share a hue: open marker = one chunk, filled = all chunks averaged.
METHODS = [
    ("weak", "Weak learner alone (gradient-boosting S-learner)", GREEN, True),
    ("cpfn_default", "CausalPFN, default large-table pipeline", BLUE, True),
    ("cpfn_placebo", "Same, weak learner's estimates replaced by random numbers", ORANGE, True),
    ("cpfn_K1_B10k_taufeat", "Estimate as an input column, one random 10K-row context", YELLOW, False),
    ("cpfn_A_B10k_taufeat", "Estimate as an input column, averaged over all 10K-row chunks", YELLOW, True),
]
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)


def main_figure():
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(METHODS)))[::-1]
    for ax, (tag, title) in zip(axes.ravel(), TABLES):
        style(ax)
        d = OV[tag]
        lo_all, hi_all = [], []
        for y, (m, _, colour, filled) in zip(ys, METHODS):
            if m not in d["methods"]:
                continue
            e = d["methods"][m]
            p, (lo, hi) = e["pooled"], e["pooled_ci95"]
            lo_all.append(lo)
            hi_all.append(hi)
            ax.plot([lo, hi], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot([p], [y], "o", markersize=8.5, markeredgewidth=2, markeredgecolor=colour, markerfacecolor=colour if filled else SURFACE, zorder=3)
            ax.annotate(f"{p:.3f}", (hi, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
        span = max(hi_all) - min(lo_all)
        ax.set_xlim(min(lo_all) - 0.06 * span, max(hi_all) + 0.24 * span)
        ax.set_ylim(-0.6, len(METHODS) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _, lab, _, _ in METHODS], color=INK, fontsize=9.5)
        k = len(d["folds"])
        ax.set_title(title + ("" if k == 5 else f"  ({k} of 5 folds)"), loc="left", fontsize=10.5, color=INK, pad=8)
    fig.suptitle("Pooled Qini (normalised) with 95% bootstrap intervals", x=0.01, ha="left", fontsize=13, color=INK, y=1.0)
    fig.text(
        0.01, -0.01,
        "Each panel has its own x-axis. Higher is better; a random ranking scores about 0.5. Open marker: one chunk; filled: all chunks averaged. "
        "Source: results/summary/overview.json.",
        color=MUTED, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.97), w_pad=1.2, h_pad=2.0)
    out = ROOT / "figures" / "main_qini.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("written", out)


def diff_figure():
    panels = [("cpfn_default", "... minus the default pipeline", BLUE), ("weak", "... minus the weak learner alone", GREEN)]
    lim = (-0.22, 0.27)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(TABLES)))[::-1]
    for ax, (other, title, colour) in zip(axes, panels):
        style(ax)
        ax.axvline(0, color=MUTED, linewidth=1)
        for y, (tag, _) in zip(ys, TABLES):
            e = OV[tag]["new_minus"][other]
            p, (lo, hi) = e["pooled"], e["ci95"]
            clear = lo > 0 or hi < 0
            ax.plot([max(lo, lim[0]), min(hi, lim[1])], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot([p], [y], "o", markersize=8.5, markeredgewidth=2, markeredgecolor=colour, markerfacecolor=colour if clear else SURFACE, zorder=3)
            label = f"{p:+.3f}"
            if lo < lim[0] or hi > lim[1]:
                label += f"   ({lo:+.2f} to {hi:+.2f}, clipped)"
                for edge, beyond in ((lim[0], lo < lim[0]), (lim[1], hi > lim[1])):
                    if beyond:
                        ax.plot([edge + (0.004 if edge < 0 else -0.004)], [y], "<" if edge < 0 else ">", color=colour, markersize=5)
            ax.annotate(label, (min(hi, lim[1]) if hi < lim[1] else p, y), xytext=(8, 0 if hi < lim[1] else 9), textcoords="offset points", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(*lim)
        ax.set_ylim(-0.6, len(TABLES) - 0.4)
        ax.set_yticks(ys)
        labels = [lab + ("" if len(OV[tag]["folds"]) == 5 else f"  ({len(OV[tag]['folds'])} of 5 folds)") for tag, lab in TABLES]
        ax.set_yticklabels(labels, color=INK, fontsize=9.5)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    fig.suptitle("Qini of estimate-as-an-input-column (all chunks averaged) ...   paired 95% bootstrap intervals", x=0.01, ha="left", fontsize=13, color=INK, y=1.0)
    fig.text(
        0.01, -0.02,
        "Right of zero: the variant with the estimate as an input column is better. Filled marker: the interval excludes zero. Source: results/summary/overview.json.",
        color=MUTED, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.96), w_pad=1.5)
    out = ROOT / "figures" / "new_vs_others.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    main_figure()
    diff_figure()
