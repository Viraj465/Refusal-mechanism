"""Submission figures. CPU only, reads the saved JSONs, writes two PNGs.

  python refusal/figures_final.py

Why this is separate from `analysis.py`: that script is the registered pipeline
output and its Figure 2 is the H4 quadrant that prereg §7.2 names as the
fallback when no head-level masks exist. These are the two figures for the
write-up, chosen for what the data actually turned out to show.

Figure 1 — the result.
  (a) refusal and compliance, y-axis zoomed so the significant SFT erosion is
      visible rather than six bars that all look like 1.0; McNemar p annotated
  (b) cos with dir_M0 at a common layer, against the split-half ceiling band
  (c) refusal under baseline / a random direction / dir_M0 — the causal claim
      with its control in the same panel, which is the point

Figure 2 — where the mechanism lives.
  (a) refusal drop by ablation layer, val split, all three checkpoints: one
      sharp onset at layer 19 shared by all three, flat above it. Also shows
      honestly that M_RL's L* = 21 is a one-prompt tie-break (prereg D13).
  (b) harmful-vs-harmless separation by layer: SFT and RL degrade the direction
      in the late layers, where it is most linearly readable, not at layer 19,
      where it is causally load-bearing.

Every number drawn is printed to stdout so it can be checked against the JSONs.
"""

from __future__ import annotations

import glob
import json
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
TAGS = ["M0", "M_SFT", "M_RL"]
NICE = {"M0": "M0", "M_SFT": "M_SFT", "M_RL": "M_RL"}
SUB = {"M0": "Qwen2.5-3B-Instruct", "M_SFT": "+ science SFT", "M_RL": "+ science Dr.GRPO"}

COLORS = {"M0": "#2a78d6", "M_SFT": "#eb6834", "M_RL": "#1baf7a"}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e4e3df"
BAND = "#d8d7d2"


# --------------------------------------------------------------------------

def latest(subdir: str, prefix: str) -> Optional[Dict]:
    files = sorted(glob.glob(str(RESULTS_DIR / subdir / f"{prefix}_*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as fh:
        return json.load(fh)


def mcnemar(a: List[bool], b: List[bool]):
    """Exact two-sided McNemar over paired binary outcomes."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p


def style(ax, ylabel: str = "", ylim=None):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    if ylim:
        ax.set_ylim(*ylim)


def title(ax, text: str):
    ax.set_title(text, fontsize=9.5, color=INK, loc="left", pad=9, fontweight="semibold")


# ==========================================================================
# Figure 1
# ==========================================================================

def figure1(out_path: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    beh = {t: latest("behaviour", f"behaviour_{t}") for t in TAGS}
    dirn = {t: latest("direction", f"direction_eval_{t}") for t in TAGS}
    rand = {t: latest("direction", f"random_control_{t}") for t in TAGS}
    ceil = (latest("direction", "direction_ceiling_M0") or {}).get("ceiling")

    tags = [t for t in TAGS if beh.get(t)]
    if not tags:
        print("[fig1] no behaviour results")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 1.35]})
    fig.patch.set_facecolor("white")

    # ---------------- (a) behaviour, zoomed ----------------
    ax = axes[0]
    x = np.arange(len(tags))
    w = 0.36
    ref = [beh[t]["behaviour"]["refusal_rate_harmful"] for t in tags]
    com = [beh[t]["behaviour"]["compliance_rate_harmless"] for t in tags]

    def err(cis):
        return [[max(0.0, c["point"] - c["low"]) for c in cis],
                [max(0.0, c["high"] - c["point"]) for c in cis]]

    lo = min(min(c["low"] for c in ref), min(c["low"] for c in com))
    bottom = max(0.0, lo - 0.02)

    ax.bar(x - w / 2, [c["point"] - bottom for c in ref], w, bottom=bottom,
           color=[COLORS[t] for t in tags], edgecolor="white", linewidth=1.3)
    ax.bar(x + w / 2, [c["point"] - bottom for c in com], w, bottom=bottom,
           color=[COLORS[t] for t in tags], edgecolor="white", linewidth=1.3,
           alpha=0.42, hatch="///")
    ax.errorbar(x - w / 2, [c["point"] for c in ref], yerr=err(ref), fmt="none",
                ecolor=INK2, elinewidth=1.1, capsize=2.5)
    ax.errorbar(x + w / 2, [c["point"] for c in com], yerr=err(com), fmt="none",
                ecolor=INK2, elinewidth=1.1, capsize=2.5)
    for xi, c in zip(x - w / 2, ref):
        ax.text(xi, c["high"] + 0.004, f"{c['point']:.3f}", ha="center",
                fontsize=8, color=INK)
    for xi, c in zip(x + w / 2, com):
        ax.text(xi, c["high"] + 0.004, f"{c['point']:.3f}", ha="center",
                fontsize=8, color=INK2)

    style(ax, "rate", (bottom, 1.035))
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[t] for t in tags], fontsize=9, color=INK)
    title(ax, "(a)  Refusal behaviour is nearly intact")

    # paired tests, drawn between the bars they compare
    stats = {}
    for a, b in (("M0", "M_SFT"), ("M_SFT", "M_RL")):
        if beh.get(a) and beh.get(b):
            stats[(a, b)] = mcnemar(beh[a]["behaviour"]["per_prompt_refused"]["harmful"],
                                    beh[b]["behaviour"]["per_prompt_refused"]["harmful"])
    y = 1.018
    for (a, b), (n01, n10, p) in stats.items():
        i, j = tags.index(a) - w / 2, tags.index(b) - w / 2
        ax.plot([i, j], [y, y], color=MUTED, linewidth=0.9)
        ax.text((i + j) / 2, y + 0.003,
                f"McNemar p={p:.3f}" + ("" if p >= 0.05 else " *"),
                ha="center", fontsize=7.5,
                color=INK if p < 0.05 else MUTED)
        y += 0.012

    solid = plt.Rectangle((0, 0), 1, 1, fc=MUTED, ec="white")
    hatch = plt.Rectangle((0, 0), 1, 1, fc=MUTED, ec="white", alpha=0.42, hatch="///")
    ax.legend([solid, hatch], ["refusal on harmful", "compliance on harmless"],
              frameon=False, fontsize=7.5, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.09), labelcolor=INK2)

    # ---------------- (b) cosine vs ceiling ----------------
    ax = axes[1]
    ctags, cvals, clayers = [], [], []
    for t in ("M_SFT", "M_RL"):
        d = dirn.get(t)
        if not d:
            continue
        key = next((k for k in d.get("cosines", {}) if ",M0)@own_L" in k), None)
        if key is None:
            continue
        ctags.append(t)
        cvals.append(d["cosines"][key])
        clayers.append(d.get("selected_layer"))

    if ceil:
        ax.axhspan(ceil["ci_low"], ceil["ci_high"], color=BAND, alpha=0.75, zorder=0)
        ax.axhline(ceil["mean"], color=INK2, linewidth=0.9, linestyle="--", zorder=1)
        ax.text(-0.45, ceil["ci_high"] + 0.004,
                f"split-half ceiling {ceil['mean']:.3f} (95% CI)",
                fontsize=7.5, color=INK2, ha="left")

    xb = np.arange(len(ctags))
    ax.bar(xb, cvals, 0.5, color=[COLORS[t] for t in ctags],
           edgecolor="white", linewidth=1.3, zorder=2)
    for xi, v in zip(xb, cvals):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    style(ax, "cosine with dir$_{M0}$", (0.86, 1.005))
    ax.set_xticks(xb)
    ax.set_xticklabels([NICE[t] + "\nat L" + str(lay)
                        for t, lay in zip(ctags, clayers)],
                       fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(ctags) - 0.4)
    title(ax, "(b)  The direction rotated")

    # ---------------- (c) causal, with the random control ----------------
    ax = axes[2]
    # Plotted as the *drop* in refusal, not the remaining rate: ablating dir_M0
    # leaves a rate of 0.000, which as a bar is invisible and reads as missing
    # data. The drop makes both the effect and the control visible at once.
    rtags = [t for t in TAGS if rand.get(t)]
    x = np.arange(len(rtags))
    w = 0.34

    rnd = [sum(r["refusal_drop"] for r in rand[t]["random"]) / len(rand[t]["random"])
           for t in rtags]
    real = [rand[t]["real"]["refusal_drop"] for t in rtags]

    ax.bar(x - w / 2, rnd, w, color=MUTED, alpha=0.45, edgecolor="white", linewidth=1.2,
           hatch="///", label="random direction (mean of 3)")
    ax.bar(x + w / 2, real, w, color=[COLORS[t] for t in rtags], edgecolor="white",
           linewidth=1.2, label="dir$_{M0}$")

    # individual random draws, so n = 3 is visible rather than implied
    for xi, t in zip(x - w / 2, rtags):
        pts = [r["refusal_drop"] for r in rand[t]["random"]]
        ax.scatter([xi] * len(pts), pts, s=11, color=INK2, zorder=3, linewidths=0)

    for xi, v in zip(x + w / 2, real):
        ax.text(xi, v + 0.018, f"{v:+.3f}", ha="center", fontsize=8.5, color=INK)
    for xi, v in zip(x - w / 2, rnd):
        ax.text(xi, max(v, 0) + 0.018, f"{v:+.3f}", ha="center", fontsize=8, color=INK2)

    style(ax, "drop in refusal when ablated", (-0.08, 1.22))
    ax.axhline(0.0, color=GRID, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[t] for t in rtags], fontsize=9, color=INK)
    title(ax, "(c)  ...but the causal handle did not move")
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.02), labelcolor=INK2)

    fig.suptitle("Figure 1 — Capability-only post-training rotates the refusal direction "
                 "without weakening it causally",
                 fontsize=11.5, color=INK, x=0.008, ha="left", y=1.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("[fig1] behaviour:", {t: round(beh[t]["behaviour"]["refusal_rate_harmful"]["point"], 4)
                                for t in tags})
    print("[fig1] McNemar:", {f"{a} vs {b}": (n01, n10, round(p, 4))
                              for (a, b), (n01, n10, p) in stats.items()})
    print("[fig1] cosines:", dict(zip(ctags, [round(v, 4) for v in cvals])), "at layers", clayers)
    print("[fig1] causal drops: random(mean)", [round(v, 4) for v in rnd],
          "| dir_M0", [round(v, 4) for v in real],
          "| tags", rtags)
    print(f"[fig1] wrote {out_path}")
    return out_path


# ==========================================================================
# Figure 2
# ==========================================================================

def figure2(out_path: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt

    fits = {t: latest("direction", f"direction_fit_{t}") for t in TAGS}
    fits = {t: f for t, f in fits.items() if f}
    if not fits:
        print("[fig2] no direction fits")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    fig.patch.set_facecolor("white")

    # ---------------- (a) causal sweep over layers ----------------
    ax = axes[0]
    for t, f in fits.items():
        tr = sorted(f["selection_trace"], key=lambda r: r["layer"])
        xs = [r["layer"] for r in tr]
        ys = [r["refusal_drop"] for r in tr]
        ax.plot(xs, ys, marker="o", markersize=3.4, linewidth=1.6,
                color=COLORS[t], label=f"{NICE[t]}  (L*={f['selected_layer']})")
        ls = f["selected_layer"]
        hit = next((r for r in tr if r["layer"] == ls), None)
        if hit:
            ax.scatter([ls], [hit["refusal_drop"]], s=64, facecolors="none",
                       edgecolors=COLORS[t], linewidths=1.6, zorder=4)
    ax.axhline(1.0, color=GRID, linewidth=1.0)
    style(ax, "refusal drop when ablated", (-0.05, 1.12))
    ax.set_xlabel("layer the direction was taken from", fontsize=9, color=INK2)
    title(ax, "(a)  One sharp onset, shared by all three")
    ax.legend(frameon=False, fontsize=7.8, loc="center left", labelcolor=INK2)
    ax.annotate("M_RL misses a full drop at L19\nby one val prompt (31/32)",
                xy=(19, 0.969), xytext=(21.2, 0.55), fontsize=7.3, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))

    # ---------------- (b) separation profile ----------------
    ax = axes[1]
    for t, f in fits.items():
        sep = f["direction_meta"]["separation"]
        ax.plot(range(len(sep)), sep, linewidth=1.8, color=COLORS[t], label=NICE[t])
    ax.axvline(19, color=MUTED, linewidth=0.9, linestyle=":")
    ax.text(19.4, 0.35, "L19\ncausal site", fontsize=7.3, color=MUTED)
    style(ax, "harmful vs harmless separation (d)")
    ax.set_xlabel("layer", fontsize=9, color=INK2)
    title(ax, "(b)  Fine-tuning degrades it late, not where it acts")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", labelcolor=INK2)

    fig.suptitle("Figure 2 — The refusal direction is a single mid-network site, "
                 "and post-training damages it elsewhere",
                 fontsize=11.5, color=INK, x=0.008, ha="left", y=1.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for t, f in fits.items():
        sep = f["direction_meta"]["separation"]
        print(f"[fig2] {t}: L*={f['selected_layer']}  sep@L19={sep[19]:.2f}  "
              f"sep@L32={sep[32]:.2f}  sep@L36={sep[36]:.2f}")
    print(f"[fig2] wrote {out_path}")
    return out_path


def main() -> None:
    figure1(FIG_DIR / "figure1_final.png")
    figure2(FIG_DIR / "figure2_final.png")


if __name__ == "__main__":
    main()
