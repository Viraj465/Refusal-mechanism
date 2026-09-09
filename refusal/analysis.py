"""
Stage 7 — analysis and the two submission figures (prereg.md §5, §6).

  Figure 1  Behaviour and the refusal direction, per checkpoint.
            (a) refusal on harmful / compliance on harmless, Wilson 95% CI
            (b) cos(dir_M0, dir_X) against the bootstrapped direction ceiling
            (c) transfer ratio against the frozen 0.5 / 0.8 thresholds

  Figure 2  Head-level retention of M0's refusal circuit, with the permutation
            null band and the split-half ceiling drawn on it, plus the IoU
            matrix. If the Stage-5 pilot gate failed and no masks exist, this
            falls back to the H4 dissociation table — which prereg §7.2
            registers in advance as an acceptable substitute, not a gap.

Also writes `analysis_summary.md`: every registered readout with its threshold
verdict, and the H4 cross-tab under both operationalisations of "mechanism".

  python analysis.py             # from real results in refusal/results/
  python analysis.py --demo      # synthetic numbers, to check layout early

Two figures maximum — that is the submission constraint, not a default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
TAGS = ["M0", "M_SFT", "M_RL"]
LABELS = {"M0": "M0\n(Qwen2.5-3B-Instruct)", "M_SFT": "M_SFT\n(+science SFT)", "M_RL": "M_RL\n(+science Dr.GRPO)"}

# Categorical slots 1-3 of the validated palette. Colour follows the entity:
# a checkpoint keeps its hue in every panel of both figures.
COLORS = {"M0": "#2a78d6", "M_SFT": "#eb6834", "M_RL": "#1baf7a"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"
NULL_BAND = "#d8d7d2"

# Frozen thresholds (prereg §5.2, §5.3), drawn on the figures as reference lines.
TRANSFER_PRESERVED, TRANSFER_MOVED = 0.80, 0.50
CEILING_FRACTION = 0.80


# ==========================================================================
# Results loading
# ==========================================================================

def latest(subdir: str, prefix: str) -> Optional[Dict]:
    """Most recent timestamped result matching a prefix. Results are never
    overwritten, so 'latest' is always well defined."""
    d = RESULTS_DIR / subdir
    if not d.is_dir():
        return None
    files = sorted(d.glob(f"{prefix}_*.json"))
    if not files:
        return None
    with files[-1].open(encoding="utf-8") as fh:
        return json.load(fh)


def find_cosine(payload: Dict, other: str) -> Optional[float]:
    """cos(this model's direction, `other`'s direction), each at its own L*."""
    cos = payload.get("cosines", {})
    for key, val in cos.items():
        if f",{other}@L" in key:      # both at their own selected layers
            return val
    for key, val in cos.items():
        if f",{other})" in key:        # fallback: both at this model's layer
            return val
    return None


def gather(demo: bool = False) -> Dict:
    if demo:
        return _demo_results()

    out: Dict = {"behaviour": {}, "direction": {}, "ceiling": {}, "controls": None}
    for tag in TAGS:
        b = latest("behaviour", f"behaviour_{tag}")
        if b:
            out["behaviour"][tag] = b
        d = latest("direction", f"direction_eval_{tag}")
        if d:
            out["direction"][tag] = d
        c = latest("direction", f"direction_ceiling_{tag}")
        if c:
            out["ceiling"][tag] = c
    out["controls"] = latest("controls", "controls_stats")
    return out


def _demo_results() -> Dict:
    """Synthetic numbers with the real schema, so layout can be checked before
    any GPU time is spent. NOT data. Every figure built this way is watermarked."""
    def ci(p):
        return {"point": p, "low": max(0, p - 0.06), "high": min(1, p + 0.06), "n": 150}

    beh = {
        "M0": (0.97, 0.95), "M_SFT": (0.71, 0.93), "M_RL": (0.88, 0.94),
    }
    out = {"behaviour": {}, "direction": {}, "ceiling": {}, "controls": None, "_demo": True}
    for tag, (r, c) in beh.items():
        out["behaviour"][tag] = {
            "tag": tag,
            "behaviour": {"refusal_rate_harmful": ci(r), "compliance_rate_harmless": ci(c)},
            "science_nts": {"M0": 21.0, "M_SFT": 68.5, "M_RL": 71.2}[tag],
        }
    for tag, (cos, ratio) in {"M_SFT": (0.71, 0.44), "M_RL": (0.90, 0.83)}.items():
        out["direction"][tag] = {
            "tag": tag,
            "cosines": {f"cos({tag}@L14,M0@L14)": cos},
            "transfer": {"transfer_ratio": ratio, "donor": "M0",
                         "verdict": "causally_preserved" if ratio >= 0.8 else
                                    "causally_moved" if ratio < 0.5 else "ambiguous"},
        }
    out["ceiling"]["M0"] = {"ceiling": {"mean": 0.94, "ci_low": 0.91, "ci_high": 0.96, "layer": 14}}
    out["controls"] = {
        "circuit_sizes": {"M0": 248, "M_SFT": 189, "M_RL": 241},
        "iou_matrix": {"M0|M0": 1.0, "M_SFT|M_SFT": 1.0, "M_RL|M_RL": 1.0,
                       "M0|M_SFT": 0.52, "M_SFT|M0": 0.52,
                       "M0|M_RL": 0.68, "M_RL|M0": 0.68,
                       "M_SFT|M_RL": 0.57, "M_RL|M_SFT": 0.57},
        "permutation_null": {"M0|M_SFT": {"p2_5": 0.28, "p97_5": 0.35, "mean": 0.31},
                             "M0|M_RL": {"p2_5": 0.33, "p97_5": 0.41, "mean": 0.37}},
        "split_half_ceiling": {"M0": 0.81, "M_SFT": 0.78, "M_RL": 0.80},
        "seed_stability": {"M0": {"jaccard": 0.88}, "M_SFT": {"jaccard": 0.85},
                           "M_RL": {"jaccard": 0.86}},
        "verdicts": {"retention": {
            "M_SFT": {"iou_vs_M0": 0.52, "null_p97_5": 0.35, "verdict": "not_preserved"},
            "M_RL": {"iou_vs_M0": 0.68, "null_p97_5": 0.41, "verdict": "preserved"}}},
    }
    return out


# ==========================================================================
# Shared styling
# ==========================================================================

def _style(ax, ylabel: str = "", ylim: Optional[Tuple[float, float]] = None):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8.5)
    if ylim:
        ax.set_ylim(*ylim)


def _label_bars(ax, bars, values, fmt="{:.2f}", dy=0.018, tops=None):
    """Direct value labels. Required, not decorative: the aqua slot sits below
    3:1 contrast on a light surface, and the palette's relief rule discharges
    that with visible labels.

    `tops` places the label above something taller than the bar (an error-bar
    cap), so the two never collide.
    """
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val is None:
            continue
        y = bar.get_height() if tops is None else max(bar.get_height(), tops[i])
        ax.text(
            bar.get_x() + bar.get_width() / 2, y + dy,
            fmt.format(val), ha="center", va="bottom",
            fontsize=8, color=INK, fontweight="medium",
        )


def _watermark(fig, results: Dict):
    if results.get("_demo"):
        fig.text(
            0.5, 0.5, "SYNTHETIC — LAYOUT ONLY", fontsize=34, color="#e34948",
            alpha=0.16, ha="center", va="center", rotation=24, fontweight="bold", zorder=10,
        )


# ==========================================================================
# Figure 1 — behaviour + direction
# ==========================================================================

def figure1(results: Dict, out_path: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    beh, dirn, ceil = results["behaviour"], results["direction"], results["ceiling"]
    if not beh:
        print("[analysis] no behaviour results — skipping Figure 1")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    fig.patch.set_facecolor("white")

    # ---- (a) behaviour ----
    ax = axes[0]
    tags = [t for t in TAGS if t in beh]
    x = np.arange(len(tags))
    w = 0.36

    def series(key):
        vals, errs = [], [[], []]
        for t in tags:
            c = beh[t]["behaviour"][key]
            vals.append(c["point"])
            errs[0].append(c["point"] - c["low"])
            errs[1].append(c["high"] - c["point"])
        return vals, errs

    refusal, r_err = series("refusal_rate_harmful")
    comply, c_err = series("compliance_rate_harmless")

    b1 = ax.bar(x - w / 2, refusal, w, color=[COLORS[t] for t in tags],
                edgecolor="white", linewidth=1.4)
    b2 = ax.bar(x + w / 2, comply, w, color=[COLORS[t] for t in tags],
                edgecolor="white", linewidth=1.4, alpha=0.45, hatch="///")
    ax.errorbar(x - w / 2, refusal, yerr=r_err, fmt="none", ecolor=INK_SECONDARY,
                elinewidth=1.1, capsize=2.5)
    ax.errorbar(x + w / 2, comply, yerr=c_err, fmt="none", ecolor=INK_SECONDARY,
                elinewidth=1.1, capsize=2.5)
    _label_bars(ax, b1, refusal, tops=[beh[t]["behaviour"]["refusal_rate_harmful"]["high"] for t in tags])
    _label_bars(ax, b2, comply, tops=[beh[t]["behaviour"]["compliance_rate_harmless"]["high"] for t in tags])

    _style(ax, "rate", (0, 1.18))
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t].split("\n")[0] for t in tags], fontsize=8.5, color=INK)
    ax.set_title("(a)  Behaviour on the test split", fontsize=9.5, color=INK,
                 loc="left", pad=10, fontweight="semibold")

    solid = plt.Rectangle((0, 0), 1, 1, fc=INK_MUTED, ec="white")
    hatched = plt.Rectangle((0, 0), 1, 1, fc=INK_MUTED, ec="white", alpha=0.45, hatch="///")
    ax.legend([solid, hatched], ["refusal on harmful", "compliance on harmless"],
              frameon=False, fontsize=7.5, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.02), labelcolor=INK_SECONDARY)

    # ---- (b) direction cosine vs ceiling ----
    ax = axes[1]
    dtags = [t for t in TAGS if t in dirn]
    cosines = [find_cosine(dirn[t], "M0") for t in dtags]
    bars = ax.bar(np.arange(len(dtags)), [c or 0 for c in cosines], 0.5,
                  color=[COLORS[t] for t in dtags], edgecolor="white", linewidth=1.4)
    _label_bars(ax, bars, cosines)

    m0c = ceil.get("M0", {}).get("ceiling")
    if m0c:
        ax.axhspan(m0c["ci_low"], m0c["ci_high"], color=NULL_BAND, zorder=0)
        ax.axhline(m0c["mean"], color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
        ax.text(-0.55, m0c["ci_high"] + 0.03,
                "split-half ceiling (95% CI)", fontsize=7, color=INK_SECONDARY, ha="left")

    _style(ax, "cosine with dir$_{M0}$", (0, 1.28))
    ax.set_xlim(-0.62, len(dtags) - 0.38)
    ax.set_xticks(np.arange(len(dtags)))
    ax.set_xticklabels([t for t in dtags], fontsize=8.5, color=INK)
    ax.set_title("(b)  Direction similarity to M0", fontsize=9.5, color=INK,
                 loc="left", pad=10, fontweight="semibold")

    # ---- (c) transfer ratio ----
    ax = axes[2]
    ratios = [dirn[t].get("transfer", {}).get("transfer_ratio") for t in dtags]
    bars = ax.bar(np.arange(len(dtags)), [r or 0 for r in ratios], 0.5,
                  color=[COLORS[t] for t in dtags], edgecolor="white", linewidth=1.4)
    _label_bars(ax, bars, ratios)

    ax.axhline(TRANSFER_PRESERVED, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.axhline(TRANSFER_MOVED, color=INK_MUTED, linewidth=1.0, linestyle=":")
    ax.text(-0.55, TRANSFER_PRESERVED + 0.03, "causally preserved  ≥0.8",
            fontsize=7, color=INK_SECONDARY, ha="left")
    ax.text(-0.55, TRANSFER_MOVED + 0.03, "causally moved  <0.5",
            fontsize=7, color=INK_SECONDARY, ha="left")

    _style(ax, "transfer ratio", (0, 1.28))
    ax.set_xlim(-0.62, len(dtags) - 0.38)
    ax.set_xticks(np.arange(len(dtags)))
    ax.set_xticklabels([t for t in dtags], fontsize=8.5, color=INK)
    ax.set_title("(c)  Ablating dir$_{M0}$ inside each model", fontsize=9.5, color=INK,
                 loc="left", pad=10, fontweight="semibold")

    fig.suptitle(
        "Figure 1 — Refusal behaviour and the refusal direction after capability-only post-training",
        fontsize=10.5, color=INK, x=0.007, ha="left", y=1.0, fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _watermark(fig, results)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[analysis] wrote {out_path}")
    return out_path


# ==========================================================================
# Figure 2 — head-level retention (or the H4 fallback)
# ==========================================================================

def figure2(results: Dict, out_path: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    controls = results.get("controls")
    if not controls or not controls.get("iou_matrix"):
        print("[analysis] no head-level masks — Figure 2 is the H4 dissociation quadrant "
              "(prereg §9 D6 scope decision)")
        return figure2_fallback(results, out_path)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")

    # ---- (a) retention with null band and ceiling ----
    ax = axes[0]
    ret = controls.get("verdicts", {}).get("retention", {})
    tags = [t for t in ("M_SFT", "M_RL") if t in ret]
    ious = [ret[t]["iou_vs_M0"] for t in tags]

    x = np.arange(len(tags))
    bars = ax.bar(x, ious, 0.36, color=[COLORS[t] for t in tags],
                  edgecolor="white", linewidth=1.4, zorder=3)
    _label_bars(ax, bars, ious)

    # permutation null band: everything at or below chance for these sizes
    nulls = controls.get("permutation_null", {})
    lo = min((n["p2_5"] for n in nulls.values()), default=None)
    hi = max((n["p97_5"] for n in nulls.values()), default=None)
    if lo is not None:
        ax.axhspan(lo, hi, color=NULL_BAND, zorder=0)
        ax.text(len(tags) - 0.45, (lo + hi) / 2, "permutation null (95%)", fontsize=7,
                color=INK_SECONDARY, ha="right", va="center")

    ceiling = controls.get("split_half_ceiling", {}).get("M0")
    if ceiling:
        ax.axhline(ceiling, color=INK_MUTED, linewidth=1.1, linestyle="--", zorder=2)
        ax.text(-0.62, ceiling + 0.02, "split-half ceiling",
                fontsize=7, color=INK_SECONDARY, ha="left")
        ax.axhline(CEILING_FRACTION * ceiling, color=INK_MUTED, linewidth=1.0,
                   linestyle=":", zorder=2)
        ax.text(-0.62, CEILING_FRACTION * ceiling + 0.02,
                "0.8 × ceiling  (retention threshold)", fontsize=7,
                color=INK_SECONDARY, ha="left")

    _style(ax, "IoU with M0's refusal circuit", (0, 1.12))
    ax.set_xlim(-0.68, len(tags) - 0.32)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=8.5, color=INK)
    ax.set_title("(a)  Retention of M0's head set", fontsize=9.5, color=INK,
                 loc="left", pad=10, fontweight="semibold")

    # ---- (b) IoU matrix ----
    ax = axes[1]
    mat_tags = [t for t in TAGS if any(k.startswith(f"{t}|") for k in controls["iou_matrix"])]
    n = len(mat_tags)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(mat_tags):
        for j, b in enumerate(mat_tags):
            v = controls["iou_matrix"].get(f"{a}|{b}", controls["iou_matrix"].get(f"{b}|{a}"))
            if v is not None:
                mat[i, j] = v

    # Sequential = one hue, light -> dark. Never a rainbow.
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", "#2a78d6", "#123a6b"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if mat[i, j] > 0.55 else INK, fontweight="medium")
    ax.set_xticks(range(n)); ax.set_xticklabels(mat_tags, fontsize=8.5, color=INK)
    ax.set_yticks(range(n)); ax.set_yticklabels(mat_tags, fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_title("(b)  Pairwise circuit IoU", fontsize=9.5, color=INK,
                 loc="left", pad=10, fontweight="semibold")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_SECONDARY, labelsize=7.5, length=0)

    sizes = controls.get("circuit_sizes", {})
    if sizes:
        fig.text(0.007, -0.03,
                 "Circuit sizes (of 576 heads): " +
                 ",  ".join(f"{k} {v}" for k, v in sizes.items()),
                 fontsize=7.5, color=INK_SECONDARY, ha="left")

    fig.suptitle(
        "Figure 2 — Head-level retention against the permutation null and the split-half ceiling",
        fontsize=10.5, color=INK, x=0.007, ha="left", y=1.0, fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _watermark(fig, results)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[analysis] wrote {out_path}")
    return out_path


def figure2_fallback(results: Dict, out_path: Path) -> Optional[Path]:
    """Figure 2 when no head-level masks exist: the prereg §4 cross-tab itself.

    Drawn as the registered 2x2 (plus an explicit 'ambiguous' column, because
    §5.2 forbids rounding an ambiguous transfer ratio to either side) with each
    checkpoint placed in its cell. A quadrant is a figure; a rendered table is
    not, and the submission asks for a graph per key experiment.
    """
    import matplotlib.pyplot as plt

    rows = h4_crosstab(results)
    if not rows:
        print("[analysis] not enough results for the H4 fallback either — skipping Figure 2")
        return None

    cols = ["preserved", "ambiguous", "moved"]        # mechanism
    rws = ["preserved", "lost"]                        # behaviour
    cell_name = {
        ("preserved", "preserved"): "re-consolidation",
        ("preserved", "moved"): "functional\nreplacement",
        ("lost", "preserved"): "latent\nmechanism",
        ("lost", "moved"): "erosion",
        ("preserved", "ambiguous"): "",
        ("lost", "ambiguous"): "",
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.patch.set_facecolor("white")

    for j, mech in enumerate(cols):
        for i, beh in enumerate(rws):
            highlight = (beh, mech) == ("preserved", "moved")
            ax.add_patch(plt.Rectangle(
                (j, len(rws) - 1 - i), 1, 1,
                facecolor="#fbf3ef" if highlight else "#fafaf9",
                edgecolor=GRID, linewidth=1.0, zorder=0,
            ))
            label = cell_name[(beh, mech)]
            if label:
                ax.text(
                    j + 0.5, len(rws) - 1 - i + 0.80, label,
                    ha="center", va="top", fontsize=8.5,
                    color=INK if highlight else INK_MUTED,
                    fontweight="semibold" if highlight else "normal",
                )

    # Place each checkpoint in its cell.
    placed: Dict[tuple, int] = {}
    for r in rows:
        tag = r["checkpoint"]
        beh = r["behaviour"].split()[0]
        mech = r["mechanism (direction)"].split()[0]
        if beh not in rws or mech not in cols:
            continue
        j, i = cols.index(mech), rws.index(beh)
        k = placed.get((i, j), 0)
        placed[(i, j)] = k + 1
        y = len(rws) - 1 - i + 0.42 - 0.22 * k
        ax.plot(j + 0.5, y, "o", markersize=9, color=COLORS.get(tag, INK),
                markeredgecolor="white", markeredgewidth=1.6, zorder=3)
        ax.text(j + 0.5, y - 0.14, tag, ha="center", va="top", fontsize=9,
                color=COLORS.get(tag, INK), fontweight="semibold", zorder=3)

    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(rws))
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels([f"mechanism\n{c}" for c in cols], fontsize=8.5, color=INK)
    ax.set_yticks([len(rws) - 1 - i + 0.5 for i in range(len(rws))])
    ax.set_yticklabels([f"behaviour\n{b}" for b in rws], fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)

    detail = ";  ".join(
        f"{r['checkpoint']}: refusal {r['behaviour'].split('(')[-1].rstrip(')')}, "
        f"transfer {r['mechanism (direction)'].split('ratio ')[-1].rstrip(')')}"
        for r in rows if "(" in r["behaviour"]
    )
    fig.suptitle(
        "Figure 2 — H4 dissociation: refusal behaviour against the refusal direction",
        fontsize=10.5, color=INK, x=0.007, ha="left", y=0.99, fontweight="semibold",
    )
    fig.text(0.007, 0.055, detail, fontsize=7.5, color=INK_SECONDARY, ha="left")
    fig.text(
        0.007, 0.005,
        "Mechanism from the transfer ratio (ablating dir$_{M0}$ inside each model): "
        "≥0.8 preserved, <0.5 moved, between = ambiguous, not rounded. "
        "Head-level operationalisation not run — scope decision, prereg §9 D6.",
        fontsize=7.5, color=INK_SECONDARY, ha="left",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    _watermark(fig, results)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[analysis] wrote {out_path}")
    return out_path


# ==========================================================================
# H4 cross-tab + summary
# ==========================================================================

def h4_crosstab(results: Dict) -> List[Dict]:
    """Behaviour x mechanism, per checkpoint, under both operationalisations.

    Behaviour is 'preserved' when the checkpoint's refusal-rate Wilson CI
    overlaps M0's. Mechanism (direction) uses the frozen transfer-ratio bands;
    mechanism (head) uses controls.py's retention verdict.
    """
    beh, dirn = results["behaviour"], results["direction"]
    if "M0" not in beh:
        return []
    m0 = beh["M0"]["behaviour"]["refusal_rate_harmful"]
    rows = []
    ret = (results.get("controls") or {}).get("verdicts", {}).get("retention", {})

    for tag in TAGS:
        if tag not in beh or tag == "M0":
            continue
        c = beh[tag]["behaviour"]["refusal_rate_harmful"]
        overlaps = not (c["high"] < m0["low"] or c["low"] > m0["high"])
        behaviour = "preserved" if overlaps else "lost"

        ratio = (dirn.get(tag, {}).get("transfer") or {}).get("transfer_ratio")
        if ratio is None:
            mech_dir = "not measured"
        elif ratio >= TRANSFER_PRESERVED:
            mech_dir = "preserved"
        elif ratio < TRANSFER_MOVED:
            mech_dir = "moved"
        else:
            mech_dir = "ambiguous"

        # prereg §4 registers the cross-tab under BOTH operationalisations, and
        # D5 makes H4 a co-headline — so the head-level cell is computed too,
        # not just carried as a column.
        head_verdict = ret.get(tag, {}).get("verdict", "not measured")
        mech_head = {
            "preserved": "preserved",
            "not_preserved": "moved",
        }.get(head_verdict, "not measured")

        rows.append({
            "checkpoint": tag,
            "behaviour": f"{behaviour}  ({c['point']:.2f} vs M0 {m0['point']:.2f})",
            "mechanism (direction)": (
                f"{mech_dir}" + (f"  (ratio {ratio:.2f})" if ratio is not None else "")
            ),
            "mechanism (head)": head_verdict,
            # "H4 cell" stays direction-derived: the direction is the spine, and
            # it is the only operationalisation that survives a pilot-gate failure.
            "H4 cell": _h4_cell(behaviour, mech_dir),
            "H4 cell (head)": _h4_cell(behaviour, mech_head),
        })
    return rows


def _h4_cell(behaviour: str, mechanism: str) -> str:
    """The prereg §4 cross-tab, applied to either operationalisation."""
    return {
        ("preserved", "preserved"): "re-consolidation",
        ("preserved", "moved"): "FUNCTIONAL REPLACEMENT",
        ("lost", "preserved"): "latent mechanism",
        ("lost", "moved"): "erosion",
    }.get((behaviour, mechanism), "undetermined")


def write_summary(results: Dict, path: Path) -> Path:
    beh, dirn, ceil = results["behaviour"], results["direction"], results["ceiling"]
    controls = results.get("controls") or {}
    L = ["# Analysis summary", ""]
    if results.get("_demo"):
        L += ["> **SYNTHETIC DEMO OUTPUT — not data.**", ""]

    L += ["## Behavioural baseline (test split)", "",
          "| Checkpoint | Refusal on harmful | Compliance on harmless | Science NTS |",
          "|---|---|---|---|"]
    for t in TAGS:
        if t not in beh:
            continue
        b = beh[t]["behaviour"]
        r, c = b["refusal_rate_harmful"], b["compliance_rate_harmless"]
        nts = beh[t].get("science_nts")
        L.append(
            f"| {t} | {r['point']:.3f} [{r['low']:.3f}, {r['high']:.3f}] | "
            f"{c['point']:.3f} [{c['low']:.3f}, {c['high']:.3f}] | "
            f"{nts if nts is None else f'{nts:.1f}%'} |"
        )

    L += ["", "## Direction level (prereg §5.2)", ""]
    m0c = ceil.get("M0", {}).get("ceiling")
    if m0c:
        L.append(
            f"Direction ceiling (M0, split-half bootstrap, layer {m0c.get('layer')}): "
            f"**{m0c['mean']:.3f}** [{m0c['ci_low']:.3f}, {m0c['ci_high']:.3f}]. "
            "Cosines are read against this, not against 1.0.\n"
        )
    L += ["| Checkpoint | cos with dir_M0 | Transfer ratio | Verdict |", "|---|---|---|---|"]
    for t in TAGS:
        if t not in dirn:
            continue
        cos = find_cosine(dirn[t], "M0")
        tr = dirn[t].get("transfer", {})
        L.append(
            f"| {t} | {'—' if cos is None else f'{cos:.3f}'} | "
            f"{tr.get('transfer_ratio', float('nan')):.3f} | {tr.get('verdict', '—')} |"
        )

    if controls.get("iou_matrix"):
        L += ["", "## Head level (prereg §5.3)", "",
              f"Circuit sizes (of {controls.get('n_total_heads', 576)} heads): "
              f"{controls.get('circuit_sizes')}", "",
              "| Checkpoint | IoU with M0 | Null p97.5 | Above null | 0.8×ceiling met | Verdict |",
              "|---|---|---|---|---|---|"]
        for tag, r in controls.get("verdicts", {}).get("retention", {}).items():
            null_p = r.get("null_p97_5")
            null_s = "—" if null_p is None else f"{null_p:.3f}"
            L.append(
                f"| {tag} | {r['iou_vs_M0']:.3f} | {null_s} | "
                f"{r.get('above_null')} | {r.get('meets_0.8_ceiling')} | {r['verdict']} |"
            )
        h2 = controls.get("verdicts", {}).get("H2")
        if h2:
            reading = {
                "supported": "Supported — report as a replication in a new regime, not a "
                             "discovery (prereg §9 D5).",
                "reversed": "**Reversed**: RL disturbs M0's refusal machinery *more* than SFT. "
                            "This contradicts the orchestrate-vs-install account of "
                            "arXiv:2510.07364 in the collateral regime — lead with it.",
                "null": "Null — inside seed noise. Calibration-positive; write it up as such.",
                "undetermined": "Undetermined — missing seed stability or retention inputs.",
            }[h2.get("outcome", "undetermined")]
            L += ["", f"**H2 (primary):** IoU(M0, M_RL) − IoU(M0, M_SFT) = "
                      f"{h2['gap']:+.3f}; largest between-seed difference = "
                      f"{h2['largest_between_seed_difference']}. {reading}"]
        stab = controls.get("seed_stability", {})
        if stab:
            L += ["", "Seed stability (Jaccard): " +
                  ", ".join(f"{k} {v['jaccard']:.3f}" for k, v in stab.items()) +
                  f"  (threshold {0.7})"]

    rows = h4_crosstab(results)
    if rows:
        L += ["", "## H4 dissociation", "",
              "Cross-tabulated under both registered operationalisations (prereg §4). "
              "Disagreement between the two columns is itself reportable — it means the "
              "direction and the head set do not localise the same thing.", "",
              "| Checkpoint | Behaviour | Mechanism (direction) | Cell (direction) | "
              "Mechanism (head) | Cell (head) |",
              "|---|---|---|---|---|---|"]
        for r in rows:
            L.append(
                f"| {r['checkpoint']} | {r['behaviour']} | {r['mechanism (direction)']} | "
                f"{r['H4 cell']} | {r['mechanism (head)']} | {r.get('H4 cell (head)', '—')} |"
            )

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[analysis] wrote {path}")
    return path


# ==========================================================================

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true",
                    help="render both figures from synthetic numbers, watermarked")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results = gather(demo=args.demo)
    suffix = "_demo" if args.demo else ""

    figure1(results, out_dir / f"figure1_behaviour_direction{suffix}.png")
    figure2(results, out_dir / f"figure2_retention{suffix}.png")
    write_summary(results, out_dir.parent / f"analysis_summary{suffix}.md")


if __name__ == "__main__":
    main()
