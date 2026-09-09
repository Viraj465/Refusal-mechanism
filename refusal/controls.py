"""
Stage 6 — the controls that make the result defensible (prereg.md §5.2, §5.3).

Without these, a head-level overlap number means nothing: the prior work's
circuits were ~50% of all heads, so two unrelated circuits of that density
already share IoU ~= 0.33 by chance.

  --mode stats    CPU, seconds. Permutation null, matched-k, split-half ceiling,
                  seed stability, IoU matrix, and the direction-level summary,
                  each scored against the thresholds frozen in prereg.md.

  --mode ablate   GPU, minutes. The fourth pilot-gate condition (prereg §7.2):
                  zero-ablate the recovered head set and check that refusal moves
                  in the predicted direction on val. dbm.py scores the other
                  three; this closes the gate.

Everything here is new code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from common import RESULTS_DIR, jaccard, load_model_and_tokenizer, save_json, set_seed  # noqa: E402

MASKS_DIR = RESULTS_DIR / "masks"

# Thresholds — frozen in prereg.md §5.3. Do not tune these.
NULL_PERCENTILE = 97.5        # an IoU is "above chance" only above this
CEILING_FRACTION = 0.80       # retention preserved needs >= 0.8 x split-half ceiling
SEED_STABILITY_MIN = 0.70     # below this, head-level conclusions are invalid
N_PERMUTATIONS = 1000


# ==========================================================================
# Loading masks
# ==========================================================================

MASK_RE = re.compile(r"^(?P<tag>.+?)_seed(?P<seed>\d+)_lam(?P<lam>[\d.]+?)(?:_half(?P<half>\d))?$")


def load_masks() -> List[Dict]:
    """Every saved mask, with its filename metadata parsed out."""
    import torch

    if not MASKS_DIR.is_dir():
        return []
    out = []
    for path in sorted(MASKS_DIR.glob("*.pt")):
        m = MASK_RE.match(path.stem)
        if not m:
            print(f"[controls] skipping unparseable mask filename: {path.name}")
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        out.append(
            {
                "path": path,
                "tag": m.group("tag"),
                "seed": int(m.group("seed")),
                "lam": float(m.group("lam")),
                "half": int(m.group("half")) if m.group("half") is not None else None,
                "values": blob["values"],
                "circuit": [tuple(h) for h in blob["circuit"]],
                "n_total": blob["values"].numel(),
            }
        )
    return out


def _primary(masks: List[Dict], tag: str, seed: int = 0) -> Optional[Dict]:
    """The main (non-split-half) mask for a checkpoint."""
    for m in masks:
        if m["tag"] == tag and m["half"] is None and m["seed"] == seed:
            return m
    return None


# ==========================================================================
# Controls
# ==========================================================================

def permutation_null(
    size_a: int, size_b: int, n_total: int, n_draws: int = N_PERMUTATIONS, seed: int = 0
) -> Dict:
    """IoU distribution for two random head subsets of the observed sizes.

    This is the number that decides whether an overlap means anything. At ~50%
    density chance IoU is ~0.33, not 0.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ious = np.empty(n_draws)
    for i in range(n_draws):
        a = rng.choice(n_total, size=size_a, replace=False)
        b = rng.choice(n_total, size=size_b, replace=False)
        inter = np.intersect1d(a, b, assume_unique=True).size
        ious[i] = inter / (size_a + size_b - inter)
    return {
        "size_a": size_a,
        "size_b": size_b,
        "n_total": n_total,
        "n_draws": n_draws,
        "mean": float(ious.mean()),
        "p2_5": float(np.percentile(ious, 2.5)),
        "p97_5": float(np.percentile(ious, NULL_PERCENTILE)),
        "max": float(ious.max()),
    }


def matched_k(mask_a: Dict, mask_b: Dict, k: int) -> Dict:
    """Top-k overlap by continuous mask value, removing the density confound.

    Two circuits of different sizes have a mechanically different IoU ceiling;
    comparing equal-size top-k sets takes that out.
    """
    import torch

    def topk(mask):
        flat = mask["values"].flatten()
        idx = torch.topk(flat, k).indices.tolist()
        n_heads = mask["values"].shape[1]
        return {(i // n_heads, i % n_heads) for i in idx}

    a, b = topk(mask_a), topk(mask_b)
    return {"k": k, "iou": jaccard(a, b), "n_shared": len(a & b)}


def run_stats(args) -> None:
    masks = load_masks()
    if not masks:
        raise SystemExit(
            f"No masks in {MASKS_DIR}. Run dbm.py --mode fit first, or (if the "
            "pilot gate failed) skip Stage 6 head-level controls entirely — "
            "prereg §7.2 registers that as an acceptable outcome."
        )

    tags = sorted({m["tag"] for m in masks})
    n_total = masks[0]["n_total"]
    print(f"[controls] {len(masks)} masks across {len(tags)} checkpoints: {tags}")

    payload: Dict = {"tags": tags, "n_total_heads": n_total, "n_masks": len(masks),
                     "thresholds": {
                         "null_percentile": NULL_PERCENTILE,
                         "ceiling_fraction": CEILING_FRACTION,
                         "seed_stability_min": SEED_STABILITY_MIN,
                     }}

    # ---- circuit sizes ----
    payload["circuit_sizes"] = {
        m["tag"]: len(m["circuit"]) for m in masks if m["half"] is None and m["seed"] == 0
    }

    # ---- IoU matrix over primary masks ----
    primaries = {t: _primary(masks, t) for t in tags}
    primaries = {t: m for t, m in primaries.items() if m is not None}
    iou_matrix = {
        f"{a}|{b}": jaccard(primaries[a]["circuit"], primaries[b]["circuit"])
        for a in primaries for b in primaries
    }
    payload["iou_matrix"] = iou_matrix

    # ---- permutation null, per observed size pair ----
    nulls = {}
    for a in primaries:
        for b in primaries:
            if a >= b:
                continue
            nulls[f"{a}|{b}"] = permutation_null(
                len(primaries[a]["circuit"]), len(primaries[b]["circuit"]),
                n_total, args.n_permutations, args.seed,
            )
    payload["permutation_null"] = nulls

    # ---- split-half ceiling ----
    ceilings = {}
    for t in tags:
        h0 = next((m for m in masks if m["tag"] == t and m["half"] == 0), None)
        h1 = next((m for m in masks if m["tag"] == t and m["half"] == 1), None)
        if h0 and h1:
            ceilings[t] = jaccard(h0["circuit"], h1["circuit"])
    payload["split_half_ceiling"] = ceilings
    if not ceilings:
        print(
            "[controls] WARNING: no split-half masks found. Retention is "
            "UNINTERPRETABLE without the ceiling (prereg §5.3) — run "
            "`dbm.py --mode fit --half 0` and `--half 1` per checkpoint."
        )

    # ---- seed stability ----
    seed_stability = {}
    for t in tags:
        seeds = sorted({m["seed"] for m in masks if m["tag"] == t and m["half"] is None})
        if len(seeds) >= 2:
            a = _primary(masks, t, seeds[0])
            b = _primary(masks, t, seeds[1])
            seed_stability[t] = {
                "seeds": seeds[:2],
                "jaccard": jaccard(a["circuit"], b["circuit"]),
            }
    payload["seed_stability"] = seed_stability
    for t, s in seed_stability.items():
        if s["jaccard"] < SEED_STABILITY_MIN:
            print(
                f"[controls] WARNING: {t} seed Jaccard {s['jaccard']:.3f} < "
                f"{SEED_STABILITY_MIN}. Per prereg §5.3, report the instability "
                "instead of the overlap for this checkpoint."
            )

    # ---- matched-k ----
    if primaries:
        k = min(len(m["circuit"]) for m in primaries.values())
        payload["matched_k"] = {
            f"{a}|{b}": matched_k(primaries[a], primaries[b], k)
            for a in primaries for b in primaries if a < b
        }
        payload["matched_k_k"] = k

    # ---- verdicts against the frozen thresholds ----
    payload["verdicts"] = _verdicts(payload, primaries)

    save_json(payload, "controls_stats", subdir="controls")
    _print_stats(payload)


def _verdicts(payload: Dict, primaries: Dict) -> Dict:
    """Score retention and H2 against prereg §5.3 — mechanically, not by eye."""
    out: Dict = {}
    base = "M0"
    if base not in primaries:
        out["note"] = f"No '{base}' mask; retention verdicts require it."
        return out

    retention = {}
    for tag in primaries:
        if tag == base:
            continue
        key = f"{base}|{tag}" if f"{base}|{tag}" in payload["iou_matrix"] else f"{tag}|{base}"
        iou = payload["iou_matrix"].get(key)
        null = payload["permutation_null"].get(f"{base}|{tag}") or \
            payload["permutation_null"].get(f"{tag}|{base}")
        ceiling = payload["split_half_ceiling"].get(base)

        above_null = iou > null["p97_5"] if (iou is not None and null) else None
        meets_ceiling = (
            iou >= CEILING_FRACTION * ceiling if (iou is not None and ceiling) else None
        )
        retention[tag] = {
            "iou_vs_M0": iou,
            "null_p97_5": null["p97_5"] if null else None,
            "above_null": above_null,
            "split_half_ceiling": ceiling,
            "meets_0.8_ceiling": meets_ceiling,
            "verdict": (
                "preserved" if (above_null and meets_ceiling)
                else "not_preserved" if (above_null is not None and meets_ceiling is not None)
                else "undetermined (missing null or ceiling)"
            ),
        }
    out["retention"] = retention

    # H2: RL retains more than SFT, by more than seed noise.
    if "M_SFT" in retention and "M_RL" in retention:
        d_sft, d_rl = retention["M_SFT"]["iou_vs_M0"], retention["M_RL"]["iou_vs_M0"]
        seed_diffs = [
            abs(1.0 - s["jaccard"]) for s in payload.get("seed_stability", {}).values()
        ]
        noise = max(seed_diffs) if seed_diffs else None
        gap = (d_rl - d_sft) if (d_rl is not None and d_sft is not None) else None

        # Three-way, not binary. A significantly NEGATIVE gap — RL disturbing
        # M0's refusal machinery more than SFT does — is not the same result as
        # a null, and prereg §9 D5 registers it as the most diagnostic outcome:
        # it would contradict the orchestrate-vs-install account of
        # arXiv:2510.07364 in the collateral regime. Collapsing it into
        # "not supported" would throw away the finding.
        if gap is None or noise is None:
            outcome = "undetermined"
        elif gap > noise:
            outcome = "supported"
        elif gap < -noise:
            outcome = "reversed"
        else:
            outcome = "null"

        out["H2"] = {
            "iou_M0_M_SFT": d_sft,
            "iou_M0_M_RL": d_rl,
            "gap": gap,
            "largest_between_seed_difference": noise,
            "outcome": outcome,
            "supported": outcome == "supported",
            "note": (
                "prereg §5.3: H2 is supported only if the gap is positive AND exceeds "
                "the largest within-checkpoint between-seed IoU difference. A gap inside "
                "seed noise is 'null', not weak support. A gap below -noise is 'reversed' "
                "— report it as the primary finding (prereg §9 D5), not as a failed H2."
            ),
        }
    return out


def _print_stats(p: Dict) -> None:
    print("\n" + "=" * 66)
    print(" CONTROLS")
    print("=" * 66)
    print(" circuit sizes:", p.get("circuit_sizes"))
    print(" split-half ceiling:", {k: round(v, 3) for k, v in p.get("split_half_ceiling", {}).items()})
    for t, s in p.get("seed_stability", {}).items():
        print(f" seed stability {t}: Jaccard {s['jaccard']:.3f}")
    for tag, r in p.get("verdicts", {}).get("retention", {}).items():
        iou = r["iou_vs_M0"]
        null = r["null_p97_5"]
        print(
            f" retention M0 vs {tag}: IoU {iou:.3f} "
            f"(null p97.5 {null:.3f})  -> {r['verdict']}"
            if iou is not None and null is not None else f" retention M0 vs {tag}: incomplete"
        )
    h2 = p.get("verdicts", {}).get("H2")
    if h2:
        note = {
            "supported": "SUPPORTED (report as replication in a new regime, not a discovery)",
            "reversed": "REVERSED — RL disturbs M0's refusal machinery MORE than SFT. "
                        "This is the diagnostic outcome (prereg §9 D5); lead with it.",
            "null": "NULL (inside seed noise) — calibration-positive, write it up as such",
            "undetermined": "UNDETERMINED (missing seed stability or retention)",
        }[h2["outcome"]]
        print(
            f" H2: gap {h2['gap']:+.3f} vs seed noise "
            f"{h2['largest_between_seed_difference']} -> {note}"
        )


# ==========================================================================
# Pilot-gate condition 4 (GPU)
# ==========================================================================

def ablate_heads_refusal(
    model, tokenizer, chats: List[str], heads: Sequence[Tuple[int, int]],
    batch_size: int = 16, max_new_tokens: int = 64,
) -> Dict:
    """Zero-ablate a head set at every position, then measure refusal rate."""
    import torch

    from common import is_refusal, wilson_ci

    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    by_layer: Dict[int, List[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    def make_hook(head_list):
        def hook(_module, args):
            x = args[0]
            b, s, hidden = x.shape
            xh = x.view(b, s, n_heads, head_dim).clone()
            for h in head_list:
                xh[:, :, h, :] = 0.0
            return (xh.view(b, s, hidden),) + args[1:]
        return hook

    handles = [
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(hs))
        for l, hs in by_layer.items()
    ]
    try:
        refusals = []
        for start in range(0, len(chats), batch_size):
            prev = tokenizer.padding_side
            tokenizer.padding_side = "left"
            enc = tokenizer(
                chats[start:start + batch_size], return_tensors="pt",
                padding=True, add_special_tokens=False,
            ).to(model.device)
            tokenizer.padding_side = prev
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            texts = tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
            refusals += [is_refusal(t) for t in texts]
    finally:
        for h in handles:
            h.remove()

    return wilson_ci(sum(refusals), len(refusals))


def run_ablate(args) -> None:
    from build_dataset import load_pairs

    set_seed(args.seed)
    masks = load_masks()
    mask = _primary(masks, args.tag, args.seed)
    if mask is None:
        raise SystemExit(f"No primary mask for tag '{args.tag}' seed {args.seed} in {MASKS_DIR}")

    pairs = load_pairs(args.split)[: args.n_pairs]
    harmful = [p["harmful_chat"] for p in pairs]
    circuit = mask["circuit"]
    print(f"[controls] ablating {len(circuit)} circuit heads in {args.tag} on {len(pairs)} val pairs")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    baseline = ablate_heads_refusal(model, tokenizer, harmful, [], args.batch_size, args.max_new_tokens)
    ablated = ablate_heads_refusal(
        model, tokenizer, harmful, circuit, args.batch_size, args.max_new_tokens
    )

    # Size-matched random control: a circuit that matters must beat a random
    # head set of the same size, otherwise the drop is just capacity damage.
    import random

    rng = random.Random(args.seed)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    random_set = rng.sample(all_heads, len(circuit))
    random_abl = ablate_heads_refusal(
        model, tokenizer, harmful, random_set, args.batch_size, args.max_new_tokens
    )

    drop = baseline["point"] - ablated["point"]
    random_drop = baseline["point"] - random_abl["point"]
    payload = {
        "tag": args.tag, "checkpoint": args.checkpoint, "split": args.split,
        "circuit_size": len(circuit), "n_pairs": len(pairs),
        "baseline_refusal": baseline,
        "ablated_refusal": ablated,
        "random_control_refusal": random_abl,
        "refusal_drop": drop,
        "random_control_drop": random_drop,
        # prereg §7.2, condition 4
        "gate_condition_4_passed": bool(drop > 0 and drop > random_drop),
        "note": (
            "Condition 4 requires refusal to move in the predicted direction AND "
            "to beat a size-matched random head set. A drop that a random set "
            "reproduces is capacity damage, not the mechanism."
        ),
    }
    print(
        f"  baseline {baseline['point']:.3f} -> circuit-ablated {ablated['point']:.3f} "
        f"(drop {drop:+.3f}); random-set drop {random_drop:+.3f}"
    )
    print(f"  pilot gate condition 4: {'PASS' if payload['gate_condition_4_passed'] else 'FAIL'}")
    save_json(payload, f"controls_ablate_{args.tag}", subdir="controls")


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["stats", "ablate"])
    ap.add_argument("--checkpoint", default=None, help="required for --mode ablate")
    ap.add_argument("--tag", default="M0")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "stats":
        run_stats(args)
    else:
        if not args.checkpoint:
            ap.error("--checkpoint is required for --mode ablate")
        run_ablate(args)


if __name__ == "__main__":
    main()
