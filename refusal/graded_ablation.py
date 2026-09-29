"""Graded ablation: alpha-scaled all-writes ablation and single-layer (L19-only)
residual-stream projection. (LW_REVIEW.md experiment 2b; prereg D15.)

THE QUESTION
The registered transfer ratio compares two endpoints of a saturated readout:
full projection out of every residual write drives refusal to ~0 for dir_M0
and for the model's own direction alike, so both drops are ~1 and the ratio is
~1 whatever the geometry. Two ways to take the readout off its ceiling:

  --variant alpha   a <- a - alpha * r (r . a) on every write, all layers and
                    positions (direction.directional_ablation with alpha).
                    alpha = 1 is the registered intervention.
  --variant single  project r out of the residual stream at hidden_states[L]
                    only (sensitivity.residual_ablation); later blocks may
                    write it back. L defaults to 19, the layer both
                    directions are read from.

Each variant is run for dir_M0 and for the model's own direction, both at the
same layer (19 by default, so M_RL's own direction is taken at 19, not at its
tie-break L* of 21; see D13). Curves are normalised by the own direction's
drop at alpha = 1 in the same variant:

    nd(alpha) = (baseline - y(alpha)) / (baseline - y_own(alpha = 1))

Summaries: alpha50 (where nd crosses 0.5) per direction, and the graded
transfer ratio AUC(nd_M0) / AUC(nd_own), with prompt-bootstrap CIs.

    python refusal/graded_ablation.py --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT --variant alpha
    python refusal/graded_ablation.py --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT --variant single
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from common import load_model_and_tokenizer, save_json, set_seed  # noqa: E402
from direction import directional_ablation, load_directions  # noqa: E402
from sensitivity import (  # noqa: E402
    auc,
    bootstrap,
    crossing,
    readout,
    refusal_token_ids,
    residual_ablation,
    m0_condition,
    seeded_subset,
)

DEFAULT_ALPHAS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
TRANSFER_PRESERVED, TRANSFER_MOVED = 0.8, 0.5   # prereg §5.2, applied to the graded ratio


def run(model, tokenizer, chats: List[str], directions: Dict[str, "object"], variant: str,
        layers: Sequence[int], alphas: Sequence[float], generate: bool = True,
        batch_size: int = 16, max_new_tokens: int = 64) -> Dict:
    token_ids = refusal_token_ids(tokenizer)
    kw = dict(generate=generate, batch_size=batch_size, max_new_tokens=max_new_tokens)

    def intervention(r, a):
        if variant == "alpha":
            return lambda: directional_ablation(model, r, 0, a)
        return lambda: residual_ablation(model, r, layers, a)

    print("[graded] baseline ...")
    baseline = readout(model, tokenizer, chats, token_ids, **kw)
    curves: Dict[str, List[Dict]] = {}
    for name, r in directions.items():
        curves[name] = []
        for a in alphas:
            res = readout(model, tokenizer, chats, token_ids, intervention(r, a), **kw)
            curves[name].append({"alpha": a, **res})
            rate = res.get("refusal_rate", {}).get("point", float("nan"))
            print(f"  {variant} {name:8s} alpha={a:.2f}: refusal {rate:.3f}, score {res['score_mean']:+.2f}")
    return {"baseline": baseline, "curves": curves, "refusal_token_ids": token_ids}


# --------------------------------------------------------------------------
# Summary (pure; CPU-tested)
# --------------------------------------------------------------------------

def _vals(res: Dict, name: str) -> Optional[List[float]]:
    if name == "rate":
        return [float(f) for f in res["flags"]] if "flags" in res else None
    return res["scores"]


def summarise(raw: Dict, own: str, donor: str, n_boot: int = 1000, seed: int = 0) -> Dict:
    out: Dict = {}
    for name in ("rate", "score"):
        base = _vals(raw["baseline"], name)
        if base is None:
            continue
        n = len(base)
        curves = {d: sorted(c, key=lambda p: p["alpha"]) for d, c in raw["curves"].items()}
        alphas = [0.0] + [p["alpha"] for p in curves[own]]
        own_end = next((p for p in curves[own] if p["alpha"] == 1.0), None)
        if own_end is None:
            raise ValueError("alpha = 1.0 is required: it anchors the normalisation")
        own_end_v = _vals(own_end, name)

        def nd(d, idx):
            b = sum(base[i] for i in idx)
            denom = b - sum(own_end_v[i] for i in idx)
            if abs(denom) < 1e-9:
                return None
            return [0.0] + [(b - sum(_vals(p, name)[i] for i in idx)) / denom for p in curves[d]]

        def a50(d, idx):
            y = nd(d, idx)
            return float("nan") if y is None else crossing(alphas, y, 0.5)

        def ratio(idx):
            yo, yd = nd(own, idx), nd(donor, idx)
            if yo is None or yd is None or auc(alphas, yo) == 0:
                return float("nan")
            return auc(alphas, yd) / auc(alphas, yo)

        full = list(range(n))
        r: Dict = {
            "alphas": alphas,
            "own_full_drop": (sum(base) - sum(own_end_v)) / n,
            "curves": {d: nd(d, full) for d in curves},
            "alpha50": {d: a50(d, full) for d in curves},
            "alpha50_ci": {d: bootstrap(lambda idx, d=d: a50(d, idx), n, n_boot, seed) for d in curves},
        }
        # A ratio over an own-direction drop indistinguishable from 0 is noise
        # (likely for the single-layer variant); D15 reports it as uninformative.
        r["own_full_drop_ci"] = bootstrap(
            lambda idx: sum(base[i] - own_end_v[i] for i in idx) / len(idx), n, n_boot, seed)
        if donor in curves and donor != own:
            ci = bootstrap(ratio, n, n_boot, seed)
            r["graded_transfer_ratio"] = ratio(full)
            r["graded_transfer_ci"] = ci
            r["verdict"] = (
                "uninformative" if not r["own_full_drop_ci"]["low"] > 0
                else "causally_preserved" if ci["low"] >= TRANSFER_PRESERVED
                else "causally_moved" if ci["high"] < TRANSFER_MOVED
                else "ambiguous"
            )
        out[name] = r
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--donor", default="M0")
    ap.add_argument("--variant", choices=["alpha", "single"], required=True)
    ap.add_argument("--layer", type=int, default=19, help="layer both directions are taken from")
    ap.add_argument("--ablate-layers", type=int, nargs="+", default=None,
                    help="single variant: hidden_states indices to project at (default: --layer)")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--no-m0-condition", action="store_true",
                    help="v2: also use test pairs where M0 did not behave as the readout needs (D16)")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    from build_dataset import load_pairs  # noqa: E402

    alphas = sorted(set(args.alphas) | {1.0})
    layers = args.ablate_layers or [args.layer]
    own_name, donor_name = f"dir_{args.tag}", f"dir_{args.donor}"
    directions = {own_name: load_directions(args.tag)["unit"][args.layer]}
    if args.donor != args.tag:
        directions[donor_name] = load_directions(args.donor)["unit"][args.layer]

    pairs = seeded_subset(
        load_pairs(args.split) if args.no_m0_condition else m0_condition(load_pairs(args.split), "harmful"),
        args.n_prompts, args.subset_seed)
    chats = [p["harmful_chat"] for p in pairs]
    print(f"[graded] {args.tag}: variant {args.variant}, {len(chats)} harmful prompts, "
          f"directions @L{args.layer}, alphas {alphas}"
          + (f", projected at hidden_states {layers}" if args.variant == "single" else ""))

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    raw = run(model, tokenizer, chats, directions, args.variant, layers, alphas,
              not args.no_generate, args.batch_size, args.max_new_tokens)
    summary = summarise(raw, own_name, donor_name, args.n_boot, args.seed)
    for name, r in summary.items():
        line = f"[graded] {name}: own full drop {r['own_full_drop']:+.3f}; alpha50 " + ", ".join(
            f"{d} {v:.3f}" for d, v in r["alpha50"].items())
        if "graded_transfer_ratio" in r:
            ci = r["graded_transfer_ci"]
            line += (f"; graded transfer {r['graded_transfer_ratio']:.3f} "
                     f"[{ci['low']:.3f}, {ci['high']:.3f}] -> {r['verdict']}")
        print(line)

    save_json({
        "tag": args.tag, "checkpoint": args.checkpoint, "donor": args.donor,
        "variant": args.variant, "layer": args.layer, "ablate_layers": layers,
        "split": args.split, "n_prompts": len(chats), "subset_seed": args.subset_seed,
        "m0_conditioned": not args.no_m0_condition,
        "prompt_ids": [p["pair_id"] for p in pairs], "alphas": alphas,
        "generate": not args.no_generate, "raw": raw, "summary": summary,
    }, f"graded_ablation_{args.variant}_{args.tag}", subdir="sensitivity")


if __name__ == "__main__":
    main()
