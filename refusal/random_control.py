"""Random-direction control for the direction-level result.

THE QUESTION THIS ANSWERS
`direction.py` shows that projecting dir_M0 out of every residual-stream write
drives refusal to ~0 in M0, M_SFT and M_RL alike (transfer ratio ~1.0). That is
only interesting if the *specific* direction matters. If projecting out an
arbitrary direction at the same site did the same thing, the finding would be
"ablating one rank-1 component anywhere breaks refusal", which is a statement
about the intervention, not about the refusal mechanism.

So: run the identical intervention with `--n-random` random unit vectors drawn
in the same d_model, at the same layer, and compare refusal drops.

Note on norm matching: directional ablation projects out the *unit* direction,
so the intervention is scale-invariant and a random unit vector is already the
matched control. Nothing is gained by rescaling to dir_M0's norm.

The harmless side is measured too. A random direction that tanks refusal *and*
tanks harmless compliance is breaking the model, not ablating a mechanism —
the same confound D9 recorded for the real direction.

  python refusal/random_control.py --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
  python refusal/random_control.py --checkpoint refusal/results/checkpoints/M_SFT \
      --tag M_SFT --donor M0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from common import load_model_and_tokenizer, save_json, set_seed  # noqa: E402
from direction import (  # noqa: E402
    _refusal_rate,
    directional_ablation,
    load_directions,
)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="checkpoint being measured")
    ap.add_argument("--donor", default=None,
                    help="direction tag to use as the real comparison (default: --tag)")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    from build_dataset import load_pairs  # noqa: E402

    pairs = load_pairs(args.split)
    harmful = [p["harmful_chat"] for p in pairs]
    harmless = [p["harmless_chat"] for p in pairs]

    donor_tag = args.donor or args.tag
    donor = load_directions(donor_tag)
    layer = donor["selected_layer"]
    real_unit = donor["unit"][layer]

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    d_model = model.config.hidden_size
    print(f"[control] {args.tag}: {len(pairs)} pairs from '{args.split}', "
          f"donor dir_{donor_tag}@L{layer}, d_model={d_model}")

    def measure(direction=None) -> Dict:
        if direction is None:
            return {
                "harmful": _refusal_rate(model, tokenizer, harmful, args.batch_size,
                                         args.max_new_tokens),
                "harmless": _refusal_rate(model, tokenizer, harmless, args.batch_size,
                                          args.max_new_tokens),
            }
        with directional_ablation(model, direction, 0):
            return {
                "harmful": _refusal_rate(model, tokenizer, harmful, args.batch_size,
                                         args.max_new_tokens),
                "harmless": _refusal_rate(model, tokenizer, harmless, args.batch_size,
                                          args.max_new_tokens),
            }

    print("[control] baseline ...")
    base = measure(None)
    b_ref = base["harmful"]["point"]
    b_over = base["harmless"]["point"]
    print(f"  baseline refusal {b_ref:.3f}, harmless refusal {b_over:.3f}")

    print(f"[control] real direction dir_{donor_tag} ...")
    real = measure(real_unit)
    real_drop = b_ref - real["harmful"]["point"]
    print(f"  refusal {real['harmful']['point']:.3f} (drop {real_drop:+.3f}), "
          f"harmless refusal {real['harmless']['point']:.3f}")

    # Random unit vectors, drawn from a dedicated generator so the draw is
    # reproducible independently of anything else the process did.
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    randoms: List[Dict] = []
    for i in range(args.n_random):
        v = torch.randn(d_model, generator=gen, dtype=torch.float32)
        v = v / v.norm()
        cos_with_real = float(torch.nn.functional.cosine_similarity(
            v, real_unit.float().cpu(), dim=0))
        print(f"[control] random vector {i + 1}/{args.n_random} "
              f"(cos with real = {cos_with_real:+.4f}) ...")
        res = measure(v)
        drop = b_ref - res["harmful"]["point"]
        randoms.append({
            "index": i,
            "cos_with_real_direction": cos_with_real,
            "refusal_harmful": res["harmful"],
            "over_refusal_harmless": res["harmless"],
            "refusal_drop": drop,
        })
        print(f"  refusal {res['harmful']['point']:.3f} (drop {drop:+.3f}), "
              f"harmless refusal {res['harmless']['point']:.3f}")

    drops = [r["refusal_drop"] for r in randoms]
    mean_rand = sum(drops) / len(drops) if drops else float("nan")
    max_rand = max(drops) if drops else float("nan")

    payload = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "donor": donor_tag,
        "layer": layer,
        "split": args.split,
        "n_pairs": len(pairs),
        "d_model": d_model,
        "seed": args.seed,
        "baseline": base,
        "real": {"refusal_harmful": real["harmful"],
                 "over_refusal_harmless": real["harmless"],
                 "refusal_drop": real_drop},
        "random": randoms,
        "summary": {
            "real_drop": real_drop,
            "random_drop_mean": mean_rand,
            "random_drop_max": max_rand,
            # The claim the writeup needs: the fitted direction does something
            # that an arbitrary direction at the same site does not.
            "real_exceeds_all_random": bool(real_drop > max_rand),
            "margin_over_best_random": real_drop - max_rand,
        },
    }
    print(f"[control] real drop {real_drop:+.3f} vs random mean {mean_rand:+.3f}, "
          f"random max {max_rand:+.3f} -> "
          f"{'SPECIFIC' if real_drop > max_rand else 'NOT SPECIFIC'}")
    save_json(payload, f"random_control_{args.tag}", subdir="direction")


if __name__ == "__main__":
    main()
