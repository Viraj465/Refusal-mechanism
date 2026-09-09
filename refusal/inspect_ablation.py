"""
Sanity check for the Stage-3 gate: print raw completions with and without the
fitted direction ablated, so a human can confirm that "refusal 0.000" means
coherent compliance and not broken generation. The substring judge cannot
tell those apart, so this must be read by eye.

  python refusal/inspect_ablation.py --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
  python refusal/inspect_ablation.py --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT --direction-tag M0

Uses the val split (never test), a seeded random subset, and writes a JSON
next to the other direction results so the check is auditable.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from common import is_refusal, load_model_and_tokenizer, save_json, set_seed  # noqa: E402
from direction import directional_ablation, load_directions  # noqa: E402


def generate(model, tokenizer, chats, max_new_tokens):
    import torch

    prev = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    tokenizer.padding_side = prev
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="checkpoint tag, for the output filename")
    ap.add_argument("--direction-tag", default=None, help="fitted direction to ablate (default: --tag)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    from build_dataset import load_pairs  # noqa: E402

    pairs = load_pairs(args.split)
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(pairs)), min(args.n, len(pairs))))
    subset = [pairs[i] for i in idx]

    d = load_directions(args.direction_tag or args.tag)
    layer = d["selected_layer"]
    unit = d["unit"][layer]

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    rows = []
    for side, chat_key, prompt_key in (("harmful", "harmful_chat", "harmful_prompt"),
                                       ("harmless", "harmless_chat", "harmless_prompt")):
        chats = [p[chat_key] for p in subset]
        base = generate(model, tokenizer, chats, args.max_new_tokens)
        with directional_ablation(model, unit, 0):
            ablated = generate(model, tokenizer, chats, args.max_new_tokens)
        for i, p, b, a in zip(idx, subset, base, ablated):
            rows.append({
                "side": side, "index": i, "pair_id": p.get("pair_id"),
                "prompt": p[prompt_key],
                "baseline": b, "baseline_judged_refusal": is_refusal(b),
                "ablated": a, "ablated_judged_refusal": is_refusal(a),
            })
            print("=" * 100)
            print(f"[{side} #{i}] {p[prompt_key]}")
            print(f"--- baseline (refusal={is_refusal(b)}):\n{b.strip()}")
            print(f"--- ablated dir_{args.direction_tag or args.tag}@L{layer} (refusal={is_refusal(a)}):\n{a.strip()}")

    save_json(
        {"checkpoint": args.checkpoint, "tag": args.tag, "direction_tag": args.direction_tag or args.tag,
         "layer": layer, "split": args.split, "seed": args.seed, "indices": idx, "rows": rows},
        f"inspect_ablation_{args.tag}", subdir="direction",
    )


if __name__ == "__main__":
    main()
