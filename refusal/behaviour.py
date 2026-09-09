"""
Stage 2 — behavioural baseline, per checkpoint (prereg.md §5.1).

Reported on the *test* split:
  - refusal rate on harmful prompts (Wilson 95% CI)
  - compliance rate on harmless prompts (Wilson 95% CI)
  - science-QA new-task score (proves the fine-tunes learned the target task)
  - expected forward KL from M0, reported separately on harmful and harmless
    prompts (prior work's Eq. 1)

Usage
  python behaviour.py --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
  python behaviour.py --checkpoint ./results/sft --tag M_SFT --ref Qwen/Qwen2.5-3B-Instruct
  python behaviour.py --checkpoint ./results/rl  --tag M_RL  --ref Qwen/Qwen2.5-3B-Instruct

`--ref` enables the KL block; omit it for M0 itself (KL from itself is 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    M0_NAME,
    is_refusal,
    load_model_and_tokenizer,
    rlrazor_path,
    save_json,
    set_seed,
    wilson_ci,
)


# --------------------------------------------------------------------------
# Refusal / compliance rates
# --------------------------------------------------------------------------

def _greedy_batch(model, tokenizer, chats: List[str], max_new_tokens: int) -> List[str]:
    import torch

    prev = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    tokenizer.padding_side = prev

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def behavioural_rates(
    model,
    tokenizer,
    pairs: List[Dict],
    batch_size: int = 16,
    max_new_tokens: int = 64,
    keep_completions: bool = False,
) -> Dict:
    """Refusal rate on the harmful side, compliance rate on the harmless side."""
    from tqdm import tqdm

    results = {"harmful": [], "harmless": []}
    completions = {"harmful": [], "harmless": []}

    for side, key in (("harmful", "harmful_chat"), ("harmless", "harmless_chat")):
        chats = [p[key] for p in pairs]
        for start in tqdm(range(0, len(chats), batch_size), desc=f"generate:{side}"):
            texts = _greedy_batch(model, tokenizer, chats[start:start + batch_size], max_new_tokens)
            for t in texts:
                results[side].append(is_refusal(t))
                if keep_completions:
                    completions[side].append(t)

    n_refuse_harmful = sum(results["harmful"])
    n_comply_harmless = sum(not r for r in results["harmless"])

    out = {
        "refusal_rate_harmful": wilson_ci(n_refuse_harmful, len(results["harmful"])),
        "compliance_rate_harmless": wilson_ci(n_comply_harmless, len(results["harmless"])),
        # Reported for completeness: refusing harmless prompts is over-refusal.
        "over_refusal_rate_harmless": wilson_ci(
            sum(results["harmless"]), len(results["harmless"])
        ),
        "per_prompt_refused": results,
    }
    if keep_completions:
        out["completions"] = completions
    return out


# --------------------------------------------------------------------------
# Expected forward KL from M0 (prior work's Eq. 1)
# --------------------------------------------------------------------------

def forward_kl_from_ref(
    model,
    ref_model,
    tokenizer,
    chats: List[str],
    max_new_tokens: int = 64,
    batch_size: int = 4,
) -> Dict[str, float]:
    """KL(P_ref || P_model) averaged over response positions.

    The response is the reference model's own greedy continuation, so the
    expectation is taken under the reference policy — the forward direction the
    prior work reports. Both models must be resident; ~12GB in bf16 for two 3B
    models.
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    kls: List[float] = []

    for start in tqdm(range(0, len(chats), batch_size), desc="forward-KL"):
        batch = chats[start:start + batch_size]

        prev = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
            ref_model.device
        )
        tokenizer.padding_side = prev
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen = ref_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            full_mask = torch.cat(
                [
                    enc["attention_mask"],
                    torch.ones_like(gen[:, prompt_len:]),
                ],
                dim=1,
            )
            ref_logits = ref_model(gen, attention_mask=full_mask).logits
            mod_logits = model(gen.to(model.device), attention_mask=full_mask.to(model.device)).logits

        # Positions prompt_len-1 .. end-1 predict the generated tokens.
        ref_lp = F.log_softmax(ref_logits[:, prompt_len - 1:-1, :].float(), dim=-1)
        mod_lp = F.log_softmax(mod_logits[:, prompt_len - 1:-1, :].float().to(ref_lp.device), dim=-1)
        per_pos = (ref_lp.exp() * (ref_lp - mod_lp)).sum(-1)  # [batch, resp_len]

        # Mask out anything the reference emitted after EOS.
        gen_tokens = gen[:, prompt_len:]
        valid = (gen_tokens != tokenizer.pad_token_id).float()
        if valid.sum() == 0:
            continue
        seq_kl = (per_pos * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        kls += seq_kl.detach().cpu().tolist()

    if not kls:
        return {"mean": float("nan"), "n": 0}
    return {
        "mean": sum(kls) / len(kls),
        "max": max(kls),
        "n": len(kls),
    }


# --------------------------------------------------------------------------
# Science-QA new-task score
# --------------------------------------------------------------------------

def science_nts(model, tokenizer, n_eval: int = 200, max_new_tokens: int = 64) -> float:
    """New-task score via the prior project's `evaluate_new_task`."""
    rlrazor_path()
    from evaluation.evaluation import evaluate_new_task  # noqa: E402

    from science_data import load_science_dataset, normalize

    _, eval_raw = load_science_dataset(seed=0, n_eval=n_eval)
    eval_ds = normalize(eval_raw)
    return float(
        evaluate_new_task(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_ds,
            eval_dataset=eval_ds,
            max_new_tokens=max_new_tokens,
            num_samples=n_eval,
        )
    )


# --------------------------------------------------------------------------
# Raw completions for the writeup
# --------------------------------------------------------------------------

def sample_completions(
    pairs: List[Dict], behaviour: Dict, n: int = 10, seed: int = 0
) -> Dict:
    """A seeded RANDOM sample of real completions, both sides, judged.

    Deliberately random and seeded rather than hand-picked: showing qualitative
    examples chosen after the fact is a way to make any result look good, and
    the sample index is recorded so the selection is reproducible and auditable.
    Both agreeing and disagreeing cases are included by construction.
    """
    import random

    rng = random.Random(seed)
    out: Dict = {
        "n_per_side": n,
        "seed": seed,
        "selection": "uniform random over the scored split, seeded; not curated",
    }
    completions = behaviour.get("completions", {})

    for side, prompt_key in (("harmful", "harmful_prompt"), ("harmless", "harmless_prompt")):
        texts = completions.get(side, [])
        if not texts:
            continue
        judged = behaviour.get("per_prompt_refused", {}).get(side, [None] * len(texts))
        idx = sorted(rng.sample(range(len(texts)), min(n, len(texts))))
        out[side] = [
            {
                "index": i,
                "pair_id": pairs[i].get("pair_id"),
                "prompt": pairs[i][prompt_key],
                "completion": texts[i],
                "judged_refusal": None if judged[i] is None else bool(judged[i]),
            }
            for i in idx
        ]
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="M0 | M_SFT | M_RL")
    ap.add_argument("--ref", default=None, help="reference model for forward KL (usually M0)")
    ap.add_argument("--split", default="test", help="registered split to score")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--kl-samples", type=int, default=50)
    ap.add_argument("--nts-samples", type=int, default=200)
    ap.add_argument("--skip-nts", action="store_true")
    ap.add_argument("--skip-kl", action="store_true")
    ap.add_argument("--no-completions", action="store_true",
                    help="skip saving the random completion sample (not recommended)")
    ap.add_argument("--n-examples", type=int, default=10,
                    help="completions per side to sample for the writeup")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
    from build_dataset import load_pairs  # noqa: E402

    pairs = load_pairs(args.split)
    print(f"[behaviour] {args.tag}: {len(pairs)} pairs from split '{args.split}'")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    payload: Dict = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_pairs": len(pairs),
        "seed": args.seed,
    }

    payload["behaviour"] = behavioural_rates(
        model, tokenizer, pairs, args.batch_size, args.max_new_tokens,
        keep_completions=not args.no_completions,
    )
    if not args.no_completions:
        payload["example_completions"] = sample_completions(
            pairs, payload["behaviour"], args.n_examples, args.seed
        )
        # The full completion lists are large and are not what the writeup uses;
        # the seeded sample above is. Drop them from the saved payload.
        payload["behaviour"].pop("completions", None)

    rr = payload["behaviour"]["refusal_rate_harmful"]
    cr = payload["behaviour"]["compliance_rate_harmless"]
    print(f"  refusal(harmful)   = {rr['point']:.3f}  [{rr['low']:.3f}, {rr['high']:.3f}]")
    print(f"  compliance(harmless)= {cr['point']:.3f}  [{cr['low']:.3f}, {cr['high']:.3f}]")

    if not args.skip_nts:
        payload["science_nts"] = science_nts(model, tokenizer, args.nts_samples)
        print(f"  science NTS        = {payload['science_nts']:.1f}%")

    if args.ref and not args.skip_kl:
        ref_model, _ = load_model_and_tokenizer(args.ref)
        subset = pairs[: args.kl_samples]
        payload["forward_kl"] = {
            "reference": args.ref,
            "harmful": forward_kl_from_ref(
                model, ref_model, tokenizer, [p["harmful_chat"] for p in subset],
                args.max_new_tokens,
            ),
            "harmless": forward_kl_from_ref(
                model, ref_model, tokenizer, [p["harmless_chat"] for p in subset],
                args.max_new_tokens,
            ),
        }
        print(
            f"  KL(harmful)={payload['forward_kl']['harmful']['mean']:.4f}  "
            f"KL(harmless)={payload['forward_kl']['harmless']['mean']:.4f}"
        )

    save_json(payload, f"behaviour_{args.tag}", subdir="behaviour")


if __name__ == "__main__":
    main()
