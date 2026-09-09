"""
Stage 1 — refusal dataset construction.

Two modes, deliberately separated so the expensive one is the only thing that
needs a GPU:

  --build    CPU. Downloads the harmful/harmless instruction sets, applies the
             Qwen chat template, writes refusal_prompts_raw.jsonl. Run locally.

  --filter   GPU. Greedy-decodes M0 on every prompt, applies the frozen Arditi
             substring judge, keeps only harmful prompts M0 refuses and harmless
             prompts it complies with, extracts the 3-token targets, length-
             matches pairs, and writes refusal_pairs.jsonl + splits.json +
             filter_report.md. Run in Phase B.

Split discipline (prereg.md §3, binding): train 200 / val 50 / test 150 pairs,
seed 0. train fits directions and masks; val chooses everything; test is only
ever scored, never used to choose.

SOURCE NOTE (deviation candidate — see prereg.md §9): prereg registers the
Arditi et al. `refusal_direction` repo as the source. That repo builds its
splits at run time from upstream datasets rather than committing them, so this
script pulls the same four underlying datasets (AdvBench, MaliciousInstruct,
HarmBench, Alpaca) from their own canonical repositories. Same data, different
mirror. Every kept prompt records its source URL in filter_report.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    DATA_DIR,
    M0_NAME,
    REFUSAL_SUBSTRINGS,
    format_chat,
    is_refusal,
    load_model_and_tokenizer,
    read_jsonl,
    save_json,
    set_seed,
    write_jsonl,
)

RAW_PATH = DATA_DIR / "refusal_prompts_raw.jsonl"
PAIRS_PATH = DATA_DIR / "refusal_pairs.jsonl"
SPLITS_PATH = DATA_DIR / "splits.json"
REPORT_PATH = DATA_DIR / "filter_report.md"
CACHE_DIR = DATA_DIR / "_cache"

# Registered split sizes (prereg.md §3).
N_TRAIN, N_VAL, N_TEST = 200, 50, 150
N_PAIRS_REQUIRED = N_TRAIN + N_VAL + N_TEST

TARGET_N_TOKENS = 3          # prereg: targets are the first ~3 tokens M0 produces
LENGTH_TOLERANCE = 2         # prereg §3: pairs matched on token length (+/- 2)
JUDGE_MAX_NEW_TOKENS = 64    # prereg §2: greedy, 64 new tokens

SOURCES = {
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "malicious_instruct": "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt",
    "harmbench": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "alpaca": "hf://tatsu-lab/alpaca",
}


# ==========================================================================
# --build : download + template  (CPU only)
# ==========================================================================

def _fetch(url: str, cache_name: str) -> str:
    """Download with an on-disk cache so re-runs are offline and reproducible."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / cache_name
    if cached.exists():
        print(f"[cache] {cache_name}")
        return cached.read_text(encoding="utf-8")

    import urllib.request

    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    cached.write_text(text, encoding="utf-8")
    return text


def load_advbench() -> List[str]:
    text = _fetch(SOURCES["advbench"], "advbench.csv")
    reader = csv.DictReader(io.StringIO(text))
    return [row["goal"].strip() for row in reader if row.get("goal", "").strip()]


def load_malicious_instruct() -> List[str]:
    text = _fetch(SOURCES["malicious_instruct"], "malicious_instruct.txt")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def load_harmbench() -> List[str]:
    """HarmBench standard behaviours only.

    'contextual' and 'copyright' behaviours carry a separate context string and
    are not self-contained single-turn instructions, so they cannot be paired
    against a plain Alpaca instruction. Dropping them keeps the counterfactual
    clean.
    """
    text = _fetch(SOURCES["harmbench"], "harmbench.csv")
    reader = csv.DictReader(io.StringIO(text))
    out = []
    for row in reader:
        if row.get("FunctionalCategory", "").strip().lower() != "standard":
            continue
        behaviour = row.get("Behavior", "").strip()
        if behaviour:
            out.append(behaviour)
    return out


def load_alpaca(limit: int = 4000) -> List[str]:
    """Alpaca instructions with no `input` field — self-contained instructions."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = []
    for row in ds:
        if row.get("input", "").strip():
            continue
        instruction = row["instruction"].strip()
        if instruction:
            out.append(instruction)
        if len(out) >= limit:
            break
    return out


def build_raw(n_harmful: int, n_harmless: int, seed: int = 0) -> Path:
    import random

    set_seed(seed)
    rng = random.Random(seed)

    harmful: List[Tuple[str, str]] = []
    for name, loader in (
        ("advbench", load_advbench),
        ("malicious_instruct", load_malicious_instruct),
        ("harmbench", load_harmbench),
    ):
        try:
            items = loader()
            harmful += [(p, name) for p in items]
            print(f"[build] {name}: {len(items)} prompts")
        except Exception as exc:  # a dead mirror must not kill the whole build
            print(f"[build] WARNING: {name} failed ({exc}); continuing without it")

    if not harmful:
        raise RuntimeError("No harmful prompts loaded from any source.")

    # The harmless pool is deliberately larger than the harmful one: pairing is
    # what loses prompts, not the filter. Measured on the built sets, an 800/800
    # pool pairs only 563 of 800 harmful prompts within the +/-2 token window,
    # while 1500+ harmless prompts pairs all 800.
    harmless = [(p, "alpaca") for p in load_alpaca(limit=max(4000, 3 * n_harmless))]
    print(f"[build] alpaca: {len(harmless)} prompts")

    # De-duplicate on exact text, preserving first-seen source.
    def dedupe(pairs):
        seen, out = set(), []
        for text, src in pairs:
            if text not in seen:
                seen.add(text)
                out.append((text, src))
        return out

    harmful, harmless = dedupe(harmful), dedupe(harmless)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    harmful = harmful[:n_harmful]
    harmless = harmless[:n_harmless]

    # Chat templating needs the tokenizer but not the model weights — this
    # downloads ~10MB of tokenizer files, no GPU involved.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(M0_NAME)

    rows = []
    for idx, (text, src) in enumerate(harmful + harmless):
        label = "harmful" if idx < len(harmful) else "harmless"
        chat = format_chat(tokenizer, text)
        rows.append(
            {
                "id": f"{label}_{idx:05d}",
                "prompt_text": text,
                "chat_formatted": chat,
                "n_prompt_tokens": len(tokenizer(chat, add_special_tokens=False).input_ids),
                "label": label,
                "source": src,
                "source_url": SOURCES.get(src, ""),
            }
        )

    write_jsonl(rows, RAW_PATH)
    print(
        f"[build] wrote {len(rows)} rows to {RAW_PATH} "
        f"({len(harmful)} harmful / {len(harmless)} harmless)"
    )
    return RAW_PATH


# ==========================================================================
# --filter : M0 behavioural filter + pairing + splits  (GPU)
# ==========================================================================

def _generate_batch(model, tokenizer, chats: List[str], max_new_tokens: int, device: str):
    """Greedy decode. Returns (completion_texts, first_generated_token_ids)."""
    import torch

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # required for correct batched generation
    enc = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    tokenizer.padding_side = prev_side

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

    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    first_ids = [row[:TARGET_N_TOKENS].tolist() for row in gen]
    return texts, first_ids


def run_filter(
    model_path: str,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    allow_short_splits: bool,
) -> None:
    from tqdm import tqdm

    set_seed(seed)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} missing — run `--build` first.")

    rows = read_jsonl(RAW_PATH)
    model, tokenizer = load_model_and_tokenizer(model_path)
    device = model.device

    kept: List[Dict] = []
    n_seen = {"harmful": 0, "harmless": 0}
    n_kept = {"harmful": 0, "harmless": 0}

    for start in tqdm(range(0, len(rows), batch_size), desc="M0 behavioural filter"):
        batch = rows[start:start + batch_size]
        texts, first_ids = _generate_batch(
            model, tokenizer, [r["chat_formatted"] for r in batch], max_new_tokens, device
        )
        for row, completion, target_ids in zip(batch, texts, first_ids):
            n_seen[row["label"]] += 1
            refused = is_refusal(completion)
            # A pair is only a valid counterfactual if the behaviours differ:
            # keep harmful prompts that are refused, harmless ones that are not.
            keep = refused if row["label"] == "harmful" else (not refused)
            if not keep:
                continue
            n_kept[row["label"]] += 1
            kept.append(
                {
                    **row,
                    "completion": completion,
                    "refused": refused,
                    "target_token_ids": target_ids,
                    "target_text": tokenizer.decode(target_ids),
                }
            )

    harmful = [r for r in kept if r["label"] == "harmful"]
    harmless = [r for r in kept if r["label"] == "harmless"]
    pairs = length_match(harmful, harmless, tolerance=LENGTH_TOLERANCE)

    print(f"[filter] kept {len(harmful)} harmful, {len(harmless)} harmless -> {len(pairs)} pairs")

    if len(pairs) < N_PAIRS_REQUIRED and not allow_short_splits:
        raise RuntimeError(
            f"Only {len(pairs)} pairs survived the filter but prereg.md §3 registers "
            f"{N_PAIRS_REQUIRED} (200/50/150). Either widen the source pull "
            f"(--n-harmful / --n-harmless) or re-run with --allow-short-splits, "
            f"which scales the splits proportionally and REQUIRES a prereg.md §9 "
            f"deviation entry."
        )

    write_jsonl(pairs, PAIRS_PATH)
    splits = make_splits(len(pairs), seed=seed)
    with SPLITS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)

    write_report(rows, kept, pairs, splits, n_seen, n_kept, model_path, max_new_tokens)
    save_json(
        {
            "model": model_path,
            "n_prompts_scored": len(rows),
            "n_seen": n_seen,
            "n_kept": n_kept,
            "n_pairs": len(pairs),
            "splits": {k: len(v) for k, v in splits.items()},
            "judge_max_new_tokens": max_new_tokens,
            "seed": seed,
        },
        "stage1_filter",
    )


def length_match(
    harmful: List[Dict], harmless: List[Dict], tolerance: int = LENGTH_TOLERANCE
) -> List[Dict]:
    """Greedily pair each harmful prompt with an unused harmless prompt of
    similar token length (prereg §3, +/- 2 tokens).

    Length matching keeps the DBM patch position and the direction's
    last-instruction-token position comparable across the two runs without any
    padding tricks.
    """
    harmless_sorted = sorted(harmless, key=lambda r: r["n_prompt_tokens"])
    used = set()
    pairs = []

    for h in sorted(harmful, key=lambda r: r["n_prompt_tokens"]):
        best, best_gap = None, None
        for i, c in enumerate(harmless_sorted):
            if i in used:
                continue
            gap = abs(c["n_prompt_tokens"] - h["n_prompt_tokens"])
            if gap > tolerance:
                # sorted by length: once we are past the window on the high side, stop
                if c["n_prompt_tokens"] > h["n_prompt_tokens"] + tolerance:
                    break
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = i, gap
                if gap == 0:
                    break
        if best is None:
            continue
        used.add(best)
        c = harmless_sorted[best]
        pairs.append(
            {
                "pair_id": f"pair_{len(pairs):05d}",
                "harmful_id": h["id"],
                "harmless_id": c["id"],
                "harmful_prompt": h["prompt_text"],
                "harmless_prompt": c["prompt_text"],
                "harmful_chat": h["chat_formatted"],
                "harmless_chat": c["chat_formatted"],
                "harmful_n_tokens": h["n_prompt_tokens"],
                "harmless_n_tokens": c["n_prompt_tokens"],
                "refusal_target_ids": h["target_token_ids"],
                "refusal_target_text": h["target_text"],
                "compliance_target_ids": c["target_token_ids"],
                "compliance_target_text": c["target_text"],
                "harmful_source": h["source"],
                "harmless_source": c["source"],
            }
        )
    return pairs


def make_splits(n_pairs: int, seed: int = 0) -> Dict[str, List[int]]:
    """Fixed seed-0 split over pair indices. Scales down proportionally if the
    filter yielded fewer than the registered 400 pairs."""
    import random

    rng = random.Random(seed)
    idx = list(range(n_pairs))
    rng.shuffle(idx)

    if n_pairs >= N_PAIRS_REQUIRED:
        n_tr, n_va = N_TRAIN, N_VAL
    else:
        scale = n_pairs / N_PAIRS_REQUIRED
        n_tr, n_va = int(N_TRAIN * scale), int(N_VAL * scale)

    return {
        "train": idx[:n_tr],
        "val": idx[n_tr:n_tr + n_va],
        "test": idx[n_tr + n_va:],
    }


def write_report(rows, kept, pairs, splits, n_seen, n_kept, model_path, max_new_tokens) -> None:
    from collections import Counter

    src_counts = Counter(r["harmful_source"] for r in pairs)
    lines = [
        "# Stage 1 filter report",
        "",
        f"- Judge model (M0): `{model_path}`",
        f"- Decoding: greedy, {max_new_tokens} new tokens",
        f"- Judge: Arditi/JailbreakBench substring matcher, {len(REFUSAL_SUBSTRINGS)} prefixes, frozen in `refusal/common.py`",
        "",
        "## Yield",
        "",
        "| Class | Scored | Kept | Yield |",
        "|---|---|---|---|",
    ]
    for label in ("harmful", "harmless"):
        seen, k = n_seen[label], n_kept[label]
        pct = f"{100 * k / seen:.1f}%" if seen else "n/a"
        lines.append(f"| {label} | {seen} | {k} | {pct} |")

    lines += [
        "",
        f"Pairs after length matching (+/-{LENGTH_TOLERANCE} tokens): **{len(pairs)}**",
        "",
        "## Splits (seed 0)",
        "",
        "| Split | Pairs | Use |",
        "|---|---|---|",
        f"| train | {len(splits['train'])} | direction fitting, mask learning |",
        f"| val | {len(splits['val'])} | layer / lambda / threshold selection |",
        f"| test | {len(splits['test'])} | reported numbers only |",
        "",
        "## Harmful source mix (paired prompts only)",
        "",
    ]
    for src, count in src_counts.most_common():
        lines.append(f"- {src}: {count} ({SOURCES.get(src, '')})")

    if len(pairs) < N_PAIRS_REQUIRED:
        lines += [
            "",
            "> **DEVIATION:** fewer pairs than the 200/50/150 registered in "
            "prereg.md §3. Splits were scaled proportionally. Log this in prereg.md §9.",
        ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[filter] wrote {REPORT_PATH}")


# ==========================================================================

def load_pairs(split: Optional[str] = None) -> List[Dict]:
    """Load pairs, optionally restricted to one registered split."""
    pairs = read_jsonl(PAIRS_PATH)
    if split is None:
        return pairs
    with SPLITS_PATH.open(encoding="utf-8") as fh:
        splits = json.load(fh)
    if split not in splits:
        raise KeyError(f"Unknown split '{split}'. Have: {list(splits)}")
    return [pairs[i] for i in splits[split]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="CPU: download + chat-template")
    ap.add_argument("--filter", action="store_true", help="GPU: M0 behavioural filter + splits")
    ap.add_argument("--model", default=M0_NAME)
    ap.add_argument("--n-harmful", type=int, default=800)
    ap.add_argument("--n-harmless", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=JUDGE_MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-short-splits", action="store_true")
    args = ap.parse_args()

    if not (args.build or args.filter):
        ap.error("pick --build (CPU) or --filter (GPU)")
    if args.build:
        build_raw(args.n_harmful, args.n_harmless, seed=args.seed)
    if args.filter:
        run_filter(
            args.model,
            args.batch_size,
            args.max_new_tokens,
            args.seed,
            args.allow_short_splits,
        )


if __name__ == "__main__":
    main()
