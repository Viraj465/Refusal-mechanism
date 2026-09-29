"""Addition-coefficient sweep: a graded causal readout of dir_M0 in M0, M_SFT
and M_RL. (LW_REVIEW.md experiment 2a; prereg D15.)

THE QUESTION
Ablation saturates (D13: every layer 19-31 gives the full drop), so it cannot
grade how well dir_M0 still addresses the refusal mechanism after fine-tuning.
Addition does not saturate at the registered coefficient: c = 1 induced
refusal on 46% (M_SFT) and 50% (M_RL) of harmless prompts. But those numbers
have no M0 baseline and are single points. This script sweeps the coefficient
on the same harmless prompts in every checkpoint and compares the curves.

    dir_M0 (raw diff-in-means, unnormalised) * c added at the output of block
    L-1 (hidden_states[L]), all positions, via direction.directional_addition.

The primary summary is the coefficient at which the curve crosses half:
  - rate:  c50 where the judged refusal rate on harmless prompts reaches 0.5
  - score: c0 where the mean first-token refusal log-odds crosses 0
and, with --summarise, the paired-bootstrap difference of those thresholds
between each fine-tune and M0 over the identical prompt subset.

--also-own adds the model's own direction at the same layer, the addition
analogue of the transfer ratio. Raw norms differ between checkpoints and are
recorded; the dir_M0 curves are the like-for-like comparison.

    python refusal/addition_sweep.py --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
    python refusal/addition_sweep.py --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT --also-own
    python refusal/addition_sweep.py --summarise
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from common import RESULTS_DIR, load_model_and_tokenizer, save_json, set_seed  # noqa: E402
from direction import directional_addition, load_directions  # noqa: E402
from sensitivity import (  # noqa: E402
    bootstrap,
    crossing,
    mean_at,
    readout,
    refusal_token_ids,
    m0_condition,
    seeded_subset,
)

DEFAULT_COEFFICIENTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
RATE_LEVEL = 0.5     # half of harmless prompts judged as refusals
SCORE_LEVEL = 0.0    # refusal opener as likely as not


def run(model, tokenizer, chats: List[str], vectors: Dict[str, "object"], layer: int,
        coefficients: Sequence[float], generate: bool = True, batch_size: int = 16,
        max_new_tokens: int = 64) -> Dict:
    """vectors: name -> raw (unnormalised) direction at `layer`."""
    token_ids = refusal_token_ids(tokenizer)
    kw = dict(generate=generate, batch_size=batch_size, max_new_tokens=max_new_tokens)

    print("[addition] coefficient 0 (baseline) ...")
    baseline = readout(model, tokenizer, chats, token_ids, **kw)
    curves: Dict[str, List[Dict]] = {}
    for name, vec in vectors.items():
        curves[name] = []
        for c in coefficients:
            if c == 0:
                res = baseline
            else:
                res = readout(model, tokenizer, chats, token_ids,
                              lambda: directional_addition(model, vec, layer, c), **kw)
            curves[name].append({"coefficient": c, **res})
            rate = res.get("refusal_rate", {}).get("point", float("nan"))
            print(f"  {name:8s} c={c:.2f}: harmless refusal {rate:.3f}, score {res['score_mean']:+.2f}")
    return {"curves": curves, "refusal_token_ids": token_ids}


# --------------------------------------------------------------------------
# Summaries (pure; CPU-tested)
# --------------------------------------------------------------------------

def _series(curve: List[Dict], name: str) -> Optional[List[List[float]]]:
    if name == "rate":
        if any("flags" not in p for p in curve):
            return None
        return [[float(f) for f in p["flags"]] for p in curve]
    return [p["scores"] for p in curve]


def _threshold(curve: List[Dict], name: str, idx: List[int]) -> float:
    xs = [p["coefficient"] for p in curve]
    ys = [mean_at(v, idx) for v in _series(curve, name)]
    return crossing(xs, ys, RATE_LEVEL if name == "rate" else SCORE_LEVEL)


def summarise_run(raw: Dict, n_boot: int = 1000, seed: int = 0) -> Dict:
    out: Dict = {}
    for dname, curve in raw["curves"].items():
        out[dname] = {}
        for name in ("rate", "score"):
            series = _series(curve, name)
            if series is None:
                continue
            n = len(series[0])
            out[dname][name] = {
                "threshold": _threshold(curve, name, list(range(n))),
                "ci": bootstrap(lambda idx: _threshold(curve, name, idx), n, n_boot, seed),
                "curve": [mean_at(v, list(range(n))) for v in series],
                "coefficients": [p["coefficient"] for p in curve],
            }
    return out


def compare(runs: Dict[str, Dict], reference: str = "M0", direction: str = "dir_M0",
            n_boot: int = 1000, seed: int = 0) -> Dict:
    """Paired bootstrap of threshold(X) - threshold(reference) on dir_M0 curves.

    Pairing is over prompts: the same resampled prompt indices are applied to
    both checkpoints, which requires the identical prompt subset (checked)."""
    if reference not in runs:
        raise KeyError(f"no addition sweep for reference '{reference}'")
    ref = runs[reference]
    out: Dict = {}
    for tag, run_ in runs.items():
        if tag == reference:
            continue
        if run_["prompt_ids"] != ref["prompt_ids"]:
            raise ValueError(f"{tag} and {reference} were measured on different prompts; "
                             "the paired comparison is undefined")
        a, b = run_["raw"]["curves"][direction], ref["raw"]["curves"][direction]
        out[tag] = {}
        for name in ("rate", "score"):
            if _series(a, name) is None or _series(b, name) is None:
                continue
            n = len(_series(a, name)[0])
            full = list(range(n))
            diff = _threshold(a, name, full) - _threshold(b, name, full)
            ci = bootstrap(lambda idx: _threshold(a, name, idx) - _threshold(b, name, idx),
                           n, n_boot, seed)
            verdict = ("indistinguishable" if ci["low"] <= 0 <= ci["high"]
                       else "needs_more" if diff > 0 else "needs_less")
            out[tag][name] = {"diff_vs_" + reference: diff, "ci": ci, "verdict": verdict}
    return out


def latest_runs(directory: Path) -> Dict[str, Dict]:
    runs: Dict[str, Dict] = {}
    for p in sorted(directory.glob("addition_sweep_*.json")):   # timestamped -> last wins
        d = json.loads(p.read_text(encoding="utf-8"))
        runs[d["tag"]] = d
    return runs


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--tag")
    ap.add_argument("--donor", default="M0")
    ap.add_argument("--layer", type=int, default=None,
                    help="default: the donor's selected layer (19 for M0), as registered")
    ap.add_argument("--coefficients", type=float, nargs="+", default=DEFAULT_COEFFICIENTS)
    ap.add_argument("--also-own", action="store_true")
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
    ap.add_argument("--summarise", action="store_true",
                    help="CPU: compare the latest sweep per checkpoint against M0")
    args = ap.parse_args()

    if args.summarise:
        runs = latest_runs(RESULTS_DIR / "sensitivity")
        cmp_ = compare(runs, args.donor, f"dir_{args.donor}", args.n_boot, args.seed)
        for tag, r in cmp_.items():
            for name, v in r.items():
                print(f"[addition] {tag} vs {args.donor} ({name}): "
                      f"diff {v['diff_vs_' + args.donor]:+.3f} "
                      f"[{v['ci']['low']:+.3f}, {v['ci']['high']:+.3f}] -> {v['verdict']}")
        save_json({"tags": sorted(runs), "comparison": cmp_}, "addition_compare", subdir="sensitivity")
        return

    if not (args.checkpoint and args.tag):
        ap.error("--checkpoint and --tag are required unless --summarise")

    set_seed(args.seed)
    from build_dataset import load_pairs  # noqa: E402

    donor = load_directions(args.donor)
    layer = args.layer if args.layer is not None else donor["selected_layer"]
    vectors = {f"dir_{args.donor}": donor["raw"][layer]}
    if args.also_own and args.tag != args.donor:
        vectors[f"dir_{args.tag}"] = load_directions(args.tag)["raw"][layer]
    norms = {k: float(v.norm()) for k, v in vectors.items()}

    coefficients = sorted(set(args.coefficients))   # crossing() scans in this order
    pairs = seeded_subset(
        load_pairs(args.split) if args.no_m0_condition else m0_condition(load_pairs(args.split), "harmless"),
        args.n_prompts, args.subset_seed)
    chats = [p["harmless_chat"] for p in pairs]
    print(f"[addition] {args.tag}: {len(chats)} harmless prompts, layer {layer}, "
          f"vectors {norms}, coefficients {coefficients}")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    raw = run(model, tokenizer, chats, vectors, layer, coefficients,
              not args.no_generate, args.batch_size, args.max_new_tokens)
    summary = summarise_run(raw, args.n_boot, args.seed)
    for dname, r in summary.items():
        for name, v in r.items():
            print(f"[addition] {dname} {name} threshold {v['threshold']:.3f} "
                  f"[{v['ci']['low']:.3f}, {v['ci']['high']:.3f}]")

    save_json({
        "tag": args.tag, "checkpoint": args.checkpoint, "donor": args.donor, "layer": layer,
        "split": args.split, "n_prompts": len(chats), "subset_seed": args.subset_seed,
        "m0_conditioned": not args.no_m0_condition,
        "prompt_ids": [p["pair_id"] for p in pairs], "vector_norms": norms,
        "coefficients": coefficients, "generate": not args.no_generate,
        "raw": raw, "summary": summary,
    }, f"addition_sweep_{args.tag}", subdir="sensitivity")


if __name__ == "__main__":
    main()
