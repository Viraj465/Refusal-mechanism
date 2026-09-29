"""Rotation dose-response: how far can the refusal direction rotate before the
registered ablation readout notices? (LW_REVIEW.md experiment 1; prereg D15.)

THE QUESTION
The headline dissociation (D11/D12) is: cos(dir_M0, dir_SFT) = 0.914 is below
the 0.9845 ceiling, yet ablating dir_M0 inside M_SFT removes refusal exactly as
well as M_SFT's own direction (transfer ratio 1.004). That only means something
if a direction at cos 0.914 from the true one *could* have failed. This script
measures it directly: inside one model, ablate

    v(c) = c * d_own + sqrt(1 - c^2) * u,     u unit, u orthogonal to d_own

for a grid of c, and read the transfer ratio drop(v)/drop(d_own) as a function
of c. The same u is used across the whole grid for a given seed, so each seed
is one smooth curve; seeds are independent draws of u.

Two kinds of axis:
  - random: u drawn uniformly on the sphere orthogonal to d_own (--n-seeds).
  - real:   u is the axis that actually carries d_own to --real-axis-to's
            direction (e.g. M_SFT -> M0). At c = c_obs this reproduces the
            registered transfer condition; below it, it extrapolates the
            rotation fine-tuning actually made. A random u is not a model of
            that rotation; the real axis is.

Analytic note, recorded so the curve can be read against it: after projecting
out v, a pure d_own component keeps a fraction 1 - c^2 of its size (0.165 at
c = 0.914). The experiment asks whether the model still refuses with that much
left, compounded over every layer's writes.

Readouts per condition (sensitivity.readout): the registered substring judge on
greedy generations and the graded first-token refusal log-odds. Harmful side
only; D14 already showed the all-writes projection of a random direction leaves
harmless behaviour intact.

    python refusal/rotation_dose.py --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT
    python refusal/rotation_dose.py --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
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
    bootstrap,
    crossing,
    orthogonal_unit,
    readout,
    real_rotation_axis,
    refusal_token_ids,
    rotated_direction,
    m0_condition,
    seeded_subset,
)

DEFAULT_COSINES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3]
OBSERVED_COS = 0.914          # cos(dir_M0, dir_SFT) @ L19, D11
TRANSFER_PRESERVED = 0.8      # prereg §5.2 "causally preserved" threshold


def run(model, tokenizer, chats: List[str], d_own, cosines: Sequence[float], seeds: Sequence[int],
        real_axis=None, generate: bool = True, batch_size: int = 16,
        max_new_tokens: int = 64) -> Dict:
    """Measure every condition. Returns {"baseline", "own", "conditions"}."""
    token_ids = refusal_token_ids(tokenizer)

    def ablate(v):
        return lambda: directional_ablation(model, v, 0)

    def one(v=None):
        kw = dict(generate=generate, batch_size=batch_size, max_new_tokens=max_new_tokens)
        if v is None:
            return readout(model, tokenizer, chats, token_ids, **kw)
        return readout(model, tokenizer, chats, token_ids, ablate(v), **kw)

    print("[rotation] baseline ...")
    baseline = one()
    print("[rotation] own direction (cos 1.0) ...")
    own = one(d_own)

    conditions: List[Dict] = []
    axes = [("random", s, orthogonal_unit(d_own, s)) for s in seeds]
    if real_axis is not None:
        axes.append(("real", None, real_axis))
    for axis, seed, u in axes:
        for c in cosines:
            if c >= 1.0:
                continue  # identical to `own` for every u
            v = rotated_direction(d_own, u, c)
            res = one(v)
            conditions.append({"axis": axis, "seed": seed, "cos": c, **res})
            rate = res.get("refusal_rate", {}).get("point", float("nan"))
            print(f"  {axis:6s} seed={seed} cos={c:.3f}: refusal {rate:.3f}, "
                  f"score {res['score_mean']:+.2f}")
    return {"baseline": baseline, "own": own, "conditions": conditions,
            "refusal_token_ids": token_ids}


# --------------------------------------------------------------------------
# Summary (pure; CPU-tested)
# --------------------------------------------------------------------------

def _per_prompt(res: Dict, readout_name: str) -> Optional[List[float]]:
    if readout_name == "rate":
        return [float(f) for f in res["flags"]] if "flags" in res else None
    return res["scores"]


def _ratio(base, own, cond, idx) -> float:
    b = sum(base[i] for i in idx)
    denom = b - sum(own[i] for i in idx)
    if abs(denom) < 1e-9:
        return float("nan")
    return (b - sum(cond[i] for i in idx)) / denom


def summarise(raw: Dict, mark: float = OBSERVED_COS, threshold: float = TRANSFER_PRESERVED,
              n_boot: int = 1000, seed: int = 0) -> Dict:
    """Transfer-ratio curves, the detection threshold c*, and the D15 verdict."""
    out: Dict = {"mark": mark, "threshold": threshold, "readouts": {}}
    conds = raw["conditions"]
    for name in ("rate", "score"):
        base = _per_prompt(raw["baseline"], name)
        own = _per_prompt(raw["own"], name)
        if base is None or own is None:
            continue
        n = len(base)
        allidx = list(range(n))

        rand = [c for c in conds if c["axis"] == "random"]
        seeds = sorted({c["seed"] for c in rand})
        cosines = sorted({c["cos"] for c in rand} | {1.0}, reverse=True)
        per = {(c["seed"], c["cos"]): _per_prompt(c, name) for c in rand}

        def mean_ratio(c, idx):
            if c >= 1.0:
                return 1.0
            vals = [_ratio(base, own, per[(s, c)], idx) for s in seeds if (s, c) in per]
            return sum(vals) / len(vals) if vals else float("nan")

        curve = []
        for c in cosines:
            seed_vals = [1.0] if c >= 1.0 else [
                _ratio(base, own, per[(s, c)], allidx) for s in seeds if (s, c) in per
            ]
            curve.append({
                "cos": c,
                "residual_fraction_1_minus_c2": 1.0 - c * c,
                "ratio_mean": sum(seed_vals) / len(seed_vals),
                "ratio_min": min(seed_vals),
                "ratio_max": max(seed_vals),
            })

        c_star = crossing(cosines, [p["ratio_mean"] for p in curve], threshold)
        c_star_ci = bootstrap(
            lambda idx: crossing(cosines, [mean_ratio(c, idx) for c in cosines], threshold),
            n, n_boot, seed,
        )
        r: Dict = {"random_curve": curve, "c_star": c_star, "c_star_ci": c_star_ci}

        if any(abs(c - mark) < 1e-9 for c in cosines):
            at = mean_ratio(mark, allidx)
            ci = bootstrap(lambda idx: mean_ratio(mark, idx), n, n_boot, seed)
            seed_vals = [_ratio(base, own, per[(s, mark)], allidx) for s in seeds if (s, mark) in per]
            if ci["low"] >= threshold:
                verdict = "undetectable"   # a random rotation this size passes as 'preserved'
            elif ci["high"] < threshold:
                verdict = "detectable"     # the readout would have flagged it
            else:
                verdict = "ambiguous"
            r["at_mark"] = {"ratio_mean": at, "ci": ci, "seed_min": min(seed_vals),
                            "seed_max": max(seed_vals), "verdict": verdict}

        real = sorted((c for c in conds if c["axis"] == "real"), key=lambda c: -c["cos"])
        if real:
            r["real_curve"] = [
                {"cos": c["cos"], "ratio": _ratio(base, own, _per_prompt(c, name), allidx)}
                for c in real
            ]
            r["real_c_star"] = crossing(
                [1.0] + [p["cos"] for p in r["real_curve"]],
                [1.0] + [p["ratio"] for p in r["real_curve"]],
                threshold,
            )
        out["readouts"][name] = r
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="model being measured; its own direction is d_own")
    ap.add_argument("--layer", type=int, default=19,
                    help="layer of d_own (default 19: common L* of M0/M_SFT, see D13)")
    ap.add_argument("--real-axis-to", default="auto",
                    help="tag whose direction defines the real rotation axis "
                         "('auto': M0 for fine-tunes, M_SFT for M0; 'none' to skip)")
    ap.add_argument("--cosines", type=float, nargs="+", default=DEFAULT_COSINES)
    ap.add_argument("--mark", type=float, default=None,
                    help="cosine to mark and add to the grid so it is measured, not interpolated "
                         f"(default: the measured c_obs of the real axis, else {OBSERVED_COS})")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--no-m0-condition", action="store_true",
                    help="v2: also use test pairs where M0 did not behave as the readout needs (D16)")
    ap.add_argument("--no-generate", action="store_true", help="refusal score only (fast)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    from build_dataset import load_pairs  # noqa: E402

    own_dirs = load_directions(args.tag)
    d_own = own_dirs["unit"][args.layer]

    ref = args.real_axis_to
    if ref == "auto":
        ref = "M_SFT" if args.tag == "M0" else "M0"
    real_axis, c_obs = None, None
    if ref != "none":
        real_axis, c_obs = real_rotation_axis(d_own, load_directions(ref)["unit"][args.layer])
        print(f"[rotation] real axis {args.tag}->{ref} @L{args.layer}: c_obs = {c_obs:.4f}")

    # The mark defaults to the measured c_obs, so the real-axis point at the mark
    # is exactly the registered transfer condition (dir_ref ablated in this model).
    mark = args.mark if args.mark is not None else (round(c_obs, 4) if c_obs else OBSERVED_COS)
    cosines = sorted(set(args.cosines) | {mark}, reverse=True)
    pairs = seeded_subset(
        load_pairs(args.split) if args.no_m0_condition else m0_condition(load_pairs(args.split), "harmful"),
        args.n_prompts, args.subset_seed)
    chats = [p["harmful_chat"] for p in pairs]
    seeds = list(range(args.n_seeds))
    n_cond = 2 + (len(cosines) - (1.0 in cosines)) * (len(seeds) + (real_axis is not None))
    print(f"[rotation] {args.tag}: {len(chats)} harmful prompts from '{args.split}', "
          f"{n_cond} conditions, cosines {cosines}")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    raw = run(model, tokenizer, chats, d_own, cosines, seeds, real_axis,
              not args.no_generate, args.batch_size, args.max_new_tokens)
    summary = summarise(raw, mark, TRANSFER_PRESERVED, args.n_boot, args.seed)

    for name, r in summary["readouts"].items():
        m = r.get("at_mark", {})
        print(f"[rotation] {name}: c* = {r['c_star']:.3f} "
              f"[{r['c_star_ci']['low']:.3f}, {r['c_star_ci']['high']:.3f}]; "
              f"ratio at cos {mark} = {m.get('ratio_mean', float('nan')):.3f} -> "
              f"{m.get('verdict', 'n/a')}")

    save_json({
        "tag": args.tag, "checkpoint": args.checkpoint, "layer": args.layer,
        "split": args.split, "n_prompts": len(chats), "subset_seed": args.subset_seed,
        "m0_conditioned": not args.no_m0_condition,
        "prompt_ids": [p["pair_id"] for p in pairs], "seeds": seeds, "cosines": cosines,
        "real_axis_to": ref, "c_obs": c_obs, "mark": mark, "generate": not args.no_generate,
        "raw": raw, "summary": summary,
    }, f"rotation_dose_{args.tag}", subdir="sensitivity")


if __name__ == "__main__":
    main()
