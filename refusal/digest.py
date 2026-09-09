"""Compact digest of every result written so far. CPU only, read-only.

Prints one short block per registered readout, taking the newest file for each
tag so reruns do not have to be cleaned up. Paste its output into the writeup
notes or into a conversation instead of screenshotting whole JSONs.

  python refusal/digest.py            # numbers only
  python refusal/digest.py --examples # + the seeded random completions
"""

from __future__ import annotations

import argparse
import glob
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TAGS = ["M0", "M_SFT", "M_RL"]


def latest(pattern):
    files = sorted(glob.glob(os.path.join(R, pattern)))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as fh:
        return json.load(fh)


def ci(d):
    if not d:
        return "-"
    return f"{d['point']:.3f} [{d['low']:.3f},{d['high']:.3f}] n={d['n']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("TRAINING")
    for stage in ("sft", "rl"):
        t = latest(f"train/train_{stage}_*.json")
        if not t:
            continue
        extra = ""
        if stage == "rl":
            extra = (f" reward_rate={t.get('rollout_reward_rate')} "
                     f"group={t.get('num_generations')} maxlen={t.get('max_completion_length')}")
        print(f"  {stage.upper():4s} init={t.get('init')} lr={t.get('lr')} "
              f"nts={t.get('science_nts')}{extra}")
    c = latest("train/chain_check_*.json")
    if c:
        print(f"  CHAIN verified={c.get('chain_verified')} "
              f"||RL-SFT||={c.get('||M_RL - M_SFT||')} ||RL-M0||={c.get('||M_RL - M0||')} "
              f"ratio={c.get('ratio_rl_sft_over_rl_m0')}")

    print("=" * 78)
    print("BEHAVIOUR (test split)")
    for tag in TAGS:
        b = latest(f"behaviour/behaviour_{tag}_*.json")
        if not b:
            continue
        bh = b["behaviour"]
        print(f"  {tag:6s} refusal(harmful)={ci(bh['refusal_rate_harmful'])}")
        print(f"         compliance(harmless)={ci(bh['compliance_rate_harmless'])}")
        print(f"         science_nts={b.get('science_nts')}")
        k = b.get("forward_kl")
        if k:
            print(f"         KL_from_{k['reference'].split('/')[-1]}: "
                  f"harmful={k['harmful']['mean']:.4f} harmless={k['harmless']['mean']:.4f}")

    print("=" * 78)
    print("DIRECTION")
    for tag in TAGS:
        f = latest(f"direction/direction_fit_{tag}_*.json")
        if f:
            trace = f.get("selection_trace", [])
            viable = [t["layer"] for t in trace if t.get("viable")]
            full = [t["layer"] for t in trace if t.get("refusal_drop", 0) >= 0.999]
            print(f"  {tag:6s} L*={f.get('selected_layer')} "
                  f"max_sep_layer={f.get('direction_meta', {}).get('layer_of_max_separation')} "
                  f"viable={len(viable)}/{len(trace)} full_drop_layers={full}")
        cg = latest(f"direction/direction_ceiling_{tag}_*.json")
        if cg:
            g = cg["ceiling"]
            print(f"         ceiling@L{g['layer']} = {g['mean']:.4f} "
                  f"[{g['ci_low']:.4f},{g['ci_high']:.4f}] ({g['n_resamples']}x)")
        e = latest(f"direction/direction_eval_{tag}_*.json")
        if e:
            print(f"         baseline_refusal={ci(e['baseline']['refusal_rate_harmful'])}")
            print(f"         ablate_own -> {ci(e['ablate_own']['refusal_rate_harmful'])} "
                  f"(drop {e['own_refusal_drop']:+.3f})")
            tr = e.get("transfer")
            if tr:
                print(f"         transfer from {tr['donor']}: ratio={tr['transfer_ratio']:.3f} "
                      f"drop={tr['transfer_drop']:+.3f} -> {tr['verdict'].upper()}")
            ad = e.get("addition")
            if ad:
                print(f"         addition: harmless refusal "
                      f"{ad['baseline_refusal_on_harmless']['point']:.3f} -> "
                      f"{ad['induced_refusal_on_harmless']['point']:.3f}")
            for k, v in (e.get("cosines") or {}).items():
                print(f"         {k} = {v:+.4f}")

    if args.examples:
        print("=" * 78)
        print("RANDOM COMPLETION SAMPLE (seeded, not curated)")
        for tag in TAGS:
            b = latest(f"behaviour/behaviour_{tag}_*.json")
            ex = (b or {}).get("example_completions")
            if not ex:
                continue
            print(f"--- {tag} (seed {ex.get('seed')}, {ex.get('n_per_side')}/side)")
            for side in ("harmful", "harmless"):
                for row in ex.get(side, [])[:3]:
                    print(f"  [{side} #{row['index']} refusal={row['judged_refusal']}] "
                          f"{row['prompt'][:70]}")
                    print(f"      {row['completion'][:160].strip()}")

    print("=" * 78)


if __name__ == "__main__":
    main()
