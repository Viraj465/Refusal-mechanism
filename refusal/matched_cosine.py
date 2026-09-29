"""Matched-layer cosines between fitted refusal directions. CPU only.
(LW_REVIEW.md experiment 3; closes the gap D13/D14 left open.)

The recorded cos(dir_M0, dir_RL) = 0.933 is at layer 21 (M_RL's tie-break L*),
while cos(dir_M0, dir_SFT) = 0.914 is at layer 19. D14 bounded the matched
value at L19 to roughly [0.84, 0.97] from the triangle inequality on angles.
This script reads the saved direction tensors and reports:

  - cos(A, B) at --layer for every pair of fitted checkpoints, with the M0
    split-half ceiling it must be read against;
  - the same cosine at every layer (the rotation profile), so "where does the
    direction move" is answered without a tie-break layer;
  - the angle triangle check for (M0, M_SFT, M_RL);
  - whether RL continued SFT's rotation: cosine between the components of
    dir_SFT and dir_RL orthogonal to dir_M0. Near 1 means RL kept SFT's
    rotation axis; near 0 means it rotated somewhere unrelated.

Needs refusal/results/directions/{M0,M_SFT,M_RL}.pt (written by
`direction.py --mode fit`; gitignored). No model, no GPU.

    python refusal/matched_cosine.py
    python refusal/matched_cosine.py --layer 19 --dir /path/to/directions
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS_DIR, save_json  # noqa: E402
from direction import DIRECTIONS_DIR  # noqa: E402


def _cos(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0))


def latest_ceiling(directory: Path) -> Optional[Dict]:
    files = sorted(directory.glob("direction_ceiling_M0_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8"))["ceiling"] if files else None


def analyse(units: Dict[str, "object"], layer: int, ceiling: Optional[Dict] = None) -> Dict:
    """units: tag -> unit-direction tensor [n_layers+1, d_model]. Pure; CPU-tested."""
    tags = sorted(units)
    n_layers = min(u.shape[0] for u in units.values())
    out: Dict = {"layer": layer, "tags": tags, "pairs": {}, "profile": {}}

    for a, b in itertools.combinations(tags, 2):
        key = f"{a}|{b}"
        c = _cos(units[a][layer], units[b][layer])
        entry: Dict = {"cos": c, "angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, c))))}
        if ceiling and ceiling.get("layer") == layer:
            # prereg §5.2: below the ceiling CI lower bound is 'degraded'
            entry["vs_ceiling"] = "preserved" if c >= ceiling["ci_low"] else "degraded"
        out["pairs"][key] = entry
        out["profile"][key] = [_cos(units[a][L], units[b][L]) for L in range(n_layers)]

    if {"M0", "M_SFT", "M_RL"} <= set(tags):
        ang = {k: out["pairs"][k]["angle_deg"] for k in ("M0|M_RL", "M0|M_SFT", "M_RL|M_SFT")}
        out["triangle"] = {
            **ang,
            # angle(M0,RL) must lie in [|a-b|, a+b] for a = angle(M0,SFT), b = angle(SFT,RL)
            "bound_deg": [abs(ang["M0|M_SFT"] - ang["M_RL|M_SFT"]), ang["M0|M_SFT"] + ang["M_RL|M_SFT"]],
        }
        d0 = units["M0"][layer].float()
        d0 = d0 / d0.norm()

        def perp(v):
            v = v.float() / v.float().norm()
            return v - (v @ d0) * d0

        ps, pr = perp(units["M_SFT"][layer]), perp(units["M_RL"][layer])
        out["rotation_axis_agreement"] = (
            _cos(ps, pr) if ps.norm() > 1e-8 and pr.norm() > 1e-8 else float("nan")
        )
    if ceiling:
        out["ceiling"] = ceiling
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--dir", type=Path, default=DIRECTIONS_DIR)
    args = ap.parse_args()

    paths = sorted(args.dir.glob("*.pt"))
    if not paths:
        sys.exit(f"[matched] no direction tensors in {args.dir}. They are gitignored: recover "
                 "them from the GPU box, or re-run `direction.py --mode fit` per checkpoint.")
    units = {}
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        units[d.get("tag", p.stem)] = d["unit"]
        print(f"[matched] {p.name}: selected layer {d.get('selected_layer')}")

    res = analyse(units, args.layer, latest_ceiling(RESULTS_DIR / "direction"))
    for k, v in res["pairs"].items():
        print(f"  cos({k}) @L{args.layer} = {v['cos']:+.4f} ({v['angle_deg']:.1f} deg)"
              + (f" [{v['vs_ceiling']}]" if "vs_ceiling" in v else ""))
    if "triangle" in res:
        lo, hi = res["triangle"]["bound_deg"]
        print(f"  angle(M0,M_RL) {res['triangle']['M0|M_RL']:.1f} deg, triangle bound [{lo:.1f}, {hi:.1f}]")
        print(f"  RL continued SFT's rotation axis? cos(perp_SFT, perp_RL) = "
              f"{res['rotation_axis_agreement']:+.3f}")
    save_json({"source_dir": str(args.dir), **res}, "matched_cosine", subdir="direction")


if __name__ == "__main__":
    main()
