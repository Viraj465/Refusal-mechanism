"""Activation-scale check for E2a (REMAINING_COMMANDS.md step 2).

E2a (prereg D15) found that M_SFT and M_RL need ~20% more of dir_M0 added at
L19 before harmless prompts get refused (c50 0.82 in M0 -> 1.01 / 0.99). One
boring explanation: fine-tuning grew the residual stream at L19, so a vector of
fixed size is simply a smaller push. This measures, per checkpoint, on the same
200 v1 harmless test prompts at the last instruction token:

  - mean ||h|| at L19            (overall activation scale)
  - own diff-in-means norm       (size of the model's own refusal vector)
  - mean projection on dir_M0    (where harmless prompts sit along dir_M0)

Reading it: if M_SFT's mean ||h|| is ~1.2x M0's, scale explains E2a; if it is
~1.0x, it does not. Always v1 data (E2a was run on v1).

    python refusal/activation_scale_check.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

os.environ["REFUSAL_DATA_PROFILE"] = "v1"  # E2a is a v1 result; must be set before `common` is imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from build_dataset import load_pairs  # noqa: E402
from common import M0_NAME, load_model_and_tokenizer, save_json  # noqa: E402
from direction import last_token_residuals, load_directions  # noqa: E402

LAYER = 19
N_PROMPTS = 200
CHECKPOINTS = [
    ("M0", M0_NAME),
    ("M_SFT", "refusal/results/checkpoints/M_SFT"),
    ("M_RL", "refusal/results/checkpoints/M_RL"),
]


def main() -> None:
    import torch

    chats = [p["harmless_chat"] for p in load_pairs("test")[:N_PROMPTS]]
    u = load_directions("M0")["unit"][LAYER].float()
    rows = {}
    for tag, ckpt in CHECKPOINTS:
        raw = load_directions(tag)["raw"][LAYER].float()
        model, tokenizer = load_model_and_tokenizer(ckpt)
        h = last_token_residuals(model, tokenizer, chats, 16)[LAYER]   # [n, d_model]
        rows[tag] = {
            "mean_resid_norm": h.norm(dim=-1).mean().item(),
            "own_diff_in_means_norm": raw.norm().item(),
            "mean_proj_on_dir_M0": (h @ u).mean().item(),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    base = rows["M0"]
    print(f"\n[scale] L{LAYER}, {len(chats)} v1 harmless test prompts, last instruction token")
    print(f"{'':6s} {'mean ||h||':>11s} {'x M0':>6s} {'own dim norm':>13s} {'x M0':>6s} {'proj dir_M0':>12s}")
    for tag, r in rows.items():
        r["resid_norm_vs_M0"] = r["mean_resid_norm"] / base["mean_resid_norm"]
        r["own_norm_vs_M0"] = r["own_diff_in_means_norm"] / base["own_diff_in_means_norm"]
        print(f"{tag:6s} {r['mean_resid_norm']:11.3f} {r['resid_norm_vs_M0']:6.3f} "
              f"{r['own_diff_in_means_norm']:13.3f} {r['own_norm_vs_M0']:6.3f} "
              f"{r['mean_proj_on_dir_M0']:+12.3f}")
    print("E2a c50 shift for comparison: M_SFT 1.011/0.817 = 1.24x, M_RL 0.985/0.817 = 1.21x "
          "(rate readout)")
    save_json({"layer": LAYER, "n_prompts": len(chats), "rows": rows},
              "activation_scale", subdir="sensitivity")


if __name__ == "__main__":
    main()
