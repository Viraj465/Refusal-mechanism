"""
Shared pieces for the readout-sensitivity experiments (LW_REVIEW.md, prereg D15).

THE PROBLEM THESE EXIST FOR
The registered causal readout (§5.2) projects a direction out of every
residual-stream write at every layer and position. Its val trace (D13) is close
to a step function: every layer 19-31 gives the full drop. A step function
cannot say whether a 0.914 rotation of the refusal direction matters, so the
headline "rotated but not weakened" (D11/D12) may be a property of the readout.
The scripts that import this module ask how much rotation, scaling or
single-site restriction the readout can resolve:

  rotation_dose.py      ablate v = cos(t) d_own + sin(t) u, sweep cos(t)
  addition_sweep.py     add c * dir_M0, sweep c, in M0 / M_SFT / M_RL
  graded_ablation.py    alpha-scaled all-writes ablation; L19-only projection
  matched_cosine.py     cos(M0, M_RL) at a common layer (CPU)

Two readouts are recorded for every condition:
  - the registered substring judge on greedy generations (binary, saturating);
  - a graded first-token refusal score: log-odds that the next token is a
    refusal opener, read from the logits at the last instruction token. This
    is Arditi et al.'s "refusal score". It needs no generation and does not
    saturate at 0/1, which is the point.

Nothing here selects anything on the test split; it only measures.
"""

from __future__ import annotations

import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from direction import _refusal_flags, directional_ablation  # noqa: E402,F401
from common import m0_condition, wilson_ci  # noqa: E402,F401

# Arditi et al. use the first token of the model's refusal openers. For Qwen's
# "I'm sorry" / "I cannot" / "As an AI" that is "I" and "As". Ids are looked up
# from the tokenizer, never hard-coded.
REFUSAL_FIRST_TOKENS = ("I", "As")


# ==========================================================================
# Prompt subset (fixed, seeded; identical across checkpoints so curves pair)
# ==========================================================================

def seeded_subset(pairs: List[Dict], n: Optional[int], seed: int = 0) -> List[Dict]:
    """A uniform, seeded, order-preserving subset. n=None or n>=len -> all."""
    if n is None or n >= len(pairs):
        return list(pairs)
    idx = sorted(random.Random(seed).sample(range(len(pairs)), n))
    return [pairs[i] for i in idx]


# ==========================================================================
# Rotation geometry
# ==========================================================================

def _unit(v):
    return v / v.norm().clamp(min=1e-12)


def orthogonal_unit(d, seed: int):
    """Random unit vector orthogonal to d, from a dedicated seeded generator."""
    import torch

    d = _unit(d.float().cpu())
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(d.shape[0], generator=gen, dtype=torch.float32)
    return _unit(u - (u @ d) * d)


def real_rotation_axis(d_own, d_other):
    """Unit axis u orthogonal to d_own such that d_other lies in span(d_own, u).

    Returns (u, c_obs) with d_other_hat = c_obs d_own + sqrt(1-c_obs^2) u, so
    rotated_direction(d_own, u, c_obs) reproduces d_other exactly. Sweeping
    cos along this axis extrapolates the rotation fine-tuning actually made.
    """
    d = _unit(d_own.float().cpu())
    o = _unit(d_other.float().cpu())
    c_obs = float(d @ o)
    return _unit(o - c_obs * d), c_obs


def rotated_direction(d, u, cos: float):
    """v = cos * d + sin * u, for unit d and unit u orthogonal to d."""
    d = _unit(d.float().cpu())
    sin = math.sqrt(max(0.0, 1.0 - cos * cos))
    return _unit(cos * d + sin * u.float().cpu())


# ==========================================================================
# Single-site residual-stream ablation
# ==========================================================================

@contextmanager
def residual_ablation(model, direction, layers: Sequence[int], alpha: float = 1.0):
    """Project the direction out of the residual stream at `layers` only.

    `layer` uses hidden_states indexing, the same convention as
    `direction.directional_addition`: layer L is the output of decoder block
    L-1, which is exactly where the direction was read from. Applied at all
    positions. Unlike `directional_ablation` (every write, every layer), later
    blocks are free to write the direction back, so this readout can come out
    partial instead of saturating.
    """
    r = direction.to(device=model.device, dtype=model.dtype)
    r = r / r.norm().clamp(min=1e-8)

    def hook(_module, _args, output):
        if isinstance(output, tuple):
            h = output[0]
            return (h - alpha * (h @ r).unsqueeze(-1) * r,) + tuple(output[1:])
        return output - alpha * (output @ r).unsqueeze(-1) * r

    targets = [model.model.embed_tokens if L == 0 else model.model.layers[L - 1] for L in layers]
    handles = [t.register_forward_hook(hook) for t in targets]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def no_intervention():
    yield


# ==========================================================================
# Readouts
# ==========================================================================

def refusal_token_ids(tokenizer) -> List[int]:
    ids = []
    for tok in REFUSAL_FIRST_TOKENS:
        enc = tokenizer(tok, add_special_tokens=False)["input_ids"]
        if enc and enc[0] not in ids:
            ids.append(int(enc[0]))
    if not ids:
        raise ValueError("tokenizer produced no ids for the refusal opener tokens")
    return ids


def log_odds_from_logits(last_logits, token_ids: Sequence[int]):
    """log p(refusal opener) - log(1 - p(refusal opener)), per row, float32."""
    import torch

    logp = torch.log_softmax(last_logits.float(), dim=-1)
    lp = torch.logsumexp(logp[:, list(token_ids)], dim=-1)
    # log(1 - p) computed stably from log p
    return lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-7))


def refusal_scores(model, tokenizer, chats: List[str], token_ids: Sequence[int],
                   batch_size: int = 16) -> List[float]:
    """Graded per-prompt refusal score at the last instruction token."""
    import torch

    out: List[float] = []
    for start in range(0, len(chats), batch_size):
        batch = chats[start:start + batch_size]
        prev = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        tokenizer.padding_side = prev
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        out += log_odds_from_logits(logits, token_ids).cpu().tolist()
    return out


def readout(model, tokenizer, chats: List[str], token_ids: Sequence[int],
            intervention: Callable = no_intervention, generate: bool = True,
            batch_size: int = 16, max_new_tokens: int = 64) -> Dict:
    """Both readouts under one intervention context. Per-prompt values are kept
    so curves can be bootstrapped over prompts afterwards."""
    with intervention():
        scores = refusal_scores(model, tokenizer, chats, token_ids, batch_size)
        flags = (
            _refusal_flags(model, tokenizer, chats, batch_size, max_new_tokens) if generate else None
        )
    res = {
        "score_mean": sum(scores) / len(scores),
        "scores": [round(s, 4) for s in scores],
    }
    if flags is not None:
        res["refusal_rate"] = wilson_ci(sum(flags), len(flags))
        res["flags"] = [bool(f) for f in flags]
    return res


# ==========================================================================
# Curve statistics (pure; CPU-tested)
# ==========================================================================

def crossing(xs: Sequence[float], ys: Sequence[float], level: float) -> float:
    """First x, scanning xs in the given order, at which ys reaches `level`,
    linearly interpolated. nan if it never does. Direction-agnostic: it finds
    the first segment whose endpoints bracket `level`."""
    for i in range(len(xs)):
        if ys[i] == level:
            return float(xs[i])
        if i and (ys[i - 1] - level) * (ys[i] - level) < 0:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Trapezoidal area; xs need not be sorted."""
    pts = sorted(zip(xs, ys))
    return float(sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:])))


def bootstrap(stat: Callable[[List[int]], float], n: int, n_boot: int = 1000,
              seed: int = 0) -> Dict:
    """Percentile bootstrap over prompt indices. `stat(idx)` recomputes the
    statistic on a resample of prompt indices; nan resamples are counted,
    not silently dropped."""
    rng = random.Random(seed)
    vals, n_nan = [], 0
    for _ in range(n_boot):
        v = stat([rng.randrange(n) for _ in range(n)])
        if v != v:
            n_nan += 1
        else:
            vals.append(v)
    if not vals:
        return {"low": float("nan"), "high": float("nan"), "n_boot": n_boot, "n_nan": n_nan}
    vals.sort()
    return {
        "low": vals[int(0.025 * len(vals))],
        "high": vals[max(0, int(0.975 * len(vals)) - 1)],
        "n_boot": n_boot,
        "n_nan": n_nan,
    }


def mean_at(values: Sequence[float], idx: List[int]) -> float:
    return sum(values[i] for i in idx) / len(idx)
