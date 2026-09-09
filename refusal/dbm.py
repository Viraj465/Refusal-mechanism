"""
Stage 5 — head-level refusal mechanism via a differentiable binary mask (DBM).

Triplet (prereg.md §2, head-level operationalisation):
    x_base   = harmful prompt  (M0 refuses)
    x_source = paired harmless prompt (M0 complies)
    y_target = the compliance target, first 3 tokens of the harmless completion

Mask semantics, stated explicitly because the prior repo's implementation used
the mirrored convention:
    a_tilde_h = (1 - m_h) * a_base_h  +  m_h * a_source_h
so m_h = 1 patches the *harmless* activation in. The circuit is
{h : m_h > 0.5} after annealing — the heads whose harmless activations, patched
into the harmful run, flip the model toward compliance.

Loss:
    -log P_geo(y_target | x_base, a_tilde)  +  lambda * sum_h m_h
with P_geo the geometric mean over the target tokens, i.e. the mean NLL.

Implementation choices, all recorded:
  - Patching site is the INPUT to o_proj, which is the concatenated per-head
    output before the output projection mixes heads. (This site is taken from
    the prior project's discovery.py; the masking, annealing, caching and loss
    below are new — the prior implementation ran everything under no_grad and
    could not be optimised.)
  - Patch POSITION is the last instruction token only (the final token of the
    chat template). Pairs are length-matched so this is a like-for-like site.
  - Source activations are cached ONCE per pair. Recomputing them each step is
    the bottleneck that makes 12 mask fits infeasible in one GPU session.
  - Annealing: m = sigmoid(logits / tau), tau 1.0 -> 0.05 over the run, then a
    hard threshold at 0.5.

Usage
  python dbm.py --mode pilot --checkpoint <path> --tag M0            # val, 1 seed
  python dbm.py --mode fit   --checkpoint <path> --tag M0 --seed 0
  python dbm.py --mode fit   --checkpoint <path> --tag M0 --half 0   # split-half
  python dbm.py --mode sweep --checkpoint <path> --tag M0            # lambda on val
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from common import RESULTS_DIR, load_model_and_tokenizer, save_json, set_seed  # noqa: E402

MASKS_DIR = RESULTS_DIR / "masks"

# prereg §7.2 pilot gate
PILOT_MIN_SIZE_FRAC = 0.05
PILOT_MAX_SIZE_FRAC = 0.60
PILOT_MIN_FAITHFULNESS = 0.80


# ==========================================================================
# Triplet batching
# ==========================================================================

@dataclass
class TripletBatch:
    """One padded batch of harmful-prompt + compliance-target sequences."""
    input_ids: "object"        # [B, S]
    attention_mask: "object"   # [B, S]
    patch_pos: "object"        # [B]      index of the last instruction token
    target_pos: "object"       # [B, T]   positions of the target tokens
    target_ids: "object"       # [B, T]
    target_mask: "object"      # [B, T]   1 where the target token is real
    pair_idx: "object"         # [B]      index into the cached source acts


def build_batches(tokenizer, pairs: List[Dict], batch_size: int, device) -> List[TripletBatch]:
    import torch

    batches = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        seqs, prompt_lens, targets = [], [], []
        for p in chunk:
            prompt_ids = tokenizer(p["harmful_chat"], add_special_tokens=False).input_ids
            target_ids = list(p["compliance_target_ids"])
            seqs.append(prompt_ids + target_ids)
            prompt_lens.append(len(prompt_ids))
            targets.append(target_ids)

        max_len = max(len(s) for s in seqs)
        max_t = max(len(t) for t in targets)
        pad = tokenizer.pad_token_id

        input_ids = torch.full((len(chunk), max_len), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
        tgt_ids = torch.zeros((len(chunk), max_t), dtype=torch.long)
        tgt_pos = torch.zeros((len(chunk), max_t), dtype=torch.long)
        tgt_mask = torch.zeros((len(chunk), max_t), dtype=torch.float)
        patch_pos = torch.zeros(len(chunk), dtype=torch.long)

        for i, (seq, plen, tgt) in enumerate(zip(seqs, prompt_lens, targets)):
            # Right padding: the patch position is an explicit per-example index,
            # so no alignment tricks are needed.
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[i, :len(seq)] = 1
            patch_pos[i] = plen - 1
            for j, tok in enumerate(tgt):
                tgt_ids[i, j] = tok
                tgt_pos[i, j] = plen + j - 1   # logits position that predicts token j
                tgt_mask[i, j] = 1.0

        batches.append(
            TripletBatch(
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                patch_pos=patch_pos.to(device),
                target_pos=tgt_pos.to(device),
                target_ids=tgt_ids.to(device),
                target_mask=tgt_mask.to(device),
                pair_idx=torch.arange(start, start + len(chunk), device=device),
            )
        )
    return batches


# ==========================================================================
# Source activation cache
# ==========================================================================

def _o_proj_modules(model) -> List:
    return [layer.self_attn.o_proj for layer in model.model.layers]


def cache_source_activations(model, tokenizer, pairs: List[Dict], batch_size: int = 8):
    """Per-head activations of the HARMLESS run at its last instruction token.

    Returns [n_pairs, n_layers, n_heads, head_dim] on the model's device, in the
    model dtype. For 400 pairs on Qwen2.5-3B this is ~59MB in bf16 — cheap, and
    it removes the dominant cost from every subsequent optimisation step.
    """
    import torch

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    out = torch.zeros(
        (len(pairs), n_layers, n_heads, head_dim), dtype=model.dtype, device=model.device
    )

    captured: Dict[int, "torch.Tensor"] = {}

    def make_hook(layer_idx):
        def hook(_module, args):
            x = args[0]  # [B, S, hidden] = concatenated per-head outputs
            b, s, _ = x.shape
            captured[layer_idx] = x.view(b, s, n_heads, head_dim)[:, -1, :, :].detach()
        return hook

    handles = [m.register_forward_pre_hook(make_hook(i)) for i, m in enumerate(_o_proj_modules(model))]
    try:
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            prev = tokenizer.padding_side
            tokenizer.padding_side = "left"  # last real token lands at index -1
            enc = tokenizer(
                [p["harmless_chat"] for p in chunk],
                return_tensors="pt", padding=True, add_special_tokens=False,
            ).to(model.device)
            tokenizer.padding_side = prev

            captured.clear()
            with torch.no_grad():
                model(**enc)
            for layer_idx in range(n_layers):
                out[start:start + len(chunk), layer_idx] = captured[layer_idx]
    finally:
        for h in handles:
            h.remove()

    return out


# ==========================================================================
# Mask + patching hooks
# ==========================================================================

class HeadMask:
    """Continuous mask over all attention heads, annealed toward binary."""

    def __init__(self, n_layers: int, n_heads: int, device, init: float = 0.0, seed: int = 0):
        import torch

        g = torch.Generator(device="cpu").manual_seed(seed)
        # Small random init breaks the symmetry between heads without biasing
        # the run toward either the base or the source activation.
        logits = init + 0.01 * torch.randn(n_layers, n_heads, generator=g)
        self.logits = logits.to(device=device, dtype=torch.float32).requires_grad_(True)
        self.n_layers, self.n_heads = n_layers, n_heads
        self.tau = 1.0

    def values(self):
        import torch

        return torch.sigmoid(self.logits / self.tau)

    def binary(self, threshold: float = 0.5):
        return (self.values() > threshold).detach()

    def circuit(self, threshold: float = 0.5) -> List[Tuple[int, int]]:
        b = self.binary(threshold).cpu()
        return [(l, h) for l in range(self.n_layers) for h in range(self.n_heads) if b[l, h]]


@contextmanager
def patch_with_mask(model, mask_values, source_acts, batch: TripletBatch, override=None):
    """Install the interpolation hooks for one forward pass.

    `mask_values` is [n_layers, n_heads] in [0,1] (differentiable), or `override`
    may supply a fixed binary tensor for faithfulness evaluation.
    """
    import torch

    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    m_all = override if override is not None else mask_values
    src = source_acts[batch.pair_idx]  # [B, n_layers, n_heads, head_dim]
    rows = torch.arange(batch.input_ids.shape[0], device=batch.input_ids.device)

    def make_hook(layer_idx):
        def hook(_module, args):
            x = args[0]
            b, s, hidden = x.shape
            xh = x.view(b, s, n_heads, head_dim).clone()
            base_at = xh[rows, batch.patch_pos]                     # [B, H, D]
            m = m_all[layer_idx].to(x.dtype).view(1, n_heads, 1)    # [1, H, 1]
            src_at = src[:, layer_idx].to(x.dtype)                  # [B, H, D]
            xh[rows, batch.patch_pos] = (1.0 - m) * base_at + m * src_at
            return (xh.view(b, s, hidden),) + args[1:]
        return hook

    handles = [m.register_forward_pre_hook(make_hook(i)) for i, m in enumerate(_o_proj_modules(model))]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def target_logprob(logits, batch: TripletBatch):
    """Mean log-prob of the target tokens = log of their geometric mean.

    Returns [B]; -this is the first loss term.
    """
    import torch

    rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    sel = logits[rows, batch.target_pos]                       # [B, T, V]
    logprobs = torch.log_softmax(sel.float(), dim=-1)
    tok_lp = logprobs.gather(-1, batch.target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
    return (tok_lp * batch.target_mask).sum(-1) / batch.target_mask.sum(-1).clamp(min=1)


# ==========================================================================
# Optimisation
# ==========================================================================

def train_mask(
    model,
    tokenizer,
    pairs: List[Dict],
    lam: float,
    n_steps: int = 300,
    batch_size: int = 8,
    lr: float = 0.1,
    tau_start: float = 1.0,
    tau_end: float = 0.05,
    seed: int = 0,
    source_acts=None,
    log_every: int = 25,
) -> Tuple[HeadMask, Dict]:
    import torch

    set_seed(seed)
    device = model.device
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    for p in model.parameters():
        p.requires_grad_(False)

    if source_acts is None:
        print("[dbm] caching source activations ...")
        source_acts = cache_source_activations(model, tokenizer, pairs, batch_size)

    batches = build_batches(tokenizer, pairs, batch_size, device)
    mask = HeadMask(n_layers, n_heads, device, seed=seed)
    opt = torch.optim.Adam([mask.logits], lr=lr)

    history = []
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        # Geometric annealing keeps the early steps soft and the late steps
        # nearly binary, so the 0.5 threshold at the end is not arbitrary.
        mask.tau = tau_start * (tau_end / tau_start) ** frac

        batch = batches[step % len(batches)]
        m = mask.values()
        with patch_with_mask(model, m, source_acts, batch):
            logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits

        nll = -target_logprob(logits, batch).mean()
        sparsity = m.sum()
        loss = nll + lam * sparsity

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            n_on = int((m > 0.5).sum().item())
            rec = {
                "step": step, "tau": mask.tau, "loss": float(loss.item()),
                "nll": float(nll.item()), "mask_sum": float(sparsity.item()), "n_heads_on": n_on,
            }
            history.append(rec)
            print(
                f"  step {step:4d}  tau {mask.tau:.3f}  loss {rec['loss']:.4f}  "
                f"nll {rec['nll']:.4f}  |C| {n_on}"
            )

    return mask, {"history": history, "lambda": lam, "n_steps": n_steps, "seed": seed}


# ==========================================================================
# Faithfulness
# ==========================================================================

def evaluate_circuit(
    model, tokenizer, pairs: List[Dict], circuit: List[Tuple[int, int]],
    batch_size: int = 8, source_acts=None,
) -> Dict:
    """F(C|M) / F(M) under the counterfactual.

    F(M)   = target probability with EVERY head patched from the source run
    F(C|M) = target probability with only the circuit heads patched
    base   = target probability with nothing patched (the harmful run itself)

    Reported both as the plain ratio (the quantity prereg §7.2 thresholds at
    0.8) and as normalised recovery (C - base) / (full - base), which is the
    more honest number when the base probability is already high.
    """
    import torch

    device = model.device
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    if source_acts is None:
        source_acts = cache_source_activations(model, tokenizer, pairs, batch_size)
    batches = build_batches(tokenizer, pairs, batch_size, device)

    zeros = torch.zeros(n_layers, n_heads, device=device)
    ones = torch.ones(n_layers, n_heads, device=device)
    circ = torch.zeros(n_layers, n_heads, device=device)
    for l, h in circuit:
        circ[l, h] = 1.0

    sums = {"base": 0.0, "full": 0.0, "circuit": 0.0}
    top1 = {"base": 0, "full": 0, "circuit": 0}
    n = 0

    with torch.no_grad():
        for batch in batches:
            bs = batch.input_ids.shape[0]
            n += bs
            for name, m in (("base", zeros), ("full", ones), ("circuit", circ)):
                with patch_with_mask(model, None, source_acts, batch, override=m):
                    logits = model(
                        input_ids=batch.input_ids, attention_mask=batch.attention_mask
                    ).logits
                lp = target_logprob(logits, batch)          # [B] mean log-prob
                sums[name] += lp.exp().sum().item()          # geometric-mean prob
                rows = torch.arange(bs, device=device)
                first_pred = logits[rows, batch.target_pos[:, 0]].argmax(-1)
                top1[name] += (first_pred == batch.target_ids[:, 0]).sum().item()

    p = {k: v / max(1, n) for k, v in sums.items()}
    t = {k: v / max(1, n) for k, v in top1.items()}
    denom = p["full"] - p["base"]
    return {
        "n_examples": n,
        "circuit_size": len(circuit),
        "p_geo": p,
        "top1_rate": t,
        "faithfulness": p["circuit"] / p["full"] if p["full"] > 0 else float("nan"),
        "normalised_recovery": (p["circuit"] - p["base"]) / denom if abs(denom) > 1e-9 else float("nan"),
        "top1_faithfulness": t["circuit"] / t["full"] if t["full"] > 0 else float("nan"),
    }


# ==========================================================================
# Persistence
# ==========================================================================

def save_mask(tag: str, mask: HeadMask, meta: Dict, name: Optional[str] = None) -> Path:
    import torch

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    name = name or f"{tag}_seed{meta.get('seed', 0)}_lam{meta.get('lambda')}"
    path = MASKS_DIR / f"{name}.pt"
    torch.save(
        {
            "tag": tag,
            "logits": mask.logits.detach().cpu(),
            "values": mask.values().detach().cpu(),   # continuous — needed for matched-k
            "binary": mask.binary().cpu(),
            "circuit": mask.circuit(),
            "meta": meta,
        },
        path,
    )
    print(f"[dbm] saved {path}")
    return path


# ==========================================================================

def _select_pairs(split: str, half: Optional[int], seed: int) -> List[Dict]:
    """Load a split; `half` selects a disjoint half for the split-half ceiling."""
    import random

    from build_dataset import load_pairs

    pairs = load_pairs(split)
    if half is None:
        return pairs
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    cut = len(idx) // 2
    chosen = idx[:cut] if half == 0 else idx[cut:2 * cut]
    return [pairs[i] for i in chosen]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["pilot", "fit", "sweep"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default=None, help="default: val for pilot/sweep, train for fit")
    ap.add_argument("--half", type=int, default=None, choices=[0, 1],
                    help="use a disjoint half of the split (split-half ceiling)")
    ap.add_argument("--lam", type=float, default=0.03)
    ap.add_argument("--lam-grid", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    split = args.split or ("train" if args.mode == "fit" else "val")
    set_seed(args.seed)

    pairs = _select_pairs(split, args.half, args.seed)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"[dbm] {args.tag} | mode={args.mode} split={split} half={args.half} pairs={len(pairs)}")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    n_total_heads = model.config.num_hidden_layers * model.config.num_attention_heads

    print("[dbm] caching source activations once for the whole run ...")
    source_acts = cache_source_activations(model, tokenizer, pairs, args.batch_size)

    payload: Dict = {
        "tag": args.tag, "checkpoint": args.checkpoint, "mode": args.mode,
        "split": split, "half": args.half, "n_pairs": len(pairs), "seed": args.seed,
        "n_total_heads": n_total_heads,
    }

    # ---------------- lambda sweep (val only) ----------------
    if args.mode == "sweep":
        sweep = []
        for lam in args.lam_grid:
            print(f"\n[sweep] lambda = {lam}")
            mask, meta = train_mask(
                model, tokenizer, pairs, lam, args.n_steps, args.batch_size, args.lr,
                seed=args.seed, source_acts=source_acts,
            )
            circuit = mask.circuit()
            faith = evaluate_circuit(
                model, tokenizer, pairs, circuit, args.batch_size, source_acts
            )
            sweep.append({"lambda": lam, "circuit_size": len(circuit), **faith})
            print(f"[sweep] lambda={lam}: |C|={len(circuit)} faithfulness={faith['faithfulness']:.3f}")
        payload["sweep"] = sweep
        # prereg: smallest lambda whose circuit faithfulness >= 0.8, same lambda
        # for all three checkpoints.
        viable = [s for s in sweep if s["faithfulness"] >= PILOT_MIN_FAITHFULNESS]
        payload["recommended_lambda"] = max(
            (s["lambda"] for s in viable), default=None
        )
        print(f"[sweep] recommended lambda = {payload['recommended_lambda']} "
              "(largest lambda still meeting faithfulness >= 0.8, i.e. sparsest viable)")
        save_json(payload, f"dbm_sweep_{args.tag}", subdir="dbm")
        return

    # ---------------- pilot / fit ----------------
    mask, meta = train_mask(
        model, tokenizer, pairs, args.lam, args.n_steps, args.batch_size, args.lr,
        seed=args.seed, source_acts=source_acts,
    )
    circuit = mask.circuit()
    faith = evaluate_circuit(model, tokenizer, pairs, circuit, args.batch_size, source_acts)

    payload["train_meta"] = meta
    payload["circuit"] = circuit
    payload["circuit_size"] = len(circuit)
    payload["circuit_frac"] = len(circuit) / n_total_heads
    payload["faithfulness"] = faith

    name = f"{args.tag}_seed{args.seed}_lam{args.lam}"
    if args.half is not None:
        name += f"_half{args.half}"
    save_mask(args.tag, mask, {**meta, "half": args.half, "split": split}, name=name)

    if args.mode == "pilot":
        # prereg §7.2: all four conditions must hold.
        converged = payload["circuit_size"] > 0
        size_ok = PILOT_MIN_SIZE_FRAC <= payload["circuit_frac"] <= PILOT_MAX_SIZE_FRAC
        faith_ok = faith["faithfulness"] >= PILOT_MIN_FAITHFULNESS
        gate = {
            "mask_converged": bool(converged),
            "size_in_range": bool(size_ok),
            "faithfulness_ok": bool(faith_ok),
            "circuit_frac": payload["circuit_frac"],
            "faithfulness": faith["faithfulness"],
            "note": "The fourth condition (ablating the found heads moves refusal "
                    "the predicted way on val) is scored separately by controls.py; "
                    "record it before declaring the gate passed.",
        }
        gate["passed_three_of_four"] = bool(converged and size_ok and faith_ok)
        payload["pilot_gate"] = gate
        print("\n[pilot gate]")
        for k, v in gate.items():
            print(f"  {k}: {v}")
        if not gate["passed_three_of_four"]:
            print(
                "\n  -> prereg §7.2 failure branch: abandon head-level analysis and "
                "submit the direction-level result. This is a registered outcome."
            )

    save_json(payload, f"dbm_{args.mode}_{args.tag}", subdir="dbm")


if __name__ == "__main__":
    main()
