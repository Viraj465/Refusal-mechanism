"""
Stage 4 — the checkpoint chain: M0 --SFT--> M_SFT --Dr.GRPO--> M_RL.

Case B of prereg.md §7.1: the prior project's checkpoints are not on disk, so
they are retrained here, with RL chained **from the SFT checkpoint** rather than
from M0. That chaining is the whole point of H2 and is verified numerically by
`--stage chain-check`.

WHY THIS DOES NOT CALL THE PRIOR REPO'S TRAINERS
`training.train_grpo` builds its reward by regex-matching the question back out
of the rendered prompt. Its four patterns look for "Question:", "\nAnswer:",
or the tool-use template. The science prompt produced by
`UnifiedDatasetInterface.from_sciknoweval` is
    "<instructions>\n<question>\n### Answer\n"
which matches none of them, so the ground truth lookup fails, every reward is
0.0, and GRPO trains on a flat reward signal. This module instead passes the
answer through as a dataset column, which TRL forwards to the reward function
as a keyword argument — no regex, no lookup, no silent failure. It also sets
the group size, completion length and beta=0 that the prior config omits.

The normaliser (`UnifiedDatasetInterface`) and the answer checker
(`check_answer_correctness`) are reused unchanged from the prior project.

Usage
  python train_chain.py --stage sft
  python train_chain.py --stage rl  --init ./results/checkpoints/M_SFT
  python train_chain.py --stage chain-check --m0 Qwen/Qwen2.5-3B-Instruct \
      --sft ./results/checkpoints/M_SFT --rl ./results/checkpoints/M_RL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    M0_NAME,
    RESULTS_DIR,
    load_model_and_tokenizer,
    rlrazor_path,
    save_json,
    set_seed,
)

CKPT_DIR = RESULTS_DIR / "checkpoints"


# --------------------------------------------------------------------------

def _prepare_science(n_eval: int, max_train: Optional[int]):
    """Normalised science data with the columns TRL needs."""
    rlrazor_path()
    from science_data import load_science_dataset, normalize

    train_raw, eval_raw = load_science_dataset(seed=0, n_eval=n_eval)
    if max_train:
        train_raw = train_raw.select(range(min(max_train, len(train_raw))))
    return normalize(train_raw), normalize(eval_raw), train_raw, eval_raw


def _push(path: Path, repo: Optional[str]) -> None:
    """Get the checkpoint off the rented box immediately (NEXT_STEPS.md §1)."""
    if not repo:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(path), repo_id=repo, repo_type="model")
    print(f"[push] {path} -> https://huggingface.co/{repo}")


# --------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------

def run_sft(args) -> Path:
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    train_ds, eval_ds, _, eval_raw = _prepare_science(args.n_eval, args.max_train)

    # TRL's completion_only_loss needs explicit prompt/completion columns; the
    # normaliser gives 'prompt' and 'answer'.
    train_ds = train_ds.map(lambda ex: {"completion": ex["answer"]})
    keep = {"prompt", "completion"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])

    model, tokenizer = load_model_and_tokenizer(args.init or M0_NAME, eval_mode=False)
    out_dir = CKPT_DIR / args.out_name

    cfg = SFTConfig(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim=args.optim,
        gradient_checkpointing=True,
        completion_only_loss=True,
        logging_steps=10,
        save_strategy="no",     # we save once, explicitly, below
        report_to="none",
        seed=args.seed,
    )
    print(
        f"[sft] effective batch = {args.per_device_batch_size * args.grad_accum}, "
        f"{len(train_ds)} examples, {args.epochs} epochs, lr {args.lr}"
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=train_ds, processing_class=tokenizer
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[sft] saved {out_dir}")
    _push(out_dir, args.push_to)

    nts = _score_nts(trainer.model, tokenizer, eval_ds, args.n_eval, args.nts_max_new_tokens)
    save_json(
        {
            "stage": "sft", "init": args.init or M0_NAME, "out": str(out_dir),
            "lr": args.lr, "epochs": args.epochs,
            "effective_batch": args.per_device_batch_size * args.grad_accum,
            "n_train": len(train_ds), "science_nts": nts, "seed": args.seed,
        },
        "train_sft",
        subdir="train",
    )
    print(f"[sft] science NTS = {nts:.1f}%  (gate: >= {args.nts_gate})")
    if nts < args.nts_gate:
        print(
            f"[sft] WARNING: NTS below the {args.nts_gate}% gate. SFT did not learn the "
            "target task; H1 is untestable in this state. Retune before spending GPU on RL."
        )
    return out_dir


# --------------------------------------------------------------------------
# Dr.GRPO from the SFT checkpoint
# --------------------------------------------------------------------------

def run_rl(args) -> Path:
    from trl import GRPOConfig, GRPOTrainer

    rlrazor_path()
    from training.training import check_answer_correctness  # noqa: E402

    set_seed(args.seed)
    if not args.init:
        raise SystemExit(
            "--init is required for the RL stage: RL must be chained from M_SFT, "
            "not from M0 (prereg.md §2). Pass the SFT checkpoint directory."
        )

    train_ds, eval_ds, _, _ = _prepare_science(args.n_eval, args.max_train)
    keep = {"prompt", "answer"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])

    stats = {"correct": 0, "total": 0}

    def reward_fn(completions, answer, **kwargs) -> List[float]:
        """Binary outcome supervision.

        `answer` arrives as a per-completion list because TRL forwards every
        remaining dataset column to the reward function, repeated across the
        generation group. No prompt parsing, so no silent lookup failure.
        """
        rewards = []
        for completion, gt in zip(completions, answer):
            ok = bool(check_answer_correctness(completion, gt))
            stats["total"] += 1
            stats["correct"] += int(ok)
            rewards.append(1.0 if ok else 0.0)
        return rewards

    model, tokenizer = load_model_and_tokenizer(args.init, eval_mode=False)
    out_dir = CKPT_DIR / args.out_name

    cfg = GRPOConfig(
        output_dir=str(out_dir / "_trainer"),
        loss_type="dr-grpo",          # Dr.GRPO: no std normalisation of advantages
        beta=0.0,                     # no explicit KL penalty (prior work's setting)
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        max_grad_norm=1.0,
        bf16=True,
        optim=args.optim,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    print(
        f"[rl] from {args.init} | group {args.num_generations} | "
        f"max_completion {args.max_completion_length} | {len(train_ds)} prompts"
    )

    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=train_ds,
        processing_class=tokenizer, reward_funcs=reward_fn,
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[rl] saved {out_dir}")
    _push(out_dir, args.push_to)

    train_acc = stats["correct"] / stats["total"] if stats["total"] else float("nan")
    nts = _score_nts(trainer.model, tokenizer, eval_ds, args.n_eval, args.nts_max_new_tokens)
    save_json(
        {
            "stage": "rl", "init": args.init, "out": str(out_dir),
            "loss_type": "dr-grpo", "beta": 0.0,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "lr": args.lr, "n_prompts": len(train_ds),
            "rollout_reward_rate": train_acc, "science_nts": nts, "seed": args.seed,
        },
        "train_rl",
        subdir="train",
    )
    print(f"[rl] rollout reward rate = {train_acc:.3f} | science NTS = {nts:.1f}%")
    if stats["total"] and stats["correct"] == 0:
        print(
            "[rl] WARNING: zero reward across every rollout. The policy learned "
            "nothing. Check the reward function against a few completions before "
            "trusting this checkpoint."
        )
    return out_dir


def _score_nts(model, tokenizer, eval_ds, n_eval: int, max_new_tokens: int) -> float:
    rlrazor_path()
    from evaluation.evaluation import evaluate_new_task  # noqa: E402

    model.eval()
    return float(
        evaluate_new_task(
            model=model, tokenizer=tokenizer, dataset=eval_ds, eval_dataset=eval_ds,
            max_new_tokens=max_new_tokens, num_samples=n_eval,
        )
    )


# --------------------------------------------------------------------------
# Chain check (prereg §7.1)
# --------------------------------------------------------------------------

def run_chain_check(args) -> None:
    """Verify RL was chained from SFT: ||th_RL - th_SFT|| << ||th_RL - th_M0||."""
    import torch
    from transformers import AutoModelForCausalLM

    def state_dict(path):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map="cpu")
        sd = {k: v.detach() for k, v in m.state_dict().items()}
        del m
        return sd

    print("[chain] loading M0 ...")
    sd0 = state_dict(args.m0)
    print("[chain] loading M_SFT ...")
    sd_sft = state_dict(args.sft)

    def l2(a, b):
        total = 0.0
        for k in a:
            if k in b and a[k].shape == b[k].shape and a[k].is_floating_point():
                total += (a[k] - b[k]).pow(2).sum().item()
        return total ** 0.5

    d_sft_m0 = l2(sd_sft, sd0)
    out = {"||M_SFT - M0||": d_sft_m0}

    if args.rl:
        print("[chain] loading M_RL ...")
        sd_rl = state_dict(args.rl)
        d_rl_sft = l2(sd_rl, sd_sft)
        d_rl_m0 = l2(sd_rl, sd0)
        out.update(
            {
                "||M_RL - M_SFT||": d_rl_sft,
                "||M_RL - M0||": d_rl_m0,
                "ratio_rl_sft_over_rl_m0": d_rl_sft / d_rl_m0 if d_rl_m0 else float("nan"),
                "chain_verified": bool(d_rl_sft < d_rl_m0),
            }
        )

    for k, v in out.items():
        print(f"  {k} = {v}")
    if args.rl and not out["chain_verified"]:
        print(
            "  -> RL is NOT closer to SFT than to M0. The RL-from-SFT design did not "
            "hold; H2 must be reworded per prereg.md §2 and logged in §9."
        )
    save_json({"m0": args.m0, "sft": args.sft, "rl": args.rl, **out}, "chain_check", subdir="train")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["sft", "rl", "chain-check"])
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--push-to", default=None, help="private HF repo id, e.g. user/mats-m-sft")

    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--optim", default="adamw_bnb_8bit",
                    help="adamw_bnb_8bit keeps a full 3B fine-tune inside 40GB")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--nts-gate", type=float, default=60.0)
    ap.add_argument("--nts-max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--num-generations", type=int, default=16)
    ap.add_argument("--max-completion-length", type=int, default=256)

    ap.add_argument("--m0", default=M0_NAME)
    ap.add_argument("--sft", default=None)
    ap.add_argument("--rl", default=None)
    args = ap.parse_args()

    # Stage defaults from the implementation plan.
    if args.stage == "sft":
        args.lr = args.lr or 3e-5
        args.epochs = args.epochs or 2
        args.out_name = args.out_name or "M_SFT"
        run_sft(args)
    elif args.stage == "rl":
        args.lr = args.lr or 2e-5
        args.epochs = args.epochs or 1
        args.max_train = args.max_train or 600
        args.out_name = args.out_name or "M_RL"
        run_rl(args)
    else:
        if not args.sft:
            raise SystemExit("--sft is required for chain-check")
        run_chain_check(args)


if __name__ == "__main__":
    main()
