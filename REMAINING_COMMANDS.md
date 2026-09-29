# Remaining commands — after the 2026-09-29 session

Every command still to run, in order. Run from the repo root.

- **PC** = your Windows machine (`E:/NeelNandaMATSProgramProject`).
- **Studio** = the Lightning L40S (`/teamspace/studios/this_studio/Refusal-Mechanism`).

The data profile defaults to **v2**. Every v1 command below sets `REFUSAL_DATA_PROFILE=v1`
explicitly; leaving it out silently reads the v2 data and results.

| # | Task | Where | Time | Status |
|---|---|---|---|---|
| 0 | Push the fixed `run_session.sh` | PC | 1 min | ready |
| 1 | Rerun v2 E3 + E1 at L27 (current v2 E1/E3 ran at L19, invalid) | Studio | ~1 h | ready |
| 2 | Activation-scale check for E2a | Studio | ~10 min | ready |
| 3 | E2b for M_RL at its own layer, L21 | Studio | ~20 min | ready |
| 4 | Noise baseline: second SFT seed + its readouts | Studio | ~2.5 h | ready (optional) |
| 5 | Save everything, stop the Studio | Studio | 5 min | ready |
| 6 | Analysis + figures | PC | 10 min | ready |
| 7 | Rescore v2 behaviour with a better judge | — | — | **no script yet** |
| 8 | D17 prereg entry, then the writeup | PC | — | text, not commands |

---

## 0. Push the fixed script (PC)

```bash
cd E:/NeelNandaMATSProgramProject
git add refusal/run_session.sh REMAINING_COMMANDS.md
git commit -m "v2 E1/E3 at v2's selected layer; remaining-commands list"
git push origin main
```

## 1. Rerun v2 E3 + E1 at L27 (Studio, ~1 h)

```bash
cd /teamspace/studios/this_studio/Refusal-Mechanism
git pull origin main

# The done-markers must still be here, or everything reruns.
ls refusal/results/.session_done/ | wc -l                       # ~45
ls refusal/results/.session_done/ | grep -E "v2_(e1|matched)"   # old names only, no *_L27

tmux new -s run
bash refusal/run_session.sh B
# expect: [skip] ... (done) lines, then "[v2] E3/E1 at M0's v2 selected layer L27",
# then only matched_cosine_L27, e1_M_SFT_L27, e1_M0_L27
```

**If the marker count is ~0** (new or reset Studio), don't run the script; run the three steps directly:

```bash
export REFUSAL_DATA_PROFILE=v2
python refusal/matched_cosine.py --layer 27
python refusal/rotation_dose.py --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT --layer 27
python refusal/rotation_dose.py --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0 --layer 27
unset REFUSAL_DATA_PROFILE
```

**Check:** the `[rotation] ... own` drop must be large. Refusal should go from ~0.95+ to near 0
under the model's own direction, unlike the ~0.99 → 0.50 at L19. If it isn't, stop: the ratios
would again rest on a weak own-direction drop.

## 2. Activation-scale check for E2a (Studio, ~10 min)

E2a found that M_SFT and M_RL need ~20% more of dir_M0 added to refuse (c50 0.82 → 1.01 / 0.99).
This tests whether that is just larger activations at L19. **If the M_SFT/M0 ratio of
`mean ||h||` is ≈ 1.2, scale explains E2a; if it is ≈ 1.0, it doesn't.**

```bash
export REFUSAL_DATA_PROFILE=v1
python - <<'PY' 2>&1 | tee refusal/results/sensitivity/activation_scale_check.txt
import gc, sys, torch
sys.path[:0] = ["refusal", "refusal/data"]
from common import load_model_and_tokenizer
from build_dataset import load_pairs
from direction import last_token_residuals, load_directions

L = 19
chats = [p["harmless_chat"] for p in load_pairs("test")[:200]]
u = load_directions("M0")["unit"][L].float()
for tag, ckpt in [("M0", "Qwen/Qwen2.5-3B-Instruct"),
                  ("M_SFT", "refusal/results/checkpoints/M_SFT"),
                  ("M_RL", "refusal/results/checkpoints/M_RL")]:
    raw = load_directions(tag)["raw"][L].float()
    model, tok = load_model_and_tokenizer(ckpt)
    h = last_token_residuals(model, tok, chats, 16)[L]          # [n, d_model], harmless, L19
    print(f"{tag:6s} mean ||h||@L{L} = {h.norm(dim=-1).mean():8.3f}   "
          f"own diff-in-means norm = {raw.norm():7.3f}   "
          f"mean proj on dir_M0 = {(h @ u).mean():+8.3f}")
    del model; gc.collect(); torch.cuda.empty_cache()
PY
unset REFUSAL_DATA_PROFILE
```

## 3. E2b for M_RL at its own selected layer, L21 (Studio, ~20 min)

E2b took M_RL's own direction at L19, but M_RL selects L21. This reruns the α-scaled ablation
with **both** directions taken at L21. If the graded transfer ratio stays > 1 here too, the
"dir_M0 beats the model's own direction" result isn't a layer artefact.

```bash
export REFUSAL_DATA_PROFILE=v1
python refusal/graded_ablation.py --checkpoint refusal/results/checkpoints/M_RL --tag M_RL \
    --variant alpha --layer 21
unset REFUSAL_DATA_PROFILE
```

Output: a new `refusal/results/sensitivity/graded_ablation_alpha_M_RL_*.json` (the file records
`"layer": 21`).

## 4. Noise baseline: a second SFT seed (Studio, ~2.5 h, optional but valuable)

Separates "chemistry SFT did this" from "SFT run-to-run noise". Same data and hyperparameters,
different training seed. (SFT on non-chemistry data at matched weight distance, the other half
of D15's perturbation null, has **no script yet**.)

```bash
# 4a. The prior project is needed for training (science data + imports).
mkdir -p nnsj
git clone https://github.com/rl-sft-circuit-research/differential-circuit-vulnerability.git \
    nnsj/differential-circuit-vulnerability
python refusal/smoke_test.py            # "science files present" must now PASS

# 4b. Train (~40 min). It prints the science-NTS gate (>= 60%); stop if it fails.
python refusal/train_chain.py --stage sft --seed 1 --out-name M_SFT_seed1

# 4c. The same v1 readouts as M_SFT (~1.5 h)
export REFUSAL_DATA_PROFILE=v1
S1=refusal/results/checkpoints/M_SFT_seed1
python refusal/behaviour.py      --checkpoint $S1 --tag M_SFT_seed1 --ref Qwen/Qwen2.5-3B-Instruct
python refusal/direction.py --mode fit  --checkpoint $S1 --tag M_SFT_seed1
python refusal/direction.py --mode eval --checkpoint $S1 --tag M_SFT_seed1 --transfer-from M0
python refusal/addition_sweep.py --checkpoint $S1 --tag M_SFT_seed1 --also-own
python refusal/addition_sweep.py --summarise                      # now compares M_SFT_seed1 too
python refusal/rotation_dose.py  --checkpoint $S1 --tag M_SFT_seed1 --real-axis-to M0
unset REFUSAL_DATA_PROFILE
```

**What to look at:** does M_SFT_seed1 land near M_SFT on cos with dir_M0 (0.914), refusal on
harmful (0.972) and E2a's shift (+0.19)? If yes, those are properties of chemistry SFT, not one
lucky run. If it's far off, report run-to-run spread as the noise floor.

## 5. Save everything, then STOP the Studio

```bash
pip freeze > requirements.lock.txt
git add -f refusal/results/directions/*.pt refusal/results/v2/directions/*.pt
git add refusal/results refusal/data/v2 logs/session_*.log requirements.lock.txt
git commit -m "D15 + D16 results, v2 E1/E3 at L27, checks"
git push origin main

# Checkpoints are never in git. If they aren't on the HF Hub yet (optionally, M_SFT_seed1 too):
hf upload <you>/mats-m-sft refusal/results/checkpoints/M_SFT .
hf upload <you>/mats-m-rl  refusal/results/checkpoints/M_RL  .
```

Then **stop the Studio**. Everything below is CPU.

## 6. Analysis + figures (PC)

```bash
cd E:/NeelNandaMATSProgramProject
git pull origin main
REFUSAL_DATA_PROFILE=v1 python refusal/analysis.py              # figures + analysis_summary.md
REFUSAL_DATA_PROFILE=v1 python refusal/digest.py --examples
```

PowerShell equivalent: `$env:REFUSAL_DATA_PROFILE = "v1"; python refusal/analysis.py`

The new lead figure, the E1 calibration curve (random-axis ratio vs cosine, c\* marked), has
**no script yet**. The data is in `refusal/results/sensitivity/rotation_dose_M_SFT_*.json`
(`summary.readouts.rate.random_curve`).

## 7. Better judge for v2 behaviour — no script yet

The v2 result that refusal *rises* after fine-tuning (0.723 → 0.759 / 0.789) rests on the substring
judge. It needs rescoring with a semantic judge (StrongREJECT grader or an LLM judge) before it's
claimed. The completions to rescore are regenerated from `refusal/data/v2/` test pairs;
`behaviour.py` doesn't save all of them.

## 8. Not commands

- **Prereg D17** (before any writeup): record E1 `undetectable` → the dissociation claim is
  withdrawn (D15 rule); E2a, E2b, E3; v2 results; the v2 L19 layer error and its fix; results of
  steps 1–4.
- **Writeup:** lead with the E1 calibration; update `WRITEUP_SCAFFOLD.md` and
  `MATS_form_answers_draft.md`, which still carry the withdrawn claim.
