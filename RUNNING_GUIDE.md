# Running guide — whole codebase, start to results (Lightning.ai L40S)

One place for every command. Run all of them **from the repo root** (the folder that
contains `prereg.md` and `refusal/`), never from inside `refusal/`.

What each experiment is and why it exists: `prereg.md` §9 **D15** (readout-sensitivity
experiments E1–E3) and **D16** (v2 dataset). Decision rules are fixed there *before* the
run; do not change a threshold after seeing a number.

| Part | Where | Time |
|---|---|---|
| 0. Push the code | your Windows PC | 2 min |
| 1. Studio setup | L40S | ~15 min |
| 2. The session (Phase A: v1 sensitivity, Phase B: v2 replication) | L40S | ~5.5 h |
| 3. Save results, stop the Studio | L40S | 5 min |
| 4. Analysis | any CPU | ~10 min |
| Appendix A. Rebuild everything from scratch (only if checkpoints are lost) | L40S | +5–6 h |

Machine: **1 × L40S 48 GB, Interruptible OFF** (nothing checkpoints mid-run, so a
pre-emption loses the running step). One GPU; more GPUs do not help.

---

## 0. Push the code (Windows PC, once)

The Studio gets code only through GitHub.

```bash
cd E:/NeelNandaMATSProgramProject
git status            # new: RUNNING_GUIDE.md, requirements.txt, .gitattributes, refusal/{sensitivity,rotation_dose,...}.py, refusal/data/v2/
git add -A
git commit -m "D15 sensitivity experiments, D16 v2 dataset, requirements, running guide"
git push origin main
```

`.gitattributes` keeps `refusal/run_session.sh` on LF line endings, so it runs on Linux
even though this PC uses `core.autocrlf=true`.

---

## 1. Studio setup (~15 min)

```bash
cd /teamspace/studios/this_studio/Refusal-Mechanism     # your repo on the Studio
git status                                               # must be clean before pulling
git pull origin main

# the new code is there
ls refusal/run_session.sh refusal/sensitivity.py refusal/rotation_dose.py \
   refusal/data/v2/sources.jsonl requirements.txt
grep -q RESULTS_ROOT refusal/common.py && echo "common.py OK" || echo "common.py is OLD"
```

### 1a. GPU and checkpoints — the one thing that can block everything

```bash
nvidia-smi                                   # NVIDIA L40S, 46068 MiB
ls refusal/results/checkpoints/M_SFT/config.json refusal/results/checkpoints/M_RL/config.json
ls refusal/results/directions/               # M0.pt / M_SFT.pt / M_RL.pt ?
```

| What you see | Do |
|---|---|
| Both `config.json` exist | Continue. |
| Missing, but you pushed them to the HF Hub earlier | `hf download <you>/mats-m-sft --local-dir refusal/results/checkpoints/M_SFT` and the same for `mats-m-rl` → `M_RL`. Check the repo names in your HF account. |
| Missing everywhere | Appendix A (retrain). Nothing except M0 can run without them. |
| `directions/` has `M0.pt`, `M_SFT.pt`, `M_RL.pt` | Those are the **original** v1 directions (gitignored, so never on GitHub). The session keeps them instead of refitting, so cos 0.914 is reproduced exactly. |
| `directions/` empty | Fine: the session refits them (~12 min). Cosines then land within ~0.01 of the registered ones; say "refit" in the writeup. |

### 1b. Install and log in

```bash
python --version                                   # 3.10+
pip install -q -r requirements.txt                 # lower bounds only: keeps the Studio's CUDA torch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True NVIDIA L40S
pip freeze > requirements.lock.txt                 # exact versions of THIS run, commit it later
hf auth login                                      # older hub versions: huggingface-cli login
```

### 1c. Smoke tests (CPU, ~2 min)

```bash
python refusal/smoke_test.py
```

Expect everything to pass except environment-only checks: `science files present`
(needs the `nnsj/` clone, Appendix A) and possibly `Qwen chat template` /
`triplet batching positions` if the tokenizer download is blocked. Any other FAIL:
stop and look before spending GPU time.

---

## 2. The session — one command (~5.5 h)

```bash
tmux new -s run
bash refusal/run_session.sh          # phase A, then phase B
# detach: Ctrl-b d        re-attach: tmux attach -t run
```

Run inside `tmux`, so closing the browser tab does not kill the job.

- **Stops at the first failure.** Fix the cause, re-run the *same command*: finished
  steps are skipped (markers in `refusal/results/.session_done/`).
- **Log:** `logs/session_<utc>.log`.
- **Only one phase:** `bash refusal/run_session.sh A` or `bash refusal/run_session.sh B`.
- **Force one step to re-run:** delete its marker, e.g.
  `rm refusal/results/.session_done/v1_e1_M_SFT`.

### What it runs

**Phase A — v1 data, the post's core (~2 h).** Results in `refusal/results/`.

| Step | Script | Output |
|---|---|---|
| Refit any missing direction (M0, M_SFT, M_RL) | `direction.py --mode fit` | `results/directions/*.pt`, `results/direction/direction_fit_*.json` |
| E3 matched-layer cosines (CPU) | `matched_cosine.py` | `results/direction/matched_cosine_*.json` |
| E1 fast pass, refusal score only | `rotation_dose.py --no-generate` | `results/sensitivity/rotation_dose_*.json` |
| **E1 rotation dose-response** (priority) | `rotation_dose.py` | `results/sensitivity/rotation_dose_{M_SFT,M0}_*.json` |
| E2b α-scaled and layer-19-only ablation | `graded_ablation.py --variant alpha/single` | `results/sensitivity/graded_ablation_*.json` |
| E2a addition sweep + cross-checkpoint comparison | `addition_sweep.py`, then `--summarise` | `results/sensitivity/addition_sweep_*.json`, `addition_compare_*.json` |

**Phase B — v2 replication (~3 h).** Data in `refusal/data/v2/`, results in
`refusal/results/v2/`; v1 files are never touched.

| Step | Script |
|---|---|
| Tokenise + chat-template the pinned v2 prompts (CPU) | `data/build_dataset.py --build` |
| M0 behavioural filter → train 200 / val 50, unfiltered test | `data/build_dataset.py --filter` |
| Direction fits, M0 ceiling | `direction.py --mode fit / ceiling` |
| Behaviour + over-refusal (XSTest safe) + KL | `behaviour.py --skip-nts` |
| Registered eval, unconditioned and M0-conditioned | `direction.py --mode eval [--m0-condition]` |
| Random-direction control (D14) | `random_control.py --m0-condition` |
| E3, E1 on v2 | `matched_cosine.py`, `rotation_dose.py` |

The science score is skipped in Phase B (`--skip-nts`): it doesn't depend on refusal data and
was already measured on v1.

### Check-in points — look, don't just wait

| When | Look at | Healthy | Stop if |
|---|---|---|---|
| ~15 min, after `fit_*` | printed `selected layer` | 19 for M0 and M_SFT; M_RL 19 or 21 (D13) | far from 19, or `No candidate direction ...` |
| after `matched_cosine` | `cos(M0,M_SFT) @L19` | ≈ 0.914 (originals kept) or within ~0.01 (refit) | far from 0.914 |
| ~25 min, after `e1_fast_*` | `[rotation] ...: c* = ...` lines | numbers, not all `nan` | every ratio `nan` (own-direction drop ≈ 0) |
| end of Phase A (~2 h 15) | `e1_M_SFT` verdict | `undetectable` / `detectable` / `ambiguous` — all are results | a crash |
| `v2 filter` | `refusal/data/v2/filter_report.md` | train 200 / val 50 | `only N filtered fit pairs` → re-run with `--fit-fraction 0.5` and log it in prereg |

---

## 3. Save results, then STOP the Studio

```bash
# Direction tensors are gitignored but small (~0.6 MB each). Force-add them this time:
# losing them is what forced the refits.
git add -f refusal/results/directions/*.pt
git add -f refusal/results/v2/directions/*.pt 2>/dev/null || true
git add refusal/results refusal/data/v2 logs/session_*.log requirements.lock.txt
git commit -m "D15 sensitivity + D16 v2 results"
git push origin main
```

Then **stop the Studio**. Everything below is CPU.

Checkpoints (`refusal/results/checkpoints/`) are never committed. If they are not
already on the HF Hub, push them before deleting the Studio:

```bash
hf upload <you>/mats-m-sft refusal/results/checkpoints/M_SFT .
hf upload <you>/mats-m-rl  refusal/results/checkpoints/M_RL  .
```

---

## 4. Analysis (any CPU, ~10 min)

```bash
cd E:/NeelNandaMATSProgramProject
git pull origin main
```

**The data profile now defaults to v2.** Every script reads `REFUSAL_DATA_PROFILE`
(`v1` or `v2`, default `v2`). For the registered v1 numbers, set `v1` explicitly:

```bash
# bash / Git Bash
REFUSAL_DATA_PROFILE=v1 python refusal/analysis.py         # figures + analysis_summary.md
REFUSAL_DATA_PROFILE=v1 python refusal/digest.py --examples
```
```powershell
# PowerShell
$env:REFUSAL_DATA_PROFILE = "v1"; python refusal/analysis.py
```

`python refusal/analysis.py` without the variable reads `refusal/results/v2/` and writes
its figures to `refusal/results/v2/figures/`. It has not been run on v2 results yet, so
check its output before using it.

### Reading the D15 results

Every `results/sensitivity/*.json` holds raw per-prompt values, a `summary` with bootstrap
CIs, and a `verdict`, for both readouts (substring judge `rate`, graded refusal `score`):

| File | Verdicts | Meaning |
|---|---|---|
| `rotation_dose_M_SFT_*` (E1) | `undetectable` / `detectable` / `ambiguous` (in `at_mark`), plus `c_star` | Would a *random* rotation to cos 0.914 still pass as "preserved"? `undetectable` = the transfer ratio cannot see a rotation this size (its CI stays above the preserved threshold); `detectable` = it would have flagged it. How this bears on D11/D12 is set by D15's rules, not here. `c_star` = the cosine where the ratio crosses the threshold. |
| `graded_ablation_{alpha,single}_*` (E2b) | `causally_moved` / `ambiguous` / `uninformative` | Graded version of the transfer ratio. `uninformative` when the own direction's drop CI includes 0. |
| `addition_compare_*` (E2a) | `indistinguishable` / `needs_more` / `needs_less` | Paired bootstrap of the addition threshold vs M0: does the checkpoint need more or less of dir_M0 added to induce refusal? |
| `direction/matched_cosine_*` (E3) | `preserved` / `degraded` vs the ceiling CI | cos(M0, M_RL) at the *same* layer as cos(M0, M_SFT). |

Record every outcome in a new prereg entry (D17) before the writeup argues anything.

---

## Appendix A — rebuild everything from scratch

Only needed if `M_SFT` / `M_RL` are lost. **Retrained checkpoints are different models**:
every D9–D14 number must then be re-measured and the retrain logged in `prereg.md` §9.

```bash
# 1. The prior project: science data + three imports (training, science score).
mkdir -p nnsj
git clone https://github.com/rl-sft-circuit-research/differential-circuit-vulnerability.git \
    nnsj/differential-circuit-vulnerability
python refusal/smoke_test.py                       # "science files present" must now PASS

# 2. v1 dataset (the committed refusal/data/refusal_pairs.jsonl + splits.json already exist;
#    rebuild ONLY if they are missing — regenerating breaks split discipline)
REFUSAL_DATA_PROFILE=v1 python refusal/data/build_dataset.py --build
REFUSAL_DATA_PROFILE=v1 python refusal/data/build_dataset.py --filter --batch-size 16

# 3. M0 direction spine (the gate) — v1
REFUSAL_DATA_PROFILE=v1 python refusal/direction.py --mode fit     --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
REFUSAL_DATA_PROFILE=v1 python refusal/direction.py --mode ceiling --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0

# 4. SFT (~40 min; gate: science NTS >= 60%), then Dr.GRPO from SFT (~3 h)
python refusal/train_chain.py --stage sft --push-to <you>/mats-m-sft
python refusal/train_chain.py --stage rl --init refusal/results/checkpoints/M_SFT \
    --num-generations 16 --max-completion-length 256 --push-to <you>/mats-m-rl
#    stop the RL run if the first log lines show "rollout reward rate 0.000"

# 5. Chain check: ||θ_RL − θ_SFT|| << ||θ_RL − θ_M0||
python refusal/train_chain.py --stage chain-check --m0 Qwen/Qwen2.5-3B-Instruct \
    --sft refusal/results/checkpoints/M_SFT --rl refusal/results/checkpoints/M_RL

# 6. Registered v1 readouts (D9–D14)
export REFUSAL_DATA_PROFILE=v1
M0=Qwen/Qwen2.5-3B-Instruct; SFT=refusal/results/checkpoints/M_SFT; RL=refusal/results/checkpoints/M_RL
python refusal/behaviour.py --checkpoint $M0  --tag M0
python refusal/behaviour.py --checkpoint $SFT --tag M_SFT --ref $M0
python refusal/behaviour.py --checkpoint $RL  --tag M_RL  --ref $M0
python refusal/direction.py --mode fit  --checkpoint $SFT --tag M_SFT
python refusal/direction.py --mode fit  --checkpoint $RL  --tag M_RL
python refusal/direction.py --mode eval --checkpoint $M0  --tag M0
python refusal/direction.py --mode eval --checkpoint $SFT --tag M_SFT --transfer-from M0
python refusal/direction.py --mode eval --checkpoint $RL  --tag M_RL  --transfer-from M0
python refusal/random_control.py --checkpoint $M0  --tag M0
python refusal/random_control.py --checkpoint $SFT --tag M_SFT --donor M0
python refusal/random_control.py --checkpoint $RL  --tag M_RL  --donor M0
python refusal/inspect_ablation.py --checkpoint $SFT --tag M_SFT --direction-tag M0
python refusal/inspect_ablation.py --checkpoint $RL  --tag M_RL  --direction-tag M0
unset REFUSAL_DATA_PROFILE

# 7. Then Part 2 (bash refusal/run_session.sh) as normal.
```

Not run by design (prereg D6): `dbm.py` and `controls.py --mode stats` (head-level
circuits). The code is kept and tested.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `$'\r': command not found` from `run_session.sh` | The file got Windows line endings: `sed -i 's/\r$//' refusal/run_session.sh` |
| `FileNotFoundError: .../data/v2/refusal_pairs.jsonl missing` | A script ran on v2 before `--build`/`--filter`. Run Phase B, or set `REFUSAL_DATA_PROFILE=v1`. |
| `[matched] no direction tensors in ...` | Directions missing: run the `fit` steps (the session does this automatically). |
| `MISSING checkpoint` from `run_session.sh` | Part 1a. |
| CUDA out of memory | Add `--batch-size 8` to the failing step's command, delete its marker, re-run. |
| `hf: command not found` | `huggingface-cli login` / `huggingface-cli download` (older `huggingface_hub`). |
| The session died mid-step | `tmux attach -t run`; if gone, re-run `bash refusal/run_session.sh` — it resumes. |
