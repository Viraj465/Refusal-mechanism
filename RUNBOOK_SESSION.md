# Runbook — sensitivity + v2 session (Lightning.ai L40S, ~5.5 h of a 9 h budget)

Machine: 1 × L40S (48 GB), **Interruptible OFF**. Use the same Studio as before: the
checkpoints are on its disk at `refusal/results/checkpoints/{M_SFT,M_RL}`.
Everything runs from the repo root. What each step does and why: prereg D15, D16.

> Short checklist. The full guide (setup decisions, results files, analysis, rebuilding
> from scratch, troubleshooting) is **`RUNNING_GUIDE.md`**.

## 0. Setup (~10 min)

```bash
cd /teamspace/studios/this_studio/Refusal-Mechanism
git pull origin main
ls refusal/results/checkpoints/M_SFT/config.json refusal/results/checkpoints/M_RL/config.json
ls refusal/results/directions/ 2>/dev/null   # if M0.pt / M_SFT.pt are here: the ORIGINAL v1
                                             # directions survived; the script keeps them
pip install -q -r requirements.txt   # lower bounds only: leaves the Studio's CUDA torch alone
huggingface-cli whoami || huggingface-cli login             # M0 + tokenizer download
```

## 1. Run (one command, in tmux so a closed browser tab does not kill it)

```bash
tmux new -s run
bash refusal/run_session.sh          # everything: phase A, then phase B
# detach: Ctrl-b d      re-attach: tmux attach -t run
```

If anything fails it stops. Fix the problem and re-run the **same command**: finished steps are
skipped (markers in `refusal/results/.session_done/`). Full log: `logs/session_*.log`.

## 2. Check-in points — look, don't just wait

| When | Look at | Healthy | Stop and ask if |
|---|---|---|---|
| ~15 min, after `fit_*` | printed `selected layer L*` | 19 for M0 and M_SFT (M_RL 19 or 21, D13) | layer far from 19, or `No candidate direction` error |
| after `matched_cosine` | `cos(M0|M_SFT) @L19` | ≈ 0.914 if originals were kept; within ~0.01 if refit | far from 0.914 |
| ~25 min, after `e1_fast_*` | `[rotation] score: c* = …` lines | a number, not all `nan` | every ratio `nan` (own-direction drop ~0) |
| end of phase A (~2 h 15) | `e1_M_SFT` verdict line | any of undetectable / detectable / ambiguous — all are results | a crash |
| `v2 filter` | `refusal/data/v2/filter_report.md` | train 200 / val 50 | `only N filtered fit pairs` → re-run filter with `--fit-fraction 0.5`, log it in prereg |

## 3. Time budget (estimated from D14's measured ~420 generations/min)

| Block | Est. |
|---|---|
| Setup + model downloads | ~25 min |
| Phase A (v1): refits, E3, E1 fast+full, E2b, E2a | ~2 h |
| Phase B (v2): build+filter, fits, ceiling, behaviour, eval ×5, random control, E3, E1 | ~3 h |
| **Total** | **~5.5 h** — leaves ~3.5 h of the 9 h budget as buffer |

Short on credits? Run `bash refusal/run_session.sh A` only — phase A alone answers the
post's headline question. Phase B can be run in a later session and resumes cleanly.

## 4. After the run — save everything, then STOP the Studio

```bash
# Direction tensors are gitignored but tiny (~0.6 MB each); commit them this time so they
# cannot be lost again (their loss is what forced the refits).
git add -f refusal/results/directions/*.pt refusal/results/v2/directions/*.pt
git add refusal/results refusal/data/v2 logs/session_*.log
git commit -m "Sensitivity (D15) + v2 replication (D16) results"
git push
```

Then stop the Studio. Everything after this (reading results, D17 entry, figures) is CPU.
