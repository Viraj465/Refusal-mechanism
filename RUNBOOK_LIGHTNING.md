# Runbook — remaining steps (Lightning.ai L40S)

Machine: 1 × L40S (48 GB), Interruptible OFF, 2.14 credits/hr.
GPU used so far ≈ 2.5 h of a 17 h budget. Everything below fits in ~1.5 h.

Run every command **from the repo root** (`~/RefusalMechanism/Refusal-mechanism` on the
Studio). Stop the Studio the moment Step 7 finishes — Steps 8–10 need no GPU.

---

## Status — what is already done

| Step | State |
|---|---|
| 0–1 Setup, prior-repo clone, HF login | done |
| 2 Behavioural filter | done — 752 pairs, splits 200/50/**502** |
| 3 Direction spine on M0 (the gate) | **PASSED** — L\* = 19, ceiling 0.9845 [0.9733, 0.9913] |
| 4 SFT | done — science 42.0 → **77.0** |
| 5 Behaviour on M0 + M_SFT | done |
| 5b Direction fit + eval on M0 + M_SFT | done (moved ahead of RL, deliberately) |
| 6 RL from M_SFT | done — reward rate 0.725, science **59.0** |
| 7 M_RL readouts + chain check | **REMAINING** |
| 8–10 Analysis, prereg, writeup | **REMAINING** (no GPU) |

Results already on disk, for reference while writing:

| Quantity | M0 | M_SFT |
|---|---|---|
| Refusal on harmful (n=502) | 0.992 | 0.972 |
| Compliance on harmless | 1.000 | 0.992 |
| Own-direction ablation drop | +0.990 | +0.968 |
| Science new-task score | 42.0 | 77.0 |
| cos with dir_M0 at L19 | — | 0.9139 |
| Transfer ratio (dir_M0 ablated inside) | — | 1.004 |

Paired McNemar M0 vs M_SFT on harmful: 12 lost, 2 gained, exact p = 0.0129.
Addition of dir_M0 into M_SFT on harmless: refusal 0.008 → 0.462.
Direction separation by layer: L19 4.15 → 3.90, L32 10.37 → 8.36, L36 10.07 → 5.71.

Prereg entries D8–D11 record all of the above. **Nothing below may change a threshold.**

---

## Step 7 — GPU, ~50 min, in this exact order

Registered readouts first, diagnostics after, so a credit or session failure costs the
least important thing.

### 7a. Behaviour on M_RL (~8 min)

```bash
python refusal/behaviour.py --checkpoint refusal/results/checkpoints/M_RL --tag M_RL \
    --ref Qwen/Qwen2.5-3B-Instruct
```

- [ ] refusal(harmful) ______   compliance(harmless) ______
- [ ] science NTS ______   KL harmful ______  harmless ______

### 7b. Direction fit on M_RL (~10 min)

```bash
python refusal/direction.py --mode fit --checkpoint refusal/results/checkpoints/M_RL --tag M_RL
```

- [ ] L\* = ______  (M0 and M_SFT both chose 19; a different layer is itself a result)

> Gate: if this raises `No candidate direction ablates refusal without destroying
> harmless compliance`, stop and tell me. It did not fire for M0 or M_SFT.

### 7c. Direction eval on M_RL (~12 min) — THE H2 READOUT

```bash
python refusal/direction.py --mode eval --checkpoint refusal/results/checkpoints/M_RL \
    --tag M_RL --transfer-from M0
```

This is the only run that computes cos(M0, M_RL) and cos(M_SFT, M_RL), because the earlier
evals ran before `M_RL.pt` existed. Do not skip it.

- [ ] cos(M_RL, M0) ______  vs ceiling [0.9733, 0.9913] and vs M_SFT's 0.9139
- [ ] transfer ratio ______  [verdict ______]
- [ ] addition: harmless refusal ______ → ______

### 7d. Science-score length check (~5 min) — explains 77.0 → 59.0

RL trained with 256-token completions; the score is measured at 64. Test whether the drop
is truncation or real degradation.

```bash
python - <<'PY'
import gc, sys, torch
sys.path.insert(0, 'refusal')
from common import load_model_and_tokenizer
from behaviour import science_nts
for tag, ckpt in [('M_SFT', 'refusal/results/checkpoints/M_SFT'),
                  ('M_RL',  'refusal/results/checkpoints/M_RL')]:
    m, t = load_model_and_tokenizer(ckpt)
    for n in (64, 256):
        print(f'{tag}  max_new_tokens={n}  NTS={science_nts(m, t, 200, n)}')
    del m; gc.collect(); torch.cuda.empty_cache()
PY
```

- [ ] M_SFT 64 ______ / 256 ______     M_RL 64 ______ / 256 ______

The **registered** score stays the 64-token one for all three checkpoints. This is an
exploratory check and is reported as such.

### 7e. Ablation transcripts for M_SFT and M_RL (~4 min)

The M0 transcripts are what found the D9 judge caveat. Get the same for the other two so
the writeup can compare ablated text across checkpoints.

```bash
python refusal/inspect_ablation.py --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT --direction-tag M0
python refusal/inspect_ablation.py --checkpoint refusal/results/checkpoints/M_RL  --tag M_RL  --direction-tag M0
```

Using `--direction-tag M0` throughout means all three are ablated with the *same* vector,
which is the causal comparison the transfer ratio scores numerically.

### 7f. Chain check (~10 min, CPU-bound, needs no GPU)

```bash
python refusal/train_chain.py --stage chain-check \
    --m0 Qwen/Qwen2.5-3B-Instruct \
    --sft refusal/results/checkpoints/M_SFT \
    --rl refusal/results/checkpoints/M_RL
```

- [ ] `chain_verified` = ______  (must be True; False means H2 is reworded per prereg §2)

### 7g. Collect and leave

```bash
python refusal/digest.py --examples
git add refusal/results refusal/train_chain.py && git commit -m "results: M_RL readouts, chain check" && git push
```

- [ ] Pushed
- [ ] **Studio stopped.** Nothing after this needs a GPU.

---

## Step 8 — Analysis, local, no GPU (~10 min)

```bash
cd E:/NeelNandaMATSProgramProject
git pull
python refusal/analysis.py
```

Produces `refusal/results/figures/figure1_behaviour_direction.png`,
`figure2_retention.png`, and `refusal/results/analysis_summary.md`.

- [ ] Recompute one headline number by hand from the JSON and note that you did. The
      McNemar test already covers the behavioural one; do the same for the transfer ratio
      (`transfer_drop / own_drop` from `direction_eval_M_RL_*.json`).

Optional, decide only after seeing Figure 2, and only if it earns a slot inside the
**two-figure limit**: a per-layer plot of direction separation for all three checkpoints,
which turns the late-layer collapse above into a picture. The numbers are already in
`direction_meta` in each fit JSON; no GPU and no new runs needed.

---

## Step 9 — Prereg D12

Record the RL numbers and both verdicts against the frozen §5.2 thresholds, before the
writeup argues anything. Send me the digest output and I will draft it.

---

## Step 10 — Writeup (see `EXECUTION.md` §4)

Google Doc, link sharing on. Executive summary ≤600 words and ≤3 pages, two figures, the
seeded random completions immediately after the summary, hours with a Toggl screenshot.
Form answers in your own voice — they are read first and used as the filter.

Lead with the finding, not the chronology:

> Science-QA fine-tuning left refusal behaviour almost intact (0.992 → 0.972, McNemar
> p = 0.013) and rotated the refusal direction well outside its own split-half ceiling
> (cos 0.914 vs 0.984 [0.973, 0.991]) — yet that direction lost **no** causal power:
> ablating M0's direction inside M_SFT removes refusal completely (transfer ratio 1.004).
> The representational readout and the causal readout disagree.

Then the RL arm answers whether that pattern moves back, holds, or breaks.

**Submission deadline: Sept 11, 11:59pm PT = Sept 12, 12:29pm IST.**

---

## Stop rules (unchanged, first to fire wins)

1. 20 h logged → write up what exists.
2. All direction-level readouts complete.
3. Deltas inside reference bands → write the null.
4. Direction gate fails → negative methodological result. (Did not fire.)
5. Case C → M0 vs M_SFT only. (Not needed; M_RL exists.)
