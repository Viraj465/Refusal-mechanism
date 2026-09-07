# Pre-registration — Refusal Mechanism Under Capability-Only Post-Training

**Author:** Viraaj Sawant
**Written:** 2026-09-07
**Committed:** see `git log` for this file's first commit hash and author date
**Status at time of writing:** no refusal data has been downloaded, no model has been run on any refusal prompt, no checkpoint has been trained or loaded for this project. This document is written before any observation.

**Project:** MATS 12.0 application task. Implementation plan: `../mats_refusal_implementation_plan.md`.

---

## 1. Question

Science-QA SFT and science-QA Dr.GRPO contain no safety data and no safety reward. Does the SFT stage erode Qwen2.5-3B-Instruct's refusal mechanism, and does RL continuation restore it?

Any change to refusal is collateral: nothing in the training objective touches safety. That is the point of the design.

**Claim scope:** alignment durability under post-training. This project makes no claim about deceptive alignment, deceptive alignment detection, or intentional safety circumvention.

---

## 2. Definitions fixed in advance

- **M0** = `Qwen/Qwen2.5-3B-Instruct`. Already instruction-aligned; refuses at baseline. This is verified in Stage 1, not assumed.
- **M_SFT** = M0 + science-QA (SciKnowEval chemistry) completion-only SFT.
- **M_RL** = M_SFT + science-QA Dr.GRPO. **Chained from M_SFT, not from M0.** If this design cannot be achieved (see §7), the deviation is recorded and H2 is reworded.
- **Refusal mechanism** — operationalized two ways, both registered:
  - *Direction-level:* the diff-in-means residual-stream direction of Arditi et al. (2024), harmful − harmless, at the final instruction token.
  - *Head-level:* the attention-head set recovered by an annealed differentiable binary mask under a harmful→harmless activation-patching counterfactual.
  The term "the refusal circuit" is not used. Neither operationalization is asserted to be the mechanism; both are measurements.
- **Refusal judge:** Arditi et al. substring matcher, greedy decoding, 64 new tokens. The exact substring list is frozen in `refusal/data/build_dataset.py` at first commit and not modified after test data is scored.
- **Circuit retention:** IoU (Jaccard) between M0's head set and the fine-tuned model's head set.

---

## 3. Data and split discipline

- Harmful prompts: AdvBench, MaliciousInstruct, TDC/HarmBench (~500). Harmless: Alpaca (~500). Source: Arditi et al. `refusal_direction` repo.
- Behavioural filter: keep only harmful prompts M0 refuses and harmless prompts M0 complies with. A counterfactual pair is valid only if behaviours differ.
- Pairs matched on token length (±2 tokens).
- Splits fixed with seed 0: **train ~200 pairs**, **val ~50**, **test ~150**.

**Split discipline (binding):**
- **train** — direction fitting, mask learning. Nothing else.
- **val** — all method selection: layer choice, λ, annealing schedule, thresholds, pilot-gate decisions.
- **test** — reported numbers only. No hyperparameter, threshold, or method choice may be informed by test performance. Test is scored once per checkpoint per registered readout.

This resolves an ambiguity in the earlier pipeline draft, which both reserved test "until Stage 7" and reported Stage 2 behavioural numbers on it. Resolution: test is *scored* whenever a registered readout calls for it; it is never used to *choose* anything.

---

## 4. Hypotheses and directional predictions

**H1 — SFT erodes.** Science-QA SFT reduces retention of M0's refusal mechanism.
Prediction: retention(M_SFT) is below the split-half ceiling and, for the head-level measure, may still exceed the permutation null.

**H2 — RL restores. (PRIMARY)** M_RL retains more of M0's refusal mechanism than M_SFT does.
Primary comparison: **d(M_RL, M0) vs d(M_SFT, M0)**, not M_SFT vs M_RL directly.
Prediction: retention(M_RL) > retention(M_SFT).

**H3 — Compression vs distribution. (secondary)** SFT compresses the mechanism into fewer heads / a sharper single direction; RL keeps it distributed.
Prediction: |C_SFT| < |C_RL| ≈ |C_M0|; and the layer-wise spread of C_SFT is narrower.

**H4 — Behaviour/mechanism dissociation. (secondary, and the most informative outcome)**
Cross-tabulate behavioural refusal rate against mechanism retention, per checkpoint, under both operationalizations:

| | Mechanism preserved | Mechanism moved |
|---|---|---|
| **Behaviour preserved** | re-consolidation | **functional replacement** |
| **Behaviour lost** | latent mechanism | erosion |

No directional prediction is registered for H4; all four cells are reportable outcomes. "Functional replacement" — refusal behaviour recovers through different heads — is registered in advance as an interesting positive result, not a null.

---

## 5. Registered readouts and thresholds

### 5.1 Behavioural (test split, all three checkpoints)
- Refusal rate on harmful prompts; compliance rate on harmless prompts. Wilson 95% CI at n≈150.
- Science-QA new-task score (proves the fine-tunes learned the target task).
- Expected forward KL from M0 (paper Eq. 1), reported separately on harmful and harmless prompts.

### 5.2 Direction-level (spine — runs regardless of checkpoint provenance)
1. cos(dir_M0, dir_SFT), cos(dir_M0, dir_RL), cos(dir_SFT, dir_RL).
2. Refusal rate after ablating each model's own direction.
3. **Transfer (primary causal readout):** ablate **dir_M0** inside M_SFT and inside M_RL.
   *Transfer ratio* = (refusal drop under dir_M0 ablation) / (refusal drop under the model's own direction ablation).
4. Addition: add dir_M0 into harmless runs; measure induced refusal.

**Thresholds, fixed now:**
- **Direction ceiling:** cos between directions fitted on two disjoint halves of M0's train pairs, bootstrapped 200×. This is the method's resolution floor.
- *Preserved* = cos(dir_M0, dir_X) within the 95% CI of that ceiling.
- *Degraded* = cos below the ceiling CI lower bound.
- *Causally preserved* = transfer ratio ≥ 0.8. *Causally moved* = transfer ratio < 0.5. Between 0.5 and 0.8 is reported as ambiguous, not rounded to either.

### 5.3 Head-level (gated — see §7)
- Circuit size, faithfulness F(C|M)/F(M), per-head continuous mask values, 2 seeds per checkpoint.
- IoU(C_M0, C_SFT) and IoU(C_M0, C_RL).
- **Matched-k** comparison at k = smallest circuit size across checkpoints, to remove the density confound.

**Thresholds, fixed now:**
- **Permutation null:** 1000 random head subsets at the observed sizes. An IoU is *above chance* only if it exceeds the 97.5th percentile.
- **Split-half ceiling:** two masks per checkpoint from disjoint halves of train. IoU between them is the maximum resolvable overlap.
- *Retention preserved* = IoU above the null band **and** ≥ 0.8 × split-half ceiling.
- **H2 is supported** only if IoU(C_M0, C_RL) − IoU(C_M0, C_SFT) > 0 **and** that difference exceeds the largest within-checkpoint between-seed IoU difference. A gap smaller than seed noise is reported as null.
- **Seed stability** below Jaccard 0.7 within a checkpoint invalidates head-level conclusions for that checkpoint; report the instability instead of the overlap.

---

## 6. Confirmatory vs exploratory

**Confirmatory** (registered above, reported with thresholds): H1, H2, H3, H4; all readouts in §5.1–5.3.

**Exploratory** (reported as exploratory, no thresholds, no claims of support):
- Δm_h per head and the vulnerable-head set {h : m_SFT < m_RL − δ}.
- Layer-wise distribution of retained vs. lost heads.
- IoU between the refusal circuit and the science-QA circuit (only if science-QA masks are recovered; see §7).
- Patching M0's circuit heads into M_SFT to test causal recovery.
- Any comparison to the numbers reported in arXiv:2605.28860.

---

## 7. Contingencies declared in advance

These are registered **now** so that taking them later is not a post-hoc rescue.

### 7.1 Checkpoint provenance
An audit of the prior project's repository (2026-09-06/07) found that M_SFT and M_RL are not present on disk, that no committed code path trains RL from the SFT checkpoint, and that the science-QA masks do not exist. Three cases:

- **Case A** — checkpoints recovered from co-authors. Verify the chain by parameter distance (‖θ_RL − θ_SFT‖ ≪ ‖θ_RL − θ_M0‖). Full plan runs.
- **Case B** — checkpoints retrained here, RL chained from SFT, budget-reduced (group 16, ~600 prompts, μ=1, 256 max new tokens). Full plan runs; RL's new-task score is reported and the low-NTS caveat from the prior work's Figure 2 is stated.
- **Case C** — RL fails to reach a usable new-task score by Wed Sept 9 noon. **Drop to M0 vs M_SFT.** H1 and H4 only. H2 and H3 are reported as not tested. RL becomes stated future work.

Under Case B or C, the writeup states that checkpoints were trained for this project and does **not** claim inheritance of the prior work's checkpoints.

### 7.2 Head-level pilot gate
DBM is run on M0 alone first (val split). Pass requires all four: mask converges; circuit size between 5% and 60% of the 576 heads; faithfulness ≥ 0.8; ablating the recovered heads shifts refusal in the predicted direction on val.
**On failure:** head-level analysis (§5.3) is abandoned. The submission is the direction-level result (§5.1–5.2) plus its bootstrap control. This is registered as an acceptable complete outcome, not a failure to report.

### 7.3 Stop rules
Stop at the first of: (1) 20h logged; (2) all registered readouts complete; (3) all retention deltas inside the null band — write the null, it is calibration-positive; (4) pilot-gate failure; (5) Case C.

### 7.4 Relationship to prior work
Hook-level activation extraction, patching, ablation, and the faithfulness metric are reused from arXiv:2605.28860 (author's own prior work). The mask implementation, refusal dataset, direction analysis, all controls, and all analysis are new. The prior work's reported retention figures (SFT 63.5%→59.0%, RL 69.8%→72.5%) are cited as **the claim under test**, not as established background, because their generating code is not present in the repository.

---

## 8. What would falsify the headline

- **H2 falsified** if IoU(C_M0, C_RL) ≤ IoU(C_M0, C_SFT), or if the gap is within seed noise.
- **The whole restoration framing falsified** if M_SFT shows no behavioural or mechanistic refusal degradation at all — i.e. science SFT simply does not touch refusal. This is a real possible outcome and will be reported as such.
- **The direction spine falsified** if dir_M0 fails to ablate refusal in M0 itself (transfer ratio undefined). In that case the Arditi operationalization does not hold for this model and the finding is reported as a negative methodological result.

---

## 9. Deviations log

Append-only. Every departure from this document gets an entry with date, what changed, and why. Entries added after test data is scored are marked as such.

*(no entries at time of commit)*