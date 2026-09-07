# MATS 12.0 — Refusal Mechanism Under Capability Post-Training
## End-to-end implementation plan (revised after code audit, Mon Sept 7 2026)

**Question:** Science-QA SFT and science-QA Dr.GRPO contain no safety data and no safety reward. Does the SFT stage erode Qwen2.5-3B-Instruct's refusal mechanism, and does RL continuation restore it anyway?

**Claim scope:** alignment durability under post-training. Never "deceptive alignment detection."

**Terminology:** *refusal mechanism* = whatever the chosen method recovers under a harmful→harmless counterfactual. Operationalize, don't assert. Not "the refusal circuit."

**Checkpoints:** M0 = Qwen2.5-3B-Instruct (already refuses). M_SFT = M0 + science-QA SFT. M_RL = M_SFT + science-QA Dr.GRPO. "Base" = M0 always.

**Deadline:** Thu Sept 11 (hard). Target completion Wed Sept 10 night. Thu is reruns + exec summary only.

**Budget:** ≤20h research + ≤2h exec summary. Track in Toggl project `MATS-refusal`: `data`, `baseline`, `train`, `dbm`, `controls`, `analysis`, `writeup`. GPU wait while doing something else does not count; time watching logs does.

---

## 0. What changed from the first draft, and why

The original pipeline assumed the 2605.28860 repo hands over three checkpoints, science-QA masks, a working DBM implementation, and diffing code. Audit of `RLRazor/` and `nnsj/` (Sept 6–7) found:

| Assumed inherited | Actual state |
|---|---|
| M_SFT, M_RL checkpoints | Not on disk in either copy. No `results/` dir. |
| Science-QA masks | Don't exist. `run_circuit_analysis.py --task science` returns empty: counterfactual builder requires the math `'0'/'1'` schema; SciKnowEval items have `prompt/question/choices/answerKey/...`. |
| DBM mask method | `DCMAnalysis.train_dcm_mask` uses mirrored interpolation `m·orig + (1−m)·cf`, no annealing, raw first-token logit, λ=0.1 dominating the loss. Coherent, but not the paper's Eq. |
| Fig. 8 IoU / diffing code | Does not exist. Only overlap code uses fixed-size top-k path-patching sets. |
| RL trained from SFT | Every committed path trains RL from base. `nu_loop.run_mu_iterations` chains correctly but is never called. |
| Dr.GRPO (`trainingv1`) | Cannot backprop: log-probs computed under `torch.no_grad()` in `rollout.py`. Advantages are std-normalized (GRPO, not Dr.GRPO). RL checkpoint never saved. TRL `train_grpo` path works and saves. |

**Consequences for this plan:**
1. Checkpoints must be recovered from co-authors or retrained. Decision point is Stage 0, today.
2. The refusal-direction analysis (Arditi) becomes the **spine** — it needs none of the missing pieces. DBM is the upgrade, gated on time and on the direction result being ambiguous.
3. Retraining is a chance to make the design match the claim: **RL from SFT**, not parallel arms.
4. Prior paper's numbers are cited as *the claim being tested*, not as established background.

Reusable and verified working from the repo: `o_proj`-input hooks and per-head extraction ([discovery.py:120-191](RLRazor/src/circuits/discovery.py#L120-L191)), `path_patch_head`, `ablate_heads`, `_compute_answer_probability` (Eq. 2 geometric mean), `compute_faithfulness`, TRL SFT ([training.py:31](RLRazor/src/training/training.py#L31)) and TRL GRPO ([training.py:521](RLRazor/src/training/training.py#L521)) with on-disk saving, `load_your_checkpoint`.

---

## 1. Pre-registration (commit before touching data)

File: `prereg.md`, committed with timestamp, before Stage 2 runs.

- **H1:** science-QA SFT reduces retention of M0's refusal mechanism.
- **H2 (primary):** M_RL retains more of M0's refusal mechanism than M_SFT does. Primary comparison is d(M_RL, M0) vs d(M_SFT, M0). Requires RL-from-SFT; if arms end up parallel, reword to "RL preserves more than SFT" and say so.
- **H3 (secondary):** SFT compresses the mechanism (fewer heads / sharper direction); RL keeps it distributed.
- **H4 (dissociation):** cross-tab behavioural refusal rate vs mechanism retention. Behaviour recovers + mechanism doesn't = functional replacement. Both recover = re-consolidation. Neither = erosion. Mechanism recovers + behaviour doesn't = latent.
- **Success/failure defined relative to controls** (permutation null, split-half ceiling, seed stability), not raw percentages.
- **Direction-level readouts registered in advance:** cosine(M0 dir, M_SFT dir), cosine(M0 dir, M_RL dir); refusal rate after ablating each model's own direction; refusal rate after ablating *M0's* direction inside M_SFT and M_RL (transfer).
- **Stage 3 gate outcome will be written down before Stage 5 starts.**

---

## 2. Stage plan

### Stage 0 — Setup and checkpoint decision (Mon Sept 7, 1.5h)

**0a. Message co-authors now.** Ask for: (i) the working copy / notebook that produced Figures 1, 4, 7, 8; (ii) M_SFT and M_RL checkpoint directories; (iii) which training path (TRL `training.py` or `trainingv1`) actually ran. This is the highest-leverage hour of the week. Do not wait on the reply; proceed with 0b–0d.

**0b. Environment.** `pip install -r RLRazor/requirements.txt`. Confirm CUDA, VRAM, and that Qwen2.5-3B-Instruct loads in bf16. Record GPU model and VRAM in the writeup's infra line.

**0c. Repo hygiene.** New top-level package `refusal/` alongside `RLRazor/`. Do not edit `RLRazor/src/circuits/discovery.py` in place; import what works, reimplement what doesn't. Keeps the disclosure line honest.

**0d. Decide checkpoint path.** Three cases:

| Case | Condition | Action |
|---|---|---|
| A | Co-authors deliver M_SFT + M_RL by Tue morning | Use them. Verify RL was from SFT (ask; or compare parameter distance M_RL→M_SFT vs M_RL→M0). Original 3-way plan runs. |
| B | No checkpoints by Tue morning | Retrain (Stage 4). SFT tonight, RL Tuesday. |
| C | Retrain but RL cannot reach useful NTS by Wed noon | Drop to **M0 vs M_SFT** (H1 only). Report RL as future work. |

Default assumption for scheduling: **Case B.** Start SFT training tonight regardless; it's cheap and it's needed under B and C.

### Stage 1 — Refusal dataset (Mon Sept 7, 2h) — checkpoint-independent

**Source:** Arditi et al. `refusal_direction` repo splits. Harmful: AdvBench, MaliciousInstruct, TDC/HarmBench (~500). Harmless: Alpaca (~500).

**Format:** apply Qwen chat template. Store per example: `prompt_text`, `chat_formatted`, `label ∈ {harmful, harmless}`, `source`.

**Behavioural filter on M0:** greedy decode, 64 new tokens. Refusal judge = Arditi substring list ("I cannot", "I can't", "I'm sorry", "As an AI", etc.). Keep only harmful prompts M0 refuses and harmless prompts M0 complies with. Record the filter yield.

**Targets:** `refusal_target` and `compliance_target` = first 3 tokens M0 produces on the kept harmful / harmless prompts respectively. Store token IDs, not just strings.

**Pairing:** match harmful↔harmless by token length (±2) so that last-token alignment is trivial and left-padding is minimal.

**Splits (fixed seed 0):** train ~200 pairs (mask learning / direction fitting), val ~50 (λ, layer selection), test ~150 (all behavioural numbers; untouched until Stage 7).

**Output:** `refusal/data/refusal_pairs.jsonl` + `splits.json` + `filter_report.md`.

### Stage 2 — Behavioural baseline (Tue Sept 8 AM, 1h once checkpoints exist)

Per checkpoint (M0, M_SFT, M_RL), test split:
- refusal rate on harmful prompts
- compliance rate on harmless prompts
- science-QA NTS (reuse `evaluate_new_task`) — proves the fine-tunes actually learned science
- expected KL from M0 on harmful prompts and on harmless prompts separately (reuse `compute_forward_kl`; paper Eq. 1)

This is Figure 1. If SFT and RL show no behavioural refusal difference, say so: the internal analysis then tests whether internals moved when behaviour did not.

### Stage 3 — Refusal-direction analysis (Tue Sept 8, 3h) — THE SPINE

Arditi method, per checkpoint, on train split:
- Residual-stream activations at the last instruction token, every layer.
- Direction_ℓ = mean(harmful) − mean(harmless) at layer ℓ.
- Pick ℓ* per model by val-set ablation effect on refusal rate (project out the direction at all positions, all layers ≥ ℓ*; Arditi's directional ablation).

**Readouts (test split):**
1. cos(dir_M0, dir_SFT), cos(dir_M0, dir_RL), cos(dir_SFT, dir_RL) at each model's ℓ* and at a common ℓ.
2. Refusal rate after ablating each model's own direction.
3. **Transfer:** ablate dir_M0 inside M_SFT and inside M_RL. Still kills refusal → mechanism preserved. Doesn't → moved.
4. **Addition:** add dir_M0 into harmless runs of each model; compliance→refusal rate. Symmetric check.
5. Direction norm and layer-of-max-separation per model (H3 at the direction level).

**Gate — write the outcome in `prereg.md` before Stage 5:**
- Direction cosines high AND transfer works in both → mechanism preserved by both; head-level analysis becomes localization only.
- Transfer works in RL, fails in SFT → restoration story; DBM justified as the *where*.
- Transfer fails in both, behaviour preserved → functional replacement candidate; DBM is the main event.
- Everything null → report as calibration-positive null; DBM still runs if time permits, but writeup leads with the direction result.

**This stage alone is a complete, defensible submission if everything after it fails.**

### Stage 4 — Training (Mon night → Tue, mostly GPU; ~1h human)

Only under Case B/C.

**SFT:** TRL path, `train_sft` in [training.py:31](RLRazor/src/training/training.py#L31). Science dataset via `load_dataset_byname('science')` + `UnifiedDatasetInterface.normalize_dataset`. lr 3e-5, effective bs 32, 2 epochs, ~2200 samples, constant-with-warmup, no weight decay. Saves to `./results/sft_lr…`. Verify NTS on 200 held-out science items before proceeding. Target NTS ≥ 60%.

**RL from SFT:** load the SFT checkpoint, then TRL `train_grpo` with `loss_type='dr-grpo'`, `beta=0`. Do **not** use `trainingv1.train_dr_grpo` (no gradient). Budget-cut config for one GPU: `num_generations=16`, `max_completion_length=256`, ~600 prompts, 1 epoch, lr 2e-5. Save to `./results/grpo_lr…`. Verify NTS; record it. If NTS < SFT's NTS − 5pp, note that RL is undertrained and interpret with the paper's Fig. 2 caveat (low-NTS regime shows little SFT/RL separation).

**Sanity on the chain:** ‖θ_RL − θ_SFT‖ ≪ ‖θ_RL − θ_M0‖. Log both.

### Stage 5 — Head-level mechanism via DBM (Wed Sept 9, 4h) — gated on Stage 3 outcome + time

Implement `refusal/dbm.py` fresh, reusing the hook layer from `discovery.py`.

**Triplet:** `x_base` = harmful prompt (refuses), `x_source` = paired harmless prompt (complies), `y_target` = compliance target (3 tokens).

**Mask semantics (paper's convention, stated explicitly):** `ã_h = (1−m_h)·a_base_h + m_h·a_source_h`. m_h=1 patches the harmless activation in. Circuit = heads with m_h > 0.5 after annealing. Loss = −log P_geo(y_target | x_base, ã) + λ·Σ m_h, with P_geo the geometric mean over the 3 target tokens (reuse `_compute_answer_probability`).

**Position alignment:** patch at the **last instruction token only**. Record this choice. Pairs are length-matched so no padding tricks needed.

**Annealing:** sigmoid with temperature τ from 1.0 → 0.05 over 300 steps, straight-through at the end; then hard threshold. Cache source activations once per triplet (do not recompute every step — the repo does, and it's the bottleneck).

**λ selection:** sweep {0.01, 0.03, 0.1} on val; choose the smallest λ whose circuit faithfulness ≥ 0.8. Same λ for all three models.

**Per checkpoint:** 2 seeds. Save the continuous mask vector (576), the binary set, circuit size, faithfulness F(C|M)/F(M) (reuse `compute_faithfulness` with zero-ablation of non-circuit heads, scored on refusal-vs-compliance top-1 at the first target token).

**Pilot gate (Wed morning, M0 only, 1 seed):** mask converges, circuit size between 5% and 60% of heads, faithfulness ≥ 0.8, ablating the found heads moves refusal rate the expected way on val. Fail → skip Stage 5/6 head-level work; writeup leads with Stage 3.

### Stage 6 — Controls (Wed Sept 9, 2.5h)

All new code, `refusal/controls.py`.

- **Matched-k:** rank heads by continuous mask value; compare top-k sets with k = smallest circuit size across models.
- **Permutation null:** 1000 random head subsets at the observed sizes; IoU distribution; report observed IoU with 95% band. At ~50% density chance IoU ≈ 0.33.
- **Split-half ceiling:** split train triplets in half, learn two masks per checkpoint (1 seed each); their IoU is the method's resolution. M0–M_SFT retention is only interpretable relative to M0–M0 split-half.
- **Seed stability:** Jaccard between the two seeds per checkpoint. Reference: 2604.04385 reports 0.92–1.0.
- **Direction-level analogue:** bootstrap the diff-in-means over train pairs (200 resamples); report cosine CI. This is the control for Stage 3 and costs minutes.

### Stage 7 — Analysis (Wed Sept 9 evening, 2h)

- **Figure 1:** behavioural refusal / compliance rates per checkpoint + direction cosines to M0 + transfer-ablation refusal rates. One panel row.
- **Figure 2:** head-level retention of M0's refusal circuit in M_SFT and M_RL, with permutation null band and split-half ceiling drawn on it; IoU matrix inset. If Stage 5 was skipped, Figure 2 becomes the H4 cross-tab.
- **H4 cross-tab:** rows = {behaviour preserved, behaviour lost}, cols = {mechanism preserved, mechanism moved}, one cell per checkpoint, using both direction-level and head-level definitions of "mechanism."
- Δm_h per head; vulnerable set {h : m_SFT < m_RL − δ}; layer-wise histogram.
- **Refusal × science-QA overlap (only if science masks are recovered from co-authors; otherwise cut):** IoU between refusal circuit and science circuit per checkpoint.
- **Optional causal check (~1h if under budget):** patch M0's circuit-head activations into M_SFT on harmful prompts; measure refusal recovery.

### Stage 8 — Stop rules

Stop at the first of:
1. 20h logged → write up what exists.
2. Stages 2–7 complete.
3. All retention deltas inside the null band → write the null; it is calibration-positive.
4. Stage 5 pilot fails → writeup is Stages 2–3 + direction-level controls only.
5. Case C triggered → M0 vs M_SFT only, all stages.

### Stage 9 — Writeup (Thu Sept 10, 2h) + exec summary (≤2h, last)

Order:
1. Question + pre-registered predictions (verbatim from `prereg.md`).
2. Infrastructure disclosure, one paragraph: "Hook-level patching and faithfulness code reused from arXiv:2605.28860 (author's prior work). Checkpoints [recovered from that project / retrained here with RL chained from SFT]. Refusal dataset, direction analysis, DBM reimplementation, all controls, and analysis are new. Hours: N (split)."
3. Inherited constraints: RL-from-SFT design; heads only; single model; prior paper's retention figures cited as the claim under test.
4. Behavioural baseline + direction analysis + gate outcome.
5. Head-level result with null band and ceiling (or: why it was skipped).
6. H4 dissociation table.
7. Limitations: one model; heads only; RL possibly undertrained (state NTS); narrow-finetune trace caveat (arXiv:2510.13900); direction-vs-circuit operationalization gap.
8. Related work: 2609.03887, 2603.23268, 2509.04259, 2609.01455, Arditi 2024, 2507.11878, 2511.07482, 2604.04385.
9. Hours, with split.

Two figures maximum. Exec summary written last.

---

## 3. Timeline

| When | What | Hours |
|---|---|---|
| **Mon Sept 7** | Stage 0 (message co-authors, env, repo layout). Stage 1 dataset + M0 filter. Kick off SFT training overnight. `prereg.md` committed. | 4 |
| **Tue Sept 8** | Verify SFT NTS. Kick off RL-from-SFT. Stage 2 behavioural baseline on M0 + M_SFT (add M_RL when ready). Stage 3 direction analysis, all readouts. Write gate outcome. | 5 |
| **Wed Sept 9** | Stage 5 pilot on M0 (AM). If pass: DBM on all three ×2 seeds (GPU) while writing Stage 6 controls. Stage 7 analysis in the evening. | 6 |
| **Thu Sept 10** | Writeup. Both figures. Exec summary last. | 3 + 2 |
| **Fri Sept 11** | Buffer: reruns, figure fixes only. Submit. | — |

Total ≈ 18h research + 2h exec summary. Slack: ~2h, all of it Wed.

---

## 4. Concrete code map

```
refusal/
  data/
    build_dataset.py      # Stage 1: download Arditi sets, chat-template, M0 filter, pair, split
    refusal_pairs.jsonl
    splits.json
    filter_report.md
  behaviour.py            # Stage 2: refusal/compliance rates, NTS, KL per checkpoint
  direction.py            # Stage 3: diff-in-means, layer select, ablate, transfer, addition
  train_chain.py          # Stage 4: TRL SFT → save → TRL Dr.GRPO from SFT → save; NTS + param-distance log
  dbm.py                  # Stage 5: annealed mask, last-token patching, cached source acts, Eq.2 loss
  controls.py             # Stage 6: matched-k, permutation null, split-half, seed Jaccard, dir bootstrap
  analysis.py             # Stage 7: IoU matrix, Δm_h, H4 cross-tab, figures
  prereg.md
  results/                # every run writes JSON here; never overwrite, always timestamp
```

Imports from `RLRazor/src/circuits/discovery.py`: `CircuitDiscovery` (for hooks, `extract_activations`, `ablate_heads`, `path_patch_head`, `_compute_answer_probability`, `compute_faithfulness`). Nothing from `DCMAnalysis` or `CrossModelCircuitAnalysis`. Nothing from `trainingv1`.

---

## 5. Fallback ladder (in order)

1. **Case A** (checkpoints recovered): original 3-way plan, all stages.
2. **Case B** (retrained, RL reaches useful NTS): 3-way plan, note RL budget cuts.
3. **Case C** (RL undertrained or fails): M0 vs M_SFT, all stages; H1 + H4 only.
4. **Stage 5 pilot fails:** direction-only submission, Stages 2–3 + direction bootstrap + H4 at the direction level.
5. **Everything fails by Wed noon:** cross-domain SciKnowEval plan (Physics–Chemistry vs Chemistry–Biology, answer-key-swap triplets, M0 only) — weakest, entirely inside existing infrastructure, still requires fixing the science schema in the counterfactual builder.

Each rung is a complete, honest submission. Never present a higher rung's framing with a lower rung's evidence.