# MATS 12.0 Application Task — Implementation Pipeline

**Project:** Safety after specialization — does capability RL restore the refusal mechanism disrupted by domain-specific SFT?

**Question:** Science-QA SFT and science-QA Dr.GRPO contain no safety data and no safety reward. Does the SFT stage erode Qwen2.5-3B-Instruct's refusal mechanism, and does RL continuation restore it anyway?

**Terminology (used throughout):** *refusal mechanism* = the attention-head set recovered by DBM under a harmful-to-harmless activation-patching counterfactual. Not "the refusal circuit" — the field currently has a single direction (Arditi 2024), multiple directions (2602.02132), and a late-layer routing pathway (2609.01455) all in play. Operationalize, don't assert.

**Checkpoint naming:** M0 = Qwen2.5-3B-Instruct (already instruction-aligned; it refuses at baseline — verify in Stage 1). M_SFT = M0 + science-QA SFT. M_RL = M_SFT + science-QA Dr.GRPO. "Base" below always means M0, never a pretrained non-aligned base.
**Pathway:** Application task (~16h research, max 20h, + ≤2h executive summary). Not the "existing research" pathway.
**Deadline:** Sept 11, 2026 (extension). Target completion Sept 10; Sept 11 is buffer for reruns only.
**Builds on:** arXiv:2605.28860 (DBM pipeline, Qwen2.5-3B base/SFT/RL checkpoints, science-QA masks). Everything below is new work.

---

## Relationship to the prior paper

Substitution, not extension. Nothing in the training trajectory touches safety, so any refusal change is collateral — that is the whole point. The paper's pipeline is: behaviour → head-level DBM circuit → diff across base/SFT/RL → report what each objective preserved. This project changes one variable — the behaviour (science QA → refusal). Model, checkpoints, mask method, diffing code, and three-way comparison are unchanged. The paper's own Limitations section asks for exactly this (broader capability domains including safety).

Inherited constraints to state in paragraph one of the writeup:
- RL was trained **from the SFT checkpoint**, so the question is "does RL continuation restore," not "RL vs SFT as parallel objectives."
- Circuits are **attention heads only** (576 heads, 36 layers × 16).
- Paper's circuits were ~50% of all heads (M0 51.6%, M_SFT 46.0%, M_RL 51.4%), so chance overlap is high and controls are mandatory.
- Open tension from the paper: RL had larger output-space KL than SFT yet preserved more internal circuitry. If it reappears on refusal, report it.
- Retention figures to quote are the published Figure 1 ones: SFT 63.5% then 59.0%; RL 69.8% then 72.5%. The GitHub README's ~68%/~52% is stale — do not cite it.

Claim scope: **alignment durability under post-training**. Do not use "deceptive alignment detection" anywhere.

---

## Stage 0 — Setup and pre-registration (1h)

**Inventory**
- Checkpoints: M0 (Qwen2.5-3B-Instruct), M_SFT, M_RL (Dr.GRPO from M_SFT).
- DBM repo: `differential-circuit-vulnerability`.
- Science-QA masks for all three checkpoints (needed in Stage 7 — do not discard).
- One GPU.

**Pre-registration file** (commit to repo with timestamp before touching data):
- H1: science-QA SFT reduces retention of M0's refusal mechanism.
- H2: M_RL retains more of M0's refusal mechanism than M_SFT does (restoration). Primary comparison is distance(M_RL, M0) vs distance(M_SFT, M0), not M_SFT vs M_RL.
- H3 (secondary): SFT compresses the refusal mechanism into fewer heads; RL keeps it distributed.
- H4 (dissociation): if refusal behaviour recovers under RL, the original mechanism may or may not recover. Readout: cross-tabulate behavioural refusal rate against mechanism retention. Behaviour-recovers-but-mechanism-doesn't = functional replacement (RL routes refusal through different heads); both recover = re-consolidation.
- Success/failure criteria defined relative to the permutation null and split-half ceiling (Stage 6), not raw percentages.

**Time tracking:** Toggl project `MATS-refusal`, entries `data`, `baseline`, `dbm`, `controls`, `analysis`, `writeup`. Rule (state it in the writeup): GPU wait time while doing something else does not count; time watching logs does. Export the report at submission and put total + split in the writeup.

---

## Stage 1 — Refusal dataset (2h)

**Source:** Arditi et al. harmful/harmless instruction sets (public in the `refusal_direction` repo). Harmful: AdvBench, MaliciousInstruct, TDC/HarmBench. Harmless: Alpaca. ~500 each.

**Behavioural filter (M0):** keep only harmful prompts the base model actually refuses and harmless prompts it complies with (greedy decode, Arditi substring refusal judge). A counterfactual pair is only valid if behaviours differ.

**Splits:**
- train — mask learning, ~200 pairs
- val — λ / layer selection, ~50
- test — all behavioural numbers, ~150. Do not touch until Stage 7.

**Format:** apply the Qwen chat template. Store `(harmful_prompt, harmless_prompt, refusal_target, compliance_target)` where targets are the first ~3 tokens the base model produces (e.g., "I cannot", "Sure, here").

---

## Stage 2 — Behavioural baseline (1h)

Per checkpoint (M0, M_SFT, M_RL) on the test split:
- refusal rate on harmful prompts
- compliance rate on harmless prompts
- expected KL from base (paper Eq. 1)

This is Figure 1 and it is free. If SFT and RL show no behavioural difference, note it: the circuit analysis then tests whether internals moved even when behaviour did not.

---

## Stage 3 — Refusal direction baseline (2h) — "the obvious simple thing"

Arditi method, per checkpoint: diff-in-means of residual-stream activations (harmful − harmless) at the last instruction token, per layer; choose the layer by val-set ablation effect.

Report:
- cosine(base direction, SFT direction) and cosine(base direction, RL direction)
- refusal rate after ablating each model's own direction
- **cross-checkpoint transfer:** ablate the *base* direction inside SFT and inside RL. Still kills refusal → mechanism preserved; doesn't → mechanism moved.

**Gate (write the outcome down before Stage 4):**
- Direction fully explains SFT-vs-RL → the writeup says so; the head circuit becomes a secondary localization of *where* the direction is written.
- Direction transfers to RL but not SFT, or drifts differently → the circuit method is justified.

Do not skip or reorder this stage. Running DBM first means defending a fancy method without showing the simple one fell short.

---

## Stage 4 — DBM triplets for refusal (2h)

Map the paper's triplet format onto refusal:
- `x_base` = harmful prompt (model refuses)
- `x_source` = matched harmless prompt (model complies)
- `y_target` = compliance target

The mask selects heads whose harmless-run activations, patched into the harmful run, flip the model to compliance — that is the refusal mechanism, found with the paper's objective: `−log P(y_target | x, ã) + λ Σ_h m_h`.

Engineering issues DBM did not face on MCQ:
- **Position alignment.** Prompts differ in length. Either patch at the last-token position only (consistent with refusal being read at the final instruction token) or left-pad so the last K positions align. Pick one; record it.
- **Target length.** Score with the geometric-mean-of-token-probs (paper Eq. 2) over the short compliance target. Do not score full generations.

**Pilot gate (by Monday night):** run DBM on base only.
- Pass = sparse binary mask converges, faithfulness in the range the paper reported, ablating found heads shifts refusal in the expected direction.
- Fail = fall back to the physics/biology cross-domain plan.

---

## Stage 5 — Circuit discovery on all three checkpoints (5h, mostly GPU)

Same λ, same annealing schedule, same triplets, fixed seeds, for base, SFT, RL. If GPU time allows, **two seeds per checkpoint** (needed for Stage 6 stability). Save per-head mask values, not just the binary set.

Outputs per checkpoint: binary circuit, mask vector over 576 heads, circuit size, faithfulness `F(C|M)/F(M)`.

---

## Stage 6 — Controls (3h) — what makes the result defensible

- **Matched-k.** Rank heads by mask value; compare top-k sets with k = smallest circuit size across the three models. Removes the density confound.
- **Permutation null.** For the observed sizes, draw 1000 random head subsets and compute the IoU distribution. Reported overlaps must sit outside the 95% band. With ~50%-density circuits, chance IoU ≈ 0.33.
- **Split-half ceiling.** Split train triplets in half, learn two masks per checkpoint; IoU between them is the maximum overlap the method can resolve. Base–SFT retention is only interpretable relative to base–base (split-half).
- **Seed stability.** Jaccard between seeds per checkpoint. Reference: "How Alignment Routes" (arXiv:2604.04385) reports 0.92–1.0 under bootstrap.

---

## Stage 7 — Analysis (2h)

- Reproduce the paper's Figure 1 logic on refusal: retention of M0's refusal mechanism in M_SFT and M_RL, with null band and split-half ceiling drawn on the plot.
- IoU matrix base/SFT/RL (paper Figure 8 code, unchanged).
- Δm_h per head; vulnerable-head set `{h : m_SFT < m_RL − δ}`; layer-wise distribution (paper Figures 5/6).
- **Free bonus (recovers the original cross-domain idea):** IoU between the *refusal* circuit and the existing *science-QA* circuit, per checkpoint. Do refusal and reasoning share heads, and does that sharing change under SFT vs RL?
- Optional if under budget (~1.5h): patch the base circuit's heads into the SFT model and measure refusal recovery — converts a correlational overlap into a causal result.

---

## Stage 8 — Stop check

Stop at the first of:
1. 20h logged — write up whatever exists.
2. Stages 2–7 complete.
3. Retention deltas inside the permutation null band — write the null result; it is calibration-positive.
4. Pilot failure at Stage 4 — switch to fallback.

---

## Stage 9 — Writeup (2h)

Order:
1. Question and pre-registered predictions.
2. Infrastructure disclosure, one line: pipeline and checkpoints from arXiv:2605.28860; refusal experiment, triplets, controls, analysis are new; hours logged.
3. Behavioural baseline + direction baseline (and the Stage 3 gate outcome).
4. Circuit result with null band and ceiling.
5. Refusal × science-QA overlap.
6. Limitations: one model, heads only, RL-from-SFT design, narrow-finetune trace caveat (arXiv:2510.13900).
7. Related work — cite the four bracketing papers: arXiv:2609.03887 (post-training methods reshape refusal circuits; SFT/reasoning-FT/ORPO only, no on-policy RL arm), arXiv:2603.23268 (SafeSeek, differentiable-mask safety circuits), arXiv:2509.04259 (RL's Razor), arXiv:2609.01455 (benign FT breaks safety routing; recovery). Also Arditi et al. 2024, Zhao et al. 2507.11878, Patel et al. 2511.07482 (same lineage).
8. Hours spent, with split.

Two figures maximum. Exec summary written last.

---

## Time budget

| Stage | Hours | Day |
|---|---|---|
| 0–1 Setup, data | 3 | Sun Sept 6 |
| 2–3 Behavioural + direction baseline | 3 | Mon Sept 7 |
| 4 Triplets + pilot | 2 | Mon Sept 7 |
| 5 DBM ×3 | 5 | Tue Sept 8 |
| 6 Controls | 3 | Wed Sept 9 |
| 7 Analysis | 2 | Wed Sept 9 |
| 9 Writeup | 2 | Thu Sept 10 |
| Buffer | — | Fri Sept 11 (reruns only) |

Total ≈ 20h.

---

## Fallback (if Stage 4 pilot fails)

Cross-domain circuit sharing on SciKnowEval: Physics–Chemistry pillar pair + Chemistry–Biology control, answer-key-swap triplets only, base/SFT/RL, same controls as Stage 6. Question: does RL continuation restore base-like cross-domain circuit sharing that SFT compressed? Weaker safety fit, but fully inside existing infrastructure.