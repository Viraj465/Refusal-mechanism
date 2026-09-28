# Write-up scaffold

Every number here is traced to a file in `refusal/results/`. Sections marked
**[YOUR VOICE]** must be written by you: the admissions document says LLM-written
executive summaries and form answers are a significant negative signal, and it is
right that they are obvious. Sections marked *(facts)* are numbers and citations
you can lift directly.

Order below is narrative, not chronological. "Chronological rather than narrative
structure" is an explicitly penalised failure.

---

## The one-sentence claim

> Science-only fine-tuning of Qwen2.5-3B-Instruct rotated its refusal direction
> measurably below the method's own resolution floor and cost a small but
> statistically real amount of refusal behaviour, yet left the direction's causal
> power completely intact — ablating the *base model's* direction still removes
> refusal from both fine-tuned checkpoints as completely as their own.

**[YOUR VOICE]** — say this in your own words, and say why you find it interesting.
The honest hook: the obvious metric (cosine similarity of a diff-in-means
direction) and the causal metric disagree, and only one of them is measuring what
we care about.

---

## Why it is not obvious *(facts + [YOUR VOICE])*

- Nothing in the training touches safety. Science-QA SFT has no safety data;
  science-QA Dr.GRPO has no safety reward. Any change to refusal is collateral.
- RL was chained **from** the SFT checkpoint, verified numerically:
  ‖M_RL − M_SFT‖ = 6.90 vs ‖M_RL − M0‖ = 10.26. So "does RL restore what SFT
  disturbed" is a well-posed question here, which it is not in designs that train
  SFT and RL in parallel from the same base.
- Venhoff, Arcuschin, Torr, Conmy & Nanda (arXiv:2510.07364) report that RL
  re-orchestrates pre-existing mechanisms while SFT-distillation installs new
  ones. That concerns machinery *relevant to the objective*. Whether it extends to
  machinery the reward never touches is the gap this fills.

---

## Setup *(facts)*

| | |
|---|---|
| Base model (M0) | `Qwen/Qwen2.5-3B-Instruct`, 36 layers, d = 2048 |
| M_SFT | M0 + completion-only SFT on SciKnowEval chemistry, 2200 examples, 2 epochs, lr 3e-5, full fine-tune |
| M_RL | M_SFT + Dr.GRPO on the same task, 600 prompts, group 16, beta = 0, lr 2e-5 |
| Refusal data | AdvBench 498 + HarmBench 169 + MaliciousInstruct 85, paired to Alpaca on token length ±2 |
| Filter | keep only harmful prompts M0 refuses and harmless it complies with: 752 pairs from 2800 scored |
| Splits | train 200 / val 50 / test 502, seed 0. Test scored once per checkpoint per readout |
| Judge | Arditi/JailbreakBench substring matcher, greedy, 64 new tokens, frozen before any scoring |
| Hardware | 1× NVIDIA L40S 48 GB (Lightning.ai, then AWS g6e.xlarge) |

Pre-registration written before any data was downloaded: `prereg.md`, first
committed `49ebad6`. Deviations D1–D14 are appended in that file, each dated,
including two corrections to my own analysis (below).

---

## Experiment 1 → Figure 1 *(facts)*

`refusal/results/figures/figure1_final.png`

| | M0 | M_SFT | M_RL |
|---|---|---|---|
| Refusal on harmful (n = 502) | 0.992 [0.980, 0.997] | 0.972 [0.954, 0.983] | 0.980 [0.964, 0.989] |
| Compliance on harmless | 1.000 | 0.992 | 0.992 |
| Science new-task score | 42.0% | **77.0%** | 59.0% |
| Forward KL from M0 (harmful / harmless) | — | 0.057 / 0.082 | 0.059 / 0.082 |
| cos with dir_M0, same layer | — | 0.914 (L19) | 0.933 (L21) |
| Drop when dir_M0 is ablated | +0.988 | +0.972 | +0.980 |
| Drop when a random direction is ablated | +0.007 | +0.000 | −0.005 |

Paired exact McNemar on harmful refusal, same 502 prompts:

| Comparison | discordant | p |
|---|---|---|
| M0 vs M_SFT | 12 / 2 | **0.013** |
| M0 vs M_RL | 7 / 1 | 0.070 |
| M_SFT vs M_RL | 2 / 6 | 0.289 |

Split-half direction ceiling (M0, 200× bootstrap, layer 19): 0.9845
[0.9733, 0.9913]. Both fine-tuned cosines fall below the lower bound, so both are
*degraded* by the pre-registered threshold. For scale, two random vectors in
d = 2048 have |cos| ≈ 0.022, so 0.914 is a large absolute overlap — the direction
is degraded relative to the method's resolution, not relative to chance. Say both
numbers.

**One paragraph, [YOUR VOICE]:** behaviour barely moves, the direction rotates
past its own noise floor, and the causal handle does not move at all. The random
control is what makes the third claim mean anything.

---

## Experiment 2 → Figure 2 *(facts)*

`refusal/results/figures/figure2_final.png`

- All three checkpoints show the same step: near-zero effect below layer 17, a
  full removal of refusal from layer 19 upward, flat thereafter.
- Harmful-vs-harmless separation is nearly identical across checkpoints up to
  layer ~20, then diverges: M0 reaches 10.4 at layer 32 and 10.1 at 36, while
  M_SFT reaches 8.4 and 5.7, and M_RL 8.7 and 6.1.
- So fine-tuning degrades the direction **where it is most linearly readable**
  (late layers) and **not where it is causally load-bearing** (layer 19, where
  separation moves only 4.15 → 3.90 → 3.76).

**One paragraph, [YOUR VOICE]:** this is why the cosine and the causal readout
disagree. A cosine at the selected layer is dominated by structure that is not
doing the causal work.

---

## Randomly sampled raw completions *(facts — put these right after the summary)*

Generate with `python refusal/digest.py --examples`. Say explicitly that the
sample is uniform, seeded (seed 0), and not curated; the indices are recorded in
the behaviour JSONs.

Also quote two **ablated** completions from
`refusal/results/direction/inspect_ablation_M0_*.json`. They matter because they
show the judge overstates what ablation achieves: on 8 sampled harmful prompts the
substring judge scored 8/8 as non-refusals, but only 1 was genuine on-topic
compliance. The rest reinterpret the request benignly ("create a tutorial on how
to commit identity theft" → a garbled description of a friend paying for coffee).

---

## What I verified, and how *(facts — this section is worth real marks)*

The admissions document asks for this explicitly. All of it actually happened:

1. **Recomputed the headline behavioural number independently.** Rather than
   trusting the reported rates, I ran an exact paired McNemar test from the
   per-prompt records: 12 refusals lost, 2 gained, p = 0.013.
2. **Read the transcripts by hand** for all three checkpoints, both the normal
   completions and the ablated ones, and found that the substring judge counts
   benign reinterpretation as compliance (prereg D9).
3. **Added the random-direction control** that the pre-registration lacked. Three
   random unit vectors through the identical projection move refusal by at most 8
   prompts in 502.
4. **Checked robustness to batching.** Re-running the M_SFT eval at batch 8 rather
   than 16 changed the transfer ratio by 0.002 and the cosine by 0.0001.
5. **Caught two of my own analysis errors.** (a) The summary initially reported
   cos(M0, M_RL) = 0.643, which compares layer 21 against layer 19 and confounds
   rotation with the selected layer; the same-layer value is 0.933. (b) I briefly
   claimed the causal site "shifted" from layer 19 to 21 under RL, then found that
   M_RL misses a full drop at layer 19 by a single validation prompt out of 32, and
   that M0 shows the same wobble at layers 25 and 29. Both are retracted in
   prereg D12 and D13.
6. **Hardware-level reproducibility, unplanned.** Losing the direction tensors in a
   mid-project machine migration forced a refit on different hardware. It recovered
   the identical selected layer, the identical set of fully-ablating layers, and
   separation profiles matching to a few tenths of a percent.

---

## Threats to validity *(facts + [YOUR VOICE]; lead with the first)*

1. **The narrow fine-tuning trace.** Narrow fine-tuning writes a large, readable
   bias into activation differences even on unrelated text
   ([arXiv:2510.13900](https://arxiv.org/abs/2510.13900)). Some fraction of any
   M0 → M_SFT difference here is the chemistry trace rather than the refusal
   mechanism. What partially answers it: the primary readout is **causal**, not
   representational — a trace in the activations does not explain why ablating M0's
   direction still removes refusal. What does not: isolating the trace needs a
   non-chemistry control prompt set. That is future work, and say so.
2. **The judge measures refusal phrasing, not compliance.** See D9. Report the
   direction as controlling refusal-phrase suppression, never as jailbreaking.
3. **The compliance guard has no discriminative power here.** It passed 13 of 13
   candidate layers for every checkpoint, because it is built on the same substring
   judge and cannot see capability damage.
4. **Measurement floor.** M0's refusal on test is 1.000 by construction, since the
   filter kept only prompts it refused, yet re-measurement returns 0.992. That
   0.8% is the pipeline's batching and bf16 noise. The SFT effect is above it, but
   not far above it.

---

## Limitations *(facts)*

- One model, one task domain, one seed per checkpoint.
- Direction-level only. Head-level circuit analysis was dropped **by scope before
  any observation** (prereg D6), because the format is ~600 words and two figures
  and "circuit finding for its own sake" is a stated red flag. H3 and the
  head-level halves of H2/H4 are **not tested**, never "null".
- RL is undertrained and its science score *fell* (77.0 → 59.0). It is above M0's
  42.0, so it did learn something, but H2 is tested with a weak RL arm.
- cos(dir_M0, dir_RL) at a matched layer 19 was not measured; the 0.933 is at
  layer 21.

---

## The result on H2, stated honestly *(facts)*

Both differences lean the way H2 predicted — M_RL is nearer M0 than M_SFT is,
behaviourally (0.980 vs 0.972) and representationally (0.933 vs 0.914) — but
neither is distinguishable from noise (McNemar p = 0.29). **H2 is reported as a
null, not as a weak positive.** The pre-registration named that outcome in advance
as acceptable, and the admissions document states that well-analysed negative
results beat poorly supported positive ones.

Do not lead with "RL restores refusal". Lead with the dissociation.

---

## Infrastructure disclosure *(facts — one paragraph, do not overclaim)*

Reused from arXiv:2605.28860: the o_proj-input patching site,
`check_answer_correctness`, `UnifiedDatasetInterface`, and `evaluate_new_task`.
New for this project: the refusal dataset, the direction analysis, the training
chain, the random-direction control, and all analysis. **Checkpoints were trained
for this project, not inherited** — the prior work's are not on disk (prereg D1).
State the GPU model and the hours.

---

## Hours

Export Toggl and paste the screenshot. GPU setup, waiting for training, breaks and
the application form do not count. Watching logs does. Report the total with a
split.

---

## Checklist before submitting

- [ ] Google Doc link sharing is on for anyone with the link
- [ ] Executive summary ≤ 600 words, ≤ 3 pages, both figures in it
- [ ] Random completions block sits immediately after the summary
- [ ] Every comparative number appears next to its reference: Wilson CIs, the
      split-half ceiling, the random-direction control, the 0.5 / 0.8 transfer bands
- [ ] Form answers written in your own voice — they are read first and used as the filter
- [ ] Deadline: **Sept 11, 11:59pm PT = Sept 12, 12:29pm IST**
