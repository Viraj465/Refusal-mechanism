# Execution guide — everything that remains

**All code is written and CPU-verified (18/19 smoke tests; the 19th needs `trl` and runs on the
box).** Nothing below requires authoring anything new. It is: rent, run these commands in this
order, read the gates, write the report.

**Scope: direction-level only** (prereg §9 D6). `dbm.py` and `controls.py --mode stats` are written
and tested but are **not run** — the submission format is ~600 words and two graphs, and "circuit
finding for its own sake" is a named red flag in the admissions document. Two figures still come out;
Figure 2 is the H4 dissociation table. **≈ 7–9 GPU-hours, ~$20.**

Companion documents: `prereg.md` (binding — hypotheses, thresholds, contingencies),
`RELATED_WORK.md` (positioning and threats, written), `NEXT_STEPS.md` (why the phases are ordered
this way).

Run everything **from the repository root**, not from inside `refusal/`.

---

## 0. Preflight — before you rent anything (15 min, free)

```bash
python refusal/smoke_test.py          # expect 18 passed, 1 skipped (needs trl)
git add .gitignore env refusal/ && git commit -m "refusal pipeline, CPU-verified; prereg D1-D7"
```

**Request the AWS quota now — it is the only item with real lead time.** New accounts have 0 vCPUs
for On-Demand G instances. Service Quotas → EC2 → *"Running On-Demand G and VR instances"* → request
**16 vCPUs**, in `us-east-1` or `us-west-2`. Approval takes hours to a day.

| Decision | Take this | Why |
|---|---|---|
| Instance | **`g6e.2xlarge`** (1 × L40S 48 GB, ~$2.24/hr) | AWS has no single-A100 box; 24 GB A10G cannot hold a full 3B fine-tune without an OOM fight |
| Full fine-tune vs LoRA | **Full FT**, `--optim adamw_bnb_8bit` | LoRA confines the weight change to a low-rank subspace — a confound for "did the mechanism move", not just an efficiency choice |
| Pricing | **On-demand, not Spot** | A Spot reclaim mid-RL costs the day |
| Co-author checkpoints | Ask today, **hard cutoff Tue noon** | A checkpoint arriving Wednesday forces re-running everything with two days left |

Set `HF_TOKEN` on the box before anything else — every checkpoint gets pushed the moment it exists.

---

## 1. Session 1 — one `g6e.2xlarge`, ~3–4h metered

```bash
# nnsj/ is gitignored, so the prior project does NOT come with your repo clone.
# Without it the science data and three imports are missing and SFT/RL/NTS all break.
mkdir -p nnsj && git -C nnsj clone <differential-circuit-vulnerability remote>

bash env/setup.sh --no-install         # DLAMI already ships torch; do not let pip downgrade it
pip install -r env/requirements-remote.txt
```

Use the **Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)**, 150 GB `gp3` root
volume. `setup.sh` ends by running the smoke tests, which fail loudly if the `nnsj/` clone is
missing.

### 1a. Behavioural filter (~25 min)

```bash
python refusal/data/build_dataset.py --filter --batch-size 16
```

Scores all 2800 prompts on M0, keeps harmful-refused + harmless-complied, length-matches, writes
`refusal_pairs.jsonl`, `splits.json` (200/50/150), `filter_report.md`.

> **Hard stop:** the script refuses to write short splits. If fewer than 400 pairs survive it exits
> and tells you so — that is deliberate. Re-run with `--allow-short-splits` only after adding a
> `prereg.md` §9 deviation entry.

**Push `refusal/data/*.jsonl` and `splits.json` off the box now.** Everything downstream depends
on this exact file; regenerating it later with a different filter run invalidates the split
discipline.

### 1b. Direction spine on M0 alone (~45 min) — THE CHEAP GATE

```bash
python refusal/direction.py --mode fit     --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
python refusal/direction.py --mode ceiling --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
```

`--mode fit` sweeps candidate layers on **val**, rejecting any direction whose ablation also
destroys harmless compliance, and prints the selected L\*.

> **Gate (prereg §8, third bullet).** If it raises
> `"No candidate direction ablates refusal without destroying harmless compliance"`, the Arditi
> operationalisation does not hold for this model. The transfer ratio is then undefined and the
> submission becomes a **negative methodological result** — registered in prereg §8, reportable, and
> far better discovered here for 45 GPU-minutes than after training two checkpoints. Write the
> outcome into `prereg.md` before continuing.

### 1c. SFT (~40 min) and its gate

```bash
python refusal/train_chain.py --stage sft --push-to <user>/mats-m-sft
```

> **Gate:** science NTS ≥ 60% on 200 held-out items. Below that, SFT did not learn the task, H1 is
> untestable, and you retune lr/epochs **before** spending anything on RL. The script prints the
> gate verdict.

### 1d. Baselines for what exists so far

```bash
python refusal/behaviour.py --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
python refusal/behaviour.py --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT \
    --ref Qwen/Qwen2.5-3B-Instruct
```

**Before stopping the instance:** push `M_SFT`, `refusal/results/`, and the dataset. **`stop`, do
not `terminate`** — the EBS volume persists and compute is free while stopped. Losing M_SFT to an
accidental terminate is the only genuinely fatal failure in this plan.

---

## 2. Session 2 — Tuesday, one `g6e.2xlarge`, ~4–5h

### RL from SFT (the long pole, ~3h unattended)

```bash
python refusal/train_chain.py --stage rl \
    --init refusal/results/checkpoints/M_SFT \
    --num-generations 16 --max-completion-length 256 \
    --push-to <user>/mats-m-rl
```

If 48 GB proves tight: `--num-generations 8 --max-completion-length 192`, and log the reduction as a
deviation. Do not silently shrink the group size. `beta=0` means no reference model is held, which
is a large part of why this fits on one card.

Watch the first 20 log lines for `rollout reward rate`. **If it is 0.000, stop the run** — the
reward is not firing and you are burning GPU on a flat signal. (This is exactly the prior repo's
bug, fixed in `train_chain.py`; a zero here means something else is wrong.)

Then:

```bash
python refusal/train_chain.py --stage chain-check \
    --m0 Qwen/Qwen2.5-3B-Instruct \
    --sft refusal/results/checkpoints/M_SFT \
    --rl refusal/results/checkpoints/M_RL
```

Confirms ‖θ_RL − θ_SFT‖ ≪ ‖θ_RL − θ_M0‖. If it does not hold, H2 must be reworded per prereg §2
and logged in §9.

### Direction readouts, same box, while RL runs elsewhere

```bash
python refusal/direction.py --mode fit  --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT
python refusal/direction.py --mode eval --checkpoint refusal/results/checkpoints/M_SFT --tag M_SFT \
    --transfer-from M0
python refusal/direction.py --mode eval --checkpoint Qwen/Qwen2.5-3B-Instruct --tag M0
```

> **Not run: `dbm.py` and `controls.py --mode stats`** (prereg §9 D6). The head-level block is a
> scope decision, not a gate failure — the submission format is ~600 words and two graphs, and
> "circuit finding for its own sake" is a stated red flag in the admissions document. The code is
> written and tested and stays in the repo. `analysis.py` produces Figure 2 as the H4 dissociation
> table automatically when no masks exist; nothing needs changing to run this way.

When M_RL lands, add its readouts:

```bash
python refusal/behaviour.py --checkpoint refusal/results/checkpoints/M_RL --tag M_RL \
    --ref Qwen/Qwen2.5-3B-Instruct
python refusal/direction.py --mode fit  --checkpoint refusal/results/checkpoints/M_RL --tag M_RL
python refusal/direction.py --mode eval --checkpoint refusal/results/checkpoints/M_RL --tag M_RL \
    --transfer-from M0
```

**End of Tuesday: write the Stage 3 gate outcome into `prereg.md`.** It is a registered step and it
is the record that the direction result was interpreted on its own terms.

**Case C checkpoint (Wed noon, prereg §7.1):** if RL has no usable NTS by then, drop to M0 vs
M_SFT, H1 + H4 only, RL as stated future work. Decide it on the clock, not on how the run feels.

---

## 3. Analysis — CPU, no GPU, from saved JSON

```bash
python refusal/analysis.py                   # both figures + analysis_summary.md
```

That is the whole analysis step. `controls.py --mode stats` is **not** run — it scores head-level
masks that this scope does not produce (prereg §9 D6). `analysis.py` handles their absence: Figure 2
becomes the H4 dissociation table, and every head-level row in the summary reads "not measured".

Outputs:

- `refusal/results/figures/figure1_behaviour_direction.png`
- `refusal/results/figures/figure2_retention.png`
- `refusal/results/analysis_summary.md` — every registered readout with its verdict, plus the H4
  cross-tab under both operationalisations

Preview the layout any time with `python refusal/analysis.py --demo` (synthetic, watermarked).

---

## 4. Writeup — Thursday, 2h + exec summary (≤2h, last)

**This is what is actually graded.** Code is optional in this application ("I'll largely use it to
give my agents context"), so the write-up is the deliverable, not the pipeline.

Two deliverables:

| Artefact | Limit | Contains |
|---|---|---|
| **Executive summary** | **1–3 pages, ~600 words max** | Problem + why it matters; the takeaway; **one paragraph + one graph per key experiment** (two experiments = two graphs); randomly sampled real completions |
| **Google Doc (findings)** | no stated limit | The full narrative below, with `analysis_summary.md` numbers |

Plus the application form answers, which are read **first** and used as a preliminary filter — write
them yourself, not with an LLM; obviously-LLM-written form answers and summaries are a named red
flag.

**Draw the prose for positioning, threats and related work from `RELATED_WORK.md` — it is written
and argued already; do not re-improvise it at 1am.**

### Structure: narrative, not chronological

**"Chronological rather than narrative structure" is an explicitly penalised failure.** Do not write
"first I built the dataset, then I trained SFT, then I measured…". Lead with the claim, then the
evidence, then what would have falsified it:

1. **The claim, in one sentence** — what happened to refusal, and what that says about whether RL's
   preservation extends to machinery the reward never touches.
2. **Why it is not obvious** — the framing from §"headline claim" below and `RELATED_WORK.md` §4.
3. **Experiment 1 → Figure 1** — behaviour and the direction, all three checkpoints. One paragraph.
4. **Experiment 2 → Figure 2** — the H4 dissociation. One paragraph.
5. **Real completions** — the seeded random sample from `example_completions` in the behaviour JSON.
   Say explicitly that it is a uniform seeded sample, not curated. Both red flags ("not looking at
   actual data", "cherry-picked examples without random sampling") are answered by this one block.
6. **What would have falsified this**, and what did not survive — the pre-registered thresholds, the
   Stage-3 gate outcome, and any null. *Well-analysed negative results beat poorly supported
   positive ones* — say so with the numbers, do not hedge.
7. **Threats to validity** — lead with the narrow-finetuning trace. `RELATED_WORK.md` §5.
8. **What was not done and why** — head-level analysis dropped by scope (prereg §9 D6), stated in
   one sentence as a decision, not an omission.
9. **Infrastructure disclosure** (below), **limitations**, **related work**, **hours with split**.

### Baselines and calibration — both are named criteria

"No baselines for comparative claims" and "overconfidence in shaky results" are listed red flags.
Every comparative number in the writeup must appear next to its reference: refusal rates with Wilson
CIs, direction cosines against the **bootstrapped split-half ceiling** (not against 1.0), transfer
ratios against the frozen 0.5 / 0.8 bands. Figure 1 already draws all three.

### The headline claim (prereg §9 D5 — settled 2026-09-07, before any observation)

Do **not** lead with "RL restores refusal". [2510.07364](https://arxiv.org/abs/2510.07364)
(Venhoff, Arcuschin, Torr, Conmy & Nanda) already reports that RL re-orchestrates pre-existing
mechanisms while SFT-distillation installs new ones, which implies the H2 ordering. Lead instead
with the question that paper does not answer:

> **Does RL's preservation extend to machinery the reward never touches?**
> Their result concerns mechanisms relevant to the training target. Nothing in the science-QA
> objective touches safety, so refusal is collateral. H4 — whether behaviour and mechanism come
> apart — is the second axis.

Report a confirmed H2 as a **replication in a new regime**, not a discovery. The informative
outcomes are H2 *falsified* and H4 *functional replacement*. Full argument: `RELATED_WORK.md` §4.

### Required paragraphs

**Infrastructure disclosure (D4 narrowed this — do not overclaim):** reused from arXiv:2605.28860
are the o_proj-input patching site, `check_answer_correctness`, `UnifiedDatasetInterface`, and
`evaluate_new_task`. New here: the refusal dataset, the direction analysis, the training chain, and
all analysis. Checkpoints were **trained for this project**, not inherited — the prior work's are
not on disk. State the GPU model and hours.

**Threats to validity must open with the trace objection, not bury it.** Narrow fine-tuning writes a
large readable bias into activation differences even on unrelated text
([2510.13900](https://arxiv.org/abs/2510.13900), co-authored by the person reading this), so some
fraction of any M0→M_SFT difference here is the chemistry trace rather than the refusal mechanism.
State what partially answers it — chiefly that **direction transfer is a causal readout, not a
representational one** — and state plainly that this is partial: isolating the trace needs a
non-chemistry control prompt set, which is future work. `RELATED_WORK.md` §5 has the argument.

**What was not done:** head-level circuit analysis was dropped by scope, before any observation
(prereg §9 D6). One sentence, framed as a decision. H3 and the head-level halves of H2/H4 are
reported as **not tested** — never as null results.

**Remaining limitations:** one model; direction-level only; RL possibly undertrained (give the NTS);
and, per D4, that parts of the inherited pipeline were rebuilt rather than reused.

**Claim scope, throughout:** alignment durability under post-training. Never "deceptive alignment
detection".

---

## 5. Stop rules (prereg §7.3) — stop at the first that fires

1. 20h logged → write up what exists. (GPU setup and waiting for training do **not** count.)
2. All registered direction-level readouts complete.
3. Deltas inside their reference bands → **write the null.** It is calibration-positive: the
   admissions document states that well-analysed negative results beat poorly supported positive
   ones.
4. Stage 3 gate fails (dir_M0 does not ablate refusal in M0) → report it as a negative
   methodological result, per prereg §8.
5. Case C → M0 vs M_SFT only, H1 + H4.

Every rung is a complete, honest submission. **Never present a higher rung's framing with a lower
rung's evidence.**

---

## 6. Budget

| Session | GPU | Wall | Notes |
|---|---|---|---|
| 1 | 1 × `g6e.2xlarge` (L40S 48 GB) | 3–4h | filter, direction gate on M0, SFT |
| 2 | 1 × `g6e.2xlarge` | 4–5h | RL from SFT, then direction + behaviour on all three |
| Analysis + writeup | none | ~5h | CPU only |

**≈ 7–9 GPU-hours, ~$18–20** on one on-demand `g6e.2xlarge` (~$2.24/hr), plus ~$3 EBS. One box, not
two — the second Tuesday box was only for the DBM pilot, which is no longer run.

Between sessions, **stop the instance, do not terminate it**: the EBS volume persists at ~$0.08/GB-
month and compute costs nothing while stopped. Push checkpoints to HF or `aws s3 sync` anyway, in
case of an accidental terminate.

Time tracking: the admissions document excludes **GPU setup and waiting for training** from the
16–20h research budget, along with reading papers and breaks. Watching logs still counts. Screenshot
the Toggl report — it is explicitly encouraged.
