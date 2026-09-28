# Runbook — Stage 0 complete → submission

**Written:** Mon 2026-09-07. **Deadline:** Thu 2026-09-11 (Fri = buffer).
Companion to `prereg.md` (frozen) and `../mats_refusal_implementation_plan.md` (the plan).
This file is operational only. It changes *when and where* things run, never *what is claimed*.

---

## 0. Where you actually are

| Stage-0 item | State |
|---|---|
| `prereg.md` committed before any observation | Done — `49ebad6`, Mon 12:11 IST |
| Repo layout (`refusal/` beside `RLRazor/`) | Done — package exists, only `prereg.md` in it |
| Prior repo available to import from | Done — `nnsj/differential-circuit-vulnerability/RLRazor/src` |
| 0b environment | **Deferred to the rented box** — do not install locally |
| 0d checkpoint decision | **Resolved: Case B.** No `results/`, no checkpoints on disk. Retrain, RL chained from SFT. |
| 0a message co-authors | Send it, but schedule as if it will never be answered (see §5) |

Verified present in the cloned repo (so the plan's imports are real, not assumed):
`train_sft` (`training.py:31`), `train_grpo(..., loss_type='dr-grpo')` (`training.py:521`),
`evaluate_new_task` (`evaluation.py:971`), `compute_forward_kl` (`evaluation.py:172`),
plus the science `.jsonl` files under `RLRazor/src/data/science/`.

**Log this in `prereg.md` §9 today**: Case B confirmed on 2026-09-07; checkpoints trained for this
project, not inherited. This is the deviation entry that keeps §7.1 honest.

---

## 1. What renting a GPU changes about the plan

Nothing about the science. Three things about the execution:

1. **The box is metered and ephemeral.** Authoring code on it is the single largest way to waste
   money and hours. Split the work: *write everything locally with no GPU, run it remotely in
   short attended bursts.*
2. **Checkpoints must leave the box the moment they exist.** Push to a private HF repo
   (`hf upload` right after each `save_pretrained`) or a persistent volume. A terminated instance
   with M_SFT on it costs you Tuesday.
3. **Toggl and the meter now diverge.** Prereg budget rule is unchanged (watching logs counts,
   walking away does not). Track GPU $ separately; it is not part of the 20h.

---

## 2. Phase A — local, no GPU — **DONE (Mon Sept 7)**

All authored and smoke-tested on CPU: **13 of 14 tests pass**, the one skip needs `trl` and
runs on the box as the last step of `env/setup.sh`.

- [x] `refusal/common.py` — paths, seeding, timestamped non-clobbering results IO, Wilson CI,
      Jaccard, and the **frozen** Arditi/JailbreakBench refusal substring list (prereg §2).
- [x] `refusal/science_data.py` — the missing science loader (see finding 1 below).
- [x] `refusal/data/build_dataset.py` — `--build` (CPU, **already run**: 800 harmful / 2000
      harmless in `refusal_prompts_raw.jsonl`) and `--filter` (GPU, Phase B).
- [x] `refusal/behaviour.py` — refusal/compliance + Wilson CI, science NTS, forward KL from M0
      reported separately on harmful and harmless.
- [x] `refusal/direction.py` — per-layer diff-in-means, ℓ* selection on val **with a harmless
      compliance guard**, directional ablation, transfer, addition, 200× bootstrap ceiling.
- [x] `refusal/dbm.py` — annealed mask over 576 heads, last-instruction-token patching, source
      activations cached once, faithfulness, prereg §7.2 pilot gate wired in.
- [x] `refusal/train_chain.py` — SFT → save → push → RL-from-SFT → save → push, plus
      `--stage chain-check` for ‖θ_RL−θ_SFT‖ vs ‖θ_RL−θ_M0‖.
- [x] `env/setup.sh` + `env/requirements-remote.txt` — bare box to ready in one command.
- [x] `refusal/smoke_test.py` — tiered, skips cleanly on missing deps.

### What the build found (logged in `prereg.md` §9 as D1–D4)

Three plan assumptions that turned out to be false:

1. **`src/data/load_data.py` does not exist.** Every entry point in the prior repo imports
   `load_dataset_byname` from it. Reimplemented as `refusal/science_data.py` over the six
   committed chemistry `.jsonl` files (2400 rows). The `UnifiedDatasetInterface` normaliser
   *does* work on that schema — verified, so SFT is not blocked.
2. **The prior repo's GRPO reward would have scored every rollout 0.0.** `train_grpo` recovers
   the question by regex from the rendered prompt; none of its four patterns match the science
   template `"<instructions>\n<question>\n### Answer\n"`, so the ground-truth lookup fails
   silently and RL trains on a flat signal. `train_chain.py` passes the answer through as a
   dataset column instead. **This one would have cost the whole Tuesday RL run.**
3. **Pairing, not the filter, is what loses prompts.** An 800/800 pool pairs only 563 of 800
   harmful prompts inside the ±2-token window; 1500+ harmless prompts pairs all 800. The
   harmless pool is now 2000, protecting the registered 200/50/150 splits against filter
   attrition. The harmful pool (800) is effectively exhausted — the three sources yield 820
   unique prompts in total.

The `--filter` pass now scores 2800 prompts rather than 1600, so budget ~25 GPU-minutes for it
rather than ~15.

---

## 3. Phase B — GPU session 1, tonight/early Tue (~3–4h metered)

One box, 40–48 GB. Order is chosen so the two cheapest gates fire first.

1. `setup.sh`, load Qwen2.5-3B-Instruct bf16, confirm it generates. (15 min)
2. **M0 behavioural filter** — `build_dataset.py --filter`, ~1000 prompts × 64 tokens. (~15 min)
   Record filter yield. This is the dataset for everything downstream; push the JSONL off the box.
3. **Direction spine on M0 alone** (~45 min). Fit dir_M0, pick ℓ*, ablate M0's own direction,
   run the 200× split-half bootstrap ceiling.
   > **Gate (prereg §8, third bullet):** if dir_M0 does not ablate refusal in M0 itself, the Arditi
   > operationalization does not hold here and the transfer ratio is undefined. Find that out now,
   > for 45 GPU-minutes, not on Wednesday.
4. **SFT** — `train_chain.py --stage sft`. ~2200 samples, 2 epochs, eff. bs 32, lr 3e-5.
   Full fine-tune, `adamw_8bit` + gradient checkpointing (~25–30 GB for a 3B).
5. **NTS gate** on 200 held-out science items. Target ≥ 60%. Below that, SFT did not learn the task
   and H1 is untestable — retune lr/epochs before spending anything on RL.
6. Push M_SFT + all JSON to HF. Terminate the box.

Deliverable at end of session 1: a filtered dataset, a validated direction method, M_SFT, and the
knowledge of whether the spine holds. If everything after this fails you still have a submission.

---

## 4. Phase C — GPU session 2, Tuesday (~4–5h)

**Rent two boxes for this window only.** RL is the long pole (600 prompts × 16 generations × 256
tokens) and runs unattended; the DBM pilot is short and attended. On one GPU the pilot sits behind
~3h of RL and Wednesday collapses. Two boxes for ~4h is the cheapest hour you buy all week.

- **Box A (80 GB, unattended):** `train_chain.py --stage rl` from M_SFT. `loss_type='dr-grpo'`,
  `beta=0`, `num_generations=16`, `max_completion_length=256`, 1 epoch, lr 2e-5. Log NTS and the
  parameter-distance chain check. Push M_RL the instant it saves.
  *If 80 GB is unavailable:* drop to `num_generations=8`, `max_completion_length=192` on 48 GB and
  log it as a deviation. Do not silently shrink the group size.
- **Box B (40–48 GB, attended):** DBM pilot on M0, val split, 1 seed — prereg §7.2's four-part gate
  (converges / size 5–60% of 576 / faithfulness ≥ 0.8 / ablation moves refusal the right way).
  Then Stage 2 behavioural + Stage 3 direction readouts on M0 and M_SFT.

**End of Tuesday, write the Stage 3 gate outcome into `prereg.md` before any Stage 5 work.**
That ordering is registered; doing it out of order costs you the pre-registration's value.

**Case C checkpoint (Wed noon, prereg §7.1):** if RL has no usable NTS by then, drop to M0 vs M_SFT,
H1 + H4 only, RL as future work. Decide it on the clock, not on how the RL run "feels".

---

## 5. Phase D — GPU session 3, Wednesday (~4–5h) — the tight one

Only if the pilot passed. The head-level plan needs **12 mask fits**:

| Fits | Purpose |
|---|---|
| 3 ckpt × 2 seeds = 6 | Main circuits + seed stability (prereg §5.3, Jaccard ≥ 0.7) |
| 3 ckpt × 2 halves = 6 | Split-half ceiling (prereg §5.3) |

Plus a λ sweep over {0.01, 0.03, 0.1} on val, M0 only, before the 12.

**This sets a hard design constraint on `dbm.py`: one fit must take ≤ 20 min.** 300 steps over ~200
triplets with source activations cached and patching at a single position should land well inside
that — but if the Phase-A implementation recomputes source activations per step (the failure mode
the audit found in the original repo), the Wednesday session does not fit and the ceiling control
is what gets cut. Budget the fits before you start, not after.

Cut order if Wednesday runs long: split-half halves → second seed → M_RL's second seed. Never cut
the permutation null (it's free, CPU) and never cut seed stability entirely — prereg §5.3 makes H2
uninterpretable without it.

---

## 6. Phase E — local, no GPU, Wed night → Thu

- `refusal/controls.py` — permutation null (1000 subsets, pure CPU, seconds), matched-k, split-half
  IoU, seed Jaccard, direction bootstrap. Write this Tuesday *while Box A runs RL*; it needs no GPU
  and no results, only the mask format.
- `refusal/analysis.py` — Figure 1 (behavioural + direction), Figure 2 (retention with null band and
  ceiling drawn on it), H4 cross-tab, Δm_h, layer histogram. Two figures maximum.
- Writeup in the prereg's order. Exec summary last, ≤2h, outside the 20h.
- Export Toggl, report total + split, state the GPU model in the infra line.

---

## 7. Three decisions to make before you rent

1. **Full fine-tune, not LoRA.** LoRA confines the weight change to a low-rank subspace, which is
   exactly the thing under test when you ask whether a mechanism moved — it is a confound, not just
   an efficiency choice. 3B full FT with an 8-bit optimizer fits in 40 GB. If VRAM forces LoRA
   anyway, it goes in `prereg.md` §9 and in Limitations, both.
2. **Two boxes Tuesday, one otherwise.** Justified in §4.
3. **Co-authors: ask, don't wait.** Send the Case-A request today. A checkpoint arriving after
   Tuesday noon cannot be used — you'd be re-running Stage 2/3 on new checkpoints with two days
   left. Set that cutoff now so the reply doesn't tempt you into a restart.

---

## 8. Honest risk list

- **Wednesday is over-subscribed.** 12 mask fits + controls + analysis in one day, with the pilot
  gate only cleared Tuesday night. §7.2's abandon-head-level clause is the real safety valve, and
  it is registered — use it without embarrassment if λ selection eats the morning.
- **RL undertrained** is likely at this budget. Prereg already handles it (report NTS, cite the
  prior work's Fig. 2 low-NTS caveat). It weakens H2's power; it does not invalidate the design.
- **M_SFT may not degrade refusal at all** (prereg §8, bullet 2). That is a reportable result, and
  the direction spine still gives a full submission. Do not let it push you into unregistered
  analyses hunting for an effect.
- **The single genuinely fatal path** is losing M_SFT or M_RL to a terminated instance. Push after
  every save. Verify the push before terminating.
