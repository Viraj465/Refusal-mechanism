# Related work and positioning

Prose the writeup draws from. Written **2026-09-07, before any observation** — this is framing, not
interpretation of results. Nothing here changes a hypothesis, readout, or threshold; the
corresponding note is `prereg.md` §9 D5.

---

## 1. What the refusal-direction line established

Arditi et al. (2024), [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717),
showed that in chat-tuned models refusal is mediated by a single residual-stream direction:
projecting it out of every residual-stream write disables refusal with little collateral damage,
and adding it induces refusal on harmless prompts. That method — diff-in-means at the last
instruction token, directional ablation, activation addition — **is** this project's Stage 3, and
its two failure modes are what the controls here defend against: a candidate direction whose
ablation simply breaks the model (handled by the harmless-compliance guard during layer selection),
and a cosine read against 1.0 rather than against the method's own resolution (handled by the
bootstrapped split-half direction ceiling).

The line has since fragmented — a single direction (Arditi), multiple directions, and a late-layer
routing pathway are all in play. Hence the terminology discipline in `prereg.md` §2: *refusal
mechanism* means whatever the chosen method recovers under a harmful→harmless counterfactual, and
the phrase "the refusal circuit" is not used.

**What the line does not ask:** what happens to that direction under post-training whose objective
says nothing about safety.

## 2. What the model-diffing line established

Venhoff, Arcuschin, Torr, Conmy & Nanda (2026),
[*Base Models Know How to Reason, Thinking Models Learn When*](https://arxiv.org/abs/2510.07364),
diff base→fine-tuned across nine pairs (four RL-trained, four SFT-distilled, one mixed) using
sentence-level SAE category vectors, and conclude that **RL primarily teaches heuristics for
orchestrating pre-existing base mechanisms, whereas SFT-distillation installs new ones** — hybrid
models recover ~76% of the RL base-to-thinking gap but only ~11% of the SFT gap.

This is the closest published result to H2 and must be engaged directly (§4 below).

Minder, Dumas et al. (2025),
[*Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning*](https://arxiv.org/abs/2504.02922),
is the methodological warning: naive model diffing produces artifacts that **falsely mark concepts
as unique to the fine-tuned model**, and correcting for them changes the conclusions. An IoU
between two head sets that each cover ~50% of all heads is exactly this class of number — it looks
like a finding and is not one. The permutation null and split-half ceiling in `controls.py` exist
for that reason and should be cited as such, not presented as generic rigour.

Turner, Soligo, Taylor, Rajamanoharan & Nanda (2025),
[*Model Organisms for Emergent Misalignment*](https://arxiv.org/abs/2506.11613), show narrow
harmful fine-tuning producing broad misalignment mediated by a single direction, inducible with a
rank-1 LoRA. This project is the mirror image — narrow *benign* fine-tuning, asking whether a later
capability stage repairs rather than breaks alignment. It is also the strongest argument for the
full-fine-tune-over-LoRA decision here: if a rank-1 adapter suffices to move alignment, a low-rank
training constraint is a confound for "did the mechanism move", not merely an efficiency choice.

## 3. The gap

Neither line asks whether a **capability-only RL stage restores a safety mechanism that a
capability-only SFT stage eroded**.

The nearest claim is [2609.01455](https://arxiv.org/abs/2609.01455) (*When Safety Routing Breaks*,
EMNLP Findings 2026), which reports that a few safety examples restore refusal because internal
safety-relevant representations are preserved. The distinction to state explicitly, because it is
the whole design: **the restoring stage here contains no safety data and no safety reward at all.**
Restoration by safety examples and restoration by a chemistry reward are different phenomena.

## 4. The strongest objection, and the answer

**The objection.** If RL re-orchestrates pre-existing machinery while SFT installs new machinery
(2510.07364), then RL should disturb pre-existing safety circuitry less than SFT — which is H2. A
confirmed H2 is therefore a weaker update than a naive reading suggests, and the paper predicting
it was written by the person reading this.

**The answer, in three parts.**

1. **Their mechanism explains why RL preserves what it *uses*. It says nothing about what happens
   to what it *ignores*.** Their claim concerns machinery relevant to the training target: RL
   re-fires reasoning mechanisms it needs. Safety circuitry is irrelevant to a chemistry reward.
   Whether preservation extends to machinery the objective never touches is a separate claim that
   happens to point the same direction. Distinguishing the two is the contribution.
2. **They do not predict H1 at all.** Nothing in that paper addresses what SFT does to machinery
   unrelated to its target. What it loosely implies is the *ordering* of SFT and RL, not the
   magnitudes, and not the collateral case. How much refusal machinery science SFT displaces when
   it has no reason to is unaddressed.
3. **Restoration is structurally untestable in their design.** Their RL and SFT models are separate
   parallel pairs. "Does RL repair what SFT broke" requires the chain M0 → M_SFT → M_RL.

Convergent evidence across units — SAE category vectors there, a direction and an attention-head
set here — is corroboration, not redundancy.

**Consequence for how results are reported** (registered in `prereg.md` §9 D5): the most
informative outcomes are now **H2 falsified** (RL disturbs safety machinery *more* than SFT, which
would contradict the orchestrate-vs-install story in the collateral regime) and **H4 functional
replacement** (refusal behaviour returns through different heads — which "re-uses pre-existing
mechanisms" does not predict). A clean H2 confirmation is the least surprising outcome and is to be
reported as a replication in a new regime, not as a discovery.

## 5. Threats to validity, argued

**The narrow-finetuning trace.** Minder, Dumas, Slocum, Casademunt, Holmes, West & Nanda (ICLR
2026), [*Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences*](https://arxiv.org/abs/2510.13900),
show that narrow fine-tuning writes a large, readable bias into activation differences — present
even on text unrelated to the fine-tuning domain, across 1B–32B models. Science-QA SFT is narrow
fine-tuning. **So some fraction of any M0→M_SFT difference measured here is the chemistry trace,
not the refusal mechanism.** This is the single strongest objection to the design and is stated in
the writeup before a reviewer raises it.

What partially answers it, and what does not:

- **Matched-k** removes the density confound: comparing equal-size top-k head sets prevents a
  diffuse bias that inflates one circuit's size from masquerading as reduced overlap.
- **The permutation null** establishes what overlap means at the observed densities, so a trace-
  driven shift has to clear chance before it is reported as anything.
- **The direction-transfer readout is causal, not representational.** Ablating dir_M0 *inside*
  M_SFT and measuring whether refusal still collapses asks whether the mechanism still routes
  through that direction — a question a diffuse additive activation bias does not obviously answer
  in either direction. This is the readout least exposed to the objection, which is one reason it
  is the spine.
- **What none of these do** is isolate the trace. A clean answer needs a third, non-chemistry
  control prompt set, or model diffing against a differently-fine-tuned checkpoint. That is stated
  as future work, not claimed.

**Other limitations** (unchanged from `prereg.md` §7): one model; attention heads only; RL possibly
undertrained — report the NTS; the direction-vs-circuit operationalisation gap; and, per §9 D4,
that parts of the inherited pipeline were rebuilt rather than reused.

## 6. Citation set for the writeup

**Directly built on / directly engaged:** Arditi et al. 2024 ([2406.11717](https://arxiv.org/abs/2406.11717));
Venhoff et al. ([2510.07364](https://arxiv.org/abs/2510.07364));
Minder et al. ([2510.13900](https://arxiv.org/abs/2510.13900));
Minder/Dumas et al. ([2504.02922](https://arxiv.org/abs/2504.02922));
Turner & Soligo et al. ([2506.11613](https://arxiv.org/abs/2506.11613)).

**Bracketing the question:** 2609.03887 (post-training methods reshape refusal circuits);
2609.01455 (benign FT breaks safety routing; recovery); 2603.23268 (SafeSeek, differentiable-mask
safety circuits); 2509.04259 (RL's Razor — the KL-minimality claim the prior work rests on);
2604.04385 (routing circuits; the 0.92–1.0 bootstrap Jaccard reference for seed stability);
2507.11878; 2511.07482; arXiv:2605.28860 (the prior work whose retention figures are the claim
under test).

> Verify every ID and author list before submission. Four of the five in the first group were
> checked on 2026-09-07; miscitation in an application to one of the authors is expensive.
