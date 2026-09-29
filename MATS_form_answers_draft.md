# MATS 12.0 form answers — DRAFT (rewrite in your own voice before pasting)

Every number below is taken from prereg.md, analysis_summary.md and the write-up docx.
Items in [BRACKETS] are things only you can fill or confirm. Do not paste any answer as-is:
the last question asks how much of this text an LLM wrote, and the answer has to be true.

Deadline: Sept 11 11:59pm PT = Sept 12 12:29pm IST.

---

## 1. What question did you try to answer?

Does capability-only post-training silently move a model's refusal mechanism? Concretely: I fine-tuned Qwen2.5-3B-Instruct on chemistry QA with no safety data and no safety reward, first by SFT and then by Dr.GRPO chained from the SFT checkpoint, and asked (H1) whether the SFT stage erodes the base model's refusal direction, (H2, primary) whether RL continuation restores it, and (H4) whether behaviour and mechanism can come apart, i.e. refusal rate stays flat while the direction underneath moves, or vice versa.

## 2. Why is this question interesting / why did you choose it?

[YOUR VOICE — the honest hook from your write-up: if you had only run the cheap representational metric you would have drawn the wrong conclusion. Say that in your words.]

Facts to build from:
- Real pipelines stack capability stages on top of safety training. If those stages move the refusal mechanism, safety evals done on the base model stop transferring, even when a behavioural refusal benchmark looks fine.
- Venhoff et al. (arXiv:2510.07364) find RL re-orchestrates existing mechanisms while SFT installs new ones, but for machinery the objective touches. Whether that holds for machinery the reward never sees (refusal, under a chemistry reward) was open.
- I had already worked on SFT-vs-RL circuit retention on this exact model (arXiv:2605.28860, 2nd author), so I could reuse the patching site and eval harness and spend the 20 hours on the new question rather than on infrastructure.

## 3. What conclusions have you reached?

On 502 held-out harmful/harmless pairs:

1. H1 (SFT erodes the direction): true representationally, false causally. cos(dir_M0, dir_SFT) = 0.914, below the split-half ceiling of 0.984 [0.973, 0.991]. But ablating M0's direction inside M_SFT still removes refusal completely (transfer ratio 1.00).
2. H2 (RL restores it): null. Refusal 0.980 vs 0.972 (p = 0.29) and cos 0.933 vs 0.914 both lean RL's way but are within noise; at a matched layer cos(dir_SFT, dir_RL) = 0.987, so RL barely moved the direction at all.
3. Behaviour moves a little, and it's real: refusal 0.992 → 0.972 (SFT) → 0.980 (RL); M0 vs M_SFT 12 lost / 2 gained, exact McNemar p = 0.013. Harmless compliance ≥ 0.992 throughout.
4. Why the two readouts disagree (layer sweep): all three checkpoints share the same causal onset at L19. Harmful-vs-harmless separation is identical up to ~L20 and only diverges late (10.4 / 8.4 / 8.7 at L32). Fine-tuning degrades the direction where it is most readable, not where it acts.
5. Specificity: three random directions through the same projection move refusal by ≤ 0.016; the fitted direction by ≥ 0.972 in every checkpoint.

## 4. Technical setup

Models. M0 = Qwen2.5-3B-Instruct (36 layers, d = 2048). M_SFT = M0 + completion-only SFT on 2,200 SciKnowEval chemistry examples (2 epochs). M_RL = M_SFT + Dr.GRPO (600 prompts, group 16, β = 0). Chaining verified in weight space: ‖M_RL − M_SFT‖ = 6.90 < ‖M_RL − M0‖ = 10.26. Trained fresh for this project. Science score 42 / 77 / 59.

Data. AdvBench + HarmBench + MaliciousInstruct harmful prompts, each paired to a length-matched Alpaca prompt. Kept only harmful prompts M0 refuses and harmless ones it complies with: 752 pairs, split 200 train / 50 val / 502 test.

What I measure.
- Refusal: JailbreakBench substring judge, greedy, 64 tokens, frozen before scoring. Wilson CIs; exact paired McNemar on the same 502 prompts.
- Refusal direction: diff-in-means (harmful − harmless) of residual-stream activations (`hidden_states[L]`, the output of decoder block L−1) at the last instruction token, fit on train, layer chosen on val as the earliest that fully removes refusal (L19 for M0 and M_SFT, L21 for M_RL).
- Representational retention: cos(dir_M0, dir_X) at a matched layer, read against a split-half bootstrap ceiling rather than against 1.0.
- Causal retention: transfer ratio = refusal drop from ablating dir_M0 inside checkpoint X ÷ drop from X's own direction. ≥ 0.8 preserved, < 0.5 moved.
- Controls: three random unit directions through the same projection; batch-size rerun; forward KL from M0.
- Layer sweep: ablation effect and harmful-vs-harmless separation per layer, per checkpoint.

Pre-registered (commit 49ebad6, before any data download), 14 dated deviations. Compute: 1× L40S 48 GB. [GPU hours: fill from logs.]

## 5. Strongest evidence against these hypotheses

Against H1 as "erosion": the causal readout flatly contradicts it. If SFT had eroded the mechanism, M0's direction should ablate refusal less well inside M_SFT. It ablates it perfectly (transfer 1.00). The cosine drop below the ceiling is real, but the layer sweep shows it comes from late layers (L32+) where the direction is readable but not where ablation acts, so the "erosion" is in a part of the representation that isn't doing the work.

Against H2: nothing distinguishes the RL arm from the SFT arm. p = 0.29 on behaviour, cos(SFT, RL) = 0.987 at a matched layer, and the two directions ablate identically. Anything I said in favour of "RL restores" would be reading noise.

Against the behavioural effect mattering: M0's re-measured refusal is 0.992 on a set filtered to prompts it refuses at 1.000, so the pipeline's own bf16/batching noise floor is ~0.8%. The SFT effect (2.0 pp) is above it but not far above it.

Against the mechanism story generally: narrow fine-tuning writes a large, readable bias into activation differences even on unrelated text (arXiv:2510.13900). Some of the M0 → M_SFT rotation is almost certainly a chemistry trace, not refusal machinery moving. A trace doesn't explain why dir_M0 still ablates refusal, but I did not isolate it.

## 6. Biggest limitations, and could I have addressed them

- One model, one domain, one seed per checkpoint. Yes, addressable: a second seed for M_SFT alone would have told me whether the 2 pp behavioural drop and the 0.914 cosine are stable. I chose a bigger test split (502 instead of the registered ~150, D10) over a second seed; in hindsight [YOUR CALL: was that the right trade?].
- The judge measures refusal phrasing, not compliance. Reading ablated completions by hand (D9): on 8 harmful val prompts the judge scored 8/8 as non-refusals but only 1 was on-topic compliance; the rest reinterpreted the request benignly (identity-theft tutorial → a story about paying for coffee). On the harmless side 6/8 were visibly degraded despite scoring "compliant". So the compliance guard passed 13/13 layers for every checkpoint and has no discriminative power. I narrowed the claim to "refusal-phrase suppression" rather than fixing it; a coherence-aware judge was in scope for time but I prioritised the random-direction control.
- cos(dir_M0, dir_RL) at a common layer 19 was never measured; the 0.933 is at L21. Triangle inequality bounds it to ~[0.84, 0.97]. This is a ~2-minute CPU job I lost to the direction tensors going missing in the machine migration. Cheap, unaddressed, stated.
- The RL arm is undertrained: science score fell 77.0 → 59.0 (still above M0's 42.0). H2 is tested with a weak RL arm. More RL steps would have cost ~[N] GPU-hours I didn't have.
- Direction-level only. The head-level DBM analysis (H3, and the head halves of H2/H4) was dropped by scope before any observation (D6). Those hypotheses are untested, not null.
- No non-chemistry control prompt set to separate the narrow-fine-tuning trace from refusal-specific rotation.

## 7. How did you use LLMs, and how did you make sure they weren't giving you slop?

[THIS ANSWER MUST BE YOURS. Below is what the project record supports; confirm, correct, and add what I can't see.]

[Your paragraph, kept as you wrote it, lightly tightened:]
I mainly used LLMs to work out which questions I could realistically answer in 20 hours, to research the two tasks I shortlisted, to get the analysis plan and code written, and then to cross-check every result I got against what the papers claimed before recording it in the prereg. I used two different LLMs and caught issues in both the research and the code output — specifics below.

What LLMs did:
- Scoping. I gave them my prior paper's infrastructure and asked what 20-hour questions it could support. Two survived: a circuit-overlap study across science domains (physics/biology) and the refusal-direction study. I picked refusal because the causal readout (ablation transfer) gave a cleaner falsification test than IoU between head sets, and kept the other as a fallback.
- Literature. Found and summarised Venhoff et al. (arXiv:2510.07364), Arditi et al. (refusal direction), and the narrow-fine-tuning trace paper (arXiv:2510.13900). I read the originals before citing any of them, and one of those checks changed the design: [FILL — e.g. the Arditi judge substring list, or the trace paper pushing you to a causal rather than cosine readout].
- Drafted the pre-registration structure and the deviations-log discipline; I wrote the hypotheses, thresholds and the tie-break rule.
- Wrote first passes of the pipeline scripts (direction.py, random_control.py, inspect_ablation.py, digest.py) [confirm which]. I reviewed the projection code by hand, since a wrong ablation site would silently produce the headline.
- Cross-checked results. After each stage I asked an LLM whether the numbers were consistent with the prereg predictions and with the cited papers' magnitudes, then verified its reading against the per-prompt JSONs myself.
- Cross-LLM checking. I ran the same analysis question through two LLMs and compared. Where they disagreed I went to the data. [FILL: one concrete disagreement and which one was right.]
- Drafted the write-up scaffold from my results files, with [YOUR VOICE] blocks left deliberately empty for me to write, because I knew LLM-written prose was a red flag in this stream.
- Drafted these form answers from my prereg and results files; I rewrote them. [Say honestly what fraction is your wording.]

What LLMs did not do: choose the question, the hypotheses, the thresholds, which controls to add, or what to retract.

Specific mistakes caught [attribute to an LLM ONLY if that is true; the record shows these as analysis-code corrections in D12/D13]:
- The first analysis summary reported cos(dir_M0, dir_RL) = 0.643. That compared layer 21 against layer 19, confounding rotation with layer selection; the same-layer value is 0.933. Caught by asking why one cosine was so far from the other two.
- A claimed "causal site shifted from L19 to L21 under RL". I pulled the full val selection traces: M_RL misses a full drop at L19 by one prompt in 32, and M0 shows the same wobble at L25 and L29. Retracted in D13.
- [Add: any code bug an LLM introduced that you caught, e.g. in the projection, the McNemar, the split seeding.]

How I kept it honest:
- Every headline number was recomputed from per-prompt JSONs, not trusted from a summary. That is how the McNemar counts (12 / 2) were produced.
- I read all the raw completions by hand, which is how I found the judge overstates ablation (D9). No metric or LLM flagged that.
- The random-direction control was not in the LLM-drafted prereg; I added it after reading the admissions doc.
- Rerun at a different batch size, and a forced refit on different hardware, both reproduced the selected layers.

## 8. Prior experience with mechanistic interpretability

Second author (of 7) on "Mechanistic origins of catastrophic forgetting: why RL preserves circuits better than SFT?" (arXiv:2605.28860, EIML workshop @ ICML 2026, Algoverse AI Research). We compared SFT vs Dr.GRPO on Qwen2.5-3B-Instruct using Differential Binary Masking for circuit discovery and introduced a head-level "differential circuit vulnerability" measure; RL retained ~72% of base circuits vs ~59% for SFT at epoch 2. [State honestly what you personally built: the DBM pipeline? the eval harness? the patching code?] Beyond that, no formal mech interp training; I have not done SAE or attribution-graph work. [Add any ARENA / reading / other hands-on work if true.]

## 9. Other than the research task, 1–3 pieces of evidence you'd do good research (≈100 words)

[Pick 2–3; be specific. Candidates from your background, your wording:]
- The ICML workshop paper above: I helped take an idea from pilot to accepted paper in a small remote team, including the benchmark retention evals across six suites.
- I ship and maintain a real tool alone (CodeTrace-AI, on PyPI, local-first codebase intelligence): fast feedback loops, my own bugs, users who report when I'm wrong.
- Contract backend/AI engineering at CoderTrails: production RAG over large PDFs, multi-agent systems, LLM cost work. Relevant because it's where I learned to distrust a metric until I've read the raw outputs.
- An Indian patent on distinguishing genuine from posed facial emotion (Application No. 202421040795): an early, self-directed empirical project.

## 10. Why Neel's stream specifically?

[Built from your draft; verified links: Arditi et al. 2024, Venhoff et al. 2510.07364, Minder et al. 2510.13900 all have Nanda as senior author. Anthropic Fellows mention removed on purpose.]

I've wanted to do research since I was a kid, and I used my B.E. to find out what kind. Maths and logic pulled me toward ML, where I built and shipped enough projects to know I liked the work. But it was Algoverse that introduced me to mechanistic interpretability, and that stuck: the idea that you can open a model and ask what it is actually computing. I worked on that programme harder than anything before it, and our team's paper on why RL preserves circuits better than SFT (arXiv:2605.28860, second author) was accepted at EIML @ ICML 2026.

That paper is why your stream specifically. Our head-level result, RL keeping ~72% of base circuits against SFT's ~59%, is the same conclusion Venhoff et al. from your group reached with steering vectors: RL orchestrates existing mechanisms, SFT installs new ones. My application project lives in the gap those two results leave, asking whether preservation holds for machinery the reward never touches, and it is built from your stream's tools: Arditi et al.'s refusal direction as the measurement, Minder et al.'s narrow-fine-tuning trace as the main threat to validity. The method, the hypothesis, and the strongest objection all came from your group, and I'd rather be red-teamed by the people who built them.

I applied to MATS once before and got through the first round but not further. The paper and this project are what I did about it.

## 11. Likelihood you will join the training program (Sept 28 – Oct 30)

[Only you can answer. Give a percentage and one sentence on constraints: timezone (IST), the CoderTrails contract, anything else.]

## 12. (Optional) Anything else about the application project

Candidates:
- The write-up was pre-registered before any data was downloaded (commit 49ebad6); D1–D7 were written before any model saw a refusal prompt, D8–D14 as data came in, none after test data was scored. The deviations log is in the repo.
- Losing the direction tensors mid-project forced a refit on different hardware, which recovered identical selected layers and separation profiles to a few tenths of a percent. Unplanned, but it's the strongest reproducibility check in the project.
- Infrastructure disclosure: reused from arXiv:2605.28860 the o_proj-input patching site (used only by the head-level DBM code, which was not run; the direction analysis reads and edits the residual stream), the correctness checker, the dataset interface and the new-task evaluator; everything refusal-specific is new.
- [Hours: total and split setup / attended training / analysis / writing, from Toggl.]
