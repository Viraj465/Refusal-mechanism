# Refusal-direction project: LessWrong readiness review (2026-09-28)

Qwen2.5-3B-Instruct; M0 -> M_SFT (chemistry SFT) -> M_RL (Dr.GRPO chained from SFT). Arditi diff-in-means refusal direction. Pre-registered (prereg.md, D1-D14).

## Main problem: the headline dissociation is probably a readout artifact
- Claim: cosine drops (0.914 < ceiling 0.984) but transfer ratio = 1.00, so "rotated but not weakened".
- Full Arditi ablation (every residual write, all layers, all positions) is close to a step function (Fig 2a: every layer 19-31 gives full drop). A step function can't resolve a 0.91 rotation.
- In-project evidence: cos(M_RL@L21, M0@L19) = 0.643, yet dir_M0@L19 removes 100% of M_RL's refusal. Retracted in D12 as a cosine comparison, but it is the best evidence that the causal readout saturates.
- The random control (cos ~ 0) only shows unrelated directions fail, not that nearby ones do.
- "Degraded where readable, not where it acts": ablation acts at every layer; L19 is only where the direction is extracted. hidden_states[36] is post-final-norm.

## Other issues
- Writeup says "diff-in-means at the o_proj input"; the code uses residual-stream hidden_states. Fix the text.
- The 2pp refusal drop (p=0.013) has no perturbation null (test set filtered to prompts M0 refuses).
- RL arm: science score fell 77 -> 59, so H2 is not a meaningful test. Keep it secondary.
- Fig 1a y-axis starts at 0.94 (exaggerates a 2pp change).
- Substring judge. Addition readout (0.462 / 0.496) has no M0 baseline, but it is the only graded causal readout.

## Positioning
- Engage 2609.01455 (When Safety Routing Breaks) centrally; cite Kissane et al. "Base LLMs refuse too", 2605.01913 (RefusalGuard, cosine-drift metric), 2602.02132 (refusal is more than one direction).
- Novel contribution: calibrating how much refusal-direction rotation ablation-based readouts can detect.

## Experiments before posting (~3-5 GPU-h on one L40S)
1. Rotation dose-response (must do): v = cos(t)*d_own + sin(t)*u, with u random unit vector orthogonal to d_own, 5 u-seeds, cos in {1,.95,.9,.85,.8,.7,.6,.5,.3}. Ablate in M0 and M_SFT and measure refusal drop on ~200 test prompts. Mark 0.914.
2. Graded readouts: addition-coefficient sweep of dir_M0 in M0/M_SFT/M_RL; single-layer (L19-only) or alpha-scaled ablation. Compare curves, not saturated endpoints.
3. Matched-layer cos(M0, M_RL)@L19 (CPU).
4. Perturbation null: a second SFT seed, plus SFT on a non-chemistry benign set at matched weight distance.
5. Optional: StrongREJECT or HarmBench-classifier judge on ablated outputs.

## Post framing
Lead with "How much can a refusal direction rotate before ablation notices?"; SFT/RL chain as the case study; RL secondary; show the retraction openly.
