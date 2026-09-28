# Analysis summary

## Behavioural baseline (test split)

| Checkpoint | Refusal on harmful | Compliance on harmless | Science NTS |
|---|---|---|---|
| M0 | 0.992 [0.980, 0.997] | 1.000 [0.992, 1.000] | 42.0% |
| M_SFT | 0.972 [0.954, 0.983] | 0.992 [0.980, 0.997] | 77.0% |
| M_RL | 0.980 [0.964, 0.989] | 0.992 [0.980, 0.997] | 59.0% |

## Direction level (prereg §5.2)

Direction ceiling (M0, split-half bootstrap, layer 19): **0.984** [0.973, 0.991]. Cosines are read against this, not against 1.0.

| Checkpoint | L\* | cos with dir_M0 (same layer) | cos (each at own L\*) | Transfer ratio | Verdict |
|---|---|---|---|---|---|
| M0 | 19 | — | — | — | — |
| M_SFT | 19 | 0.914 | 0.914 | 1.002 | causally_preserved |
| M_RL | 21 | 0.933 | 0.643 | 1.000 | causally_preserved |

Source runs: M0 `direction_eval_M0_20260909T092315Z.json`; M_SFT `direction_eval_M_SFT_20260909T182043Z.json`; M_RL `direction_eval_M_RL_20260909T160401Z.json`. Where a stability rerun exists it is the newest file and is used above; the originally scored values are the ones recorded in the pre-registration.

### Random-direction control (added; not a §5.2 registered readout)

The same projection, run with random unit vectors at the same layer. If an arbitrary direction removed refusal too, the result would be about the intervention rather than the mechanism.

| Checkpoint | baseline refusal | dir_M0 drop | random drops | max random | specific? |
|---|---|---|---|---|---|
| M0 | 0.990 | **+0.988** | +0.004, +0.000, +0.016 | +0.016 | yes |
| M_SFT | 0.972 | **+0.972** | +0.000, -0.002, +0.002 | +0.002 | yes |
| M_RL | 0.980 | **+0.980** | -0.004, -0.010, +0.000 | +0.000 | yes |

Largest effect any random direction had on refusal: +0.016. Harmless-side refusal under random ablation is reported in the per-run JSONs; it stays near baseline, so the random directions are not breaking the model either.

The same-layer column reads both directions at the checkpoint's own L\*, and is the rotation measure. The own-L\* column takes each direction at the layer that model selected, so it mixes rotation with a shift in the selected layer; it differs from the same-layer value only when a checkpoint selects a different layer from M0. Read the ceiling above against the same-layer column, and note that the ceiling itself was bootstrapped at M0's layer.

## H4 dissociation

Cross-tabulated under both registered operationalisations (prereg §4). Disagreement between the two columns is itself reportable — it means the direction and the head set do not localise the same thing.

| Checkpoint | Behaviour | Mechanism (direction) | Cell (direction) | Mechanism (head) | Cell (head) |
|---|---|---|---|---|---|
| M_SFT | preserved  (0.97 vs M0 0.99) | preserved  (ratio 1.00) | re-consolidation | not measured | undetermined |
| M_RL | preserved  (0.98 vs M0 0.99) | preserved  (ratio 1.00) | re-consolidation | not measured | undetermined |
