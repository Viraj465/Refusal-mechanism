# Analysis summary

> **SYNTHETIC DEMO OUTPUT — not data.**

## Behavioural baseline (test split)

| Checkpoint | Refusal on harmful | Compliance on harmless | Science NTS |
|---|---|---|---|
| M0 | 0.970 [0.910, 1.000] | 0.950 [0.890, 1.000] | 21.0% |
| M_SFT | 0.710 [0.650, 0.770] | 0.930 [0.870, 0.990] | 68.5% |
| M_RL | 0.880 [0.820, 0.940] | 0.940 [0.880, 1.000] | 71.2% |

## Direction level (prereg §5.2)

Direction ceiling (M0, split-half bootstrap, layer 14): **0.940** [0.910, 0.960]. Cosines are read against this, not against 1.0.

| Checkpoint | cos with dir_M0 | Transfer ratio | Verdict |
|---|---|---|---|
| M_SFT | 0.710 | 0.440 | causally_moved |
| M_RL | 0.900 | 0.830 | causally_preserved |

## Head level (prereg §5.3)

Circuit sizes (of 576 heads): {'M0': 248, 'M_SFT': 189, 'M_RL': 241}

| Checkpoint | IoU with M0 | Null p97.5 | Above null | 0.8×ceiling met | Verdict |
|---|---|---|---|---|---|
| M_SFT | 0.520 | 0.350 | None | None | not_preserved |
| M_RL | 0.680 | 0.410 | None | None | preserved |

Seed stability (Jaccard): M0 0.880, M_SFT 0.850, M_RL 0.860  (threshold 0.7)

## H4 dissociation

Cross-tabulated under both registered operationalisations (prereg §4). Disagreement between the two columns is itself reportable — it means the direction and the head set do not localise the same thing.

| Checkpoint | Behaviour | Mechanism (direction) | Cell (direction) | Mechanism (head) | Cell (head) |
|---|---|---|---|---|---|
| M_SFT | lost  (0.71 vs M0 0.97) | moved  (ratio 0.44) | erosion | not_preserved | erosion |
| M_RL | preserved  (0.88 vs M0 0.97) | preserved  (ratio 0.83) | re-consolidation | preserved | re-consolidation |
