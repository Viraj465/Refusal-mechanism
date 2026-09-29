# Stage 1 filter report (data profile v2)

- Judge model (M0): `Qwen/Qwen2.5-3B-Instruct`
- Decoding: greedy, 64 new tokens; frozen substring judge (13 prefixes)
- Fit pool = 40% of each label, drawn before scoring; only the fit pool is filtered.
- **Test is not filtered on M0 behaviour.** Per-pair `harmful_m0_refused` / `harmless_m0_refused` record M0's verdict.

## M0 behaviour on the full pool

| Class | Scored | Refused by M0 |
|---|---|---|
| harmful | 1142 | 826 (72.3%) |
| harmless | 3000 | 47 (1.6%) |

M0 refusal by harmful source:

- donotanswer: 446/666 refused (https://raw.githubusercontent.com/Libr-AI/do-not-answer/main/datasets/data_en.csv)
- strongreject: 224/276 refused (https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv)
- xstest_unsafe: 156/200 refused (https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv)

## Splits (seed 0)

| Split | Pairs | Filtered on M0? | Use |
|---|---|---|---|
| train | 200 | yes | direction fitting |
| val | 50 | yes | layer / threshold selection |
| test | 607 | **no** | reported numbers only |

Fit pairs available 300, unused 50. M0 refuses 0.720 of test harmful prompts.

Harmful source mix in test:

- donotanswer: 407
- xstest_unsafe: 115
- strongreject: 85
