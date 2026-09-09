# Stage 1 filter report

- Judge model (M0): `Qwen/Qwen2.5-3B-Instruct`
- Decoding: greedy, 64 new tokens
- Judge: Arditi/JailbreakBench substring matcher, 13 prefixes, frozen in `refusal/common.py`

## Yield

| Class | Scored | Kept | Yield |
|---|---|---|---|
| harmful | 800 | 752 | 94.0% |
| harmless | 2000 | 1957 | 97.8% |

Pairs after length matching (+/-2 tokens): **752**

## Splits (seed 0)

| Split | Pairs | Use |
|---|---|---|
| train | 200 | direction fitting, mask learning |
| val | 50 | layer / lambda / threshold selection |
| test | 502 | reported numbers only |

## Harmful source mix (paired prompts only)

- advbench: 498 (https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv)
- harmbench: 169 (https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv)
- malicious_instruct: 85 (https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt)
