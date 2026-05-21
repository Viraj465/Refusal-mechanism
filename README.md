# Mechanistic Origins of Catastrophic Forgetting

Code for the paper **"Mechanistic origins of catastrophic forgetting: why RL preserves circuits better than SFT?"**

This repository contains the experiments behind our central question: can RL's retention advantage over SFT be explained by stronger preservation of task-relevant internal circuits? We introduce **differential circuit vulnerability**, a head-level measure of how much a circuit degrades under fine-tuning, and use it to compare RL (Dr.GRPO) and SFT on Qwen2.5-3B-Instruct adapted to scientific question answering.

## Key Findings

- **SFT compresses, RL distributes.** SFT collapses adaptation into a small set of "critical specialist" heads, discarding much of the base circuit (~265 heads retained), while RL spreads adaptation across a broader subgraph (~296 heads), close to the base model's 297.
- **RL preserves more of the base circuit.** RL retains ~68% of base heads vs. ~52% for SFT, a gap that widens to 13.5 percentage points over two epochs of high new-task-score training.
- **Output drift ≠ internal forgetting.** RL has higher output-space KL divergence yet preserves more internal computation, suggesting KL alone may not predict internal forgetting.

## Setup

```bash
pip install -r requirements.txt
```

Primary model: `Qwen/Qwen2.5-3B-Instruct` (downloaded from Hugging Face).

## Reproducing the Experiments

All commands are run from inside the `RLRazor/` directory.

**1. Train the SFT baseline** (completion-only cross-entropy on Task A):
```bash
python -m src.trainingv1.train_sft_baseline
```

**2. Refine with RL** (Dr.GRPO; group size 64, µ=2, no explicit KL penalty):
```bash
python -m src.trainingv1.train_dr_grpo
```

**3. Run circuit analysis** (Differential Binary Masking circuit discovery):
```bash
python run_circuit_analysis.py
```

**4. Generate plots** (retention trajectories, Pareto, etc.):
```bash
python -m src.trainingv1.plot_rls_razor
```

You can chain training in one shot:
```bash
python -m src.trainingv1.train_sft_baseline && python -m src.trainingv1.train_dr_grpo
```

## Repository Structure
```
RLRazor/
├── run_circuit_analysis.py      # Circuit discovery + cross-model comparison
├── src/
│   ├── circuits/                # DBM masking, discovery, faithfulness
│   ├── training/                # SFT and Dr.GRPO training
│   ├── trainingv1/              # Training entry points + plotting
│   ├── models/                  # Model loading, regularization
│   ├── data/                    # Science / math / tool datasets
│   ├── evaluation/              # Retention benchmark evaluation
│   └── visualization/           # Circuit + result plots
```
## Datasets & Benchmarks

- **Task A (fine-tuning):** SciKnowEval (science Q&A)
- **Task B (retention):** HellaSwag, TruthfulQA, MMLU, IFEval, WinoGrande, HumanEval

## License

See [LICENSE](LICENSE).
