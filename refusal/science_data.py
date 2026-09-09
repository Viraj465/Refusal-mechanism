"""
Science-QA (SciKnowEval chemistry) loader.

WHY THIS FILE EXISTS: the prior project's every entry point calls
`from data.load_data import load_dataset_byname`, but `src/data/load_data.py`
is **not present** in the repository — only the six chemistry `.jsonl` files
under `src/data/science/` and the normaliser in `src/data/dataset_utils.py`
survive. This module is the missing loader, reimplemented over those files.

The on-disk schema (verified) is:
    prompt: {default: str}   question: str   answer: str
    type: str   domain: str   details: {...}
    answerKey: ""            choices: {text: [], label: []}
which `UnifiedDatasetInterface.detect_format` reads as 'sciknoweval' and routes
through the open-ended branch (answerKey empty, choices empty).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import RLRAZOR_ROOT, read_jsonl

SCIENCE_DIR = RLRAZOR_ROOT / "src" / "data" / "science"

# All six chemistry task files. Order is fixed so shuffling with a seed is
# reproducible regardless of filesystem listing order.
SCIENCE_FILES: List[str] = [
    "balancing_chemical_equation.jsonl",
    "molar_weight_calculation.jsonl",
    "molecular_property_calculation.jsonl",
    "molecule_structure_prediction.jsonl",
    "reaction_prediction.jsonl",
    "retrosynthesis.jsonl",
]


def load_science_raw(files: Optional[List[str]] = None) -> List[Dict]:
    """Read the raw SciKnowEval chemistry rows, tagged with their task file."""
    files = files or SCIENCE_FILES
    rows: List[Dict] = []
    for fname in files:
        path = SCIENCE_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Science task file missing: {path}. Expected it inside the "
                "differential-circuit-vulnerability clone."
            )
        for row in read_jsonl(path):
            row["_source_file"] = fname
            rows.append(row)
    if not rows:
        raise RuntimeError("No science rows loaded.")
    return rows


def load_science_dataset(
    seed: int = 0,
    n_train: Optional[int] = None,
    n_eval: int = 200,
    files: Optional[List[str]] = None,
):
    """Return (train_dataset, eval_dataset) as raw-schema HuggingFace Datasets.

    Raw schema is returned deliberately: `train_sft`/`train_grpo` and
    `evaluate_new_task` all call `UnifiedDatasetInterface` themselves and
    detect the format from the *raw* keys.

    The eval split is held out before any training sample is taken, so the
    new-task score reported in the writeup is on unseen items.
    """
    from datasets import Dataset

    rows = load_science_raw(files)
    ds = Dataset.from_list(rows).shuffle(seed=seed)

    if n_eval >= len(ds):
        raise ValueError(f"n_eval={n_eval} but only {len(ds)} science rows exist.")

    eval_ds = ds.select(range(n_eval))
    train_ds = ds.select(range(n_eval, len(ds)))
    if n_train is not None:
        train_ds = train_ds.select(range(min(n_train, len(train_ds))))
    return train_ds, eval_ds


def science_task_counts(files: Optional[List[str]] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for fname in files or SCIENCE_FILES:
        path = SCIENCE_DIR / fname
        counts[fname] = sum(1 for _ in path.open(encoding="utf-8") if _.strip()) if path.exists() else 0
    return counts


def normalize(dataset):
    """Apply the prior project's normaliser (question/answer/text/prompt)."""
    from data.dataset_utils import UnifiedDatasetInterface  # noqa: E402

    return UnifiedDatasetInterface.normalize_dataset(dataset)


if __name__ == "__main__":
    counts = science_task_counts()
    total = sum(counts.values())
    print(f"Science chemistry rows: {total}")
    for k, v in counts.items():
        print(f"  {v:5d}  {k}")
