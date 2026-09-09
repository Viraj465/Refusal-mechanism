"""
Phase-A CPU smoke tests.

Everything here is cheap and none of it needs a GPU. The point is to catch, on
a laptop and for free, the failures that would otherwise surface at 2am on a
metered box: a missing dataset file, a schema the normaliser rejects, a pairing
routine that silently returns nothing, a reward function that scores every
rollout zero.

Tests are tiered by dependency and skip cleanly when a tier's imports are
absent, so this runs both locally (stdlib + datasets) and on the remote box
(everything). Run it as the last step of env/setup.sh.

    python refusal/smoke_test.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

PASS, FAIL, SKIP = [], [], []


def have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check(name: str, requires: tuple = ()):
    """Decorator: register a check, skipping it if a dependency is absent."""
    def deco(fn):
        missing = [m for m in requires if not have(m)]
        if missing:
            SKIP.append((name, f"needs {', '.join(missing)}"))
            return fn
        try:
            detail = fn()
            PASS.append((name, detail or ""))
        except AssertionError as exc:
            FAIL.append((name, str(exc)))
        except Exception:
            FAIL.append((name, traceback.format_exc(limit=3)))
        return fn
    return deco


# ==========================================================================
# Tier A — stdlib / datasets only
# ==========================================================================

@check("science files present", requires=())
def _science_files():
    from science_data import SCIENCE_DIR, science_task_counts

    assert SCIENCE_DIR.is_dir(), f"missing science dir: {SCIENCE_DIR}"
    counts = science_task_counts()
    total = sum(counts.values())
    assert total > 0, "no science rows found"
    empty = [k for k, v in counts.items() if v == 0]
    assert not empty, f"empty science files: {empty}"
    return f"{total} rows across {len(counts)} chemistry tasks"


@check("science schema -> normaliser", requires=("datasets",))
def _science_schema():
    """The pre-rental check. If the SFT path cannot consume the science schema,
    the whole Case-B plan stalls, and it must not be discovered on a GPU."""
    from common import rlrazor_path

    rlrazor_path()
    from data.dataset_utils import UnifiedDatasetInterface

    from science_data import load_science_raw

    rows = load_science_raw()[:5]
    fmt = UnifiedDatasetInterface.detect_format(rows[0])
    assert fmt == "sciknoweval", f"expected sciknoweval, got {fmt}"

    norm = UnifiedDatasetInterface.normalize_example(rows[0], fmt)
    for key in ("question", "answer", "text", "prompt"):
        assert key in norm, f"normaliser dropped '{key}'"
        assert str(norm[key]).strip(), f"normaliser produced empty '{key}'"
    assert norm["text"] == norm["prompt"] + norm["answer"], (
        "text != prompt + answer; completion-only loss would mask the wrong span"
    )
    return f"format={fmt}, prompt+answer round-trips"


@check("science dataset split", requires=("datasets",))
def _science_split():
    from science_data import load_science_dataset

    train, evl = load_science_dataset(seed=0, n_eval=50)
    assert len(evl) == 50, f"eval split is {len(evl)}"
    assert len(train) > 0, "empty train split"
    q_eval = {r["question"] for r in evl}
    q_train = {r["question"] for r in train.select(range(min(200, len(train))))}
    assert not (q_eval & q_train), "eval questions leak into train"
    return f"train={len(train)}, eval={len(evl)}, no leakage"


@check("refusal judge", requires=())
def _judge():
    from common import REFUSAL_SUBSTRINGS, is_refusal

    refusals = [
        "I'm sorry, but I can't help with that.",
        "I cannot provide instructions for that.",
        "As an AI language model, I must decline.",
    ]
    compliances = [
        "Sure, here is a recipe for banana bread:",
        "Photosynthesis is the process by which plants",
        "1. Preheat the oven to 180C.",
    ]
    for t in refusals:
        assert is_refusal(t), f"missed refusal: {t!r}"
    for t in compliances:
        assert not is_refusal(t), f"false refusal: {t!r}"
    return f"{len(REFUSAL_SUBSTRINGS)} frozen prefixes, 6/6 correct"


@check("length-matched pairing", requires=())
def _pairing():
    from build_dataset import length_match

    def row(i, n, label):
        return {
            "id": f"{label}_{i}", "prompt_text": f"p{i}", "chat_formatted": f"c{i}",
            "n_prompt_tokens": n, "label": label, "source": "synthetic",
            "target_token_ids": [1, 2, 3], "target_text": "xyz",
        }

    harmful = [row(i, 20 + i, "harmful") for i in range(10)]
    harmless = [row(i, 20 + i, "harmless") for i in range(10)]
    pairs = length_match(harmful, harmless, tolerance=2)
    assert len(pairs) == 10, f"expected 10 pairs, got {len(pairs)}"
    for p in pairs:
        gap = abs(p["harmful_n_tokens"] - p["harmless_n_tokens"])
        assert gap <= 2, f"pair violates the +/-2 window: {gap}"
    used = [p["harmless_id"] for p in pairs]
    assert len(set(used)) == len(used), "a harmless prompt was reused across pairs"

    # No overlap in length ranges -> no pairs at all, not a silent bad match.
    far = [row(i, 100 + i, "harmless") for i in range(10)]
    assert length_match(harmful, far, tolerance=2) == [], "matched across a 80-token gap"
    return "10/10 matched, reuse-free, rejects out-of-window"


@check("split discipline", requires=())
def _splits():
    from build_dataset import N_TEST, N_TRAIN, N_VAL, make_splits

    s = make_splits(400, seed=0)
    assert len(s["train"]) == N_TRAIN and len(s["val"]) == N_VAL and len(s["test"]) == N_TEST, (
        f"registered sizes violated: {[len(v) for v in s.values()]}"
    )
    all_idx = s["train"] + s["val"] + s["test"]
    assert len(set(all_idx)) == 400, "splits overlap or lose pairs"
    assert make_splits(400, seed=0) == s, "splits are not reproducible at seed 0"

    short = make_splits(200, seed=0)
    assert sum(len(v) for v in short.values()) == 200, "short-yield splits lose pairs"
    return "200/50/150 exact, disjoint, reproducible; scales on short yield"


@check("statistics", requires=())
def _stats():
    from common import jaccard, wilson_ci

    ci = wilson_ci(75, 150)
    assert abs(ci["point"] - 0.5) < 1e-9
    assert ci["low"] < 0.5 < ci["high"], "CI does not bracket the point estimate"
    assert wilson_ci(150, 150)["high"] <= 1.0, "CI exceeds 1"
    assert wilson_ci(0, 0)["n"] == 0, "empty sample should not raise"
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert jaccard(set(), set()) != jaccard({1}, {1})  # nan vs 1.0
    return "Wilson CI + Jaccard behave at the edges"


@check("results IO never overwrites", requires=())
def _results_io():
    import json
    import time

    from common import RESULTS_DIR, save_json

    a = save_json({"probe": 1}, "_smoketest")
    time.sleep(1.05)  # timestamps have 1s resolution
    b = save_json({"probe": 2}, "_smoketest")
    assert a != b, "two runs wrote the same results file"
    assert json.loads(a.read_text())["probe"] == 1, "first result was clobbered"
    for p in RESULTS_DIR.glob("_smoketest_*.json"):
        p.unlink()
    return "timestamped, non-clobbering"


# ==========================================================================
# Tier B — needs transformers (tokenizer only, no weights)
# ==========================================================================

@check("Qwen chat template", requires=("transformers",))
def _chat_template():
    from transformers import AutoTokenizer

    from common import M0_NAME, format_chat

    tok = AutoTokenizer.from_pretrained(M0_NAME)
    chat = format_chat(tok, "How do I bake bread?")
    assert "How do I bake bread?" in chat, "instruction lost in templating"
    assert chat.rstrip().endswith("assistant"), (
        f"template does not end at the assistant turn; last 40 chars: {chat[-40:]!r}"
    )
    ids = tok(chat, add_special_tokens=False).input_ids
    assert len(ids) > 10, "suspiciously short tokenisation"
    return f"{len(ids)} tokens, ends at the assistant generation prompt"


# ==========================================================================
# Tier C — needs torch (no model weights)
# ==========================================================================

@check("mask annealing", requires=("torch",))
def _mask_math():
    import torch

    from dbm import HeadMask

    m = HeadMask(4, 8, device="cpu", seed=0)
    m.tau = 1.0
    soft = m.values().detach()
    assert soft.shape == (4, 8)
    assert 0.4 < float(soft.mean()) < 0.6, "init is not near 0.5"

    m.tau = 0.05
    hard = m.values().detach()
    # Annealing must push values toward the ends, or the 0.5 threshold at the
    # end of training is arbitrary.
    assert float((hard - 0.5).abs().mean()) > float((soft - 0.5).abs().mean()), (
        "annealing did not sharpen the mask"
    )
    m.logits.data[0, 0] = 10.0
    m.logits.data[0, 1] = -10.0
    circ = m.circuit()
    assert (0, 0) in circ and (0, 1) not in circ, "threshold picks the wrong side"

    # Gradients must reach the logits — the prior repo's version ran under
    # no_grad and could not be optimised at all.
    loss = m.values().sum()
    loss.backward()
    assert m.logits.grad is not None and float(m.logits.grad.abs().sum()) > 0, (
        "no gradient reaches the mask logits"
    )
    return "sharpens under annealing, thresholds correctly, differentiable"


@check("triplet batching positions", requires=("torch", "transformers"))
def _batching():
    import torch
    from transformers import AutoTokenizer

    from common import M0_NAME, format_chat
    from dbm import build_batches

    tok = AutoTokenizer.from_pretrained(M0_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pairs = []
    for i, (h, c) in enumerate([("Explain X" + " word" * i, "Explain Y") for i in range(3)]):
        pairs.append(
            {
                "harmful_chat": format_chat(tok, h),
                "harmless_chat": format_chat(tok, c),
                "compliance_target_ids": tok("Sure, here", add_special_tokens=False).input_ids[:3],
            }
        )

    batches = build_batches(tok, pairs, batch_size=3, device="cpu")
    b = batches[0]
    assert b.input_ids.shape[0] == 3

    for i, p in enumerate(pairs):
        plen = len(tok(p["harmful_chat"], add_special_tokens=False).input_ids)
        assert int(b.patch_pos[i]) == plen - 1, "patch position is not the last instruction token"
        # target token j sits at plen+j and is predicted from plen+j-1
        for j in range(int(b.target_mask[i].sum())):
            assert int(b.target_pos[i, j]) == plen + j - 1, "target logit position is off by one"
            assert int(b.input_ids[i, plen + j]) == int(b.target_ids[i, j]), (
                "target token is not where the position index says it is"
            )
    return "patch + target positions correct under ragged right padding"


@check("target log-prob = geometric mean", requires=("torch",))
def _geo_mean():
    import torch

    from dbm import TripletBatch, target_logprob

    vocab = 16
    logits = torch.zeros(1, 6, vocab)
    logits[0, 0, 3] = 100.0   # predicts token 3 with prob ~1
    logits[0, 1, 5] = 100.0

    batch = TripletBatch(
        input_ids=torch.zeros(1, 6, dtype=torch.long),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        patch_pos=torch.tensor([0]),
        target_pos=torch.tensor([[0, 1]]),
        target_ids=torch.tensor([[3, 5]]),
        target_mask=torch.tensor([[1.0, 1.0]]),
        pair_idx=torch.tensor([0]),
    )
    lp = target_logprob(logits, batch)
    assert float(lp.exp()) > 0.99, f"confident targets should score ~1, got {float(lp.exp())}"

    batch.target_mask = torch.tensor([[1.0, 0.0]])
    lp_masked = target_logprob(logits, batch)
    assert float(lp_masked.exp()) > 0.99, "masking a target changed a correct score"
    return "geometric mean correct, padding-masked"


@check("directional ablation is a projection", requires=("torch",))
def _ablation_math():
    import torch

    r = torch.randn(64)
    r = r / r.norm()
    a = torch.randn(4, 7, 64)
    ablated = a - (a @ r).unsqueeze(-1) * r

    assert float((ablated @ r).abs().max()) < 1e-4, "residual component along r survives"
    twice = ablated - (ablated @ r).unsqueeze(-1) * r
    assert torch.allclose(ablated, twice, atol=1e-5), "ablation is not idempotent"
    orth = torch.randn(64)
    orth = orth - (orth @ r) * r
    proj = a @ orth
    assert torch.allclose(proj, ablated @ orth, atol=1e-4), "ablation disturbed the orthogonal complement"
    return "idempotent projection, orthogonal complement preserved"


@check("completion sample is random and seeded", requires=())
def _completion_sample():
    """The admissions doc penalises 'cherry-picked qualitative examples without
    random sampling' and 'not looking at actual data'. The sample must be
    uniform, seeded, reproducible, and must carry the judge's verdict."""
    from behaviour import sample_completions

    pairs = [
        {"pair_id": f"pair_{i:05d}", "harmful_prompt": f"h{i}", "harmless_prompt": f"c{i}"}
        for i in range(150)
    ]
    behaviour = {
        "completions": {
            "harmful": [f"refusal text {i}" for i in range(150)],
            "harmless": [f"compliance text {i}" for i in range(150)],
        },
        "per_prompt_refused": {
            "harmful": [i % 5 != 0 for i in range(150)],
            "harmless": [False] * 150,
        },
    }

    a = sample_completions(pairs, behaviour, n=10, seed=0)
    b = sample_completions(pairs, behaviour, n=10, seed=0)
    c = sample_completions(pairs, behaviour, n=10, seed=1)

    assert len(a["harmful"]) == 10 and len(a["harmless"]) == 10, "wrong sample size"
    assert a == b, "sample is not reproducible at a fixed seed"
    assert [r["index"] for r in a["harmful"]] != [r["index"] for r in c["harmful"]], \
        "seed has no effect — sample may not be random"
    assert [r["index"] for r in a["harmful"]] != list(range(10)), \
        "sample looks like the first n, not a random draw"

    for row in a["harmful"]:
        assert row["completion"] == f"refusal text {row['index']}", "completion/index misaligned"
        assert row["prompt"] == f"h{row['index']}", "prompt/index misaligned"
        assert isinstance(row["judged_refusal"], bool), "judge verdict missing"

    # Must not silently drop a side when completions were not kept.
    empty = sample_completions(pairs, {"completions": {}, "per_prompt_refused": {}}, 10, 0)
    assert "harmful" not in empty, "fabricated a sample with no completions"
    return "uniform, seeded, reproducible, index-aligned, judge verdict attached"


@check("permutation null is calibrated", requires=("numpy",))
def _perm_null():
    """The number that decides whether any overlap means anything. At ~50%
    density two unrelated circuits already share IoU ~0.33 — if this control
    were wrong, every retention claim in the writeup would be too."""
    from controls import permutation_null

    n = permutation_null(288, 288, 576, n_draws=300, seed=0)
    assert 0.30 < n["mean"] < 0.36, f"chance IoU at 50% density should be ~0.33, got {n['mean']:.3f}"
    assert n["p97_5"] > n["mean"], "97.5th percentile below the mean"

    sparse = permutation_null(29, 29, 576, n_draws=300, seed=0)
    assert sparse["mean"] < n["mean"], "sparser circuits should have a lower chance overlap"
    return f"50% density -> {n['mean']:.3f} (p97.5 {n['p97_5']:.3f}); 5% -> {sparse['mean']:.3f}"


@check("controls end-to-end", requires=("torch", "numpy"))
def _controls_e2e():
    """Write synthetic masks, run the real stats path, check the verdicts fire."""
    import torch

    import controls
    from controls import MASKS_DIR, load_masks

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    torch.manual_seed(0)

    def fake(tag, seed, half, on_heads):
        vals = torch.rand(36, 16) * 0.4
        flat = vals.flatten()
        flat[on_heads] = 0.6 + torch.rand(len(on_heads)) * 0.4
        vals = flat.view(36, 16)
        circuit = [(i // 16, i % 16) for i in range(576) if vals.flatten()[i] > 0.5]
        name = f"{tag}_seed{seed}_lam0.03" + (f"_half{half}" if half is not None else "")
        path = MASKS_DIR / f"{name}.pt"
        torch.save({"tag": tag, "logits": vals, "values": vals, "binary": vals > 0.5,
                    "circuit": circuit, "meta": {}}, path)
        written.append(path)

    base = list(range(0, 240))
    try:
        fake("M0", 0, None, base)
        fake("M0", 1, None, base[:230] + list(range(300, 310)))
        fake("M0", 0, 0, base[:220]); fake("M0", 0, 1, base[:215] + list(range(400, 405)))
        fake("M_SFT", 0, None, base[:120] + list(range(300, 380)))
        fake("M_RL", 0, None, base[:200] + list(range(300, 340)))

        masks = load_masks()
        assert len(masks) == 6, f"expected 6 masks, parsed {len(masks)}"
        assert {m["tag"] for m in masks} == {"M0", "M_SFT", "M_RL"}, "tag parsing wrong"
        assert sum(1 for m in masks if m["half"] is not None) == 2, "half suffix not parsed"

        class A:
            n_permutations, seed = 200, 0

        controls.run_stats(A())
        latest = sorted((controls.RESULTS_DIR / "controls").glob("controls_stats_*.json"))[-1]
        import json
        p = json.loads(latest.read_text())
        latest.unlink()

        ret = p["verdicts"]["retention"]
        assert "M_SFT" in ret and "M_RL" in ret, "retention verdicts missing"
        assert ret["M_RL"]["iou_vs_M0"] > ret["M_SFT"]["iou_vs_M0"], "IoU ordering wrong"
        assert p["verdicts"]["H2"]["gap"] > 0, "H2 gap should be positive on this fixture"
        assert p["verdicts"]["H2"]["outcome"] == "supported", "H2 outcome should be supported here"
        assert p["split_half_ceiling"]["M0"] > 0.8, "split-half ceiling not computed"
        assert 0 < p["seed_stability"]["M0"]["jaccard"] < 1, "seed stability out of range"
        assert p["matched_k_k"] == min(p["circuit_sizes"].values()), "matched-k used the wrong k"
        return (
            f"IoU M0-SFT {ret['M_SFT']['iou_vs_M0']:.2f} < M0-RL {ret['M_RL']['iou_vs_M0']:.2f}, "
            f"null p97.5 {ret['M_RL']['null_p97_5']:.2f}, H2 gap {p['verdicts']['H2']['gap']:+.2f}"
        )
    finally:
        for path in written:
            path.unlink(missing_ok=True)


@check("H2 three-way outcome", requires=("numpy",))
def _h2_outcome():
    """A reversed gap must not collapse into 'null'. Under prereg §9 D5, RL
    disturbing refusal machinery MORE than SFT is the diagnostic result — the
    one that would contradict arXiv:2510.07364 in the collateral regime."""
    from controls import _verdicts

    primaries = {"M0": None, "M_SFT": None, "M_RL": None}

    def payload(iou_sft, iou_rl, jaccard=0.95):
        return {
            "iou_matrix": {"M0|M_SFT": iou_sft, "M0|M_RL": iou_rl},
            "permutation_null": {"M0|M_SFT": {"p97_5": 0.33}, "M0|M_RL": {"p97_5": 0.33}},
            "split_half_ceiling": {"M0": 0.85},
            "seed_stability": {t: {"jaccard": jaccard} for t in primaries},
        }

    # noise = |1 - 0.95| = 0.05
    got = {
        "supported": _verdicts(payload(0.40, 0.60), primaries)["H2"],
        "reversed": _verdicts(payload(0.60, 0.40), primaries)["H2"],
        "null": _verdicts(payload(0.50, 0.52), primaries)["H2"],
    }
    for expected, h2 in got.items():
        assert h2["outcome"] == expected, f"expected {expected}, got {h2['outcome']} (gap {h2['gap']:+.3f})"
    assert got["reversed"]["supported"] is False, "'reversed' must not read as supported"
    assert got["null"]["supported"] is False, "'null' must not read as supported"

    undet = _verdicts(
        {**payload(0.4, 0.6), "seed_stability": {}}, primaries
    )["H2"]
    assert undet["outcome"] == "undetermined", "missing seed stability should be undetermined"
    return "supported / reversed / null / undetermined all distinguished"


@check("figures render", requires=("matplotlib", "numpy"))
def _figures():
    import matplotlib
    matplotlib.use("Agg")

    import tempfile

    from analysis import figure1, figure2, gather, h4_crosstab, write_summary

    results = gather(demo=True)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        f1 = figure1(results, d / "f1.png")
        f2 = figure2(results, d / "f2.png")
        summary = write_summary(results, d / "s.md")
        assert f1 and f1.exists() and f1.stat().st_size > 20_000, "figure 1 did not render"
        assert f2 and f2.exists() and f2.stat().st_size > 20_000, "figure 2 did not render"
        assert "H4 dissociation" in summary.read_text(encoding="utf-8"), "summary missing H4 table"

    # Demo numbers: M_SFT refusal 0.71 vs M0 0.97 with non-overlapping CIs =
    # behaviour lost; transfer 0.44 = mechanism moved -> erosion.
    cells = {r["checkpoint"]: r["H4 cell"] for r in h4_crosstab(results)}
    assert cells["M_SFT"] == "erosion", f"H4 logic wrong: {cells}"
    assert cells["M_RL"] == "re-consolidation", f"H4 logic wrong: {cells}"

    # The cell prereg §4 registers as the interesting positive result must be
    # reachable: behaviour recovers, but through a different mechanism.
    import copy

    fr = copy.deepcopy(results)
    fr["behaviour"]["M_SFT"]["behaviour"]["refusal_rate_harmful"] = {
        "point": 0.96, "low": 0.90, "high": 0.99, "n": 150
    }
    fr_cells = {r["checkpoint"]: r["H4 cell"] for r in h4_crosstab(fr)}
    assert fr_cells["M_SFT"] == "FUNCTIONAL REPLACEMENT", f"H4 logic wrong: {fr_cells}"

    # prereg §4 requires the cross-tab under BOTH operationalisations, so the
    # head-level cell must be computed, not just carried as a text column.
    head_cells = {r["checkpoint"]: r.get("H4 cell (head)") for r in h4_crosstab(results)}
    assert head_cells["M_SFT"] == "erosion", f"head-level H4 wrong: {head_cells}"
    assert head_cells["M_RL"] == "re-consolidation", f"head-level H4 wrong: {head_cells}"
    no_head = {
        r["checkpoint"]: r.get("H4 cell (head)")
        for r in h4_crosstab(dict(results, controls=None))
    }
    assert set(no_head.values()) == {"undetermined"}, f"expected undetermined: {no_head}"

    # Figure 2 must degrade to the H4 table when the pilot gate failed.
    no_masks = dict(results, controls=None)
    with tempfile.TemporaryDirectory() as td:
        f2b = figure2(no_masks, Path(td) / "f2b.png")
        assert f2b and f2b.exists(), "figure 2 fallback did not render"
    return "both figures + fallback, H4 cells correct"


# ==========================================================================
# Tier D — needs trl (the prior repo's trainers import it at module level)
# ==========================================================================

@check("science reward function", requires=("torch", "trl"))
def _reward():
    """Guards the bug this project had to route around: the prior repo's GRPO
    reward parses the question back out of the prompt with regexes that do not
    match the science template, so every reward is 0.0."""
    from common import rlrazor_path

    rlrazor_path()
    from training.training import check_answer_correctness

    from science_data import load_science_raw

    rows = load_science_raw()[:20]
    hits = sum(1 for r in rows if check_answer_correctness(r["answer"], r["answer"]))
    assert hits == len(rows), f"ground truth scored against itself: {hits}/{len(rows)}"

    wrong = sum(1 for r in rows if check_answer_correctness("completely unrelated text", r["answer"]))
    assert wrong <= len(rows) * 0.2, f"reward fires on unrelated text {wrong}/{len(rows)} times"
    return f"{hits}/{len(rows)} self-match, {wrong}/{len(rows)} false positives"


# ==========================================================================

def main() -> int:
    print("=" * 68)
    print(" Phase-A smoke tests")
    print("=" * 68)

    for name, detail in PASS:
        print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))
    for name, reason in SKIP:
        print(f"  SKIP  {name}  -- {reason}")
    for name, err in FAIL:
        print(f"  FAIL  {name}")
        for line in str(err).strip().splitlines():
            print(f"          {line}")

    print("-" * 68)
    print(f" {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if SKIP:
        print(" (skipped tiers run on the remote box once torch/trl are installed)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
