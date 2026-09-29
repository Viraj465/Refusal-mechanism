#!/usr/bin/env bash
# One L40S session: D15 sensitivity experiments on v1, then the D16 v2 replication.
#
#   bash refusal/run_session.sh            # run everything, in priority order
#   bash refusal/run_session.sh A          # phase A only (v1: the post's core)
#   bash refusal/run_session.sh B          # phase B only (v2 replication)
#
# Resumable: every finished step drops a marker in refusal/results/.session_done/,
# and a re-run skips it. Stops at the first failure (fix, then re-run the same
# command). Log: logs/session_<utc>.log. Run from the repo root.

set -euo pipefail
cd "$(dirname "$0")/.."

M0="Qwen/Qwen2.5-3B-Instruct"
SFT="refusal/results/checkpoints/M_SFT"
RL="refusal/results/checkpoints/M_RL"
DONE="refusal/results/.session_done"
PHASE="${1:-all}"
mkdir -p "$DONE" logs
LOG="logs/session_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
T0=$(date +%s)

step() {  # step <profile> <name> <command...>
  local prof="$1" name="$2"; shift 2
  local marker="$DONE/${prof}_${name}"
  if [[ -f "$marker" ]]; then echo "[skip] $prof/$name (done)"; return; fi
  echo; echo "=== [$prof] $name  (elapsed $(( ($(date +%s) - T0) / 60 )) min) ==="
  echo "+ $*"
  REFUSAL_DATA_PROFILE="$prof" "$@"
  touch "$marker"
}

fit_if_missing() {  # fit_if_missing <profile> <tag> <checkpoint>
  local prof="$1" tag="$2" ckpt="$3" dir="refusal/results/directions"
  [[ "$prof" == "v1" ]] || dir="refusal/results/$prof/directions"
  if [[ -f "$dir/$tag.pt" ]]; then
    echo "[keep] $dir/$tag.pt exists — using it, not refitting"
    touch "$DONE/${prof}_fit_${tag}"; return
  fi
  step "$prof" "fit_$tag" python refusal/direction.py --mode fit --checkpoint "$ckpt" --tag "$tag"
}

# ---------------------------------------------------------------- preflight
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for c in "$SFT" "$RL"; do
  [[ -f "$c/config.json" ]] || { echo "MISSING checkpoint $c — stop, nothing else can run"; exit 1; }
done
step v1 smoke python refusal/smoke_test.py || true   # env-only FAILs (no nnsj/HF) are tolerated

# ======================================================= PHASE A — v1 (core)
if [[ "$PHASE" == "all" || "$PHASE" == "A" ]]; then
  fit_if_missing v1 M0 "$M0"
  fit_if_missing v1 M_SFT "$SFT"
  fit_if_missing v1 M_RL "$RL"
  step v1 matched_cosine python refusal/matched_cosine.py

  # E1 fast pass: refusal score only, minutes. Read its printout before the long run.
  step v1 e1_fast_M_SFT python refusal/rotation_dose.py --checkpoint "$SFT" --tag M_SFT --no-generate
  step v1 e1_fast_M0    python refusal/rotation_dose.py --checkpoint "$M0"  --tag M0    --no-generate
  # E1 full (priority): M_SFT first — it holds the registered transfer condition.
  step v1 e1_M_SFT python refusal/rotation_dose.py --checkpoint "$SFT" --tag M_SFT
  step v1 e1_M0    python refusal/rotation_dose.py --checkpoint "$M0"  --tag M0

  # E2b graded ablation
  step v1 e2b_alpha_M0    python refusal/graded_ablation.py --checkpoint "$M0"  --tag M0    --variant alpha
  step v1 e2b_alpha_M_SFT python refusal/graded_ablation.py --checkpoint "$SFT" --tag M_SFT --variant alpha
  step v1 e2b_alpha_M_RL  python refusal/graded_ablation.py --checkpoint "$RL"  --tag M_RL  --variant alpha
  for t in M0 M_SFT M_RL; do
    c="$M0"; [[ $t == M_SFT ]] && c="$SFT"; [[ $t == M_RL ]] && c="$RL"
    step v1 "e2b_single_$t" python refusal/graded_ablation.py --checkpoint "$c" --tag "$t" --variant single
  done

  # E2a addition sweep, then the paired cross-checkpoint comparison (CPU)
  step v1 e2a_M0    python refusal/addition_sweep.py --checkpoint "$M0"  --tag M0
  step v1 e2a_M_SFT python refusal/addition_sweep.py --checkpoint "$SFT" --tag M_SFT --also-own
  step v1 e2a_M_RL  python refusal/addition_sweep.py --checkpoint "$RL"  --tag M_RL  --also-own
  step v1 e2a_compare python refusal/addition_sweep.py --summarise
fi

# ================================================== PHASE B — v2 replication
if [[ "$PHASE" == "all" || "$PHASE" == "B" ]]; then
  # `if`, not `a || step ...`: bash ignores `set -e` inside a function called
  # from an || list, so a crashed build/filter would be marked done and skipped.
  if [[ ! -f refusal/data/v2/refusal_prompts_raw.jsonl ]]; then
    step v2 build python refusal/data/build_dataset.py --build
  fi
  if [[ ! -f refusal/data/v2/refusal_pairs.jsonl ]]; then
    step v2 filter python refusal/data/build_dataset.py --filter
  fi

  fit_if_missing v2 M0 "$M0"
  fit_if_missing v2 M_SFT "$SFT"
  fit_if_missing v2 M_RL "$RL"
  step v2 ceiling_M0 python refusal/direction.py --mode ceiling --checkpoint "$M0" --tag M0

  # registered §5.1 readouts (science score is data-independent: measured in v1, skipped)
  step v2 beh_M0    python refusal/behaviour.py --checkpoint "$M0"  --tag M0 --skip-nts
  step v2 beh_M_SFT python refusal/behaviour.py --checkpoint "$SFT" --tag M_SFT --ref "$M0" --skip-nts
  step v2 beh_M_RL  python refusal/behaviour.py --checkpoint "$RL"  --tag M_RL  --ref "$M0" --skip-nts

  # registered §5.2 readouts: unconditioned (D16), then M0-conditioned
  step v2 eval_M0    python refusal/direction.py --mode eval --checkpoint "$M0"  --tag M0
  step v2 eval_M_SFT python refusal/direction.py --mode eval --checkpoint "$SFT" --tag M_SFT --transfer-from M0
  step v2 eval_M_RL  python refusal/direction.py --mode eval --checkpoint "$RL"  --tag M_RL  --transfer-from M0
  step v2 evalc_M_SFT python refusal/direction.py --mode eval --checkpoint "$SFT" --tag M_SFT --transfer-from M0 --m0-condition
  step v2 evalc_M_RL  python refusal/direction.py --mode eval --checkpoint "$RL"  --tag M_RL  --transfer-from M0 --m0-condition

  # D14 control, conditioned to match v1's filtered test
  step v2 rc_M0    python refusal/random_control.py --checkpoint "$M0"  --tag M0    --m0-condition
  step v2 rc_M_SFT python refusal/random_control.py --checkpoint "$SFT" --tag M_SFT --donor M0 --m0-condition
  step v2 rc_M_RL  python refusal/random_control.py --checkpoint "$RL"  --tag M_RL  --donor M0 --m0-condition

  step v2 matched_cosine python refusal/matched_cosine.py
  step v2 e1_M_SFT python refusal/rotation_dose.py --checkpoint "$SFT" --tag M_SFT
  step v2 e1_M0    python refusal/rotation_dose.py --checkpoint "$M0"  --tag M0
fi

echo; echo "=== finished phase $PHASE in $(( ($(date +%s) - T0) / 60 )) min. Log: $LOG ==="
echo "Now save results (see RUNBOOK_SESSION.md, 'After the run'), then STOP the Studio."
