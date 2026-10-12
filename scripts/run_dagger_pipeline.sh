#!/usr/bin/env bash
# DAgger rollouts with human interventions, end to end on the workstation (docs/interventions.md):
#   download -> frame store -> splits -> returns cut at each takeover -> value training -> values -> labels.
# Resumable: every stage skips what is already done (a killed training run resumes from checkpoints/last).
#
#   scripts/remote.sh bg dagger "bash scripts/run_dagger_pipeline.sh"
#   bash scripts/run_dagger_pipeline.sh --data-only      # stop after the returns (no GPU needed)
set -euo pipefail
cd "$(dirname "$0")/.."
export RECAP_DATASET=actuator_unboxing_1h_HQ_dagger_rollouts
export PATH="$HOME/.local/bin:$PATH"
RUN=vf_dagger_i1060_s0
TAG=i1060
CACHE="data/cache/$RECAP_DATASET"
PY="uv run python"
stage() { echo "[$(date '+%F %T')] === $* ==="; }

stage "download (meta, data, 3 x 240 videos)"
$PY scripts/01_download.py
stage "tables + frame store"
$PY scripts/01_extract_frames.py --source local --episodes all
if [ ! -f "$CACHE/splits.json" ]; then
    stage "splits (stratified on number of interventions x length)"
    $PY scripts/01_make_splits.py --stratify interventions --seed 0
fi
if [ ! -f "$CACHE/returns_$TAG.json" ]; then
    stage "returns cut at each takeover"
    $PY scripts/02_compute_intervention_returns.py
fi
[ -f "$CACHE/returns_$TAG.json" ] || { echo "expected $CACHE/returns_$TAG.json (C_interv changed?)"; exit 1; }
[ "${1:-}" = "--data-only" ] && { stage "data ready"; exit 0; }

if [ ! -f "runs/$RUN/summary.json" ]; then
    resume=""
    [ -d "runs/$RUN/checkpoints/last" ] && resume="--resume"
    stage "value training $RUN $resume"
    $PY scripts/04_train_value.py --config configs/train_dagger.yaml $resume
fi
if [ ! -f "runs/$RUN/values.parquet" ]; then
    stage "values for every frame (best checkpoint)"
    $PY scripts/06_infer_values.py --run "$RUN" --ckpt best --batch-size 256 --num-workers 12
fi
stage "advantages + labels (human frames forced positive)"
$PY scripts/06_compute_advantages.py --run "$RUN" --returns-tag "$TAG" --segment-len 50
stage "done"
