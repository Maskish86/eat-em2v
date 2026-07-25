#!/usr/bin/env bash
set -euo pipefail

# Pre-flight probes for the prosody pretext task (see prosody-multitask-plan.md).
#
# Runs BEFORE spending a pretraining run:
#   A1. prosody summary  (15 dims, raw descriptors)      -> IEMOCAP linear probe
#   A2. prosody contour  (192 dims, the training target) -> IEMOCAP linear probe
#   B.  ridge from frozen-EAT mean-pooled features to the summary -> per-descriptor R^2
#
# Compare A's WA against the MEAN PER-FOLD MAJORITY RATE that step A1 prints --
# not against 25%. IEMOCAP's 4-class merge is imbalanced.
#
# Usage: scripts/preflight_prosody.sh [path/to/env]
# The env file is the same one the downstream eval uses (IEMOCAP_ROOT, FEAT_PREFIX, ...).

if [ -z "${DATA_ROOT:-}" ]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  DATA_ROOT="$(dirname "${REPO_DIR}")"
  export DATA_ROOT
  echo "[INFO] DATA_ROOT not set; inferred as ${DATA_ROOT}"
fi

ENV_FILE="${1:-${DATA_ROOT}/eat-em2v/baselines/configs/eval_downstream_iemocap_eat_em2v.env}"
if [ ! -f "${ENV_FILE}" ]; then
  echo "Env file not found: ${ENV_FILE}" >&2
  exit 1
fi

MAIN_ENV="${DATA_ROOT}/eat-em2v/.env"
[ -f "${MAIN_ENV}" ] && source "${MAIN_ENV}"
export WANDB_API_KEY

# shellcheck disable=SC1090
source "${ENV_FILE}"

pip install --no-cache-dir wandb fairseq==0.12.2 soundfile torchaudio h5py tensorboardX scikit_learn timm

cd "${DATA_ROOT}"/eat-em2v
git submodule update --init --recursive
export PYTHONPATH="${DATA_ROOT}/eat-em2v:${PYTHONPATH:-}"
export WANDB_DIR="${DATA_ROOT}/wandb"

PROSODY_PREFIX="${PROSODY_PREFIX:-${FEAT_PREFIX%/*}/prosody}"
PREFLIGHT_OUT="${PREFLIGHT_OUT:-${OUTPUT_DIR%/}/preflight}"
mkdir -p "$(dirname "${PROSODY_PREFIX}")" "${PREFLIGHT_OUT}"

echo
echo "=============================================================="
echo " Pre-flight A1 -- prosody summary (raw descriptors, 15 dims)"
echo "=============================================================="
python baselines/downstream/preflight_prosody_features.py \
  --iemocap_root "${IEMOCAP_ROOT}" \
  --output_prefix "${PROSODY_PREFIX}_summary" \
  --variant summary \
  --batch_size "${PREFLIGHT_BATCH_SIZE:-16}" \
  --num_workers 4 \
  | tee "${PREFLIGHT_OUT}/a1_summary_features.log"

python baselines/downstream/eval_downstream_iemocap.py \
  --feat_prefix "${PROSODY_PREFIX}_summary" \
  --batch_size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-100}" \
  --lr "${LR:-5e-4}" \
  --max_lr "${MAX_LR:-1e-3}" \
  --step_size_up "${STEP_SIZE_UP:-10}" \
  --weight_decay "${WEIGHT_DECAY:-1e-5}" \
  --eval_is_test \
  --seed "${SEED:-42}" \
  --output_dir "${PREFLIGHT_OUT}/summary" \
  --device "${DEVICE:-cuda}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_GROUP:+--wandb_group "${WANDB_GROUP}-preflight"} \
  --wandb_name "preflight-prosody-summary"

echo
echo "=============================================================="
echo " Pre-flight A2 -- prosody contour (training target, 192 dims)"
echo "=============================================================="
python baselines/downstream/preflight_prosody_features.py \
  --iemocap_root "${IEMOCAP_ROOT}" \
  --output_prefix "${PROSODY_PREFIX}_contour" \
  --variant contour \
  --prosody_norm "${PROSODY_NORM:-instance}" \
  --batch_size "${PREFLIGHT_BATCH_SIZE:-16}" \
  --num_workers 4 \
  | tee "${PREFLIGHT_OUT}/a2_contour_features.log"

python baselines/downstream/eval_downstream_iemocap.py \
  --feat_prefix "${PROSODY_PREFIX}_contour" \
  --batch_size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-100}" \
  --lr "${LR:-5e-4}" \
  --max_lr "${MAX_LR:-1e-3}" \
  --step_size_up "${STEP_SIZE_UP:-10}" \
  --weight_decay "${WEIGHT_DECAY:-1e-5}" \
  --eval_is_test \
  --seed "${SEED:-42}" \
  --output_dir "${PREFLIGHT_OUT}/contour" \
  --device "${DEVICE:-cuda}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_GROUP:+--wandb_group "${WANDB_GROUP}-preflight"} \
  --wandb_name "preflight-prosody-contour-${PROSODY_NORM:-instance}"

echo
echo "=============================================================="
echo " Pre-flight B -- prosody decodability from the encoder"
echo "=============================================================="
if [ ! -f "${FEAT_PREFIX}.npy" ]; then
  echo "[ERROR] ${FEAT_PREFIX}.npy not found." >&2
  echo "        Run scripts/eval_donwnstream_iemocap.sh first to extract encoder features," >&2
  echo "        or point FEAT_PREFIX at an existing extraction." >&2
  exit 1
fi

python baselines/downstream/preflight_prosody_r2.py \
  --feat_prefix "${FEAT_PREFIX}" \
  --prosody_prefix "${PROSODY_PREFIX}_summary" \
  --label "${BACKBONE_TYPE:-eat_original}" \
  --output_json "${PREFLIGHT_OUT}/prosody_r2_${BACKBONE_TYPE:-eat_original}.json" \
  | tee "${PREFLIGHT_OUT}/b_prosody_r2.log"

echo
echo "Done. Decision rule (prosody-multitask-plan.md, 'Pre-flight A'):"
echo "  both WA at the per-fold majority rate       -> STOP, direction is dead"
echo "  contour above floor, summary at floor       -> emotion is in the contour;"
echo "                                                 expect prosody_norm=instance"
echo "  both above floor                            -> proceed; corpus mode worth the job"
echo
echo "Store the Pre-flight B numbers next to the 67.07% frozen-EAT reference in"
echo "experiment-summary-dino-lr.md -- they are the baseline for success requirement 2."
