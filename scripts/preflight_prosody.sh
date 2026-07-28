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

# corpus mode needs the six constants from compute_prosody_corpus_stats.py;
# compute_prosody asserts on them, so fail here with a usable message instead.
CONTOUR_NORM_ARGS=()
if [ "${PROSODY_NORM:-instance}" = "corpus" ]; then
  if [ -n "${PROSODY_CORPUS_STATS_JSON:-}" ]; then
    # Two lines, read separately. A single space-joined line cannot be split with
    # `read -r A B`: the last variable absorbs every remaining field, so the stds
    # would end up appended to the means.
    { read -r PROSODY_CORPUS_MEAN; read -r PROSODY_CORPUS_STD; } < <(
      python - "${PROSODY_CORPUS_STATS_JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(" ".join(f"{v:.6f}" for v in d["prosody_corpus_mean"]))
print(" ".join(f"{v:.6f}" for v in d["prosody_corpus_std"]))
PY
    )
  fi
  if [ -z "${PROSODY_CORPUS_MEAN:-}" ] || [ -z "${PROSODY_CORPUS_STD:-}" ]; then
    echo "[ERROR] PROSODY_NORM=corpus needs corpus statistics." >&2
    echo "        Set PROSODY_CORPUS_STATS_JSON to the output of" >&2
    echo "        scripts/compute_prosody_corpus_stats.py, or set" >&2
    echo "        PROSODY_CORPUS_MEAN / PROSODY_CORPUS_STD to three floats each." >&2
    exit 1
  fi
  # shellcheck disable=SC2206
  CONTOUR_NORM_ARGS=(--corpus_mean ${PROSODY_CORPUS_MEAN} --corpus_std ${PROSODY_CORPUS_STD})
  echo "corpus stats: mean=[${PROSODY_CORPUS_MEAN}] std=[${PROSODY_CORPUS_STD}]"
fi

python baselines/downstream/preflight_prosody_features.py \
  --iemocap_root "${IEMOCAP_ROOT}" \
  --output_prefix "${PROSODY_PREFIX}_contour" \
  --variant contour \
  --prosody_norm "${PROSODY_NORM:-instance}" \
  ${CONTOUR_NORM_ARGS[@]+"${CONTOUR_NORM_ARGS[@]}"} \
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

# _summary_all carries the probe-only candidates too, so the R2 table and the
# joined view cover them. Falls back to _summary if A did not emit it.
B_TARGET="${PROSODY_PREFIX}_summary_all"
[ -f "${B_TARGET}.npy" ] || B_TARGET="${PROSODY_PREFIX}_summary"

python baselines/downstream/preflight_prosody_r2.py \
  --feat_prefix "${FEAT_PREFIX}" \
  --prosody_prefix "${B_TARGET}" \
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
