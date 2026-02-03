#!/usr/bin/env bash
set -euo pipefail

# Usage: baselines/scripts/eval_donwnstream_iemocap.sh [path/to/env]
ENV_FILE="${1:-baselines/configs/eval_donwnstream_iemocap.env}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Env file not found: ${ENV_FILE}" >&2
  exit 1
fi

# If DATA_ROOT is not set, infer it from repo location
if [ -z "${DATA_ROOT:-}" ]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  DATA_ROOT="$(dirname "${REPO_DIR}")"
  export DATA_ROOT
  echo "[INFO] DATA_ROOT not set; inferred as ${DATA_ROOT}"
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

pip install --no-cache-dir wandb fairseq==0.12.2 soundfile torchaudio h5py tensorboardX scikit_learn timm

cd "${DATA_ROOT}"/eat-em2v
git submodule update --init --recursive
export PYTHONPATH="${DATA_ROOT}/eat-em2v:${PYTHONPATH:-}"

# Step 1: Extract frozen features if missing
if [ ! -f "${FEAT_PREFIX}.npy" ] || [ ! -f "${FEAT_PREFIX}.lengths" ] || [ ! -f "${FEAT_PREFIX}.emo" ]; then
  echo "Feature files not found; extracting with ${BACKBONE_TYPE:-eat_original}..."
  python baselines/downstream/extract_eat_iemocap_features.py \
    --iemocap_root "${IEMOCAP_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --backbone_type "${BACKBONE_TYPE:-eat_original}" \
    ${EM2V_CFG:+--em2v_cfg "${EM2V_CFG}"} \
    --output_prefix "${FEAT_PREFIX}" \
    --batch_size "${BATCH_SIZE:-128}" \
    --num_workers 4 \
    --target_length 1024 \
    --n_mels 128 \
    --patch_size 16 \
    --norm_mean -4.268 \
    --norm_std 4.569 \
    --device cuda \
    --seed "${SEED:-42}"
else
  echo "Feature files already present; skipping extraction."
fi

# Step 2: Downstream evaluation on cached features
OUT_DIR="${OUTPUT_DIR%/}/${BACKBONE_TYPE:-eat_original}"
mkdir -p "${OUT_DIR}"

python baselines/downstream/eval_downstream_iemocap.py \
  --feat_prefix "${FEAT_PREFIX}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-100}" \
  --lr "${LR:-5e-4}" \
  --max_lr "${MAX_LR:-1e-3}" \
  --step_size_up "${STEP_SIZE_UP:-10}" \
  --weight_decay "${WEIGHT_DECAY:-1e-5}" \
  --eval_is_test \
  --seed "${SEED:-42}" \
  --output_dir "${OUT_DIR}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_NAME:+--wandb_name "${WANDB_NAME}"} \
  ${WANDB_GROUP:+--wandb_group "${WANDB_GROUP}"}
