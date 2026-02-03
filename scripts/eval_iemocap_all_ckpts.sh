#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/eval_iemocap_all_ckpts.sh [path/to/env]
ENV_FILE="${1:-baselines/configs/eval_iemocap_all_ckpts.env}"

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

# Allow either CKPT_DIR or legacy CHECKPOINT to drive the sweep.
if [ -z "${CKPT_DIR:-}" ] && [ -n "${CHECKPOINT:-}" ]; then
  CKPT_DIR="$(dirname "${CHECKPOINT}")"
fi

# Prefer OUT_ROOT; fall back to OUTPUT_DIR if provided.
if [ -n "${OUT_ROOT:-}" ]; then
  OUT_ROOT_ARG="${OUT_ROOT}"
elif [ -n "${OUTPUT_DIR:-}" ]; then
  OUT_ROOT_ARG="${OUTPUT_DIR}"
else
  OUT_ROOT_ARG=""
fi

python baselines/downstream/eval_iemocap_all_ckpts.py \
  ${CKPT_DIR:+--ckpt_dir "${CKPT_DIR}"} \
  ${OUT_ROOT_ARG:+--out_root "${OUT_ROOT_ARG}"} \
  ${IEMOCAP_ROOT:+--iemocap_root "${IEMOCAP_ROOT}"} \
  ${EVERY:+--every "${EVERY}"} \
  ${BACKBONE_TYPE:+--backbone_type "${BACKBONE_TYPE}"} \
  ${EM2V_CFG:+--em2v_cfg "${EM2V_CFG}"} \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_GROUP:+--wandb_group "${WANDB_GROUP}"}
