#!/usr/bin/env bash
set -euo pipefail

# Locate where per-dim variance collapses in the EAT-em2v student->decoder
# pipeline (explains pred_var << target_var). No training; one masked forward.
#
# Usage: bash scripts/inspect_pred_var.sh [path/to/env]
# Reuses the IEMOCAP eval env file for CHECKPOINT, IEMOCAP_ROOT, DEVICE,
# BATCH_SIZE. EM2V_CFG is optional (the em2v checkpoint carries its config
# embedded); set it only if your checkpoint lacks one. Override any var
# inline, e.g.  CHECKPOINT=/workspace/ckpts/26784368.pt bash scripts/...

# If DATA_ROOT is not set, infer it from repo location
if [ -z "${DATA_ROOT:-}" ]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  DATA_ROOT="$(dirname "${REPO_DIR}")"
  export DATA_ROOT
  echo "[INFO] DATA_ROOT not set; inferred as ${DATA_ROOT}"
fi

ENV_FILE="${1:-${DATA_ROOT}/eat-em2v/baselines/configs/eval_donwnstream_iemocap_eat_em2v.env}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Env file not found: ${ENV_FILE}" >&2
  exit 1
fi

MAIN_ENV="${DATA_ROOT}/eat-em2v/.env"
[ -f "${MAIN_ENV}" ] && source "${MAIN_ENV}"

# shellcheck disable=SC1090
source "${ENV_FILE}"

pip install --no-cache-dir wandb fairseq==0.12.2 soundfile torchaudio h5py tensorboardX scikit_learn timm

cd "${DATA_ROOT}"/eat-em2v
git submodule update --init --recursive
export PYTHONPATH="${DATA_ROOT}/eat-em2v:${PYTHONPATH:-}"

python -m baselines.downstream.inspect_pred_var \
  --checkpoint "${CHECKPOINT}" \
  ${EM2V_CFG:+--em2v_cfg "${EM2V_CFG}"} \
  --iemocap_root "${IEMOCAP_ROOT}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --clone_batch "${CLONE_BATCH:-2}" \
  --device "${DEVICE:-cuda}"
