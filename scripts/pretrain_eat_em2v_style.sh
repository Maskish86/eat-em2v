#!/bin/bash
set -e

if [ -z "${DATA_ROOT:-}" ]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  DATA_ROOT="$(dirname "${REPO_DIR}")"
  export DATA_ROOT
  echo "[INFO] DATA_ROOT not set; inferred as ${DATA_ROOT}"
fi

pip install --no-cache-dir wandb fairseq==0.11.2 soundfile torchaudio h5py tensorboardX scikit_learn timm

cd "${DATA_ROOT}"/eat-em2v
git submodule update --init --recursive

export PYTHONPATH="${DATA_ROOT}/eat-em2v:${PYTHONPATH}"
export WANDB_RUN_ID_FILE="${DATA_ROOT}/wandb/run_id"

CHECKPOINT_DIR="${DATA_ROOT}/checkpoints/ser_pretrained"
RESTORE_ARG=()
if [ -f "${CHECKPOINT_DIR}/checkpoint_last.pt" ]; then
  echo "Auto-resume: found ${CHECKPOINT_DIR}/checkpoint_last.pt"
  RESTORE_ARG=("checkpoint.restore_file=${CHECKPOINT_DIR}/checkpoint_last.pt")
fi

MANIFEST_ROOT="${DATA_ROOT}/manifests/pretrain"
python "${DATA_ROOT}/eat-em2v/scripts/build_wav_manifest.py" \
  --root "${DATA_ROOT}/datasets" \
  --out "${MANIFEST_ROOT}/train.tsv"

python -m fairseq_cli.hydra_train -m \
    --config-dir ${DATA_ROOT}/eat-em2v/baselines/configs \
    --config-name pretrain_eat_em2v_style \
    common.user_dir=baselines \
    checkpoint.save_dir="${CHECKPOINT_DIR}" \
    "${RESTORE_ARG[@]}" \
    task.data=${MANIFEST_ROOT} \
    task.h5_format=False \
