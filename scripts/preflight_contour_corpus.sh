#!/usr/bin/env bash
set -euo pipefail

# Contour probe under prosody_norm=corpus (see plan/preflight-results.md).
#
# WHY: the first preflight gave summary WA 53.92 vs instance-contour WA 40.06.
# Two things differ between those arms -- the summary keeps absolute level while
# the contour is instance-normalised, AND the summary is 5 functionals per
# descriptor while the contour is the raw 64-step series. This run holds the
# representation fixed (contour, 192 dims) and changes only the normalisation, so
# the gap can be attributed.
#
#   result near 53.9  -> the gap was LEVEL DELETION; train with prosody_norm=corpus
#   result near 40.1  -> the gap was SUMMARISATION; instance stays viable
#   in between        -> both contribute; report the split
#
# Deliberately NOT scripts/preflight_prosody.sh with PROSODY_NORM=corpus: that
# re-runs A1 and B for nothing and overwrites _contour.npy, the ablation JSON and
# the PREFLIGHT_OUT logs from the instance run being compared against.
#
# Usage:
#   scripts/preflight_contour_corpus.sh [path/to/env] [--smoke|--stats-only|--skip-stats]
#
#   --smoke       corpus statistics over 500 utterances only, then stop. Run this
#                 first: compute_prosody_corpus_stats.py has never been executed,
#                 and bad constants are SILENT -- they shift the target uniformly
#                 forever with no error and no visible effect on the loss curve.
#   --stats-only  full corpus statistics, then stop (the 1-3 hour part)
#   --skip-stats  reuse an existing stats JSON, go straight to the probe

MODE=""
ENV_FILE_ARG=""
for a in "$@"; do
  case "$a" in
    --smoke|--stats-only|--skip-stats) MODE="$a" ;;
    -*) echo "unknown flag: $a" >&2; exit 1 ;;
    *) ENV_FILE_ARG="$a" ;;
  esac
done

if [ -z "${DATA_ROOT:-}" ]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  DATA_ROOT="$(dirname "${REPO_DIR}")"
  export DATA_ROOT
  echo "[INFO] DATA_ROOT not set; inferred as ${DATA_ROOT}"
fi

ENV_FILE="${ENV_FILE_ARG:-${DATA_ROOT}/eat-em2v/baselines/configs/eval_downstream_iemocap_eat_original.env}"
[ -f "${ENV_FILE}" ] || { echo "Env file not found: ${ENV_FILE}" >&2; exit 1; }

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

MANIFEST="${PROSODY_MANIFEST:-${DATA_ROOT}/manifests/pretrain/train.tsv}"
STATS_JSON="${PROSODY_CORPUS_STATS_JSON:-${DATA_ROOT}/prosody_corpus_stats.json}"
PROSODY_PREFIX="${PROSODY_PREFIX:-${FEAT_PREFIX%/*}/prosody}"
PREFLIGHT_OUT="${PREFLIGHT_OUT:-${OUTPUT_DIR%/}/preflight}"
OUT_PREFIX="${PROSODY_PREFIX}_contour_corpus"     # NOT _contour: keep the instance run
mkdir -p "$(dirname "${PROSODY_PREFIX}")" "${PREFLIGHT_OUT}"

if [ -n "${WANDB_PROJECT:-}" ]; then
  echo "[INFO] W&B: project=${WANDB_PROJECT} group=${WANDB_GROUP:-<none>}-preflight"
  [ -n "${WANDB_API_KEY:-}" ] || echo "[WARN] WANDB_API_KEY is empty (${MAIN_ENV} missing?)"
else
  echo "[WARN] WANDB_PROJECT unset; this run will not log to W&B."
fi
echo "[INFO] manifest   : ${MANIFEST}"
echo "[INFO] stats json : ${STATS_JSON}"
echo "[INFO] out prefix : ${OUT_PREFIX}"

# --- sanity check on the six constants ------------------------------------
# Bad constants are the failure this whole script is exposed to: they do not
# raise, they offset and mis-scale the target uniformly and permanently.
check_stats () {
  python - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m, s = d["prosody_corpus_mean"], d["prosody_corpus_std"]
names = d.get("descriptors", ["log_energy", "centroid", "flux"])
print(f"  utterances     : {d.get('n_utterances')}")
print(f"  no-valid-patch : {d.get('n_utterances_with_no_valid_patch')}")
print(f"  valid patches  : {d.get('n_valid_patches')}")
for n, mu, sd in zip(names, m, s):
    print(f"  {n:<12} mean={mu:>10.4f}  std={sd:>9.4f}")
bad = []
if d.get("n_utterances_with_no_valid_patch"):
    bad.append("some utterances had NO valid patch -- pad detection or manifest is wrong")
# logsumexp over 128 mel bins, NOT a per-bin value: logsumexp(x) >= max(x) >=
# mean(x), and the dataset's S_log mean is -4.268, so this can never be below
# ~-4.3 and there is no reason for it to be negative at all -- the sum adds up
# to ln(128) ~ 4.85 on top of the largest bin. An earlier version of this check
# required < 0 and would have rejected correct statistics.
if not (-5.0 < m[0] < 25.0):
    bad.append(f"log_energy mean {m[0]:.3f} is outside [-5, 25]; logsumexp over "
               "128 bins of S_log (mean ~-4.27) should land near 0. Step 0 or "
               "norm='none' is likely wrong.")
if not (0 <= m[1] <= 127):
    bad.append(f"centroid mean {m[1]:.3f} is outside the mel-bin range 0-127")
if min(s) <= 0:
    bad.append(f"non-positive std in {s}")
if bad:
    print("\n[FAIL] " + "\n[FAIL] ".join(bad)); sys.exit(1)
print("\n[OK] constants look like raw descriptors")
PY
}

# --- step 1: corpus statistics --------------------------------------------
if [ "${MODE}" = "--smoke" ]; then
  echo; echo "=== corpus statistics (SMOKE, 500 utterances) ==="
  python scripts/compute_prosody_corpus_stats.py \
    --manifest "${MANIFEST}" \
    --output_json "${STATS_JSON%.json}_SMOKE.json" \
    --max_utts 500 \
    ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
    ${WANDB_GROUP:+--wandb_group "${WANDB_GROUP}-preflight"} \
    | tee "${PREFLIGHT_OUT}/corpus_stats_smoke.log"
  echo; echo "--- sanity check ---"
  check_stats "${STATS_JSON%.json}_SMOKE.json"
  echo; echo "Smoke test passed. Now run the full pass (1-3h, use tmux/nohup):"
  echo "  scripts/preflight_contour_corpus.sh ${ENV_FILE} --stats-only"
  exit 0
fi

if [ "${MODE}" != "--skip-stats" ]; then
  if [ -f "${STATS_JSON}" ]; then
    echo; echo "=== corpus statistics already present, reusing ==="
    check_stats "${STATS_JSON}"
  else
    echo; echo "=== corpus statistics (FULL corpus -- expect 1-3 hours) ==="
    python scripts/compute_prosody_corpus_stats.py \
      --manifest "${MANIFEST}" \
      --output_json "${STATS_JSON}" \
    ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
    ${WANDB_GROUP:+--wandb_group "${WANDB_GROUP}-preflight"} \
      | tee "${PREFLIGHT_OUT}/corpus_stats.log"
    echo; echo "--- sanity check ---"
    check_stats "${STATS_JSON}"
  fi
else
  [ -f "${STATS_JSON}" ] || { echo "--skip-stats given but ${STATS_JSON} does not exist" >&2; exit 1; }
  check_stats "${STATS_JSON}"
fi

if [ "${MODE}" = "--stats-only" ]; then
  echo; echo "Stats written. Re-run with --skip-stats for the probe."
  exit 0
fi

# Two lines, read separately: `read -r A B` gives the last variable every
# remaining field, which would append the stds to the means.
{ read -r CORPUS_MEAN; read -r CORPUS_STD; } < <(
  python - "${STATS_JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(" ".join(f"{v:.6f}" for v in d["prosody_corpus_mean"]))
print(" ".join(f"{v:.6f}" for v in d["prosody_corpus_std"]))
PY
)
echo "[INFO] corpus mean=[${CORPUS_MEAN}] std=[${CORPUS_STD}]"

# --- step 2: contour features under corpus norm ---------------------------
echo; echo "=============================================================="
echo " Contour probe -- prosody_norm=corpus (192 dims)"
echo "=============================================================="
# shellcheck disable=SC2086
python baselines/downstream/preflight_prosody_features.py \
  --iemocap_root "${IEMOCAP_ROOT}" \
  --output_prefix "${OUT_PREFIX}" \
  --variant contour \
  --prosody_norm corpus \
  --corpus_mean ${CORPUS_MEAN} \
  --corpus_std ${CORPUS_STD} \
  --batch_size "${PREFLIGHT_BATCH_SIZE:-16}" \
  --num_workers 4 \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_GROUP:+--wandb_group "${WANDB_GROUP}-preflight"} \
  --wandb_name "preflight-descriptors-contour-corpus" \
  | tee "${PREFLIGHT_OUT}/contour_corpus_features.log"

# --- step 3: probe it ------------------------------------------------------
python baselines/downstream/eval_downstream_iemocap.py \
  --feat_prefix "${OUT_PREFIX}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-100}" \
  --lr "${LR:-5e-4}" \
  --max_lr "${MAX_LR:-1e-3}" \
  --step_size_up "${STEP_SIZE_UP:-10}" \
  --weight_decay "${WEIGHT_DECAY:-1e-5}" \
  --eval_is_test \
  --seed "${SEED:-42}" \
  --output_dir "${PREFLIGHT_OUT}/contour_corpus" \
  --device "${DEVICE:-cuda}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_GROUP:+--wandb_group "${WANDB_GROUP}-preflight"} \
  --wandb_name "preflight-prosody-contour-corpus" \
  | tee "${PREFLIGHT_OUT}/contour_corpus_eval.log"

echo
echo "Compare the Average WA against the two runs already recorded:"
echo "  summary          (15 dims, level retained)  WA 53.92  UA 54.59"
echo "  contour instance (192 dims, level deleted)  WA 40.06  UA 38.27"
echo "  contour corpus   (192 dims, level retained) <- this run"
echo
echo "  near 53.9 -> the gap was LEVEL DELETION; train with prosody_norm=corpus"
echo "  near 40.1 -> the gap was SUMMARISATION; instance stays viable"
echo "  between   -> both contribute; report the split"
echo
echo "Record the result in plan/preflight-results.md."
