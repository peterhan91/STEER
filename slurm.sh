#!/bin/bash
#SBATCH --job-name=mimic_cdm
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

set -euo pipefail

APPTAINER_IMAGE="${APPTAINER_IMAGE:-/cbica/projects/CXR/containers/pytorch_25.12.sif}"
APPTAINER_OVERLAY="${APPTAINER_OVERLAY:-/cbica/projects/CXR/containers/pytorch_25.12_overlay.img}"
# Default to mounting the shared overlay read-only so multiple jobs can run concurrently.
# Override APPTAINER_OVERLAY_MODE=rw if you really need a writable overlay file.
APPTAINER_OVERLAY_MODE="${APPTAINER_OVERLAY_MODE:-ro}"
APPTAINER_WRITABLE_TMPFS="${APPTAINER_WRITABLE_TMPFS:-1}"
APPTAINER_BIND="${APPTAINER_BIND:-/cbica/projects/CXR:/workspace}"
PROJECTS_BIND="${PROJECTS_BIND:-/cbica/projects:/cbica/projects}"
CONTAINER_HOME="${CONTAINER_HOME:-/workspace}"
CONTAINER_REPO_PATH="${CONTAINER_REPO_PATH:-${CONTAINER_HOME}/codes/MIMIC-Clinical-Decision-Making-Framework}"
CONTAINER_VENV="${CONTAINER_VENV:-${CONTAINER_HOME}/venvs/torch}"
HF_HOME_IN_CONTAINER="${HF_HOME_IN_CONTAINER:-${CONTAINER_HOME}/.cache/huggingface}"
PY_ENTRY="${PY_ENTRY:-run.py}"
HF_MODEL_ID="${HF_MODEL_ID:-}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"

DATA_ROOT_III="${DATA_ROOT_III:-${CONTAINER_HOME}/dropbox/CDM_III}"
DATA_ROOT_IV="${DATA_ROOT_IV:-${CONTAINER_HOME}/dropbox/CDM_IV}"
PLAIN_REPO="${PLAIN_REPO:-${CONTAINER_HOME}/codes/MIMIC-Plain}"
LOG_DIR="${LOG_DIR:-${CONTAINER_REPO_PATH}/outputs}"

XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CONTAINER_HOME}/.cache}"
NLTK_DATA="${NLTK_DATA:-${CONTAINER_HOME}/nltk_data}"
JOB_TMP="${JOB_TMP:-${CONTAINER_HOME}/scratch/SLURM_${SLURM_JOB_ID:-$$}}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${JOB_TMP}/torch_extensions}"

# Disease selection (arg1)
DISEASE="${1:-}"
if [[ -z "$DISEASE" ]]; then
  echo "Usage: sbatch slurm.sh <disease> [extra args]" >&2
  exit 1
fi
shift

case "$DISEASE" in
  aortic_valve_disorders|mitral_valve_disorders|congestive_heart_failure|myocardial_infarction)
    DATA_PATH="$DATA_ROOT_III"
    REF_RANGES_JSON="${PLAIN_REPO}/itemid_ref_ranges_III.json"
    LAB_MAP_PKL="${PLAIN_REPO}/lab_test_mapping_III.pkl"
    ;;
  *)
    DATA_PATH="$DATA_ROOT_IV"
    REF_RANGES_JSON="${PLAIN_REPO}/itemid_ref_ranges_IV.json"
    LAB_MAP_PKL="${PLAIN_REPO}/lab_test_mapping_IV.pkl"
    ;;
esac

HADM_PKL="${DATA_PATH}/${DISEASE}_hadm_info_first_diag.pkl"

PYTHON_EXTRA_ARGS=""
if [[ $# -gt 0 ]]; then
    for arg in "$@"; do
        PYTHON_EXTRA_ARGS+=" $(printf '%q' "$arg")"
    done
fi

MODEL_ARG="--hf-model-id \"$HF_MODEL_ID\""
if [[ "$PYTHON_EXTRA_ARGS" == *"--hf-model-id"* ]]; then
    MODEL_ARG=""
elif [[ -z "$HF_MODEL_ID" ]]; then
    echo "ERROR: Set HF_MODEL_ID or pass --hf-model-id in extra args." >&2
    exit 1
fi

APPTAINER_CMD=(
    apptainer exec
    --nv
    --overlay "$APPTAINER_OVERLAY:${APPTAINER_OVERLAY_MODE}"
    --bind "$APPTAINER_BIND"
)

if [[ "$APPTAINER_WRITABLE_TMPFS" == "1" ]]; then
    # Give each job its own ephemeral writable layer even when overlay is mounted read-only.
    APPTAINER_CMD+=(--writable-tmpfs)
fi

if [[ -n "${PROJECTS_BIND:-}" ]]; then
    APPTAINER_CMD+=(--bind "$PROJECTS_BIND")
fi

APPTAINER_CMD+=("$APPTAINER_IMAGE")

echo "--- Starting MIMIC CDM run via $PY_ENTRY ---"

"${APPTAINER_CMD[@]}" bash -lc "
set -euo pipefail
unset CC CXX
export HOME=\"$CONTAINER_HOME\"
export HF_HOME=\"$HF_HOME_IN_CONTAINER\"
export TRANSFORMERS_CACHE=\"${HF_HOME_IN_CONTAINER}/transformers\"
export HF_DATASETS_CACHE=\"${HF_HOME_IN_CONTAINER}/datasets\"
export XDG_CACHE_HOME=\"${XDG_CACHE_HOME}\"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export NLTK_DATA=\"${NLTK_DATA}\"
export TMPDIR=\"${JOB_TMP}\"
export TORCH_EXTENSIONS_DIR=\"${TORCH_EXTENSIONS_DIR}\"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=\"${CONTAINER_REPO_PATH}:${PYTHONPATH:-}\"

mkdir -p \"$LOG_DIR\" \"$HF_HOME_IN_CONTAINER\" \"$XDG_CACHE_HOME\" \"$NLTK_DATA\" \
  \"$JOB_TMP\" \"$TORCH_EXTENSIONS_DIR\"
cd \"$CONTAINER_REPO_PATH\"
source \"$CONTAINER_VENV/bin/activate\"

python \"$PY_ENTRY\" \
  --paths cbica \
  --pathology \"$DISEASE\" \
  --hadm-pkl \"$HADM_PKL\" \
  --lab-map-pkl \"$LAB_MAP_PKL\" \
  --ref-ranges-json \"$REF_RANGES_JSON\" \
  --local-logging-dir \"$LOG_DIR\" \
  --reasoning-effort \"$REASONING_EFFORT\" \
  ${MODEL_ARG}${PYTHON_EXTRA_ARGS}
"

echo "--- Finished MIMIC CDM run ($PY_ENTRY) ---"
