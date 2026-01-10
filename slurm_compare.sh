#!/bin/bash
#SBATCH --job-name=mimic_cdm_compare
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-compare-%j.out
#SBATCH --error=slurm-compare-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

set -euo pipefail

APPTAINER_IMAGE="${APPTAINER_IMAGE:-/cbica/projects/CXR/containers/pytorch_25.12.sif}"
APPTAINER_OVERLAY="${APPTAINER_OVERLAY:-/cbica/projects/CXR/containers/pytorch_25.12_overlay.img}"
APPTAINER_OVERLAY_MODE="${APPTAINER_OVERLAY_MODE:-ro}"
APPTAINER_WRITABLE_TMPFS="${APPTAINER_WRITABLE_TMPFS:-1}"
APPTAINER_BIND="${APPTAINER_BIND:-/cbica/projects/CXR:/workspace}"
PROJECTS_BIND="${PROJECTS_BIND:-/cbica/projects:/cbica/projects}"
CONTAINER_HOME="${CONTAINER_HOME:-/workspace}"
CONTAINER_REPO_PATH="${CONTAINER_REPO_PATH:-${CONTAINER_HOME}/codes/MIMIC-Clinical-Decision-Making-Framework}"
CONTAINER_VENV="${CONTAINER_VENV:-${CONTAINER_HOME}/venvs/torch}"
HF_HOME_IN_CONTAINER="${HF_HOME_IN_CONTAINER:-${CONTAINER_HOME}/.cache/huggingface}"
PY_ENTRY="${PY_ENTRY:-run.py}"
HF_MODEL_ID="${HF_MODEL_ID:-google/medgemma-27b-text-it}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
SAMPLE_COUNT="${SAMPLE_COUNT:-20}"
SEED="${SEED:-2023}"

DATA_ROOT_III="${DATA_ROOT_III:-${CONTAINER_HOME}/dropbox/CDM_III}"
DATA_ROOT_IV="${DATA_ROOT_IV:-${CONTAINER_HOME}/dropbox/CDM_IV}"
PLAIN_REPO="${PLAIN_REPO:-${CONTAINER_HOME}/codes/MIMIC-Plain}"
LOG_DIR="${LOG_DIR:-${CONTAINER_REPO_PATH}/outputs}"

XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CONTAINER_HOME}/.cache}"
NLTK_DATA="${NLTK_DATA:-${CONTAINER_HOME}/nltk_data}"
JOB_TMP="${JOB_TMP:-${CONTAINER_HOME}/scratch/SLURM_${SLURM_JOB_ID:-$$}}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${JOB_TMP}/torch_extensions}"

DISEASE="${1:-cholecystitis}"

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

APPTAINER_CMD=(
    apptainer exec
    --nv
    --overlay "$APPTAINER_OVERLAY:${APPTAINER_OVERLAY_MODE}"
    --bind "$APPTAINER_BIND"
)

if [[ "$APPTAINER_WRITABLE_TMPFS" == "1" ]]; then
    APPTAINER_CMD+=(--writable-tmpfs)
fi

if [[ -n "${PROJECTS_BIND:-}" ]]; then
    APPTAINER_CMD+=(--bind "$PROJECTS_BIND")
fi

APPTAINER_CMD+=("$APPTAINER_IMAGE")

echo "--- Starting compare run for ${DISEASE} ---"

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

RUN_TAG=\$(date +%Y%m%d-%H%M%S)
COMPARE_DIR=\"${LOG_DIR}/compare/${DISEASE}/\${RUN_TAG}\"
REACT_LOG_DIR=\"\${COMPARE_DIR}/react\"
PLANNER_LOG_DIR=\"\${COMPARE_DIR}/planner\"
SAMPLE_IDS_FILE=\"\${COMPARE_DIR}/sample_ids.txt\"
MODEL_TAG=\"${HF_MODEL_ID##*/}\"
export SAMPLE_IDS_FILE

mkdir -p \"\$COMPARE_DIR\" \"\$REACT_LOG_DIR\" \"\$PLANNER_LOG_DIR\" \"$HF_HOME_IN_CONTAINER\" \"$XDG_CACHE_HOME\" \"$NLTK_DATA\" \
  \"$JOB_TMP\" \"$TORCH_EXTENSIONS_DIR\"
cd \"$CONTAINER_REPO_PATH\"
source \"$CONTAINER_VENV/bin/activate\"

python - <<'PY'
import os
import pickle
import random
import pathlib
hadm_pkl = pathlib.Path(\"${HADM_PKL}\")
out_path = pathlib.Path(os.environ[\"SAMPLE_IDS_FILE\"])
seed = int(\"${SEED}\")
sample_n = int(\"${SAMPLE_COUNT}\")
with hadm_pkl.open(\"rb\") as f:
    data = pickle.load(f)
keys = list(data.keys())
if sample_n > 0 and sample_n < len(keys):
    rng = random.Random(seed)
    keys = rng.sample(keys, sample_n)
out_path.write_text(\"\\n\".join(str(k) for k in keys))
print(f\"Wrote {len(keys)} sample ids to {out_path}\")
PY

python \"$PY_ENTRY\" \
  --paths cbica \
  --pathology \"$DISEASE\" \
  --hadm-pkl \"$HADM_PKL\" \
  --lab-map-pkl \"$LAB_MAP_PKL\" \
  --ref-ranges-json \"$REF_RANGES_JSON\" \
  --local-logging-dir \"\$REACT_LOG_DIR\" \
  --reasoning-effort \"$REASONING_EFFORT\" \
  --agent-type zeroshot \
  --hf-model-id \"$HF_MODEL_ID\" \
  patient_list_path=\"\$SAMPLE_IDS_FILE\"

python \"$PY_ENTRY\" \
  --paths cbica \
  --pathology \"$DISEASE\" \
  --hadm-pkl \"$HADM_PKL\" \
  --lab-map-pkl \"$LAB_MAP_PKL\" \
  --ref-ranges-json \"$REF_RANGES_JSON\" \
  --local-logging-dir \"\$PLANNER_LOG_DIR\" \
  --reasoning-effort \"$REASONING_EFFORT\" \
  --agent-type plannerjudge \
  --hf-model-id \"$HF_MODEL_ID\" \
  planner=GPTOss20BPlanner \
  patient_list_path=\"\$SAMPLE_IDS_FILE\"

REACT_RESULTS=\$(ls -td \"\$REACT_LOG_DIR/${DISEASE}/\$MODEL_TAG\"/*/results.json | head -1)
PLANNER_RESULTS=\$(ls -td \"\$PLANNER_LOG_DIR/${DISEASE}/\$MODEL_TAG\"/*/results.json | head -1)

python - <<'PY'
import json, statistics, pathlib
react_path = pathlib.Path(\"${REACT_RESULTS}\")
planner_path = pathlib.Path(\"${PLANNER_RESULTS}\")

def summarize(path):
    data = json.loads(path.read_text())
    acc = []
    rounds = []
    for v in data.values():
        if isinstance(v, dict):
            if v.get(\"diagnosis_accuracy\") is not None:
                acc.append(v.get(\"diagnosis_accuracy\"))
            eval_scores = (v.get(\"evaluation\") or {}).get(\"scores\") or {}
            if eval_scores.get(\"Rounds\") is not None:
                rounds.append(eval_scores.get(\"Rounds\"))
    return {
        \"n\": len(data),
        \"acc_mean\": statistics.mean(acc) if acc else None,
        \"rounds_mean\": statistics.mean(rounds) if rounds else None,
    }

def summarize_planner(path):
    data = json.loads(path.read_text())
    mod_rates = []
    for v in data.values():
        if isinstance(v, dict) and v.get(\"step_modification_rate\") is not None:
            mod_rates.append(v.get(\"step_modification_rate\"))
    out = summarize(path)
    out[\"mod_rate_mean\"] = statistics.mean(mod_rates) if mod_rates else None
    return out

react = summarize(react_path)
planner = summarize_planner(planner_path)
print(\"React summary:\", react)
print(\"Planner+Judge summary:\", planner)
PY
"

echo "--- Finished compare run for ${DISEASE} ---"
