#!/usr/bin/env bash
set -euo pipefail

d=${1:-aortic_valve_disorders}

root=/cbica/projects/CXR
repo="${root}/codes/STEER"
plain="${root}/codes/MIMIC-Plain"
log="${repo}/outputs"

if [[ "$d" == "aortic_valve_disorders" || "$d" == "mitral_valve_disorders" || "$d" == "congestive_heart_failure" || "$d" == "myocardial_infarction" ]]; then
  data="${root}/dropbox/CDM_III"
  ref="${plain}/itemid_ref_ranges_III.json"
  map="${plain}/lab_test_mapping_III.pkl"
else
  data="${root}/dropbox/CDM_IV"
  ref="${plain}/itemid_ref_ranges_IV.json"
  map="${plain}/lab_test_mapping_IV.pkl"
fi

module load cuda/11.8 2>/dev/null || true
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate mimic-cdm

mkdir -p "$log"
cd "$repo"

python run.py \
  --paths cbica \
  --pathology "$d" \
  --hadm-pkl "${data}/${d}.pkl" \
  --lab-map-pkl "$map" \
  --ref-ranges-json "$ref" \
  --local-logging-dir "$log" \
  --hf-model-id meta-llama/Llama-3.3-70B-Instruct

