#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/pssm/01_run_psiblast_batch.sh \
#     /home/nemophila/projects/protein_bert/pssm_work/sample_manifest.csv \
#     /home/nemophila/projects/protein_bert/blast_db/uniref50 \
#     8

MANIFEST_CSV="${1:?sample_manifest.csv is required}"
BLAST_DB="${2:?blast db prefix path is required}"
N_JOBS="${3:-8}"
TASKS_INPUT="${4:-}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
EVALUE="${EVALUE:-0.001}"
MAX_TARGET_SEQS="${MAX_TARGET_SEQS:-2000}"
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"

WORK_ROOT="$(dirname "$MANIFEST_CSV")"
LOG_DIR="${WORK_ROOT}/logs"
PSSM_DIR="${WORK_ROOT}/pssm"
FAILED_FILE="${FAILED_FILE:-${LOG_DIR}/failed_ids.txt}"
TASKS_FILE="${WORK_ROOT}/psiblast_tasks.tsv"

mkdir -p "${LOG_DIR}" "${PSSM_DIR}"
: > "${FAILED_FILE}"

if [[ -z "${TASKS_INPUT}" ]]; then
python - <<'PY' "${MANIFEST_CSV}" "${TASKS_FILE}"
import pandas as pd
import sys

manifest = pd.read_csv(sys.argv[1])
manifest[["sample_id", "fasta_path", "pssm_path"]].to_csv(
    sys.argv[2], sep="\t", index=False, header=False
)
print(f"Wrote tasks: {sys.argv[2]}")
PY
else
  TASKS_FILE="${TASKS_INPUT}"
fi

if ! command -v parallel >/dev/null 2>&1; then
  echo "GNU parallel is required. Install it first."
  exit 1
fi

run_one() {
  local sample_id="$1"
  local fasta_path="$2"
  local pssm_path="$3"
  local pssm_tmp="${pssm_path}.tmp"

  mkdir -p "$(dirname "$pssm_path")"
  if [[ -s "$pssm_path" ]]; then
    echo "[SKIP] ${sample_id}"
    return 0
  fi

  if psiblast \
    -query "$fasta_path" \
    -db "$BLAST_DB" \
    -num_iterations "$NUM_ITERATIONS" \
    -evalue "$EVALUE" \
    -max_target_seqs "$MAX_TARGET_SEQS" \
    -num_threads "$THREADS_PER_JOB" \
    -out_ascii_pssm "$pssm_tmp" \
    -out /dev/null >/dev/null 2>&1; then
    if [[ -s "$pssm_tmp" ]]; then
      mv "$pssm_tmp" "$pssm_path"
      echo "[OK] ${sample_id}"
      return 0
    fi
  fi

  rm -f "$pssm_tmp"
  echo "$sample_id" >> "${FAILED_FILE}"
  echo "[FAIL] ${sample_id}"
  return 0
}

export BLAST_DB NUM_ITERATIONS EVALUE MAX_TARGET_SEQS THREADS_PER_JOB FAILED_FILE
export -f run_one

parallel --colsep '\t' -j "${N_JOBS}" \
  run_one {1} {2} {3} :::: "${TASKS_FILE}" \
  | tee "${LOG_DIR}/psiblast_batch.log"

echo "Failed ids saved to: ${FAILED_FILE}"

