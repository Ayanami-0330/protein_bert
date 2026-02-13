#!/usr/bin/env bash
set -euo pipefail

# Retry only failed sample ids from a previous batch.
# Usage:
#   bash scripts/pssm/02_retry_failed.sh \
#     /home/nemophila/projects/protein_bert/pssm_work/sample_manifest.csv \
#     /home/nemophila/projects/protein_bert/pssm_work/logs/failed_ids.txt \
#     /home/nemophila/projects/protein_bert/blast_db/uniref50 \
#     4

MANIFEST_CSV="${1:?sample_manifest.csv is required}"
FAILED_IDS_FILE="${2:?failed_ids.txt is required}"
BLAST_DB="${3:?blast db prefix path is required}"
N_JOBS="${4:-4}"

WORK_ROOT="$(dirname "$MANIFEST_CSV")"
TASKS_FILE="${WORK_ROOT}/retry_tasks.tsv"
LOG_DIR="${WORK_ROOT}/logs"
RETRY_FAILED_FILE="${LOG_DIR}/retry_failed_ids.txt"
mkdir -p "${LOG_DIR}"

if [[ ! -s "${FAILED_IDS_FILE}" ]]; then
  echo "No failed ids to retry: ${FAILED_IDS_FILE}"
  exit 0
fi

python - <<'PY' "${MANIFEST_CSV}" "${FAILED_IDS_FILE}" "${TASKS_FILE}"
import pandas as pd
import sys

manifest = pd.read_csv(sys.argv[1])
failed = pd.read_csv(sys.argv[2], header=None, names=["sample_id"])
task = manifest.merge(failed, on="sample_id", how="inner")
task[["sample_id", "fasta_path", "pssm_path"]].drop_duplicates().to_csv(
    sys.argv[3], sep="\t", index=False, header=False
)
print(f"Retry tasks: {len(task)}")
PY

NUM_ITERATIONS="${NUM_ITERATIONS:-3}" \
EVALUE="${EVALUE:-0.001}" \
MAX_TARGET_SEQS="${MAX_TARGET_SEQS:-2000}" \
THREADS_PER_JOB="${THREADS_PER_JOB:-1}" \
FAILED_FILE="${RETRY_FAILED_FILE}" \
bash "$(dirname "$0")/01_run_psiblast_batch.sh" "${MANIFEST_CSV}" "${BLAST_DB}" "${N_JOBS}" "${TASKS_FILE}"

