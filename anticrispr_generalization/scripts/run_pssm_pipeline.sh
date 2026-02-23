#!/usr/bin/env bash
# 在 anticrispr_generalization 的 pssm_work 上运行 PSSM 流水线（调用 protein_bert 的脚本与 BLAST DB）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PB="${PROJECT_ROOT}/../protein_bert"
MANIFEST="${PROJECT_ROOT}/pssm_work/sample_manifest.csv"
WORK_ROOT="${PROJECT_ROOT}/pssm_work"
BLAST_DB="${PB}/blast_db/uniref50"
N_JOBS="${N_JOBS:-8}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Run first: python scripts/prepare_fasta_manifest.py"
  exit 1
fi

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/pssm" "${WORK_ROOT}/features"

# Stage 1: PSI-BLAST
echo "[1/4] PSI-BLAST batch..."
conda run -n tf24pb bash "${PB}/scripts/pssm/01_run_psiblast_batch.sh" \
  "${MANIFEST}" "${BLAST_DB}" "${N_JOBS}" || true

# Stage 2: Retry failed
FAILED="${WORK_ROOT}/logs/failed_ids.txt"
if [[ -s "${FAILED}" ]]; then
  echo "[2/4] Retrying failed..."
  conda run -n tf24pb bash "${PB}/scripts/pssm/02_retry_failed.sh" \
    "${MANIFEST}" "${FAILED}" "${BLAST_DB}" 4 || true
else
  echo "[2/4] No failed ids."
fi

# Stage 3: Extract 1110-dim features
echo "[3/4] Extract PSSM features..."
conda run -n tf24pb python "${PB}/scripts/pssm/03_extract_rpssm_pssmac.py" \
  --manifest-csv "${MANIFEST}" --work-root "${WORK_ROOT}"

# Stage 4: Build feature cache (1110)
echo "[4/4] Build feature cache..."
conda run -n tf24pb python "${PB}/scripts/pssm/04_build_feature_cache.py" \
  --manifest-csv "${MANIFEST}" --work-root "${WORK_ROOT}" --variants 1110

echo "Done. Features: ${WORK_ROOT}/features/pssm_features_1110.parquet"
