#!/bin/bash
# PSSM 流程独立启动脚本
# 完全独立于 SSH/Cursor，使用 nohup 后台运行

set -euo pipefail

# 统一使用环境变量 PSSM_WORK_ROOT 作为 PSSM 工作根目录，默认为仓库外的安全路径
WORK_ROOT="${PSSM_WORK_ROOT:-/home/nemophila/data/pssm_work}"
RESULTS_ROOT="${PB_RESULTS_ROOT:-/home/nemophila/data/protein_bert_results}"
LOG_DIR="${WORK_ROOT}/logs"
MAIN_LOG="${LOG_DIR}/pipeline_main.log"
DB_PREFIX="${BLAST_DB_PREFIX:-/home/nemophila/data/blast_db/uniref50}"
MANIFEST="${WORK_ROOT}/sample_manifest.csv"

mkdir -p "${LOG_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================" | tee -a "${MAIN_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PSSM Pipeline Started" | tee -a "${MAIN_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================" | tee -a "${MAIN_LOG}"

# 阶段1: PSI-BLAST 批处理
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1: PSI-BLAST batch processing..." | tee -a "${MAIN_LOG}"
THREADS_PER_JOB=1 \
NUM_ITERATIONS=3 \
EVALUE=0.001 \
MAX_TARGET_SEQS=2000 \
conda run -n tf24pb bash "/home/nemophila/projects/protein_bert/scripts/pssm/01_run_psiblast_batch.sh" \
  "${MANIFEST}" "${DB_PREFIX}" 20 >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1: Completed" | tee -a "${MAIN_LOG}"

# 阶段2: 重试失败的任务
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2: Retrying failed tasks..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb bash "/home/nemophila/projects/protein_bert/scripts/pssm/02_retry_failed.sh" \
  "${MANIFEST}" "${LOG_DIR}/failed_ids.txt" "${DB_PREFIX}" 20 >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2: Completed" | tee -a "${MAIN_LOG}"

# 阶段3: 特征提取
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 3: Feature extraction..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb python "/home/nemophila/projects/protein_bert/scripts/pssm/03_extract_rpssm_pssmac.py" \
  --manifest-csv "${MANIFEST}" --work-root "${WORK_ROOT}" >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 3: Completed" | tee -a "${MAIN_LOG}"

# 阶段4: 构建特征缓存
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 4: Building feature cache..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb python "/home/nemophila/projects/protein_bert/scripts/pssm/04_build_feature_cache.py" \
  --manifest-csv "${MANIFEST}" --work-root "${WORK_ROOT}" --variants 310,710,1110 >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 4: Completed" | tee -a "${MAIN_LOG}"

# 阶段5: 运行实验
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 5: Running Exp15~Exp17 experiments..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb jupyter nbconvert --to notebook --execute --inplace \
  "/home/nemophila/projects/protein_bert/anticrispr_demo.ipynb" >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 5: Completed" | tee -a "${MAIN_LOG}"

# 阶段6: 生成结论报告
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 6: Generating plan reports..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb python "/home/nemophila/projects/protein_bert/scripts/pssm/06_generate_plan_reports.py" \
  --features-dir "${WORK_ROOT}/features" \
  --results-dir "${RESULTS_ROOT}/plan_reports" >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 6: Completed" | tee -a "${MAIN_LOG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================" | tee -a "${MAIN_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All Stages Completed Successfully!" | tee -a "${MAIN_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================" | tee -a "${MAIN_LOG}"
