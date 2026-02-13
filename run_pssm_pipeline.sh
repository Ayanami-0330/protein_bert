#!/bin/bash
# PSSM 流程独立启动脚本
# 完全独立于 SSH/Cursor，使用 nohup 后台运行

set -euo pipefail

LOG_DIR="/home/nemophila/projects/protein_bert/pssm_work/logs"
MAIN_LOG="${LOG_DIR}/pipeline_main.log"
DB_PREFIX="/home/nemophila/projects/protein_bert/blast_db/uniref50"
MANIFEST="/home/nemophila/projects/protein_bert/pssm_work/sample_manifest.csv"
WORK_ROOT="/home/nemophila/projects/protein_bert/pssm_work"

mkdir -p "${LOG_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================" | tee -a "${MAIN_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PSSM Pipeline Started" | tee -a "${MAIN_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================" | tee -a "${MAIN_LOG}"

# 阶段1: PSI-BLAST 批处理
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1: PSI-BLAST batch processing..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb bash "/home/nemophila/projects/protein_bert/scripts/pssm/01_run_psiblast_batch.sh" \
  "${MANIFEST}" "${DB_PREFIX}" 16 >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1: Completed" | tee -a "${MAIN_LOG}"

# 阶段2: 重试失败的任务
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2: Retrying failed tasks..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb bash "/home/nemophila/projects/protein_bert/scripts/pssm/02_retry_failed.sh" \
  "${MANIFEST}" "${LOG_DIR}/failed_ids.txt" "${DB_PREFIX}" 8 >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2: Completed" | tee -a "${MAIN_LOG}"

# 阶段3: 特征提取
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 3: Feature extraction..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb python "/home/nemophila/projects/protein_bert/scripts/pssm/03_extract_rpssm_pssmac.py" \
  --manifest-csv "${MANIFEST}" --work-root "${WORK_ROOT}" >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 3: Completed" | tee -a "${MAIN_LOG}"

# 阶段4: 构建特征缓存
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 4: Building feature cache..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb python "/home/nemophila/projects/protein_bert/scripts/pssm/04_build_feature_cache.py" \
  --manifest-csv "${MANIFEST}" --work-root "${WORK_ROOT}" >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 4: Completed" | tee -a "${MAIN_LOG}"

# 阶段5: 运行实验
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 5: Running Exp15 experiments..." | tee -a "${MAIN_LOG}"
conda run -n tf24pb jupyter nbconvert --to notebook --execute --inplace \
  "/home/nemophila/projects/protein_bert/anticrispr_demo.ipynb" >> "${MAIN_LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 5: Completed" | tee -a "${MAIN_LOG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================" | tee -a "${MAIN_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All Stages Completed Successfully!" | tee -a "${MAIN_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================" | tee -a "${MAIN_LOG}"
