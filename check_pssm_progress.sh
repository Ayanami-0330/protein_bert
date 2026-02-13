#!/bin/bash
# PSSM 流水线进度监控脚本
# 使用：bash check_pssm_progress.sh

echo "==============================================="
echo "PSSM 流水线进度监控"
echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================="

echo -e "\n[1] 活跃进程"
echo "---"
ps -ef | grep -E "makeblastdb|psiblast|01_run_psiblast_batch|03_extract_rpssm|04_build_feature_cache|nbconvert" | grep -v grep | head -10 || echo "无相关进程"

echo -e "\n[2] 数据库状态"
echo "---"
DB_PREFIX="/home/nemophila/projects/protein_bert/blast_db/uniref50"
if [ -s "${DB_PREFIX}.pal" ] || [ -s "${DB_PREFIX}.00.psq" ]; then
    echo "✓ BLAST 数据库已建立（分卷格式）"
    echo "  - pal 文件: $([ -s "${DB_PREFIX}.pal" ] && echo 已生成 || echo 未生成)"
    part_count=$(ls "${DB_PREFIX}".*.psq 2>/dev/null | wc -l)
    echo "  - psq 分卷数: ${part_count}"
else
    echo "✗ BLAST 数据库尚未完成（仍在建立中）"
    if [ -f "/home/nemophila/projects/protein_bert/blast_db/uniref50.fasta" ]; then
        echo "  - uniref50.fasta 已解压: $(du -sh /home/nemophila/projects/protein_bert/blast_db/uniref50.fasta | cut -f1)"
    fi
fi

echo -e "\n[3] PSSM 生成进度"
echo "---"
PSSM_DIR="/home/nemophila/projects/protein_bert/pssm_work/pssm"
if [ -d "$PSSM_DIR" ]; then
    pssm_count=$(ls "$PSSM_DIR"/*.pssm 2>/dev/null | wc -l)
    echo "已生成 PSSM 文件: $pssm_count / 1393"
    echo "进度: $(echo "scale=2; $pssm_count * 100 / 1393" | bc)%"
    
    # 统计成功/失败/跳过
    LOG_FILE="/home/nemophila/projects/protein_bert/pssm_work/logs/psiblast_batch.log"
    if [ -f "$LOG_FILE" ]; then
        ok_count=$(grep -c '\[OK\]' "$LOG_FILE" 2>/dev/null || echo 0)
        fail_count=$(grep -c '\[FAIL\]' "$LOG_FILE" 2>/dev/null || echo 0)
        skip_count=$(grep -c '\[SKIP\]' "$LOG_FILE" 2>/dev/null || echo 0)
        echo "  - 成功: $ok_count"
        echo "  - 失败: $fail_count"
        echo "  - 跳过: $skip_count"
        echo "  - 日志最后 3 行:"
        tail -3 "$LOG_FILE"
    fi
else
    echo "PSSM 目录尚未创建（还未进入 PSI-BLAST 阶段）"
fi

echo -e "\n[4] 特征提取状态"
echo "---"
if [ -f "/home/nemophila/projects/protein_bert/pssm_work/features/feature_status.csv" ]; then
    echo "✓ 特征提取已完成"
    wc -l /home/nemophila/projects/protein_bert/pssm_work/features/feature_status.csv
else
    echo "✗ 特征提取尚未开始"
fi

echo -e "\n[5] 特征缓存状态"
echo "---"
for dim in 310 710 1110; do
    if [ -f "/home/nemophila/projects/protein_bert/pssm_work/features/pssm_features_${dim}.parquet" ]; then
        echo "✓ pssm_features_${dim}.parquet 已构建"
    elif [ -f "/home/nemophila/projects/protein_bert/pssm_work/features/pssm_features_${dim}.csv" ]; then
        echo "✓ pssm_features_${dim}.csv 已构建"
    else
        echo "✗ pssm_features_${dim} 尚未构建"
    fi
done

echo -e "\n[6] Exp15~Exp17 实验结果"
echo "---"
if [ -f "/home/nemophila/projects/protein_bert/pssm_work/features/exp_results.csv" ]; then
    echo "✓ 实验已完成（exp_results.csv）"
    head -8 /home/nemophila/projects/protein_bert/pssm_work/features/exp_results.csv
else
    echo "✗ 实验尚未开始或尚未汇总（exp_results.csv 缺失）"
fi

echo -e "\n[7] 磁盘空间"
echo "---"
df -h /home/nemophila/projects/protein_bert | tail -1

echo -e "\n[8] 后台任务日志文件"
echo "---"
echo "数据库下载建库日志: /home/nemophila/.cursor/projects/home-nemophila-projects/terminals/787696.txt"
echo "串行总流程日志: /home/nemophila/.cursor/projects/home-nemophila-projects/terminals/972930.txt"

echo -e "\n==============================================="
echo "提示：使用 'tail -f' 可以实时查看日志"
echo "例如：tail -f /home/nemophila/projects/protein_bert/pssm_work/logs/psiblast_batch.log"
echo "==============================================="
