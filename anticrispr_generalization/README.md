# Anti-CRISPR 泛化验证项目

在 **Fusion_PSSM1110** 上，用 2023–2025 年新发现的 32 个 anti-CRISPR 蛋白验证模型泛化能力。

- **训练**：与 `protein_bert/fusion_confusion_matrix_demo.ipynb` 完全一致（同一数据、同一配置、同一流程），在 protein_bert 的 train/test 上训练得到模型与 scaler。
- **验证**：对新发现的 32 个 Acr 提取 1110 维 PSSM（与训练集相同流水线），用训练时的 scaler 变换后输入模型预测，统计被正确判为 Acr 的比例等。

所有工作目录：`/home/nemophila/projects/anticrispr_generalization`（不修改 `protein_bert` 仓库）。

## 流程概览

1. **序列**：`data/new_acr_sequences.csv`（列：name, seq, label）。运行 `python scripts/fetch_sequences.py` 尝试从 UniProt/NCBI 检索；缺失可手动补齐。**重复运行脚本只会补空位，不会覆盖已有序列。**
2. **FASTA 与 manifest**：`conda run -n tf24pb python scripts/prepare_fasta_manifest.py` → 按 CSV 生成 `pssm_work/sample_manifest.csv` 与 `pssm_work/fasta/`（仅含有序列的条目）。
3. **PSSM 特征**：`bash scripts/run_pssm_pipeline.sh`（调用 protein_bert 的 BLAST DB 与 01/02/03/04），得到 `pssm_work/features/pssm_features_1110.parquet`。**补全或新增序列后，需重跑本步与上一步，新蛋白才会有 1110 维 PSSM。**
4. **训练与泛化评估**：用 conda 环境 `tf24pb` 打开 `train_and_validate.ipynb`，顺序运行：在 protein_bert 数据上训练 Fusion_PSSM1110，保存模型与 scaler 到 `outputs/`；再加载新 Acr 的 1110 维特征并预测，输出分数表与 Recall。

## 依赖

- 使用与 protein_bert 相同的 conda 环境（如 `tf24pb`）。
- 需可访问 protein_bert 的 BLAST 库与预训练模型路径（见 `config.py`）。
