# Anti-CRISPR LM Baseline Comparison

新增入口：`lm_baseline_comparison.ipynb`

## 目标

在统一任务协议下比较以下冻结蛋白语言模型：
- ProteinBERT
- ProtT5 (`Rostlab/prot_t5_xl_uniref50`)
- ESM-2 (`facebook/esm2_t30_150M_UR50D`)
- Ankh-large（默认 `ElnaggarLab/ankh2-large`）

统一评估输出：`AUC, AUPRC, F1, MCC, Brier, ECE, Threshold`。

## 协议（与现有实验一致）

- 数据读取：`proteinbert.pssm_fusion.load_anticrispr_with_ids`
- 划分方式：`train_test_split(..., test_size=0.1, stratify=train_df['label'], random_state=22)`
- 评估函数：`proteinbert.pssm_fusion.evaluate_binary` + `find_best_threshold`

## 实现要点

- 每个 LM 先提取句子级表示（冻结大模型，仅前向推理）。
- 将各 LM 的原始表示通过 train-set PCA 压缩到 128 维，缓存为 `cache/lm_baseline/<MODEL>/*_128.npy`。
- ESM-2 当前使用 150M 版本（hidden size=640）；切换 checkpoint 时建议清理旧 `*_raw.npy`，或依赖维度校验逻辑自动重算，避免缓存误复用。
- 在 128 维表示上训练统一 head：
  - `LayerNormalization -> Dense(128, relu) -> Dropout(0.3) -> Dense(1, sigmoid)`
- 在验证集寻找最优阈值（F1），在测试集报告最终指标。

## 与后续实验衔接

- 本对比仅新增 notebook 与缓存，不改动既有实验 notebook。
- 后续 `anticrispr_demo.ipynb`、PSSM-only、Fusion 等实验保持原路径。
- 以本对比最优模型作为后续 backbone；若 ProteinBERT 最优，则沿用 ProteinBERT 作为默认 backbone。
