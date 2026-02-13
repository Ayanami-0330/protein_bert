---
name: PSSM融合后续实验计划
overview: 以当前已产出结果为固定基线，不扩seed不复现论文全流程，聚焦新增 PSSM-composition 与 DPC-PSSM 并验证其对 ProteinBERT 融合效果的增益。
todos:
  - id: freeze-current-baseline
    content: 固定当前 Baseline/RPSSM110/PSSM310 结果作为后续唯一对照基线
    status: pending
  - id: extract-acrpred4-features
    content: 实现并缓存 PSSM-composition 与 DPC-PSSM 特征，并与现有样本对齐
    status: pending
  - id: run-focused-fusion-ablation
    content: 在固定seed集合上完成“现有310 + 新特征组合”的聚焦消融实验
    status: pending
  - id: decision-report
    content: 基于AUPRC/MCC与校准指标给出是否纳入新特征的结论
    status: pending
isProject: false
---

# ProteinBERT-PSSM 下一阶段实验计划

## 目标

- 保持当前产出作为固定基线（不再扩 seed、不过度回滚历史配置）。
- 在 ProteinBERT 融合路线中，重点验证新增 `PSSM-composition` 与 `DPC-PSSM` 是否带来稳定增益。
- 以最小改动完成“特征增强”而非“整套论文复现”。

## 现状基线与关键发现

- 当前结果来自 [anticrispr_demo.ipynb](/home/nemophila/projects/protein_bert/anticrispr_demo.ipynb)：
  - `Baseline_ProteinBERT` 约 `AUC 0.887 / AUPRC 0.605`
  - `Ablation_RPSSM_110` 约 `AUC 0.898 / AUPRC 0.657`
  - `Exp15_PSSM310` 约 `AUC 0.901 / AUPRC 0.649`
- 结论：PSSM 融合有效，下一步重点是“在现有310维基础上继续引入 AcrPred 指出的高价值 PSSM 特征”。
- 融合实现位于 [proteinbert/pssm_fusion.py](/home/nemophila/projects/protein_bert/proteinbert/pssm_fusion.py)，当前采用“多层 global 表示瓶颈压缩 + PSSM 分支等宽融合”。

## 实验阶段

### 阶段A：冻结当前基线（轻量）

- 固定当前三组结果作为基线：`Baseline_ProteinBERT`、`Ablation_RPSSM_110`、`Exp15_PSSM310`。
- 不新增 seed，不重做历史路线对齐。
- 交付：一份“基线锁定说明”（后续新增实验统一对照这三组）。

### 阶段B：新增特征提取（核心）

- 在 [scripts/pssm/03_extract_rpssm_pssmac.py](/home/nemophila/projects/protein_bert/scripts/pssm/03_extract_rpssm_pssmac.py) 基础上扩展提取：
  - `PSSM-composition`（400）
  - `DPC-PSSM`（400）
- 在缓存构建脚本 [scripts/pssm/04_build_feature_cache.py](/home/nemophila/projects/protein_bert/scripts/pssm/04_build_feature_cache.py) 增加多缓存产出，至少支持：
  - `PSSM310`（现有）
  - `PSSM710 = 310 + PSSM-composition(400)`
  - `PSSM1110 = 310 + DPC-PSSM(400) + PSSM-composition(400)`（可按资源选择）
- 交付：新特征缓存与维度映射说明（feat 区间到特征语义）。

### 阶段C：聚焦消融实验（不扩seed）

- 在 notebook 中保持当前 seed 集合与训练协议，新增以下实验组：
  - `Exp16_ProteinBERT_PSSM710`（310 + PSSM-composition）
  - `Exp17_ProteinBERT_PSSM1110`（310 + PSSM-composition + DPC-PSSM）
  - （可选）`Exp18_ProteinBERT_PSSM510`（仅 PSSM-AC + PSSM-comp 或仅 DPC-PSSM）用于定位贡献来源
- 统一输出 `AUC/AUPRC/F1/MCC/Brier/ECE`，按 `AUPRC mean` 排序。
- 交付：`exp_results.csv` 与 `exp_summary.csv`（含 Exp15~Exp17/18）。

### 阶段D：结论判定与保留策略

- 判定规则：
  - 主指标：相对 `Exp15_PSSM310` 的 `AUPRC mean` 与 `MCC mean` 同时不下降；
  - 次指标：`MCC` 与 `ECE` 至少一项改善。
- 若 `Exp16/17` 优于 `Exp15`：将其纳入后续主线并继续轻量超参优化。
- 若无提升：保留 `Exp15` 或 `RPSSM110` 为主线，终止大规模加维特征尝试。

## 产出清单

- 主实验 notebook（新增 Exp16/Exp17，沿用当前基线）。
- 多缓存文件（310/710/1110）与特征维度说明。
- 一页对照结论：`Baseline vs RPSSM110 vs PSSM310 vs PSSM710 vs PSSM1110`。

## 执行顺序建议

1. 阶段A（0.5天）
2. 阶段B（1-2天）
3. 阶段C（1-2天）
4. 阶段D（0.5天）

