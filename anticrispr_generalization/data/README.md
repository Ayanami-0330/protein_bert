# 数据说明

- **new_acr_list.txt**：32 个新发现 Acr 名称列表。
- **new_acr_sequences.csv**：列 `name`, `seq`, `label`。`label` 固定为 1（均为 Acr）。
  - 运行 `python scripts/fetch_sequences.py` 会从 UniProt 尝试拉取序列并写回本文件；部分条目可能仍为空。
  - 缺失序列可从以下途径补齐后手动填入 `seq` 列：
    - [Anti-CRISPRdb](http://guolab.whu.edu.cn/anti-CRISPRdb/)
    - [AcrDB](https://bcb.unl.edu/AcrDB/)
    - 各文献正文或 Supplementary 中的 FASTA/序列表（见 `protein_bert/docs/new_anticrispr_proteins_2023_2025.md` 中的文献链接）
