# LM Baseline 模型参数量对比

本目录下四种基线模型的参数量统计与来源说明。

---

## 一、参数量汇总表

| 模型 | 脚本 | Hugging Face / 来源 | 参数量 | 相对 ProteinBERT |
|------|------|---------------------|--------|------------------|
| **ProteinBERT** | `run_proteinbert_baseline.py` | Brandes (Keras, 本地 dump) | **15,981,321** (15.98M) | 1× |
| **ProtT5** | `run_prott5_baseline.py` | `Rostlab/prot_t5_xl_uniref50` | 1,200,000,000 (1,200M) | ~75× |
| **ESM2** | `run_esm2_baseline.py` | `facebook/esm2_t30_150M_UR50D` | 150,000,000 (150M) | ~9× |
| **Ankh** | `run_ankh_baseline.py` | `ElnaggarLab/ankh-large` (优先) | 1,900,000,000 (1,900M) | ~119× |
| | | `ElnaggarLab/ankh-base` (备选) | 736,000,000 (736M) | ~46× |

---

## 二、各模型参数量来源

### 1. ProteinBERT（本项目实测）

- **统计方式**：`load_pretrained_model()` → `create_model(seq_len=512)` → `model.count_params()`
- **环境**：`lm-hf`，TensorFlow/Keras
- **dump 路径**：`~/proteinbert_models/default.pkl` 或 `PROJECT_DIR/proteinbert_models/default.pkl`
- **结果**：**15,981,321** (15.98M)

### 2. ProtT5

- **模型**：`Rostlab/prot_t5_xl_uniref50`
- **来源**：Nature Communications 2024, Table 1（Fine-tuning protein language models boosts predictions across diverse tasks）
- **参数量**：1,200 M (encoder)
- **架构**：T5 Encoder-Decoder，24 层，d_model=1024

### 3. ESM2

- **模型**：`facebook/esm2_t30_150M_UR50D`
- **来源**：模型名含 "150M"；Nature Communications 2024, Table 1
- **参数量**：150 M
- **架构**：Encoder-only，30 层，hidden_size=640

### 4. Ankh

- **模型**：`ElnaggarLab/ankh-large`（脚本优先加载），备选 `ankh-base`
- **来源**：Nature Communications 2024, Table 1
- **参数量**：Ankh Large 1,900 M；Ankh Base 736 M
- **架构**：T5 Encoder-Decoder，48 层

---

## 三、统计脚本（可选自测）

在 `lm-hf` 环境中运行以下命令可重新统计 ProteinBERT 参数量：

```bash
cd /home/nemophila/projects/protein_bert && conda run -n lm-hf python -c "
from proteinbert import load_pretrained_model
pmg, _ = load_pretrained_model(download_model_dump_if_not_exists=False, validate_downloading=False)
model = pmg.create_model(seq_len=512, compile=False, init_weights=True)
print('ProteinBERT backbone:', model.count_params(), 'params')
"
```

统计 ProtT5 / ESM2 / Ankh（需联网下载模型）：

```python
import torch
from transformers import T5EncoderModel, AutoModel

# ProtT5
m = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
print('ProtT5:', sum(p.numel() for p in m.parameters()))

# ESM2
m = AutoModel.from_pretrained('facebook/esm2_t30_150M_UR50D')
print('ESM2:', sum(p.numel() for p in m.parameters()))

# Ankh
m = T5EncoderModel.from_pretrained('ElnaggarLab/ankh-large')
print('Ankh-large:', sum(p.numel() for p in m.parameters()))
```

---

## 四、文献引用

- **Nature Communications 2024**：Fine-tuning protein language models boosts predictions across diverse tasks. *Nat Commun* **15**, 51844 (2024). Table 1. https://doi.org/10.1038/s41467-024-51844-2
