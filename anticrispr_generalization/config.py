# 路径与配置：所有工作均在 /home/nemophila/projects 下，不修改 protein_bert

import os

# 本项目根目录（必须为 /home/nemophila/projects/anticrispr_generalization）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# protein_bert 仓库路径（只读：加载数据、脚本、BLAST DB、预训练模型）
PROTEIN_BERT_ROOT = "/home/nemophila/projects/protein_bert"
BENCHMARKS_DIR = os.path.join(PROTEIN_BERT_ROOT, "anticrispr_benchmarks")
PROTEIN_BERT_PSSM_SCRIPTS = os.path.join(PROTEIN_BERT_ROOT, "scripts", "pssm")
BLAST_DB_PREFIX = os.path.join(PROTEIN_BERT_ROOT, "blast_db", "uniref50")
PROTEINBERT_MODELS_DIR = os.path.join(PROTEIN_BERT_ROOT, "proteinbert_models")

# 本项目数据与工作目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NEW_ACR_CSV = os.path.join(DATA_DIR, "new_acr_sequences.csv")
WORK_ROOT = os.path.join(PROJECT_ROOT, "pssm_work")
MANIFEST_CSV = os.path.join(WORK_ROOT, "sample_manifest.csv")
FEATURES_DIR = os.path.join(WORK_ROOT, "features")
PSSM_CACHE_1110 = os.path.join(FEATURES_DIR, "pssm_features_1110.parquet")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
SAVED_MODEL_DIR = os.path.join(OUTPUTS_DIR, "saved_model")
SAVED_SCALER_NPY = os.path.join(OUTPUTS_DIR, "scaler_fit.npy")

SEED = 22
VARIANT = "1110"
