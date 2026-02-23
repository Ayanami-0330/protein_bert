#!/usr/bin/env python3
"""Add PSSM-only and ProteinBERT-only sections to train_and_validate.ipynb"""
import json

with open("train_and_validate.ipynb", "r") as f:
    nb = json.load(f)

# 1. Update cell 1: add LogisticRegression and proteinbert imports
cell1 = nb["cells"][1]
src = cell1["source"]
if isinstance(src, str):
    src = [src]
new_src = []
for i, line in enumerate(src):
    new_src.append(line)
    if "from sklearn.preprocessing import StandardScaler" in line and i + 1 < len(src):
        if "LogisticRegression" not in (src[i+1] if isinstance(src[i+1], str) else ""):
            new_src.append("from sklearn.linear_model import LogisticRegression\n")
    if "    attach_pssm_features,\n" in line and "OutputSpec" not in "".join(src):
        new_src.append("    OutputSpec,\n")
        new_src.append("    OutputType,\n")
        new_src.append("    FinetuningModelGenerator,\n")
        new_src.append("    finetune,\n")
    if "from proteinbert.pssm_fusion import _build_late_fusion_model" in line:
        if "get_model_with_hidden" not in "".join(src):
            new_src.append("from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs\n")
nb["cells"][1]["source"] = new_src

# 2. New cells to insert after cell 7 (index 7, the "新发现 Acr 泛化验证" code cell)
seq_len_val = 512  # same as cfg.seq_len

new_cells = [
    # --- 仅用 1110 维 PSSM ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 仅用 1110 维 PSSM 特征\n",
            "\n",
            "与 demo 中 Ablation_RPSSM_1110 逻辑一致：仅用 1110 维 PSSM 训练 LogisticRegression，在验证集上找最优阈值，输出测试集 AUC/ACC/AUPRC、混淆矩阵，以及对 11 个新蛋白的预测。"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 与 Fusion 使用相同划分 (rng_train, rng_valid 已在上面定义)\n",
            "x_tr = rng_train[feature_cols].to_numpy(dtype=np.float32)\n",
            "x_va = rng_valid[feature_cols].to_numpy(dtype=np.float32)\n",
            "x_te = test_df[feature_cols].to_numpy(dtype=np.float32)\n",
            "scaler_pssm = StandardScaler()\n",
            "x_tr = scaler_pssm.fit_transform(x_tr)\n",
            "x_va = scaler_pssm.transform(x_va)\n",
            "x_te = scaler_pssm.transform(x_te)\n",
            "y_va = rng_valid[\"label\"].astype(int).to_numpy()\n",
            "\n",
            "clf_pssm = LogisticRegression(max_iter=2000, solver=\"liblinear\", random_state=SEED)\n",
            "clf_pssm.fit(x_tr, rng_train[\"label\"].astype(int).to_numpy())\n",
            "va_prob_pssm = clf_pssm.predict_proba(x_va)[:, 1]\n",
            "best_thr_pssm = find_best_threshold(y_va, va_prob_pssm)\n",
            "y_prob_pssm = clf_pssm.predict_proba(x_te)[:, 1]\n",
            "y_pred_pssm = (y_prob_pssm >= best_thr_pssm).astype(int)\n",
            "\n",
            "auc_pssm = roc_auc_score(y_test, y_prob_pssm)\n",
            "acc_pssm = accuracy_score(y_test, y_pred_pssm)\n",
            "auprc_pssm = average_precision_score(y_test, y_prob_pssm)\n",
            "print(\"PSSM-only (1110 dim, seed=22) — Test set metrics\")\n",
            "print(f\"  AUC:   {auc_pssm:.4f}\")\n",
            "print(f\"  ACC:   {acc_pssm:.4f}\")\n",
            "print(f\"  AUPRC: {auprc_pssm:.4f}\")\n",
            "cm_pssm = confusion_matrix(y_test, y_pred_pssm)\n",
            "cm_pssm_df = pd.DataFrame(cm_pssm, index=[\"Non-Acr\", \"Acr\"], columns=[\"Non-Acr\", \"Acr\"])\n",
            "cm_pssm_df.index.name = \"True\"\n",
            "cm_pssm_df.columns.name = \"Predicted\"\n",
            "print(\"\\nConfusion matrix\")\n",
            "display(cm_pssm_df)"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# PSSM-only：11 个新蛋白预测\n",
            "if not os.path.exists(MANIFEST_PATH):\n",
            "    print(\"No manifest. Run: python scripts/prepare_fasta_manifest.py\")\n",
            "elif not os.path.exists(NEW_CACHE) and not os.path.exists(NEW_CACHE_CSV):\n",
            "    print(\"No new Acr PSSM cache. Run: bash scripts/run_pssm_pipeline.sh\")\n",
            "else:\n",
            "    new_cache_path = NEW_CACHE if os.path.exists(NEW_CACHE) else NEW_CACHE_CSV\n",
            "    new_feat_df, _ = load_feature_cache(new_cache_path)\n",
            "    manifest_df = pd.read_csv(MANIFEST_PATH)\n",
            "    new_df_pssm = manifest_df.merge(new_feat_df, on=\"sample_id\", how=\"inner\")\n",
            "    if new_df_pssm.empty:\n",
            "        print(\"No overlapping sample_id between manifest and feature cache.\")\n",
            "    else:\n",
            "        x_new_pssm = new_df_pssm[feature_cols].to_numpy(dtype=np.float32)\n",
            "        x_new_pssm = scaler_pssm.transform(x_new_pssm)\n",
            "        prob_new_pssm = clf_pssm.predict_proba(x_new_pssm)[:, 1]\n",
            "        pred_new_pssm = (prob_new_pssm >= best_thr_pssm).astype(int)\n",
            "        new_df_pssm = new_df_pssm.copy()\n",
            "        new_df_pssm[\"score\"] = prob_new_pssm\n",
            "        new_df_pssm[\"predicted\"] = pred_new_pssm\n",
            "        display_cols = [\"name\", \"sample_id\", \"score\", \"predicted\"]\n",
            "        if \"name\" not in new_df_pssm.columns:\n",
            "            display_cols = [c for c in display_cols if c in new_df_pssm.columns]\n",
            "        print(\"PSSM-only — New Acr prediction (true label = Acr for all)\")\n",
            "        display(new_df_pssm[display_cols])\n",
            "        recall_pssm = pred_new_pssm.mean()\n",
            "        n = len(pred_new_pssm)\n",
            "        print(f\"\\nCorrectly predicted as Acr: {pred_new_pssm.sum()}/{n} — Recall = {recall_pssm:.4f}\")"
        ],
        "outputs": [],
        "execution_count": None
    },
    # --- 仅用 finetune 过的 ProteinBERT ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 仅用 finetune 过的 ProteinBERT\n",
            "\n",
            "与 demo 中 Stage1 逻辑一致：仅用序列 finetune ProteinBERT（无 PSSM），在验证集上找最优阈值，输出测试集 AUC/ACC/AUPRC、混淆矩阵，以及对 11 个新蛋白的预测。"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "output_type = OutputType(False, \"binary\")\n",
            "output_spec = OutputSpec(output_type, [0, 1])\n",
            "mg_bert = FinetuningModelGenerator(\n",
            "    pmg,\n",
            "    output_spec=output_spec,\n",
            "    pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,\n",
            "    dropout_rate=0.3,\n",
            "    head_type=\"classification\",\n",
            "    loss_type=\"bce\",\n",
            "    lr=1e-4,\n",
            ")\n",
            "finetune(\n",
            "    mg_bert,\n",
            "    enc,\n",
            "    output_spec,\n",
            "    rng_train[\"seq\"].tolist(),\n",
            "    rng_train[\"label\"].tolist(),\n",
            "    rng_valid[\"seq\"].tolist(),\n",
            "    rng_valid[\"label\"].tolist(),\n",
            "    seq_len=512,\n",
            "    batch_size=8,\n",
            "    max_epochs_per_stage=8,\n",
            "    begin_with_frozen_pretrained_layers=True,\n",
            "    n_final_epochs=0,\n",
            ")\n",
            "model_bert = mg_bert.create_model(512)\n",
            "X_valid_bert = enc.encode_X(rng_valid[\"seq\"].tolist(), 512)\n",
            "valid_prob_bert = model_bert.predict(X_valid_bert, batch_size=8, verbose=0).reshape(-1)\n",
            "best_thr_bert = find_best_threshold(y_valid, valid_prob_bert)\n",
            "X_test_bert = enc.encode_X(test_df[\"seq\"].tolist(), 512)\n",
            "y_prob_bert = model_bert.predict(X_test_bert, batch_size=8, verbose=0).reshape(-1)\n",
            "y_pred_bert = (y_prob_bert >= best_thr_bert).astype(int)\n",
            "\n",
            "auc_bert = roc_auc_score(y_test, y_prob_bert)\n",
            "acc_bert = accuracy_score(y_test, y_pred_bert)\n",
            "auprc_bert = average_precision_score(y_test, y_prob_bert)\n",
            "print(\"ProteinBERT-only (finetune, seed=22) — Test set metrics\")\n",
            "print(f\"  AUC:   {auc_bert:.4f}\")\n",
            "print(f\"  ACC:   {acc_bert:.4f}\")\n",
            "print(f\"  AUPRC: {auprc_bert:.4f}\")\n",
            "cm_bert = confusion_matrix(y_test, y_pred_bert)\n",
            "cm_bert_df = pd.DataFrame(cm_bert, index=[\"Non-Acr\", \"Acr\"], columns=[\"Non-Acr\", \"Acr\"])\n",
            "cm_bert_df.index.name = \"True\"\n",
            "cm_bert_df.columns.name = \"Predicted\"\n",
            "print(\"\\nConfusion matrix\")\n",
            "display(cm_bert_df)"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ProteinBERT-only：11 个新蛋白预测（仅需序列，与上面 NEW_WORK / MANIFEST 一致）\n",
            "if not os.path.exists(MANIFEST_PATH):\n",
            "    print(\"No manifest. Run: python scripts/prepare_fasta_manifest.py\")\n",
            "elif not os.path.exists(NEW_CACHE) and not os.path.exists(NEW_CACHE_CSV):\n",
            "    print(\"No new Acr PSSM cache. Run: bash scripts/run_pssm_pipeline.sh\")\n",
            "else:\n",
            "    new_cache_path = NEW_CACHE if os.path.exists(NEW_CACHE) else NEW_CACHE_CSV\n",
            "    new_feat_df, _ = load_feature_cache(new_cache_path)\n",
            "    manifest_df = pd.read_csv(MANIFEST_PATH)\n",
            "    new_df_bert = manifest_df.merge(new_feat_df, on=\"sample_id\", how=\"inner\")\n",
            "    if new_df_bert.empty:\n",
            "        print(\"No overlapping sample_id between manifest and feature cache.\")\n",
            "    else:\n",
            "        max_aa = 512 - 2\n",
            "        seqs_new_bert = [\n",
            "            str(s)[:max_aa] if len(str(s)) > max_aa else str(s)\n",
            "            for s in new_df_bert[\"seq\"].tolist()\n",
            "        ]\n",
            "        X_new_bert = enc.encode_X(seqs_new_bert, 512)\n",
            "        prob_new_bert = model_bert.predict(X_new_bert, batch_size=8, verbose=0).reshape(-1)\n",
            "        pred_new_bert = (prob_new_bert >= best_thr_bert).astype(int)\n",
            "        new_df_bert = new_df_bert.copy()\n",
            "        new_df_bert[\"score\"] = prob_new_bert\n",
            "        new_df_bert[\"predicted\"] = pred_new_bert\n",
            "        display_cols = [\"name\", \"sample_id\", \"score\", \"predicted\"]\n",
            "        if \"name\" not in new_df_bert.columns:\n",
            "            display_cols = [c for c in display_cols if c in new_df_bert.columns]\n",
            "        print(\"ProteinBERT-only — New Acr prediction (true label = Acr for all)\")\n",
            "        display(new_df_bert[display_cols])\n",
            "        recall_bert = pred_new_bert.mean()\n",
            "        n = len(pred_new_bert)\n",
            "        print(f\"\\nCorrectly predicted as Acr: {pred_new_bert.sum()}/{n} — Recall = {recall_bert:.4f}\")"
        ],
        "outputs": [],
        "execution_count": None
    },
]

# 在“11 新蛋白”的两个 code cell 开头加上路径定义，便于独立运行
for c in new_cells:
    if c["cell_type"] == "code":
        src_text = "".join(c.get("source", []))
        if "11 个新蛋白" in src_text or "11 个新蛋白预测" in src_text:
            s = c["source"]
            prefix = [
                "NEW_WORK = f\"{PROJECT_ROOT}/pssm_work\"\n",
                "NEW_CACHE = f\"{NEW_WORK}/features/pssm_features_1110.parquet\"\n",
                "NEW_CACHE_CSV = f\"{NEW_WORK}/features/pssm_features_1110.csv\"\n",
                "MANIFEST_PATH = f\"{NEW_WORK}/sample_manifest.csv\"\n",
                "\n",
            ]
            c["source"] = prefix + s

for c in new_cells:
    nb["cells"].append(c)

with open("train_and_validate.ipynb", "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Patched: added imports and 6 new cells (PSSM-only + ProteinBERT-only).")
