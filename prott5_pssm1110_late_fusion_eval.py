#!/usr/bin/env python3
from pathlib import Path
import os
import re
import json
import copy
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from huggingface_hub import snapshot_download

from proteinbert.pssm_fusion import (
    load_anticrispr_with_ids,
    load_feature_cache,
    attach_pssm_features,
    evaluate_binary,
    find_best_threshold,
)

SEED = 22
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
elif DEVICE == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

PROJECT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = PROJECT_DIR / "anticrispr_benchmarks"
WORK_ROOT = Path(os.environ.get("PSSM_WORK_ROOT", "/home/nemophila/data/pssm_work"))
FEAT_DIR = WORK_ROOT / "features"
OUT_DIR = PROJECT_DIR / "cache" / "prott5_pssm1110_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    seq_len: int = 512
    batch_size: int = 8
    frozen_epochs: int = 6
    unfrozen_epochs: int = 12
    frozen_lr: float = 1e-4
    unfrozen_lr: float = 2e-5
    patience: int = 4
    lr_plateau_patience: int = 1
    lr_plateau_factor: float = 0.25
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0
    global_dropout: float = 0.3
    pssm_dropout: float = 0.3
    fusion_dropout: float = 0.3
    pssm_hidden_dim: int = 128
    global_hidden_dim: int = 128
    global_bottleneck_dim: int = 64
    fusion_hidden_dim: int = 128
    use_all_hidden_states: bool = True


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_features(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_seq_for_t5(seq: str) -> str:
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYUZOBX]", "X", str(seq).upper())
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


def pick_pssm_cache() -> Path:
    parquet_path = FEAT_DIR / "pssm_features_1110.parquet"
    csv_path = FEAT_DIR / "pssm_features_1110.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"PSSM cache not found: {parquet_path} or {csv_path}")


def load_data_and_pssm() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_base_df, test_base_df = load_anticrispr_with_ids(str(BENCHMARKS_DIR), benchmark_name="anticrispr_binary")
    cache_path = pick_pssm_cache()
    feature_df, feature_cols = load_feature_cache(str(cache_path))
    train_df = attach_pssm_features(train_base_df, feature_df, feature_cols)
    test_df = attach_pssm_features(test_base_df, feature_df, feature_cols)
    return train_df, test_df, list(feature_cols)


def prepare_local_prott5_dir() -> str:
    return snapshot_download(
        repo_id=MODEL_NAME,
        allow_patterns=[
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "pytorch_model.bin",
        ],
        ignore_patterns=["*.safetensors"],
    )


class FusionDataset(Dataset):
    def __init__(self, seqs: List[str], pssm: np.ndarray, labels: np.ndarray):
        self.seqs = seqs
        self.pssm = torch.from_numpy(pssm.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.pssm[idx], self.labels[idx]


def make_collate(tokenizer, max_length: int):
    def _collate(batch):
        seqs, pssm, labels = zip(*batch)
        toks = tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return toks["input_ids"], toks["attention_mask"], torch.stack(pssm), torch.stack(labels)

    return _collate


class ProtT5PSSMLateFusion(nn.Module):
    def __init__(
        self,
        encoder: T5EncoderModel,
        pssm_dim: int,
        global_hidden_dim: int,
        pssm_hidden_dim: int,
        global_bottleneck_dim: int,
        fusion_hidden_dim: int,
        global_dropout: float,
        pssm_dropout: float,
        fusion_dropout: float,
        use_all_hidden_states: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_all_hidden_states = bool(use_all_hidden_states)
        emb_dim = int(self.encoder.config.d_model)

        # Use all hidden layers via a learnable scalar mix, then combine with the last layer
        # to mimic hidden/global multi-source fusion in a controlled way.
        self.layer_mix_logits: Optional[nn.Parameter]
        if self.use_all_hidden_states:
            n_encoder_layers = int(getattr(self.encoder.config, "num_layers", 0))
            if n_encoder_layers <= 0:
                raise ValueError("Unable to infer encoder layer count for hidden-state aggregation.")
            # +1 includes embeddings output, matching HF hidden_states convention.
            self.layer_mix_logits = nn.Parameter(torch.zeros(n_encoder_layers + 1, dtype=torch.float32))
            emb_input_dim = emb_dim * 2
        else:
            self.layer_mix_logits = None
            emb_input_dim = emb_dim

        self.emb_branch = nn.Sequential(
            nn.LayerNorm(emb_input_dim),
            nn.Linear(emb_input_dim, global_bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(global_dropout),
            nn.Linear(global_bottleneck_dim, global_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(global_hidden_dim),
        )

        self.pssm_branch = nn.Sequential(
            nn.LayerNorm(pssm_dim),
            nn.Linear(pssm_dim, pssm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(pssm_dropout),
            nn.LayerNorm(pssm_hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(global_hidden_dim + pssm_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    @staticmethod
    def masked_mean(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def encode_global(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.use_all_hidden_states,
            return_dict=True,
        )
        pooled_last = self.masked_mean(out.last_hidden_state, attention_mask)

        if not self.use_all_hidden_states:
            return pooled_last

        if out.hidden_states is None or self.layer_mix_logits is None:
            raise RuntimeError("Hidden states are required but unavailable for scalar-mix aggregation.")

        stacked = torch.stack(out.hidden_states, dim=0)
        mix_w = torch.softmax(self.layer_mix_logits, dim=0).to(stacked.dtype)
        mixed_hidden = (mix_w[:, None, None, None] * stacked).sum(dim=0)
        pooled_mixed = self.masked_mean(mixed_hidden, attention_mask)
        return torch.cat([pooled_last, pooled_mixed], dim=1)

    def forward(self, input_ids, attention_mask, pssm):
        pooled = self.encode_global(input_ids, attention_mask)
        pooled = pooled.to(self.emb_branch[0].weight.dtype)
        pssm = pssm.to(self.pssm_branch[0].weight.dtype)
        emb_feat = self.emb_branch(pooled)
        pssm_feat = self.pssm_branch(pssm)
        fused = torch.cat([emb_feat, pssm_feat], dim=1)
        logits = self.fusion(fused).squeeze(1)
        return logits


def set_encoder_trainable(model: ProtT5PSSMLateFusion, trainable: bool):
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def run_epoch(
    model: ProtT5PSSMLateFusion,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    train: bool,
    grad_clip_norm: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        model.train()
    else:
        model.eval()

    losses = []
    all_probs = []
    all_labels = []
    skipped_non_finite = 0

    for input_ids, attention_mask, pssm, labels in loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        pssm = pssm.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(train):
            logits = model(input_ids, attention_mask, pssm)
            labels = labels.to(logits.dtype)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                if train:
                    skipped_non_finite += 1
                    continue
                raise RuntimeError("Non-finite loss encountered during evaluation")

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())
        losses.append(float(loss.item()))

    probs = np.array(np.concatenate(all_probs), dtype=np.float32) if len(all_probs) > 0 else np.array([], dtype=np.float32)
    y = np.array(np.concatenate(all_labels), dtype=np.float32) if len(all_labels) > 0 else np.array([], dtype=np.float32)
    if skipped_non_finite > 0:
        print(f"skipped_non_finite_batches={skipped_non_finite}")
    return float(np.mean(losses)) if len(losses) else 0.0, y, probs


def train_stage(
    model: ProtT5PSSMLateFusion,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    grad_clip_norm: float,
    lr_plateau_patience: int,
    lr_plateau_factor: float,
    min_lr: float,
) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_plateau_factor,
        patience=lr_plateau_patience,
        min_lr=min_lr,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, train=True, grad_clip_norm=grad_clip_norm)
        va_loss, _, _ = run_epoch(model, valid_loader, optimizer, criterion, train=False, grad_clip_norm=None)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} lr={current_lr:.2e}")
        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("early stopping triggered")
                break

    model.load_state_dict(best_state)
    return best_val


def main() -> None:
    set_seed(SEED)
    cfg = TrainConfig()

    train_df, test_df, feature_cols = load_data_and_pssm()
    sub_train, sub_valid = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=SEED,
    )

    x_tr_pssm = sanitize_features(sub_train[feature_cols].to_numpy(dtype=np.float32))
    x_va_pssm = sanitize_features(sub_valid[feature_cols].to_numpy(dtype=np.float32))
    x_te_pssm = sanitize_features(test_df[feature_cols].to_numpy(dtype=np.float32))

    scaler = StandardScaler().fit(x_tr_pssm)
    x_tr_pssm = sanitize_features(scaler.transform(x_tr_pssm))
    x_va_pssm = sanitize_features(scaler.transform(x_va_pssm))
    x_te_pssm = sanitize_features(scaler.transform(x_te_pssm))

    y_train = sub_train["label"].astype(int).to_numpy()
    y_valid = sub_valid["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    seq_train = [normalize_seq_for_t5(str(s)) for s in sub_train["seq"].astype(str).tolist()]
    seq_valid = [normalize_seq_for_t5(str(s)) for s in sub_valid["seq"].astype(str).tolist()]
    seq_test = [normalize_seq_for_t5(str(s)) for s in test_df["seq"].astype(str).tolist()]

    if len(seq_train) != len(y_train) or len(seq_valid) != len(y_valid) or len(seq_test) != len(y_test):
        raise ValueError(
            f"Cardinality mismatch seq(train/valid/test)=({len(seq_train)},{len(seq_valid)},{len(seq_test)}) "
            f"vs y=({len(y_train)},{len(y_valid)},{len(y_test)})"
        )

    local_model_dir = prepare_local_prott5_dir()
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(
        local_model_dir,
        local_files_only=True,
        torch_dtype=DTYPE,
        use_safetensors=False,
    )
    encoder = encoder.to(DEVICE)

    model = ProtT5PSSMLateFusion(
        encoder=encoder,
        pssm_dim=x_tr_pssm.shape[1],
        global_hidden_dim=cfg.global_hidden_dim,
        pssm_hidden_dim=cfg.pssm_hidden_dim,
        global_bottleneck_dim=cfg.global_bottleneck_dim,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        global_dropout=cfg.global_dropout,
        pssm_dropout=cfg.pssm_dropout,
        fusion_dropout=cfg.fusion_dropout,
        use_all_hidden_states=cfg.use_all_hidden_states,
    ).to(DEVICE)

    collate_fn = make_collate(tokenizer, cfg.seq_len)
    train_loader = DataLoader(FusionDataset(seq_train, x_tr_pssm, y_train), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(FusionDataset(seq_valid, x_va_pssm, y_valid), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(FusionDataset(seq_test, x_te_pssm, y_test), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    print("\nStage 1: freeze pretrained encoder, train fusion head")
    set_encoder_trainable(model, trainable=False)
    stage1_best_val = train_stage(
        model,
        train_loader,
        valid_loader,
        epochs=cfg.frozen_epochs,
        lr=cfg.frozen_lr,
        patience=cfg.patience,
        grad_clip_norm=cfg.grad_clip_norm,
        lr_plateau_patience=cfg.lr_plateau_patience,
        lr_plateau_factor=cfg.lr_plateau_factor,
        min_lr=cfg.min_lr,
    )
    stage1_best_state = copy.deepcopy(model.state_dict())

    print("\nStage 2: unfreeze encoder, full fine-tuning")
    set_encoder_trainable(model, trainable=True)
    stage2_best_val = train_stage(
        model,
        train_loader,
        valid_loader,
        epochs=cfg.unfrozen_epochs,
        lr=cfg.unfrozen_lr,
        patience=cfg.patience,
        grad_clip_norm=cfg.grad_clip_norm,
        lr_plateau_patience=cfg.lr_plateau_patience,
        lr_plateau_factor=cfg.lr_plateau_factor,
        min_lr=cfg.min_lr,
    )

    # Keep global best across both stages to avoid stage-2 overfitting regressions.
    if stage2_best_val >= stage1_best_val:
        model.load_state_dict(stage1_best_state)
        print(f"restore stage1 best (val_loss={stage1_best_val:.4f}) over stage2 best (val_loss={stage2_best_val:.4f})")
    else:
        print(f"keep stage2 best (val_loss={stage2_best_val:.4f})")

    criterion = nn.BCEWithLogitsLoss()
    _, y_valid_eval, valid_prob = run_epoch(model, valid_loader, optimizer=None, criterion=criterion, train=False)
    _, y_test_eval, test_prob = run_epoch(model, test_loader, optimizer=None, criterion=criterion, train=False)

    best_thr: float = float(find_best_threshold(y_valid_eval.astype(int), valid_prob))
    metrics = evaluate_binary(y_test_eval.astype(int), test_prob, threshold=best_thr)

    y_pred_raw = test_prob >= best_thr
    if isinstance(y_pred_raw, np.ndarray):
        y_pred = y_pred_raw.astype(np.int32)
    else:
        y_pred = np.array([int(y_pred_raw)], dtype=np.int32)
    cm = confusion_matrix(y_test_eval.astype(int), y_pred)

    print("\nProtT5 + PSSM1110 (late fusion, frozen->unfrozen) — Test metrics")
    for k in ["AUC", "AUPRC", "F1", "MCC", "Brier", "ECE", "Threshold"]:
        print(f"{k}: {metrics[k]:.6f}")

    print("\nConfusion matrix (test)")
    print(cm)
    print("labels: rows=true [0,1], cols=pred [0,1]")

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(OUT_DIR / "confusion_matrix_test.csv")
    pd.DataFrame({"y_true": y_test_eval.astype(int), "y_prob": test_prob, "y_pred": y_pred}).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
