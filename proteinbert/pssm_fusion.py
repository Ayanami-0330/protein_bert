import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from .conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

@dataclass
class FusionTrainConfig:
    seq_len: int = 512
    batch_size: int = 8
    frozen_epochs: int = 6
    unfrozen_epochs: int = 12
    frozen_lr: float = 1e-4
    unfrozen_lr: float = 2e-5
    patience: int = 4
    pssm_dropout: float = 0.3
    global_dropout: float = 0.3
    pssm_hidden_dim: int = 128
    global_hidden_dim: int = 128
    global_bottleneck_dim: int = 64
    fusion_hidden_dim: int = 128
    use_hidden_global_concat: bool = True


def ensure_sample_ids(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = [f"{split_name}_{i:06d}" for i in range(len(out))]
    return out


def load_anticrispr_with_ids(benchmarks_dir: str, benchmark_name: str = "anticrispr_binary") -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(benchmarks_dir, f"{benchmark_name}.train.csv")
    test_path = os.path.join(benchmarks_dir, f"{benchmark_name}.test.csv")
    train_df = pd.read_csv(train_path).dropna().drop_duplicates().reset_index(drop=True)
    test_df = pd.read_csv(test_path).dropna().drop_duplicates().reset_index(drop=True)
    return ensure_sample_ids(train_df, "train"), ensure_sample_ids(test_df, "test")


def load_feature_cache(feature_cache_path: str) -> Tuple[pd.DataFrame, List[str]]:
    if feature_cache_path.endswith(".parquet"):
        feat_df = pd.read_parquet(feature_cache_path)
    elif feature_cache_path.endswith(".csv"):
        feat_df = pd.read_csv(feature_cache_path)
    else:
        raise ValueError("Feature cache must be parquet or csv.")

    if "sample_id" not in feat_df.columns:
        raise ValueError("feature cache missing sample_id column.")

    feature_cols = [c for c in feat_df.columns if c.startswith("feat_")]
    if len(feature_cols) == 0:
        raise ValueError("feature cache missing feat_* columns.")
    return feat_df, feature_cols


def attach_pssm_features(
    seq_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    out = seq_df.merge(feature_df[["sample_id", *feature_cols]], on="sample_id", how="left")
    out.loc[:, feature_cols] = out.loc[:, feature_cols].fillna(fill_value)
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = ids == b
        if np.any(m):
            conf = float(np.mean(y_prob[m]))
            acc = float(np.mean(y_true[m]))
            ece += (np.sum(m) / n) * abs(acc - conf)
    return float(ece)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_cls = (y_prob >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_cls)),
        "MCC": float(matthews_corrcoef(y_true, y_cls)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": expected_calibration_error(y_true, y_prob, n_bins=10),
        "Threshold": float(threshold),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: Optional[Iterable[float]] = None) -> float:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in grid:
        cur = f1_score(y_true, (y_prob >= thr).astype(int))
        if cur > best_f1:
            best_f1 = cur
            best_thr = float(thr)
    return best_thr


def _encode_x(input_encoder, seqs: Sequence[str], seq_len: int, pssm_feats: np.ndarray) -> List[np.ndarray]:
    tokenized, annotations = input_encoder.encode_X(seqs, seq_len)
    return [tokenized, annotations, pssm_feats.astype(np.float32)]


def _build_late_fusion_model(
    pretrained_model_generator,
    seq_len: int,
    pssm_dim: int,
    freeze_pretrained_layers: bool,
    cfg: FusionTrainConfig,
):
    base_model = pretrained_model_generator.create_model(seq_len, compile=False, init_weights=True)
    if cfg.use_hidden_global_concat:
        # Keep consistent with the baseline notebook by using concatenated hidden/global representation.
        base_model = get_model_with_hidden_layers_as_outputs(base_model)
    if freeze_pretrained_layers:
        for layer in base_model.layers:
            layer.trainable = False
    _, global_output = base_model.output

    pssm_input = keras.layers.Input(shape=(pssm_dim,), name="pssm_input")
    # Compress high-dim global representation first, then balance branch widths before fusion.
    global_branch = keras.layers.LayerNormalization(name="global_ln_in")(global_output)
    global_branch = keras.layers.Dense(cfg.global_bottleneck_dim, activation="relu", name="global_bottleneck")(global_branch)
    global_branch = keras.layers.Dropout(cfg.global_dropout, name="global_drop")(global_branch)
    global_branch = keras.layers.Dense(cfg.global_hidden_dim, activation="relu", name="global_dense")(global_branch)
    global_branch = keras.layers.LayerNormalization(name="global_ln_out")(global_branch)
    pssm_branch = keras.layers.LayerNormalization(name="pssm_ln")(pssm_input)
    pssm_branch = keras.layers.Dense(cfg.pssm_hidden_dim, activation="relu", name="pssm_dense")(pssm_branch)
    pssm_branch = keras.layers.Dropout(cfg.pssm_dropout, name="pssm_drop")(pssm_branch)
    pssm_branch = keras.layers.LayerNormalization(name="pssm_ln_out")(pssm_branch)

    fused = keras.layers.Concatenate(name="late_fusion")([global_branch, pssm_branch])
    fused = keras.layers.Dense(cfg.fusion_hidden_dim, activation="relu", name="fusion_dense")(fused)
    fused = keras.layers.Dropout(cfg.pssm_dropout, name="fusion_drop")(fused)
    out = keras.layers.Dense(1, activation="sigmoid", name="output")(fused)

    model = keras.models.Model(inputs=base_model.inputs + [pssm_input], outputs=out)
    return model


def run_finetune_with_pssm(
    pretrained_model_generator,
    input_encoder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    seed: int = 42,
    cfg: FusionTrainConfig = FusionTrainConfig(),
) -> Dict[str, float]:
    rng_train, rng_valid = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=seed
    )

    x_train = rng_train[feature_cols].to_numpy(dtype=np.float32)
    x_valid = rng_valid[feature_cols].to_numpy(dtype=np.float32)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    y_train = rng_train["label"].astype(int).to_numpy()
    y_valid = rng_valid["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    X_train = _encode_x(input_encoder, rng_train["seq"].tolist(), cfg.seq_len, x_train)
    X_valid = _encode_x(input_encoder, rng_valid["seq"].tolist(), cfg.seq_len, x_valid)
    X_test = _encode_x(input_encoder, test_df["seq"].tolist(), cfg.seq_len, x_test)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience, restore_best_weights=True
        )
    ]

    model = _build_late_fusion_model(
        pretrained_model_generator,
        seq_len=cfg.seq_len,
        pssm_dim=len(feature_cols),
        freeze_pretrained_layers=True,
        cfg=cfg,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.frozen_lr),
        loss="binary_crossentropy",
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=cfg.frozen_epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.unfrozen_lr),
        loss="binary_crossentropy",
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=cfg.unfrozen_epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    valid_prob = model.predict(X_valid, batch_size=cfg.batch_size, verbose=0).reshape(-1)
    thr = find_best_threshold(y_valid, valid_prob)
    test_prob = model.predict(X_test, batch_size=cfg.batch_size, verbose=0).reshape(-1)
    return evaluate_binary(y_test, test_prob, threshold=thr)

