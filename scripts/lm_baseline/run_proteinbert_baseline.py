#!/usr/bin/env python3
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from proteinbert.pssm_fusion import load_anticrispr_with_ids, evaluate_binary, find_best_threshold

SEED = 22
MODEL_KEY = "ProteinBERT_freezed"
SEQ_LEN = 512


@dataclass
class HeadTrainConfig:
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 40
    patience: int = 3


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_paths() -> Tuple[Path, Path]:
    benchmarks_dir = PROJECT_DIR / "anticrispr_benchmarks"
    cache_dir = PROJECT_DIR / "cache" / "lm_baseline" / MODEL_KEY
    cache_dir.mkdir(parents=True, exist_ok=True)
    return benchmarks_dir, cache_dir


def load_splits(benchmarks_dir: Path):
    train_base_df, test_df = load_anticrispr_with_ids(str(benchmarks_dir), benchmark_name="anticrispr_binary")
    train_df, valid_df = train_test_split(
        train_base_df,
        test_size=0.1,
        stratify=train_base_df["label"],
        random_state=SEED,
    )
    for df in (train_df, valid_df, test_df):
        df["label"] = df["label"].astype(int)
    return train_df, valid_df, test_df


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))


def load_or_compute(path: Path, builder):
    if path.exists():
        return np.array(np.load(path), dtype=np.float32)
    value = builder()
    save_npy(path, value)
    return value


def sanitize_features(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_pca_128(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray):
    x_train = sanitize_features(x_train)
    x_valid = sanitize_features(x_valid)
    x_test = sanitize_features(x_test)

    n_components = min(128, x_train.shape[1], x_train.shape[0])
    pca = PCA(n_components=n_components, random_state=SEED)
    tr = pca.fit_transform(x_train)
    va = pca.transform(x_valid)
    te = pca.transform(x_test)

    if n_components < 128:
        pad = 128 - n_components
        tr = np.pad(tr, ((0, 0), (0, pad)))
        va = np.pad(va, ((0, 0), (0, pad)))
        te = np.pad(te, ((0, 0), (0, pad)))

    return tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32)


def encode_proteinbert_global(seqs, seq_len: int = SEQ_LEN, batch_size: int = 8) -> np.ndarray:
    pretrained_model_generator, input_encoder = load_pretrained_model()
    base_model = pretrained_model_generator.create_model(seq_len, compile=False, init_weights=True)
    model = get_model_with_hidden_layers_as_outputs(base_model)

    encoded_x = input_encoder.encode_X(seqs, seq_len)
    _, global_repr = model.predict(encoded_x, batch_size=batch_size, verbose=0)
    return sanitize_features(global_repr)


def build_shared_head() -> keras.Model:
    x_in = keras.layers.Input(shape=(128,), name="lm_emb_128")
    x = keras.layers.LayerNormalization(name="ln")(x_in)
    x = keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs=x_in, outputs=out)


def train_and_evaluate_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: HeadTrainConfig = HeadTrainConfig(),
) -> Dict[str, float]:
    model = build_shared_head()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.lr), loss="binary_crossentropy")

    callbacks: Any = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    valid_prob = model.predict(x_valid, batch_size=cfg.batch_size, verbose=0).reshape(-1)
    best_thr = find_best_threshold(y_valid, valid_prob)

    test_prob = model.predict(x_test, batch_size=cfg.batch_size, verbose=0).reshape(-1)
    return evaluate_binary(y_test, test_prob, threshold=best_thr)


def upsert_results_row(results_path: Path, model_key: str, metrics: Dict[str, float]) -> pd.DataFrame:
    row: Dict[str, Any] = {"Model": model_key}
    row.update(metrics)

    if results_path.exists():
        df = pd.read_csv(results_path)
        df = df[df["Model"] != model_key]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    cols = ["Model", "AUC", "AUPRC", "F1", "MCC", "Brier", "ECE", "Threshold"]
    records = [{c: row[c] for c in cols} for _, row in df[cols].iterrows()]
    records = sorted(records, key=lambda row: (row["AUC"], row["AUPRC"], row["F1"]), reverse=True)
    df = pd.DataFrame(records, columns=cols).reset_index(drop=True)
    df.to_csv(results_path, index=False)
    return df


def main() -> None:
    set_seed(SEED)
    benchmarks_dir, cache_dir = get_paths()
    train_df, valid_df, test_df = load_splits(benchmarks_dir)

    split_seqs = {
        "train": train_df["seq"].astype(str).tolist(),
        "valid": valid_df["seq"].astype(str).tolist(),
        "test": test_df["seq"].astype(str).tolist(),
    }
    labels = {
        "train": train_df["label"].to_numpy(dtype=int),
        "valid": valid_df["label"].to_numpy(dtype=int),
        "test": test_df["label"].to_numpy(dtype=int),
    }

    raw = {}
    for split_name, seqs in split_seqs.items():
        raw_path = cache_dir / f"{split_name}_raw.npy"
        raw[split_name] = load_or_compute(raw_path, lambda seqs=seqs: encode_proteinbert_global(seqs))
        print(MODEL_KEY, split_name, raw[split_name].shape)

    if len(raw["train"]) != len(labels["train"]) or len(raw["valid"]) != len(labels["valid"]) or len(raw["test"]) != len(labels["test"]):
        raise ValueError(
            f"Cardinality mismatch raw(train/valid/test)=({len(raw['train'])},{len(raw['valid'])},{len(raw['test'])}) "
            f"vs y=({len(labels['train'])},{len(labels['valid'])},{len(labels['test'])})"
        )

    x_train_128, x_valid_128, x_test_128 = fit_pca_128(raw["train"], raw["valid"], raw["test"])
    save_npy(cache_dir / "train_128.npy", x_train_128)
    save_npy(cache_dir / "valid_128.npy", x_valid_128)
    save_npy(cache_dir / "test_128.npy", x_test_128)

    metrics = train_and_evaluate_head(
        x_train_128,
        labels["train"],
        x_valid_128,
        labels["valid"],
        x_test_128,
        labels["test"],
    )

    metrics_path = cache_dir / "metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)

    results_path = PROJECT_DIR / "cache" / "lm_baseline" / "lm_baseline_results.csv"
    results_df = upsert_results_row(results_path, MODEL_KEY, metrics)

    print("saved metrics:", metrics_path)
    print("updated results:", results_path)
    print(results_df)


if __name__ == "__main__":
    main()
