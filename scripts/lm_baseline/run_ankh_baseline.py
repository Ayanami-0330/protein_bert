#!/usr/bin/env python3
from pathlib import Path
import sys
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

import torch
from transformers import AutoTokenizer, T5EncoderModel

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from proteinbert.pssm_fusion import load_anticrispr_with_ids, evaluate_binary, find_best_threshold

SEED = 22
MODEL_KEY = "Ankh_freezed"
ANKH_MODEL_CANDIDATES = [
    "ElnaggarLab/ankh-large",
    "ElnaggarLab/ankh-base",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
_ANKH_CACHE: Dict[str, Any] = {"name": None, "tokenizer": None, "model": None}


@dataclass
class HeadTrainConfig:
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 40
    patience: int = 3


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)


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


def normalize_seq_for_lm(seq: str) -> str:
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYUZOBX]", "X", str(seq).upper())
    return re.sub(r"[UZOB]", "X", seq)


def to_t5_input(seq: str) -> str:
    return " ".join(list(normalize_seq_for_lm(seq)))


def masked_mean(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-6)
    return summed / denom


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


def load_ankh_components():
    if _ANKH_CACHE["model"] is not None:
        return _ANKH_CACHE["name"], _ANKH_CACHE["tokenizer"], _ANKH_CACHE["model"]

    last_error = None
    for candidate in ANKH_MODEL_CANDIDATES:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            try:
                model = T5EncoderModel.from_pretrained(candidate, torch_dtype=DTYPE, use_safetensors=True)
            except Exception:
                model = T5EncoderModel.from_pretrained(candidate, torch_dtype=DTYPE)
            model = model.to(DEVICE)
            model.eval()
            _ANKH_CACHE["name"] = candidate
            _ANKH_CACHE["tokenizer"] = tokenizer
            _ANKH_CACHE["model"] = model
            print(f"Using Ankh checkpoint: {candidate}")
            return candidate, tokenizer, model
        except Exception as exc:
            last_error = exc
            print(f"Skip Ankh checkpoint {candidate}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No available Ankh checkpoint found. Tried: "
        + ", ".join(ANKH_MODEL_CANDIDATES)
        + f". Last error: {last_error}"
    )


def encode_ankh_sentence(seqs, batch_size: int = 2, max_length: int = 1024):
    _, tokenizer, model = load_ankh_components()
    if tokenizer is None or model is None:
        raise RuntimeError("Ankh components are not initialized.")

    outputs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = [to_t5_input(s) for s in seqs[i : i + batch_size]]
            toks = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            out = model(**toks)
            pooled = masked_mean(out.last_hidden_state, toks["attention_mask"])
            outputs.append(pooled.detach().cpu().float().numpy())

    return sanitize_features(np.array(np.concatenate(outputs, axis=0), dtype=np.float32))


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
        raw[split_name] = load_or_compute(raw_path, lambda seqs=seqs: encode_ankh_sentence(seqs))
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

    print("DEVICE:", DEVICE)
    print("saved metrics:", metrics_path)
    print("updated results:", results_path)
    print(results_df)


if __name__ == "__main__":
    main()
