#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


AA_COLUMNS = 20
LAG = 10

# 11 reduced AA groups for RPSSM-like compact profiling.
AA_GROUPS = [
    [0, 4],       # A,C
    [3, 6],       # D,E
    [13, 16],     # F,Y
    [7, 8],       # G,H
    [9, 10],      # I,K
    [11, 12],     # L,M
    [14, 15],     # N,P
    [1, 2],       # R,N
    [5, 17],      # Q,V
    [18, 19],     # W,Y-like tail group
    [0, 1, 2, 3], # mixed group for stability
]


def parse_pssm_ascii(pssm_path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    with pssm_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            # Typical PSSM body line starts with index + residue + 20 scores.
            if len(parts) < 22:
                continue
            if not parts[0].isdigit():
                continue
            try:
                vals = [float(x) for x in parts[2:22]]
            except ValueError:
                continue
            rows.append(vals)
    if len(rows) == 0:
        raise ValueError(f"no valid PSSM rows in {pssm_path}")
    return np.asarray(rows, dtype=np.float32)


def sigmoid_norm(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def autocov_features(x: np.ndarray, lag: int) -> np.ndarray:
    # x: [L, D]
    L, D = x.shape
    feats = []
    mean = x.mean(axis=0)
    centered = x - mean
    for d in range(D):
        col = centered[:, d]
        for k in range(1, lag + 1):
            if L <= k:
                feats.append(0.0)
            else:
                feats.append(float(np.mean(col[:-k] * col[k:])))
    return np.asarray(feats, dtype=np.float32)


def build_rpssm_110(pssm: np.ndarray, lag: int = LAG) -> np.ndarray:
    p = sigmoid_norm(pssm)
    reduced_cols = []
    for group in AA_GROUPS:
        valid = [idx for idx in group if idx < p.shape[1]]
        reduced_cols.append(p[:, valid].mean(axis=1))
    reduced = np.stack(reduced_cols, axis=1)  # [L, 11]
    feats = autocov_features(reduced, lag=lag)  # 11*10=110
    if feats.shape[0] != 110:
        raise ValueError(f"RPSSM dim mismatch: {feats.shape[0]}")
    return feats


def build_pssm_ac_200(pssm: np.ndarray, lag: int = LAG) -> np.ndarray:
    p = sigmoid_norm(pssm)
    feats = autocov_features(p[:, :AA_COLUMNS], lag=lag)  # 20*10=200
    if feats.shape[0] != 200:
        raise ValueError(f"PSSM-AC dim mismatch: {feats.shape[0]}")
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RPSSM(110)+PSSM-AC(200) features.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--lag", type=int, default=LAG)
    args = parser.parse_args()

    work_root = Path(args.work_root)
    feat_dir = work_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    status_path = work_root / "features" / "feature_status.csv"

    manifest = pd.read_csv(args.manifest_csv)
    rows = []
    ok_count = 0
    for row in manifest.itertuples(index=False):
        pssm_path = Path(row.pssm_path)
        out_npy = feat_dir / f"{row.sample_id}.npy"
        status = {"sample_id": row.sample_id, "pssm_ok": 0, "feature_ok": 0, "retry_count": 0}

        try:
            pssm = parse_pssm_ascii(pssm_path)
            rpssm = build_rpssm_110(pssm, lag=args.lag)
            pssmac = build_pssm_ac_200(pssm, lag=args.lag)
            feat = np.concatenate([rpssm, pssmac], axis=0).astype(np.float32)
            if feat.shape[0] != 310:
                raise ValueError(f"expected 310 dims, got {feat.shape[0]}")
            np.save(out_npy, feat)
            status["pssm_ok"] = 1
            status["feature_ok"] = 1
            ok_count += 1
        except Exception:
            pass

        rows.append(status)

    status_df = pd.DataFrame(rows)
    status_df.to_csv(status_path, index=False)
    print(f"Feature extraction done. ok={ok_count}/{len(status_df)}")
    print(f"Status table: {status_path}")


if __name__ == "__main__":
    main()

