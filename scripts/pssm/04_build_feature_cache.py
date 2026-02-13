#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature cache from sample npy files.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--work-root", required=True)
    args = parser.parse_args()

    work_root = Path(args.work_root)
    features_dir = work_root / "features"
    cache_parquet = features_dir / "pssm_features_310.parquet"
    cache_npy = features_dir / "pssm_features_310.npy"
    cache_ids = features_dir / "pssm_feature_ids.csv"

    manifest = pd.read_csv(args.manifest_csv)

    mat = []
    ids = []
    for row in manifest.itertuples(index=False):
        feat_path = features_dir / f"{row.sample_id}.npy"
        if not feat_path.exists():
            continue
        feat = np.load(feat_path)
        if feat.shape[0] != 310:
            continue
        ids.append(row.sample_id)
        mat.append(feat.astype(np.float32))

    if len(mat) == 0:
        raise RuntimeError("No valid feature vectors found.")

    mat = np.stack(mat, axis=0)
    cols = [f"feat_{i:03d}" for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "sample_id", ids)

    # Preferred output format.
    try:
        df.to_parquet(cache_parquet, index=False)
        print(f"Wrote parquet: {cache_parquet}")
    except Exception as exc:
        fallback_csv = str(cache_parquet).replace(".parquet", ".csv")
        df.to_csv(fallback_csv, index=False)
        print(f"Parquet unavailable ({exc}); wrote csv fallback: {fallback_csv}")

    np.save(cache_npy, mat)
    pd.DataFrame({"sample_id": ids}).to_csv(cache_ids, index=False)
    print(f"Wrote npy: {cache_npy}")
    print(f"Wrote ids: {cache_ids}")
    print(f"Shape: {mat.shape}")


if __name__ == "__main__":
    main()

