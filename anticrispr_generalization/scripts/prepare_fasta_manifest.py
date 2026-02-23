#!/usr/bin/env python3
"""
从 data/new_acr_sequences.csv（列 name, seq, label）生成 pssm_work 下的 FASTA 与 sample_manifest.csv。
与 protein_bert 的 00_prepare_fasta 输出格式一致，便于后续用 protein_bert 的 01/02/03/04 脚本。
"""
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WORK_ROOT = os.path.join(PROJECT_ROOT, "pssm_work")
CSV_PATH = os.path.join(DATA_DIR, "new_acr_sequences.csv")
SPLIT_NAME = "new_acr"


def main():
    if not os.path.isfile(CSV_PATH):
        print(f"Missing {CSV_PATH}. Run scripts/fetch_sequences.py first and fill missing sequences.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if "name" not in df.columns or "seq" not in df.columns:
        print("CSV must have columns: name, seq, label", file=sys.stderr)
        sys.exit(1)
    df = df.dropna(subset=["seq"]).drop_duplicates(subset=["name"]).reset_index(drop=True)
    df["seq"] = df["seq"].astype(str).str.strip()
    # 去掉空序列或过短
    df = df[df["seq"].str.len() >= 10].reset_index(drop=True)
    if df.empty:
        print("No valid sequences (non-empty, len>=10).", file=sys.stderr)
        sys.exit(1)

    # sample_id: 用 name 简化，避免空格等
    sample_ids = []
    for n in df["name"]:
        sid = str(n).replace(" ", "_").replace("/", "_")
        if not sid.replace("_", "").isalnum():
            sid = "".join(c if c.isalnum() or c == "_" else "_" for c in sid)
        sample_ids.append(f"{SPLIT_NAME}_{sid}")
    df["sample_id"] = sample_ids

    fasta_dir = os.path.join(WORK_ROOT, "fasta")
    pssm_dir = os.path.join(WORK_ROOT, "pssm")
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(pssm_dir, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        seq = row["seq"]
        fasta_path = os.path.join(fasta_dir, f"{sid}.fa")
        pssm_path = os.path.join(pssm_dir, f"{sid}.pssm")
        with open(fasta_path, "w", encoding="utf-8") as f:
            f.write(f">{sid}\n{seq}\n")
        rows.append({
            "sample_id": sid,
            "split": SPLIT_NAME,
            "label": int(row.get("label", 1)),
            "seq": seq,
            "name": row["name"],
            "fasta_path": fasta_path,
            "pssm_path": pssm_path,
        })

    manifest_df = pd.DataFrame(rows)
    manifest_path = os.path.join(WORK_ROOT, "sample_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Wrote manifest: {manifest_path} ({len(manifest_df)} samples)")
    print(f"FASTA dir: {fasta_dir}")


if __name__ == "__main__":
    main()
