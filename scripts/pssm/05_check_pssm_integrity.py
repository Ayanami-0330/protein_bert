#!/usr/bin/env python3
"""检查所有 .pssm 与 manifest 中序列长度是否一致（序列中的空格不参与计数）。"""
import csv
import sys
from pathlib import Path


def count_pssm_rows(pssm_path: Path) -> int:
    count = 0
    with open(pssm_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 22 and parts[0].isdigit():
                count += 1
    return count


def main():
    manifest_path = Path(__file__).resolve().parents[1] / "pssm_work" / "sample_manifest.csv"
    if len(sys.argv) > 1:
        manifest_path = Path(sys.argv[1])
    if not manifest_path.exists():
        print("Manifest not found:", manifest_path)
        sys.exit(1)

    issues = []
    with open(manifest_path) as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["sample_id"]
            seq = row.get("seq", "")
            pssm_path = Path(row.get("pssm_path", ""))
            if not pssm_path.exists():
                continue
            size = pssm_path.stat().st_size
            if size == 0:
                issues.append((sid, "空文件"))
                continue
            # 残基数：去掉空格后的长度（与 PSI-BLAST 输出行数一致）
            residue_len = len(seq.replace(" ", "").strip())
            if residue_len == 0:
                continue
            rows = count_pssm_rows(pssm_path)
            if rows < 0:
                issues.append((sid, "无法解析"))
            elif rows != residue_len:
                issues.append((sid, f"行数不匹配 PSSM={rows} 残基数={residue_len}"))

    if not issues:
        print("OK: 所有已存在的 .pssm 与序列残基数一致。")
        return
    print("发现异常:", len(issues))
    for sid, msg in issues[:50]:
        print(" ", sid, msg)
    if len(issues) > 50:
        print(" ... 共", len(issues), "个")
    sys.exit(1)


if __name__ == "__main__":
    main()
