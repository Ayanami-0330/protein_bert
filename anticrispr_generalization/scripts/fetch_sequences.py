#!/usr/bin/env python3
"""
从 UniProt / NCBI 精确获取 32 个新发现 Acr 的氨基酸序列，写出 data/new_acr_sequences.csv。
策略：严格查询（名称 + anti-CRISPR）、JSON 校验命中名称、序列去重（同一序列只赋给一个 name）。
列：name, seq, label。缺失留空，需从文献/补充材料或 Anti-CRISPRdb 手动补齐。
"""
import os
import re
import sys
import time
import urllib.parse

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LIST_FILE = os.path.join(DATA_DIR, "new_acr_list.txt")
OUT_CSV = os.path.join(DATA_DIR, "new_acr_sequences.csv")

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TIMEOUT = 25
# 降低 NCBI 429 概率：主循环间隔、调用 NCBI 前额外等待、429 后重试退避
DELAY_BETWEEN_NAMES = 1.2
DELAY_BEFORE_NCBI = 1.0
DELAY_NCBI_EFETCH = 0.5
NCBI_429_SLEEP = 10
NCBI_MAX_RETRIES = 2

# 已知可超长的 Acr，不因长度过滤
KNOWN_LONG_ACR = {"AcrVA5", "AcrVIB1", "AcrIIA1"}


def _normalize_sequence(seq: str) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(c.upper() for c in (seq or "") if c.upper() in allowed)


def _entry_name_matches(entry: dict, want_name: str) -> bool:
    """检查 UniProt 条目的 protein/gene 名称是否包含目标 Acr 名（不区分大小写）。"""
    want = (want_name or "").strip()
    if not want:
        return False
    want_upper = want.upper()
    # proteinDescription.recommendedName.fullName.value
    try:
        rec = (entry.get("proteinDescription") or {}).get("recommendedName") or {}
        full = (rec.get("fullName") or {}).get("value") or ""
        if want_upper in full.upper():
            return True
    except Exception:
        pass
    for alt in (entry.get("proteinDescription") or {}).get("alternativeNames") or []:
        full = (alt.get("fullName") or {}).get("value") or ""
        if want_upper in full.upper():
            return True
    # genes[].geneName.value
    for g in entry.get("genes") or []:
        gn = (g.get("geneName") or {}).get("value") or ""
        if want_upper in gn.upper():
            return True
    return False


def _get_sequence_from_entry(entry: dict) -> str:
    seq = (entry.get("sequence") or {}).get("value") or ""
    return _normalize_sequence(seq)


def _uniprot_search_strict(name: str, used_seqs: set) -> str:
    """
    严格查询：名称 + anti-CRISPR，返回 JSON，只接受 protein/gene 名称含 name 的条目，
    且序列未在 used_seqs 中（避免多 name 共用同一序列）。
    """
    # 查询：(名称) AND (anti-crispr 或 anti CRISPR)
    query = f'({name}) AND (anti-crispr OR "anti CRISPR")'
    params = {"query": query, "format": "json", "size": 15}
    try:
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for entry in (data.get("results") or []):
            if not _entry_name_matches(entry, name):
                continue
            seq = _get_sequence_from_entry(entry)
            if len(seq) < 20:
                continue
            if name not in KNOWN_LONG_ACR and len(seq) > 600:
                continue
            if seq in used_seqs:
                continue
            return seq
        return ""
    except Exception as e:
        print(f"  [{name}] UniProt strict error: {e}", file=sys.stderr)
        return ""


def _uniprot_search_name_only(name: str, used_seqs: set) -> str:
    """
    放宽：仅按名称查，仍要求命中条目的 protein/gene 名称包含 name，且描述/标题含 anti 或 crispr。
    """
    query = f'({name})'
    params = {"query": query, "format": "json", "size": 20}
    try:
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for entry in (data.get("results") or []):
            if not _entry_name_matches(entry, name):
                continue
            seq = _get_sequence_from_entry(entry)
            if len(seq) < 20:
                continue
            if name not in KNOWN_LONG_ACR and len(seq) > 600:
                continue
            if seq in used_seqs:
                continue
            return seq
        return ""
    except Exception as e:
        print(f"  [{name}] UniProt name-only error: {e}", file=sys.stderr)
        return ""


def _parse_fasta_text(text: str) -> str:
    if not text or not text.strip().startswith(">"):
        return ""
    lines = text.strip().splitlines()
    seq_lines = [ln.strip() for ln in lines[1:] if ln and not ln.startswith(">")]
    return "".join(seq_lines) if seq_lines else ""


def _ncbi_protein_search(name: str, used_seqs: set) -> str:
    """NCBI protein：term 含 name + anti-CRISPR；遇 429 则等待后重试，避免请求过频。"""
    time.sleep(DELAY_BEFORE_NCBI)
    term = f"{name} anti-CRISPR"
    try:
        for _ in range(NCBI_MAX_RETRIES + 1):
            r = requests.get(
                NCBI_ESEARCH,
                params={"db": "protein", "term": term, "retmax": 10, "retmode": "json"},
                timeout=TIMEOUT,
            )
            if r.status_code == 429:
                time.sleep(NCBI_429_SLEEP)
                continue
            r.raise_for_status()
            break
        else:
            return ""  # 全部 429
        data = r.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        for fid in ids:
            time.sleep(DELAY_NCBI_EFETCH)
            for _ in range(NCBI_MAX_RETRIES + 1):
                r2 = requests.get(
                    NCBI_EFETCH,
                    params={"db": "protein", "id": fid, "rettype": "fasta", "retmode": "text"},
                    timeout=TIMEOUT,
                )
                if r2.status_code == 429:
                    time.sleep(NCBI_429_SLEEP)
                    continue
                r2.raise_for_status()
                break
            else:
                continue
            raw = r2.text
            lines = raw.strip().splitlines()
            if lines:
                title = lines[0].upper()
                if "ANTI" not in title or "CRISPR" not in title or name.upper() not in title:
                    continue
            seq = _normalize_sequence(_parse_fasta_text(raw))
            if len(seq) < 20:
                continue
            if name not in KNOWN_LONG_ACR and len(seq) > 600:
                continue
            if seq in used_seqs:
                continue
            return seq
        return ""
    except Exception as e:
        print(f"  [{name}] NCBI error: {e}", file=sys.stderr)
        return ""


def main():
    import csv

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LIST_FILE, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]

    # 若已有 CSV：只对“当前无序列”的条目请求 API，保留已有序列（含手动填写）
    existing = {}
    if os.path.isfile(OUT_CSV):
        with open(OUT_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n = (row.get("name") or "").strip()
                s = (row.get("seq") or "").strip()
                if n:
                    existing[n] = {"seq": s, "label": int(row.get("label", 1))}

    used_seqs = set()
    rows = []
    for i, name in enumerate(names):
        seq = ""
        if name in existing and existing[name]["seq"] and len(_normalize_sequence(existing[name]["seq"])) >= 20:
            seq = _normalize_sequence(existing[name]["seq"])
            used_seqs.add(seq)
            status = f"{len(seq)} aa (kept)"
        else:
            seq = _uniprot_search_strict(name, used_seqs)
            if not seq:
                seq = _uniprot_search_name_only(name, used_seqs)
            if not seq:
                seq = _ncbi_protein_search(name, used_seqs)
            if seq:
                used_seqs.add(seq)
            status = f"{len(seq)} aa" if seq else "missing"
        rows.append({"name": name, "seq": seq, "label": 1})
        print(f"[{i+1}/{len(names)}] {name}: {status}")
        time.sleep(DELAY_BETWEEN_NAMES)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "seq", "label"])
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["seq"])
    print(f"\nWrote {OUT_CSV}: {n_ok}/{len(rows)} sequences. Re-run: only missing entries refetched; existing (incl. manual) kept. Dedup: one sequence per name.")


if __name__ == "__main__":
    main()
