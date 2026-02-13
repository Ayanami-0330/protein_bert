#!/usr/bin/env bash
set -euo pipefail

# Download UniRef50 FASTA and build a local BLAST database.
# All data stays under /home/nemophila/projects/protein_bert by default.
#
# Usage:
#   bash scripts/pssm/00_download_uniref50.sh
#
# Optional env:
#   DB_ROOT=/home/nemophila/projects/protein_bert/blast_db
#   UNIREF50_FASTA_URL=https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz

DB_ROOT="${DB_ROOT:-/home/nemophila/projects/protein_bert/blast_db}"
UNIREF50_FASTA_URL="${UNIREF50_FASTA_URL:-https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz}"

mkdir -p "${DB_ROOT}"
FA_GZ="${DB_ROOT}/uniref50.fasta.gz"
FA="${DB_ROOT}/uniref50.fasta"
DB_PREFIX="${DB_ROOT}/uniref50"

if ! command -v wget >/dev/null 2>&1; then
  echo "wget is required."
  exit 1
fi
if ! command -v makeblastdb >/dev/null 2>&1; then
  echo "makeblastdb is required (install blast+)."
  exit 1
fi

echo "Downloading UniRef50 to ${FA_GZ}"
wget -c -O "${FA_GZ}" "${UNIREF50_FASTA_URL}"

if [[ ! -s "${FA}" ]]; then
  echo "Decompressing ${FA_GZ}"
  gunzip -c "${FA_GZ}" > "${FA}"
fi

echo "Building BLAST database at ${DB_PREFIX}"
makeblastdb -in "${FA}" -dbtype prot -out "${DB_PREFIX}"
blastdbcmd -db "${DB_PREFIX}" -info

echo "Done."

