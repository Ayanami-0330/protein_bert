import sys
import csv

def fasta_to_csv(fasta_path, csv_path, label):
    """
    Convert a FASTA file to ProteinBERT-compatible CSV.
    Each sequence gets the same label.
    """
    records = []

    with open(fasta_path, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    records.append(seq)
                    seq = ''
            else:
                seq += line
        if seq:
            records.append(seq)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'seq'])
        for seq in records:
            writer.writerow([label, seq])

    print(f'Saved {len(records)} sequences to {csv_path}')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python fasta_to_csv.py <input.fasta> <output.csv> <label>')
        sys.exit(1)

    fasta_path = sys.argv[1]
    csv_path = sys.argv[2]
    label = int(sys.argv[3])

    fasta_to_csv(fasta_path, csv_path, label)
