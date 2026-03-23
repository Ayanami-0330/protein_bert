import numpy as np

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

CTD_GROUPS = {
    "hydrophobicity": ["RKEDQN", "GASTPHY", "CLVIMFW"],
    "van_der_waals": ["GASTPD", "CPNVEQIL", "KMHFRYW"],
    "polarity": ["LIFWCMVY", "PATGS", "HQRKNED"],
    "polarizability": ["GASDT", "CPNVEQIL", "KMHFRYW"],
    "charge": ["KRH", "DE", "AGSTCPNQYFIMLWV"],
    "secondary_structure": ["EALMQKRH", "VIYCWFT", "GNPSD"],
    "solvent_accessibility": ["ALFCGIVW", "RKQEND", "MPSTHY"],
}

KD_HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Molecular weight of amino acids (average, Daltons)
AA_MW = {
    'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.03, 'H': 155.16, 'I': 131.18, 'K': 146.19, 'L': 131.18,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19,
}

# pK values for isoelectric point estimation
AA_PK = {
    'D': 3.65, 'E': 4.25, 'C': 8.18, 'Y': 10.07,
    'H': 6.00, 'K': 10.53, 'R': 12.48,
}

# Flexibility scale (Vihinen & Mantsala, 1989)
AA_FLEX = {
    'A': 0.357, 'C': 0.346, 'D': 0.511, 'E': 0.497, 'F': 0.314,
    'G': 0.544, 'H': 0.323, 'I': 0.462, 'K': 0.466, 'L': 0.365,
    'M': 0.295, 'N': 0.463, 'P': 0.509, 'Q': 0.493, 'R': 0.529,
    'S': 0.507, 'T': 0.444, 'V': 0.386, 'W': 0.305, 'Y': 0.420,
}


def _clean_seq(seq):
    seq = str(seq).strip().upper()
    return "".join([aa for aa in seq if aa in AA_INDEX])


def _aac(seq):
    seq = _clean_seq(seq)
    if not seq:
        return np.zeros(len(AA_LIST), dtype = np.float32)
    counts = np.zeros(len(AA_LIST), dtype = np.float32)
    for aa in seq:
        counts[AA_INDEX[aa]] += 1
    return counts / len(seq)


def _aac_segment(seq, n = 20):
    seq = _clean_seq(seq)
    if not seq:
        return np.zeros(len(AA_LIST), dtype = np.float32), np.zeros(len(AA_LIST), dtype = np.float32)
    n = min(n, len(seq))
    n_seg = seq[:n]
    c_seg = seq[-n:]
    return _aac(n_seg), _aac(c_seg)


def _paac(seq, lambda_value = 3, w = 0.05):
    seq = _clean_seq(seq)
    if not seq:
        return np.zeros(len(AA_LIST) + lambda_value, dtype = np.float32)
    values = np.array([KD_HYDRO.get(aa, 0.0) for aa in seq], dtype = np.float32)
    mean = values.mean()
    std = values.std() if values.std() > 0 else 1.0
    values = (values - mean) / std
    theta = []
    for lam in range(1, lambda_value + 1):
        if len(values) <= lam:
            theta.append(0.0)
            continue
        theta.append(np.mean((values[:-lam] - values[lam:]) ** 2))
    theta = np.array(theta, dtype = np.float32)
    denom = 1.0 + w * theta.sum()
    aac = _aac(seq) / denom
    paac = (w * theta) / denom
    return np.concatenate([aac, paac], axis = 0)


def _ctd(seq):
    seq = _clean_seq(seq)
    if not seq:
        return np.zeros(21 * len(CTD_GROUPS), dtype = np.float32)
    features = []
    for _, groups in CTD_GROUPS.items():
        group_map = {}
        for gi, group in enumerate(groups, start = 1):
            for aa in group:
                group_map[aa] = gi
        g_seq = [group_map.get(aa, 0) for aa in seq if aa in group_map]
        length = len(g_seq)
        if length == 0:
            features.extend([0.0] * 21)
            continue
        comp = [g_seq.count(1) / length, g_seq.count(2) / length, g_seq.count(3) / length]
        trans_counts = {"12": 0, "13": 0, "23": 0}
        for i in range(length - 1):
            a, b = g_seq[i], g_seq[i + 1]
            if a == b:
                continue
            key = "%d%d" % (min(a, b), max(a, b))
            if key in trans_counts:
                trans_counts[key] += 1
        trans = [
            trans_counts["12"] / (length - 1) if length > 1 else 0.0,
            trans_counts["13"] / (length - 1) if length > 1 else 0.0,
            trans_counts["23"] / (length - 1) if length > 1 else 0.0,
        ]
        dist = []
        for g in [1, 2, 3]:
            positions = [i + 1 for i, val in enumerate(g_seq) if val == g]
            if not positions:
                dist.extend([0.0] * 5)
                continue
            percentiles = [0, 25, 50, 75, 100]
            for p in percentiles:
                k = int(np.ceil(p / 100.0 * len(positions)))
                k = max(1, k)
                pos = positions[k - 1]
                dist.append(pos / length)
        features.extend(comp + trans + dist)
    return np.array(features, dtype = np.float32)


def _dpc(seq):
    """Dipeptide Composition (DPC): 400 dimensions.
    Captures short-range sequence order between adjacent amino acids.
    This is one of the key features used in AcrPred.
    """
    seq = _clean_seq(seq)
    n = len(seq)
    if n <= 1:
        return np.zeros(400, dtype = np.float32)
    counts = np.zeros(400, dtype = np.float32)
    for i in range(n - 1):
        idx1 = AA_INDEX.get(seq[i], -1)
        idx2 = AA_INDEX.get(seq[i + 1], -1)
        if idx1 >= 0 and idx2 >= 0:
            counts[idx1 * 20 + idx2] += 1
    total = max(n - 1, 1)
    return counts / total


def _cksaap(seq, k = 3):
    """Composition of k-Spaced Amino Acid Pairs (CKSAAP).
    For a given k, count pairs of amino acids separated by k residues.
    Returns 400 * (k+1) dimensional feature vector for k=0,1,...,k.
    To keep dimension manageable, only use k=0,1,2,3 -> 1600 dims.
    """
    seq = _clean_seq(seq)
    n = len(seq)
    all_features = []
    for gap in range(k + 1):
        counts = np.zeros(400, dtype = np.float32)
        total = max(n - gap - 1, 1)
        for i in range(n - gap - 1):
            idx1 = AA_INDEX.get(seq[i], -1)
            idx2 = AA_INDEX.get(seq[i + gap + 1], -1)
            if idx1 >= 0 and idx2 >= 0:
                counts[idx1 * 20 + idx2] += 1
        all_features.append(counts / total if total > 0 else counts)
    return np.concatenate(all_features, axis = 0)


def _physicochemical_global(seq):
    """Global physicochemical statistics: 15 dimensions.
    Captures sequence-level properties: length, weight, charge,
    hydrophobicity stats, flexibility stats, amino acid group fractions.
    """
    seq = _clean_seq(seq)
    if not seq:
        return np.zeros(15, dtype = np.float32)
    n = len(seq)

    # 1. Log-normalized length
    log_len = np.log1p(n)

    # 2. Average molecular weight
    avg_mw = sum(AA_MW.get(aa, 120.0) for aa in seq) / n

    # 3. Net charge at pH 7
    charge = 0.0
    for aa in seq:
        if aa in ('R', 'K'):
            charge += 1.0
        elif aa == 'H':
            charge += 0.1
        elif aa in ('D', 'E'):
            charge -= 1.0
    charge_per_res = charge / n

    # 4. Hydrophobicity statistics (Kyte-Doolittle)
    hydro = np.array([KD_HYDRO.get(aa, 0.0) for aa in seq], dtype = np.float32)
    hydro_mean = hydro.mean()
    hydro_std = hydro.std() if n > 1 else 0.0
    hydro_max = hydro.max()
    hydro_min = hydro.min()
    frac_hydrophobic = float((hydro > 0).mean())

    # 5. Flexibility statistics
    flex = np.array([AA_FLEX.get(aa, 0.4) for aa in seq], dtype = np.float32)
    flex_mean = flex.mean()
    flex_std = flex.std() if n > 1 else 0.0

    # 6. Amino acid group fractions
    aromatic = sum(1 for aa in seq if aa in 'FWY') / n
    aliphatic = sum(1 for aa in seq if aa in 'AILV') / n
    polar_uncharged = sum(1 for aa in seq if aa in 'STNQ') / n

    return np.array([
        log_len, avg_mw, charge_per_res,
        hydro_mean, hydro_std, hydro_max, hydro_min, frac_hydrophobic,
        flex_mean, flex_std,
        aromatic, aliphatic, polar_uncharged,
        float(n), float(charge),
    ], dtype = np.float32)


def extract_features(seqs, nc_len = 20, paac_lambda = 3):
    """Extract comprehensive hand-crafted features from protein sequences.

    Features included:
        CTD (Composition/Transition/Distribution): 147 dims
        AAC (Amino Acid Composition): 20 dims
        PAAC (Pseudo Amino Acid Composition): 23 dims
        N-terminal AAC: 20 dims
        C-terminal AAC: 20 dims
        DPC (Dipeptide Composition): 400 dims
        Physicochemical statistics: 15 dims
    Total: 645 dimensions
    """
    features = []
    for seq in seqs:
        ctd = _ctd(seq)                                     # 147
        aac = _aac(seq)                                     # 20
        paac = _paac(seq, lambda_value = paac_lambda)       # 23
        n_aac, c_aac = _aac_segment(seq, n = nc_len)       # 20 + 20
        dpc = _dpc(seq)                                     # 400
        physico = _physicochemical_global(seq)               # 15
        features.append(np.concatenate([ctd, aac, paac, n_aac, c_aac, dpc, physico], axis = 0))
    return np.vstack(features)
