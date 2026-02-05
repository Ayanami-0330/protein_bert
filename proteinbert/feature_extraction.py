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


def extract_features(seqs, nc_len = 20, paac_lambda = 3):
    features = []
    for seq in seqs:
        ctd = _ctd(seq)
        aac = _aac(seq)
        paac = _paac(seq, lambda_value = paac_lambda)
        n_aac, c_aac = _aac_segment(seq, n = nc_len)
        features.append(np.concatenate([ctd, aac, paac, n_aac, c_aac], axis = 0))
    return np.vstack(features)
