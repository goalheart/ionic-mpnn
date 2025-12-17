# utils/mp_utils.py
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-6

def r2_numpy(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + EPS)

def pad_sequences_1d(seq_list, max_len, pad_val=0):
    return np.array(
        [s + [pad_val] * (max_len - len(s)) for s in seq_list],
        dtype=np.int32
    )

def preprocess_edges_and_bonds(edge_list, bond_list, max_edges):
    processed_edges, processed_bonds = [], []

    for edges, bonds in zip(edge_list, bond_list):
        e2, b2 = [], []
        for (src, tgt), bond_id in zip(edges, bonds):
            e2.append([src, tgt])
            b2.append(bond_id)
            e2.append([tgt, src])
            b2.append(bond_id)
        processed_edges.append(e2)
        processed_bonds.append(b2)

    max_len = max_edges * 2

    processed_edges = [
        e + [[0, 0]] * (max_len - len(e)) if len(e) < max_len else e[:max_len]
        for e in processed_edges
    ]
    processed_bonds = [
        b + [0] * (max_len - len(b)) if len(b) < max_len else b[:max_len]
        for b in processed_bonds
    ]

    return (
        np.array(processed_edges, dtype=np.int32),
        np.array(processed_bonds, dtype=np.int32)
    )

def plot_loss(history, title):
    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (scaled)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
