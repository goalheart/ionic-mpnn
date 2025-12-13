# ============================================================
# Melting point prediction – TRANSFER LEARNING FROM VISCOSITY
# Figure 2(c) in paper – Modified with temperature input (Scheme A)
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from models.layers import (
    BondMatrixMessage, GatedUpdate, GlobalSumPool, Reduce
)

EPS = 1e-6

# ============================================================
# Utils
# ============================================================

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
    edges_out, bonds_out = [], []

    for edges, bonds in zip(edge_list, bond_list):
        e2, b2 = [], []
        for (src, tgt), bid in zip(edges, bonds):
            e2.append([src, tgt])
            b2.append(bid)
            e2.append([tgt, src])
            b2.append(bid)
        edges_out.append(e2)
        bonds_out.append(b2)

    max_len = max_edges * 2

    edges_out = [
        e + [[0, 0]] * (max_len - len(e)) if len(e) < max_len else e[:max_len]
        for e in edges_out
    ]
    bonds_out = [
        b + [0] * (max_len - len(b)) if len(b) < max_len else b[:max_len]
        for b in bonds_out
    ]

    return (
        np.array(edges_out, dtype=np.int32),
        np.array(bonds_out, dtype=np.int32)
    )

# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    DATA = "data/melting_point_id_data.pkl"
    VOCAB = "data/vocab.pkl"
    VISC_MODEL_PATH = "models/viscosity_final.keras"

    with open(DATA, "rb") as f:
        data = pickle.load(f)
    with open(VOCAB, "rb") as f:
        vocab = pickle.load(f)

    atom_vocab_size = vocab["atom_vocab_size"] + 1
    bond_vocab_size = vocab["bond_vocab_size"] + 1

    pair_ids = [d["pair_id"] for d in data]

    cat_atoms = [[a + 1 for a in d["cation"]["atom_ids"]] for d in data]
    cat_bonds = [[b + 1 for b in d["cation"]["bond_ids"]] for d in data]
    cat_edges = [d["cation"]["edge_indices"] for d in data]

    an_atoms  = [[a + 1 for a in d["anion"]["atom_ids"]] for d in data]
    an_bonds  = [[b + 1 for b in d["anion"]["bond_ids"]] for d in data]
    an_edges  = [d["anion"]["edge_indices"] for d in data]

    # --------------------------------------------------------
    # Target: melting point (z-score, PAPER)
    # --------------------------------------------------------
    raw_mp = np.array([d["mp"] for d in data], dtype=np.float32)
    mp_mean = raw_mp.mean()
    mp_std  = raw_mp.std() + 1e-6

    y = (raw_mp - mp_mean) / mp_std
    y = y.reshape(-1, 1)

    indices = np.arange(len(data))

    # --------------------------------------------------------
    # SPLIT (paper default: sample-level)
    # --------------------------------------------------------
    idx_train, idx_tmp = train_test_split(
        indices, test_size=0.30, random_state=42
    )
    idx_dev, idx_test = train_test_split(
        idx_tmp, test_size=0.50, random_state=42
    )

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------
    max_atoms = max(max(map(len, cat_atoms)), max(map(len, an_atoms)))
    max_edges = max(max(map(len, cat_edges)), max(map(len, an_edges)))

    def build_inputs(idxs):
        ce, cb = preprocess_edges_and_bonds(
            [cat_edges[i] for i in idxs],
            [cat_bonds[i] for i in idxs],
            max_edges
        )
        ae, ab = preprocess_edges_and_bonds(
            [an_edges[i] for i in idxs],
            [an_bonds[i] for i in idxs],
            max_edges
        )
        # Use dummy temperature = 0 (as in original model)
        temperature = np.zeros((len(idxs), 1), dtype=np.float32)
        return {
            "cat_atom": pad_sequences_1d([cat_atoms[i] for i in idxs], max_atoms),
            "cat_bond": cb,
            "cat_connectivity": ce,
            "an_atom": pad_sequences_1d([an_atoms[i] for i in idxs], max_atoms),
            "an_bond": ab,
            "an_connectivity": ae,
            "temperature": temperature,  # ✅ Now included as input
        }

    x_train = build_inputs(idx_train)
    x_dev   = build_inputs(idx_dev)
    x_test  = build_inputs(idx_test)

    y_train = y[idx_train]
    y_dev   = y[idx_dev]
    y_test  = y[idx_test]

    # ========================================================
    # Load viscosity model & extract base feature
    # ========================================================
    visc_model = keras.models.load_model(
        VISC_MODEL_PATH,
        compile=False
    )

    # Extract output before the final viscosity head (assumed to be layer[-4])
    mixed_output = visc_model.layers[-4].output

    base_model = Model(
        inputs=visc_model.inputs,
        outputs=mixed_output,
        name="frozen_viscosity_base"
    )

    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False

    # Get reference to temperature input from base model
    # NOTE: viscosity model must have included 'temperature' as an input!
    temperature_input = None
    for inp in base_model.inputs:
        if inp.name.startswith('temperature'):
            temperature_input = inp
            break
    if temperature_input is None:
        raise ValueError("Viscosity model does not contain 'temperature' input. "
                         "Please ensure the original model was trained with temperature.")

    # ========================================================
    # Build new melting point head WITH temperature
    # ========================================================
    # Feature from frozen base
    ion_feature = base_model.output

    # New dense layer on top
    x = Dense(256, activation="relu", name="mp_dense_1")(ion_feature)
    x = Dropout(0.2, name="mp_dropout")(x)

    # 🔥 Concatenate temperature
    x = Concatenate(name="concat_temp")([x, temperature_input])

    # Final prediction
    mp_out = Dense(1, name="melting_out")(x)

    model = Model(
        inputs=base_model.inputs,  # includes temperature
        outputs=mp_out,
        name="melting_transfer"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse"
    )

    model.summary()

    # ========================================================
    # Train
    # ========================================================
    history = model.fit(
        x_train, y_train,
        validation_data=(x_dev, y_dev),
        epochs=1000,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
        ],
        verbose=1
    )

    # ========================================================
    # Evaluation (paper format)
    # ========================================================
    def eval_split(name, x_, y_):
        pred = model.predict(x_).flatten()
        y_true = y_.flatten() * mp_std + mp_mean
        y_pred = pred * mp_std + mp_mean
        print(
            f"{name}: R2={r2_numpy(y_true, y_pred):.4f}, "
            f"MAE={np.mean(np.abs(y_true - y_pred)):.4f}"
        )
        return y_true, y_pred

    ytr_true, ytr_pred = eval_split("Train", x_train, y_train)
    ydv_true, ydv_pred = eval_split("Dev",   x_dev,   y_dev)
    yts_true, yts_pred = eval_split("Test",  x_test,  y_test)

    # ========================================================
    # Figure 2(c) – Transfer learning parity plot
    # ========================================================
    plt.figure(figsize=(5, 5))

    plt.scatter(
        ytr_true,
        ytr_pred,
        s=10,
        alpha=0.6,
        color="#2E7D32",
        label="Train"
    )

    plt.scatter(
        ydv_true,
        ydv_pred,
        s=18,
        alpha=0.6,
        color="#A5D6A7",
        label="Validation"
    )

    low = min(ytr_true.min(), ydv_true.min(), ytr_pred.min(), ydv_pred.min())
    high = max(ytr_true.max(), ydv_true.max(), ytr_pred.max(), ydv_pred.max())
    plt.plot([low, high], [low, high], "k--", linewidth=1)

    plt.xlabel("Experimental melting point (K)")
    plt.ylabel("Predicted melting point (K)")
    plt.title("Melting point prediction (transfer learning)")

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("./results/figure2_c_melting_point_transfer.png", dpi=300)
    plt.close()

    print("Saved Figure 2(c): ./results/figure2_c_melting_point_transfer.png")

# ============================================================
if __name__ == "__main__":
    main()