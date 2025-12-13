# train_viscosity.py
# 训练粘度预测模型：基于离子对 SMILES 预测粘度
# 依据论文 "Predicting Ionic Liquid Materials Properties from Chemical Structure"
# ============================================================
# Viscosity prediction – FINAL STABLE PAPER-CONSISTENT VERSION
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
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

def plot_loss(history, out_path="loss_curve_viscosity.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training curve (viscosity)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

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

# ============================================================
# Viscosity head (PHYSICALLY CONSTRAINED)
# log_eta = A + B / (T + C)
# ============================================================

@keras.saving.register_keras_serializable(package="ionic_mpnn")
def get_A(x):
    return keras.ops.expand_dims(x[:, 0], -1)

@keras.saving.register_keras_serializable(package="ionic_mpnn")
def get_B(x):
    # B ∈ [0, 20]
    return keras.ops.clip(
        keras.ops.nn.softplus(keras.ops.expand_dims(x[:, 1], -1)),
        0.0, 20.0
    )

@keras.saving.register_keras_serializable(package="ionic_mpnn")
def get_C(x):
    # C ∈ [0.1, 50]
    return keras.ops.clip(
        keras.ops.nn.softplus(keras.ops.expand_dims(x[:, 2], -1)),
        0.1, 50.0
    )

@keras.saving.register_keras_serializable(package="ionic_mpnn")
def compute_log_eta(inputs):
    A, B, T, C = inputs
    return A + B / (T + C + 1e-3)

# ============================================================
# Model (PAPER SETTINGS)
# ============================================================

def build_model(
    atom_vocab_size,
    bond_vocab_size,
    atom_dim=32,
    bond_dim=8,
    fp_size=32,
    mixing_size=20,
    num_steps=4
):
    # ---------- Inputs ----------
    cat_atom = Input(shape=(None,), dtype=tf.int32, name="cat_atom")
    cat_bond = Input(shape=(None,), dtype=tf.int32, name="cat_bond")
    cat_conn = Input(shape=(None, 2), dtype=tf.int32, name="cat_connectivity")

    an_atom = Input(shape=(None,), dtype=tf.int32, name="an_atom")
    an_bond = Input(shape=(None,), dtype=tf.int32, name="an_bond")
    an_conn = Input(shape=(None, 2), dtype=tf.int32, name="an_connectivity")

    T_input = Input(shape=(1,), dtype=tf.float32, name="temperature")

    # ---------- Embeddings ----------
    atom_emb_layer = Embedding(atom_vocab_size, atom_dim, mask_zero=True)
    bond_emb_layer = Embedding(bond_vocab_size, bond_dim, mask_zero=True)

    def encode(atom_ids, bond_ids, conn, prefix):
        atom_emb = atom_emb_layer(atom_ids)
        bond_emb = bond_emb_layer(bond_ids)
        h = atom_emb

        for i in range(num_steps):
            m = BondMatrixMessage(atom_dim, bond_dim, name=f"{prefix}_bmm_{i}")(
                [h, bond_emb, conn]
            )
            agg = Reduce(name=f"{prefix}_reduce_{i}")([m, conn[:, :, 1], h])
            h = GatedUpdate(atom_dim)([h, agg])

        fp = GlobalSumPool()([h, atom_ids])
        fp = Dense(fp_size, activation="relu", kernel_regularizer=l2(1e-5))(fp)
        return fp

    fp_cat = encode(cat_atom, cat_bond, cat_conn, "cat")
    fp_an  = encode(an_atom,  an_bond,  an_conn,  "an")

    cat_proj = Dense(mixing_size, activation="relu")(fp_cat)
    an_proj  = Dense(mixing_size, activation="relu")(fp_an)

    # PAPER: element-wise sum
    mixed = Lambda(lambda x: x[0] + x[1])([cat_proj, an_proj])

    visc_params = Dense(3)(mixed)
    A = Lambda(get_A)(visc_params)
    B = Lambda(get_B)(visc_params)
    C = Lambda(get_C)(visc_params)

    # Temperature scaling (physical)
    T_scaled = Lambda(lambda t: t / 100.0)(T_input)

    log_eta = Lambda(compute_log_eta)([A, B, T_scaled, C])

    model = Model(
        inputs=[
            cat_atom, cat_bond, cat_conn,
            an_atom, an_bond, an_conn,
            T_input
        ],
        outputs=log_eta
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss="mse"
    )
    return model

# ============================================================
# Main
# ============================================================

def main():
    DATA = "data/viscosity_id_data.pkl"
    VOCAB = "data/vocab.pkl"

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

    T = np.array([d["T"] for d in data], np.float32)[:, None]
    y = np.array([d["log_eta"] for d in data], np.float32)

    indices = np.arange(len(data))

    # ========================================================
    # SPLIT (DEFAULT = PAPER LEAKAGE)
    # ========================================================
    idx_train, idx_tmp = train_test_split(indices, test_size=0.30, random_state=42)
    idx_dev, idx_test  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    # --------- STRICT NO-LEAKAGE (UNCOMMENT) ----------
    # unique_pairs = np.unique(pair_ids)
    # p_train, p_tmp = train_test_split(unique_pairs, test_size=0.30, random_state=42)
    # p_dev, p_test  = train_test_split(p_tmp, test_size=0.50, random_state=42)
    # idx_train = [i for i, p in enumerate(pair_ids) if p in p_train]
    # idx_dev   = [i for i, p in enumerate(pair_ids) if p in p_dev]
    # idx_test  = [i for i, p in enumerate(pair_ids) if p in p_test]

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
        return {
            "cat_atom": pad_sequences_1d([cat_atoms[i] for i in idxs], max_atoms),
            "cat_bond": cb,
            "cat_connectivity": ce,
            "an_atom": pad_sequences_1d([an_atoms[i] for i in idxs], max_atoms),
            "an_bond": ab,
            "an_connectivity": ae,
            "temperature": T[idxs]
        }

    x_train = build_inputs(idx_train)
    x_dev   = build_inputs(idx_dev)
    x_test  = build_inputs(idx_test)

    y_train, y_dev, y_test = y[idx_train], y[idx_dev], y[idx_test]

    model = build_model(atom_vocab_size, bond_vocab_size)

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
    # Save final viscosity model (for transfer learning)
    # ========================================================
    os.makedirs("models", exist_ok=True)
    model.save("models/viscosity_final.keras", save_format="keras_v3")
    print("Saved viscosity model to models/viscosity_final.keras")

    plot_loss(history, "./results/loss_viscosity.png")

    for name, x_, y_ in [
        ("Train", x_train, y_train),
        ("Dev",   x_dev,   y_dev),
        ("Test",  x_test,  y_test)
    ]:
        pred = model.predict(x_).flatten()
        print(
            f"{name}: R2={r2_numpy(y_, pred):.4f}, "
            f"MAE={np.mean(np.abs(y_ - pred)):.4f}"
        )

    # ===============================
    # Figure 2(a): Viscosity Train + Dev
    # ===============================

    y_train_pred = model.predict(x_train).flatten()
    y_dev_pred   = model.predict(x_dev).flatten()

    plt.figure(figsize=(5, 5))

    # Train: 深橘色
    plt.scatter(
        y_train,
        y_train_pred,
        s=10,
        alpha=0.6,
        color="#FF8B32",
        label="Train"
    )

    # Dev: 浅橘色
    plt.scatter(
        y_dev,
        y_dev_pred,
        s=18,
        alpha=0.6,
        color="#FFD582BE",
        label="Validation"
    )

    # y = x reference line
    low = min(y_train.min(), y_dev.min(), y_train_pred.min(), y_dev_pred.min())
    high = max(y_train.max(), y_dev.max(), y_train_pred.max(), y_dev_pred.max())
    plt.plot([low, high], [low, high], "k--", linewidth=1)

    plt.xlabel("Experimental log(viscosity)")
    plt.ylabel("Predicted log(viscosity)")
    plt.title("Viscosity prediction (Figure 2a)")

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("./results/figure2_a_viscosity.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
