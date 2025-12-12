# train_mp_from_pkl.py
# 训练熔点预测模型：基于离子对 SMILES 预测熔点（K）
# 依据论文 "Predicting Ionic Liquid Materials Properties from Chemical Structure"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# nfp-style layers
from models.layers import BondMatrixMessage, GatedUpdate, GlobalSumPool, Reduce

EPS = 1e-8

# ---------------------------
# Metrics
# ---------------------------
def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + EPS)

# ---------------------------
# Padding helpers
# ---------------------------
def pad_sequences_1d(seq_list, max_len=None, pad_val=0):
    if max_len is None:
        max_len = max(len(s) for s in seq_list)
    padded = [s + [pad_val] * (max_len - len(s)) for s in seq_list]
    return np.array(padded, dtype=np.int32)

def preprocess_edges_and_bonds(edge_list, bond_list, max_edges=None):
    processed_edges = []
    processed_bonds = []
    for edges, bonds in zip(edge_list, bond_list):
        de = []
        db = []
        for (s, t), b in zip(edges, bonds):
            de.append([s, t])
            db.append(b)
            de.append([t, s])
            db.append(b)
        processed_edges.append(de)
        processed_bonds.append(db)

    # pad
    if max_edges is None:
        max_len = max(len(e) for e in processed_edges)
    else:
        max_len = max_edges * 2

    proc_e = [
        e + [[0,0]] * (max_len - len(e)) if len(e) < max_len else e[:max_len]
        for e in processed_edges
    ]
    proc_b = [
        b + [0] * (max_len - len(b)) if len(b) < max_len else b[:max_len]
        for b in processed_bonds
    ]

    return np.array(proc_e, dtype=np.int32), np.array(proc_b, dtype=np.int32)

# ---------------------------
# GNN Model
# ---------------------------
def build_model(atom_vocab_size, bond_vocab_size,
                atom_dim=32, bond_dim=8, fp_size=32, mixing_size=20,
                num_steps=4, dropout_rate=0.2):

    # Inputs
    cat_atom_in = Input(shape=(None,), dtype=tf.int32, name='cat_atom')
    cat_bond_in = Input(shape=(None,), dtype=tf.int32, name='cat_bond')
    cat_conn_in = Input(shape=(None, 2), dtype=tf.int32, name='cat_connectivity')

    an_atom_in = Input(shape=(None,), dtype=tf.int32, name='an_atom')
    an_bond_in = Input(shape=(None,), dtype=tf.int32, name='an_bond')
    an_conn_in = Input(shape=(None, 2), dtype=tf.int32, name='an_connectivity')

    atom_embedding = Embedding(atom_vocab_size, atom_dim, mask_zero=True, name="atom_emb")
    bond_embedding = Embedding(bond_vocab_size, bond_dim, mask_zero=True, name="bond_emb")

    # ---------------------------
    # CATION GNN
    # ---------------------------
    cat_atom_emb = atom_embedding(cat_atom_in)
    cat_bond_emb = bond_embedding(cat_bond_in)
    atom_state = cat_atom_emb

    for i in range(num_steps):
        msg = BondMatrixMessage(atom_dim, bond_dim, name=f"cat_bmm_{i}")(
            [atom_state, cat_bond_emb, cat_conn_in]
        )
        agg = Reduce(name=f"cat_reduce_{i}")([msg, cat_conn_in[:,:,1], atom_state])
        atom_state = GatedUpdate(atom_dim, dropout_rate=dropout_rate,
                                 name=f"cat_update_{i}")([atom_state, agg])

    fp_cat = GlobalSumPool()([atom_state, cat_atom_in])
    fp_cat = Dense(fp_size, activation="relu", kernel_regularizer=l2(1e-5))(fp_cat)
    fp_cat = Dropout(dropout_rate)(fp_cat)

    # ---------------------------
    # ANION GNN
    # ---------------------------
    an_atom_emb = atom_embedding(an_atom_in)
    an_bond_emb = bond_embedding(an_bond_in)
    atom_state = an_atom_emb

    for i in range(num_steps):
        msg = BondMatrixMessage(atom_dim, bond_dim, name=f"an_bmm_{i}")(
            [atom_state, an_bond_emb, an_conn_in]
        )
        agg = Reduce(name=f"an_reduce_{i}")([msg, an_conn_in[:,:,1], atom_state])
        atom_state = GatedUpdate(atom_dim, dropout_rate=dropout_rate,
                                 name=f"an_update_{i}")([atom_state, agg])

    fp_an = GlobalSumPool()([atom_state, an_atom_in])
    fp_an = Dense(fp_size, activation="relu", kernel_regularizer=l2(1e-5))(fp_an)
    fp_an = Dropout(dropout_rate)(fp_an)

    # ---------------------------
    # Combine
    # ---------------------------
    cat_proj = Dense(mixing_size, activation="relu")(fp_cat)
    an_proj = Dense(mixing_size, activation="relu")(fp_an)

    combined = Lambda(lambda x: x[0] + x[1])([cat_proj, an_proj])

    # ---------------------------
    # Regression head → melting point (K)
    # ---------------------------
    mp_pred = Dense(1, name="mp_pred")(combined)

    model = Model(
        inputs=[cat_atom_in, cat_bond_in, cat_conn_in,
                an_atom_in, an_bond_in, an_conn_in],
        outputs=mp_pred
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="mse",
        metrics=[r2_metric]
    )
    return model

# ---------------------------
# Visualization
# ---------------------------
def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()

def visualize_results(history, model, x_train, y_train, x_dev, y_dev):
    print("Generating visualizations...")

    y_pred_train = model.predict(x_train).flatten()
    y_pred_dev = model.predict(x_dev).flatten()

    # 1. Training curve
    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Training Curve")
    save_plot("results/mp_training_curve.png")

    # 2. Pred vs True
    plt.figure(figsize=(6,6))
    plt.scatter(y_train, y_pred_train, s=8, alpha=0.3, label="Train")
    plt.scatter(y_dev, y_pred_dev, s=15, alpha=0.6, label="Val")
    low, high = min(y_train.min(), y_dev.min()), max(y_train.max(), y_dev.max())
    plt.plot([low, high], [low, high], "k--")
    plt.xlabel("True MP (K)")
    plt.ylabel("Predicted MP (K)")
    plt.title("Pred vs True")
    plt.legend()
    save_plot("results/mp_pred_vs_true.png")

    # 3. Residual Plot
    residuals = y_pred_dev - y_dev
    plt.figure()
    plt.scatter(y_dev, residuals, alpha=0.6, s=10)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("True MP")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    save_plot("results/mp_residual_plot.png")

    # ----------------------
    # 4. Residual histogram
    # ----------------------
    plt.figure()
    plt.hist(residuals, bins=40, alpha=0.8, color="orange")
    plt.xlabel("Residual (K)")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    save_plot("results/mp_residual_hist.png")

    # ----------------------
    # 5. Distribution comparison
    # ----------------------
    plt.figure()
    plt.hist(y_dev, bins=40, alpha=0.6, label="True")
    plt.hist(y_pred_dev, bins=40, alpha=0.6, label="Predicted")
    plt.legend()
    plt.title("Distribution of True vs Predicted (MP in K)")
    save_plot("results_mp/mp_dist_compare.png")

    print("All visualizations saved to results_mp/")


# ---------------------------
# Main Training Flow
# ---------------------------
def main():
    with open("data/melting_point_id_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    atom_vocab_size = vocab["atom_vocab_size"]
    bond_vocab_size = vocab["bond_vocab_size"]

    pair_ids = [d["pair_id"] for d in data]
    cat_atom_ids = [d["cation"]["atom_ids"] for d in data]
    cat_bond_ids = [d["cation"]["bond_ids"] for d in data]
    cat_edges = [d["cation"]["edge_indices"] for d in data]

    an_atom_ids = [d["anion"]["atom_ids"] for d in data]
    an_bond_ids = [d["anion"]["bond_ids"] for d in data]
    an_edges = [d["anion"]["edge_indices"] for d in data]

    labels = np.array([d["mp"] for d in data], dtype=np.float32)

    # ---------------------------
    # Pair-level split
    # ---------------------------
    uniq = list(dict.fromkeys(pair_ids))
    train_pairs, dev_pairs = train_test_split(uniq, test_size=0.2, random_state=42)

    train_mask = np.array([pid in train_pairs for pid in pair_ids])
    dev_mask = np.array([pid in dev_pairs for pid in pair_ids])

    max_atoms = max(max(len(x) for x in cat_atom_ids), max(len(x) for x in an_atom_ids))
    max_edges = max(max(len(x) for x in cat_edges), max(len(x) for x in an_edges))

    # ---------------------------
    # Prepare train/dev tensors
    # ---------------------------
    def build_split(mask):
        cat_atom = pad_sequences_1d([cat_atom_ids[i] for i in range(len(mask)) if mask[i]], max_atoms)
        cat_edge, cat_bond = preprocess_edges_and_bonds(
            [cat_edges[i] for i in range(len(mask)) if mask[i]],
            [cat_bond_ids[i] for i in range(len(mask)) if mask[i]],
            max_edges
        )
        an_atom = pad_sequences_1d([an_atom_ids[i] for i in range(len(mask)) if mask[i]], max_atoms)
        an_edge, an_bond = preprocess_edges_and_bonds(
            [an_edges[i] for i in range(len(mask)) if mask[i]],
            [an_bond_ids[i] for i in range(len(mask)) if mask[i]],
            max_edges
        )
        y = labels[mask]
        return (cat_atom, cat_bond, cat_edge, an_atom, an_bond, an_edge, y)

    cat_atom_train, cat_bond_train, cat_edge_train, \
    an_atom_train, an_bond_train, an_edge_train, y_train = build_split(train_mask)

    cat_atom_dev, cat_bond_dev, cat_edge_dev, \
    an_atom_dev, an_bond_dev, an_edge_dev, y_dev = build_split(dev_mask)

    # ---------------------------
    # Model
    # ---------------------------
    model = build_model(atom_vocab_size+1, bond_vocab_size+1)
    model.summary()

    # Callbacks
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 200 == 0:
            return lr * 0.5
        return lr

    x_train = {
        "cat_atom": cat_atom_train,
        "cat_bond": cat_bond_train,
        "cat_connectivity": cat_edge_train,
        "an_atom": an_atom_train,
        "an_bond": an_bond_train,
        "an_connectivity": an_edge_train
    }
    x_dev = {
        "cat_atom": cat_atom_dev,
        "cat_bond": cat_bond_dev,
        "cat_connectivity": cat_edge_dev,
        "an_atom": an_atom_dev,
        "an_bond": an_bond_dev,
        "an_connectivity": an_edge_dev
    }

    history = model.fit(
        x_train, y_train,
        validation_data=(x_dev, y_dev),
        epochs=1000,
        batch_size=32,
        callbacks=[
            LearningRateScheduler(lr_schedule),
            EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
        ],
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/mp_final.keras")
    print("Saved model to models/mp_final.keras")

    # -------- Final R2 --------
    y_pred = model.predict(x_dev).flatten()
    ss_tot = np.sum((y_dev - np.mean(y_dev))**2)
    ss_res = np.sum((y_dev - y_pred)**2)
    R2 = 1 - ss_res / (ss_tot + EPS)
    MAE = np.mean(np.abs(y_dev - y_pred))

    print(f"MP_Dev R2: {R2:.4f}, MP_MAE(K): {MAE:.4f}")

    visualize_results(history, model, x_train, y_train, x_dev, y_dev)


if __name__ == "__main__":
    main()
