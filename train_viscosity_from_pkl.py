# train_viscosity_from_pkl.py
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
from models.layers import BondMatrixMessage, GatedUpdate, GlobalSumPool, Reduce
import matplotlib.pyplot as plt
# -------------------
EPS = 1e-6
# -------------------

def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + EPS)

def pad_sequences_1d(seq_list, max_len=None, pad_val=0):
    if max_len is None:
        max_len = max(len(s) for s in seq_list)
    padded = [s + [pad_val] * (max_len - len(s)) for s in seq_list]
    return np.array(padded, dtype=np.int32)

def preprocess_edges_and_bonds(edge_list, bond_list, max_edges=None):
    """
    Create directed edges (src,tgt) and bond ids. Use 0 for padding.
    """
    processed_edges = []
    processed_bonds = []
    for edges, bonds in zip(edge_list, bond_list):
        directed_edges = []
        directed_bonds = []
        for (src, tgt), bond_id in zip(edges, bonds):
            directed_edges.append([src, tgt])
            directed_bonds.append(bond_id)
            directed_edges.append([tgt, src])
            directed_bonds.append(bond_id)
        processed_edges.append(directed_edges)
        processed_bonds.append(directed_bonds)

    if max_edges is None:
        max_len = max(len(e) for e in processed_edges)
    else:
        max_len = max_edges * 2

    # pad with (0,0) and bond id 0 (embedding mask_zero=True will ensure zeros)
    processed_edges = [e + [[0,0]] * (max_len - len(e)) if len(e) < max_len else e[:max_len] for e in processed_edges]
    processed_bonds = [b + [0] * (max_len - len(b)) if len(b) < max_len else b[:max_len] for b in processed_bonds]

    return np.array(processed_edges, dtype=np.int32), np.array(processed_bonds, dtype=np.int32)

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

    T_input = Input(shape=(1,), dtype=tf.float32, name='temperature')

    # Shared embeddings (mask_zero -> index 0 used as padding)
    atom_embedding = Embedding(atom_vocab_size, atom_dim, mask_zero=True, name='atom_embedding')
    bond_embedding = Embedding(bond_vocab_size, bond_dim, mask_zero=True, name='bond_embedding')

    # --- cation encoder ---
    cat_atom_emb = atom_embedding(cat_atom_in)    # (B, N, D)
    cat_bond_emb = bond_embedding(cat_bond_in)    # (B, E, D_b)

    atom_state = cat_atom_emb
    for i in range(num_steps):
        messages = BondMatrixMessage(atom_dim, bond_dim, name=f'cat_bmm_{i}')([atom_state, cat_bond_emb, cat_conn_in])
        aggregated = Reduce(name=f'cat_reduce_{i}')([messages, cat_conn_in[:,:,1], atom_state])
        atom_state = GatedUpdate(atom_dim, dropout_rate=dropout_rate, name=f'cat_update_{i}')([atom_state, aggregated])

    fp_cat = GlobalSumPool()([atom_state, cat_atom_in])
    fp_cat = Dense(fp_size, activation='relu', kernel_regularizer=l2(1e-5))(fp_cat)
    fp_cat = Dropout(dropout_rate)(fp_cat)

    # --- anion encoder (shared embeddings & same architecture) ---
    an_atom_emb = atom_embedding(an_atom_in)
    an_bond_emb = bond_embedding(an_bond_in)

    atom_state = an_atom_emb
    for i in range(num_steps):
        messages = BondMatrixMessage(atom_dim, bond_dim, name=f'an_bmm_{i}')([atom_state, an_bond_emb, an_conn_in])
        aggregated = Reduce(name=f'an_reduce_{i}')([messages, an_conn_in[:,:,1], atom_state])
        atom_state = GatedUpdate(atom_dim, dropout_rate=dropout_rate, name=f'an_update_{i}')([atom_state, aggregated])

    fp_an = GlobalSumPool()([atom_state, an_atom_in])
    fp_an = Dense(fp_size, activation='relu', kernel_regularizer=l2(1e-5))(fp_an)
    fp_an = Dropout(dropout_rate)(fp_an)

    # combine
    cat_proj = Dense(mixing_size, activation='relu')(fp_cat)
    an_proj = Dense(mixing_size, activation='relu')(fp_an)
    combined = Lambda(lambda x: x[0] + x[1])([cat_proj, an_proj])

    # Viscosity head => output 3 params A, B, C
    visc_params = Dense(3, name='visc_params')(combined)
    # Use inverse temperature (1/T) as input to head
    T_inv = Lambda(lambda x: 1.0 / (x + 1e-6), name='T_inv')(T_input)

    # Map to log_eta = A + B / (T + C)  -> but we will implement with parameters
    # Split params
    A = Lambda(lambda x: tf.expand_dims(x[:, 0], -1))(visc_params)
    B = Lambda(lambda x: tf.expand_dims(x[:, 1], -1))(visc_params)
    C = Lambda(lambda x: tf.expand_dims(x[:, 2], -1))(visc_params)
    # ensure C is positive by softplus to avoid divide by negative small values
    C_pos = Lambda(lambda x: tf.nn.softplus(x) + 1e-6)(C)

    # compute log_eta_pred
    log_eta_pred = Lambda(lambda x: x[0] + tf.divide(x[1], x[2] + x[3]))([A, B, T_input, C_pos])
    # (we keep T_input + C to match A + B/(T+C) form)

    model = Model(
        inputs=[cat_atom_in, cat_bond_in, cat_conn_in,
                an_atom_in, an_bond_in, an_conn_in,
                T_input],
        outputs=log_eta_pred
    )

    opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse', metrics=[r2_metric])

    return model


def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def visualize_results(model, x_train, y_train, x_dev, y_dev):
    print("Generating visualizations...")

    # predict
    y_pred_train = model.predict(x_train).flatten()
    y_pred_dev = model.predict(x_dev).flatten()

    # ----------------------
    # 1. Train/Val Loss curve
    # ----------------------
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("Training Curve")
    save_plot("results/training_curve.png")

    # ----------------------
    # 2. Pred vs True (log10)
    # ----------------------
    plt.figure(figsize=(6,6))
    plt.scatter(y_train, y_pred_train, s=8, alpha=0.3, label='Train')
    plt.scatter(y_dev, y_pred_dev, s=15, alpha=0.6, label='Val')
    low = min(y_train.min(), y_dev.min())
    high = max(y_train.max(), y_dev.max())
    plt.plot([low, high], [low, high], 'k--')
    plt.xlabel("True Log10(η)")
    plt.ylabel("Predicted Log10(η)")
    plt.legend()
    plt.title("Predicted vs True")
    save_plot("results/pred_vs_true.png")

    # ----------------------
    # 3. Residual plot
    # ----------------------
    residuals = y_pred_dev - y_dev
    plt.figure()
    plt.scatter(y_dev, residuals, alpha=0.6, s=10)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("True Log10(η)")
    plt.ylabel("Residual")
    plt.title("Residual Plot (Val)")
    save_plot("results/residuals.png")

    # ----------------------
    # 4. Residual histogram
    # ----------------------
    plt.figure()
    plt.hist(residuals, bins=40, color="orange", alpha=0.8)
    plt.xlabel("Residual (log10 cP)")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    save_plot("results/residual_hist.png")

    # ----------------------
    # 5. Distribution comparison
    # ----------------------
    plt.figure()
    plt.hist(y_dev, bins=40, alpha=0.6, label="True")
    plt.hist(y_pred_dev, bins=40, alpha=0.6, label="Predicted")
    plt.legend()
    plt.title("Distribution of True vs Predicted")
    save_plot("results/dist_compare.png")

    print("All figures saved in results/")


def main():
    # load data
    with open('data/viscosity_id_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    atom_vocab_size = vocab['atom_vocab_size']
    bond_vocab_size = vocab['bond_vocab_size']

    pair_ids = [rec['pair_id'] for rec in data]
    cat_atom_ids = [rec['cation']['atom_ids'] for rec in data]
    cat_bond_ids = [rec['cation']['bond_ids'] for rec in data]
    cat_edges    = [rec['cation']['edge_indices'] for rec in data]
    an_atom_ids  = [rec['anion']['atom_ids'] for rec in data]
    an_bond_ids  = [rec['anion']['bond_ids'] for rec in data]
    an_edges     = [rec['anion']['edge_indices'] for rec in data]
    temps = np.array([rec['T'] for rec in data], dtype=np.float32)[:, None]
    labels = np.array([rec['log_eta'] for rec in data], dtype=np.float32)

    unique_pairs = list(dict.fromkeys(pair_ids))
    train_pairs, dev_pairs = train_test_split(unique_pairs, test_size=0.2, random_state=42)
    train_mask = np.array([pid in train_pairs for pid in pair_ids])
    dev_mask = np.array([pid in dev_pairs for pid in pair_ids])

    max_atoms = max(max(len(x) for x in cat_atom_ids), max(len(x) for x in an_atom_ids))
    max_bonds = max(max(len(x) for x in cat_bond_ids), max(len(x) for x in an_bond_ids))
    max_edges = max(max(len(x) for x in cat_edges),    max(len(x) for x in an_edges))

    cat_atom_train = pad_sequences_1d([cat_atom_ids[i] for i in range(len(cat_atom_ids)) if train_mask[i]], max_atoms, pad_val=0)
    cat_edge_train, cat_bond_train = preprocess_edges_and_bonds(
        [cat_edges[i] for i in range(len(cat_edges)) if train_mask[i]],
        [cat_bond_ids[i] for i in range(len(cat_bond_ids)) if train_mask[i]],
        max_edges
    )
    an_atom_train = pad_sequences_1d([an_atom_ids[i] for i in range(len(an_atom_ids)) if train_mask[i]], max_atoms, pad_val=0)
    an_edge_train, an_bond_train = preprocess_edges_and_bonds(
        [an_edges[i] for i in range(len(an_edges)) if train_mask[i]],
        [an_bond_ids[i] for i in range(len(an_bond_ids)) if train_mask[i]],
        max_edges
    )
    T_train = temps[train_mask]
    y_train = labels[train_mask]

    cat_atom_dev = pad_sequences_1d([cat_atom_ids[i] for i in range(len(cat_atom_ids)) if dev_mask[i]], max_atoms, pad_val=0)
    cat_edge_dev, cat_bond_dev = preprocess_edges_and_bonds(
        [cat_edges[i] for i in range(len(cat_edges)) if dev_mask[i]],
        [cat_bond_ids[i] for i in range(len(cat_bond_ids)) if dev_mask[i]],
        max_edges
    )
    an_atom_dev = pad_sequences_1d([an_atom_ids[i] for i in range(len(an_atom_ids)) if dev_mask[i]], max_atoms, pad_val=0)
    an_edge_dev, an_bond_dev = preprocess_edges_and_bonds(
        [an_edges[i] for i in range(len(an_edges)) if dev_mask[i]],
        [an_bond_ids[i] for i in range(len(an_bond_ids)) if dev_mask[i]],
        max_edges
    )
    T_dev = temps[dev_mask]
    y_dev = labels[dev_mask]

    print(f"Train / Val sizes: {len(y_train)} / {len(y_dev)}")

    model = build_model(atom_vocab_size+1, bond_vocab_size+1,
                        atom_dim=32, bond_dim=8, fp_size=32, mixing_size=20, num_steps=4, dropout_rate=0.2)
    model.summary()

    # callbacks
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 200 == 0:
            return lr * 0.5
        return lr

    lr_callback = LearningRateScheduler(lr_schedule)
    early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, min_delta=1e-5)

    x_train = {
        'cat_atom': cat_atom_train,
        'cat_bond': cat_bond_train,
        'cat_connectivity': cat_edge_train,
        'an_atom': an_atom_train,
        'an_bond': an_bond_train,
        'an_connectivity': an_edge_train,
        'temperature': T_train
    }
    x_dev = {
        'cat_atom': cat_atom_dev,
        'cat_bond': cat_bond_dev,
        'cat_connectivity': cat_edge_dev,
        'an_atom': an_atom_dev,
        'an_bond': an_bond_dev,
        'an_connectivity': an_edge_dev,
        'temperature': T_dev
    }

    history = model.fit(
        x_train, y_train,
        validation_data=(x_dev, y_dev),
        epochs=1000,
        batch_size=32,
        callbacks=[lr_callback, early_stop],
        verbose=1
    )

    os.makedirs('models', exist_ok=True)
    model.save('models/viscosity_final.keras')
    print("Saved model to models/viscosity_final.keras")

    # compute exact R2 on dev
    y_pred_dev = model.predict(x_dev, batch_size=32).flatten()
    mean_y = np.mean(y_dev)
    ss_tot = np.sum((y_dev - mean_y)**2)
    ss_res = np.sum((y_dev - y_pred_dev)**2)
    R2 = 1 - ss_res / (ss_tot + EPS)
    mae = np.mean(np.abs(y_dev - y_pred_dev))
    print(f"Dev R2: {R2:.4f}, MAE(log10 cP): {mae:.4f}")
    visualize_results(model, x_train, y_train, x_dev, y_dev)


if __name__ == '__main__':
    main()
