# train_viscosity.py
# 训练粘度预测模型：基于离子对 SMILES 预测粘度
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
from models.layers import BondMatrixMessage, GatedUpdate, GlobalSumPool, Reduce
import matplotlib.pyplot as plt

EPS = 1e-6

def r2_numpy(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + EPS)

def combine_proj(inputs):
    return inputs[0] + inputs[1]

def pad_sequences_1d(seq_list, max_len=None, pad_val=0):
    if max_len is None:
        max_len = max(len(s) for s in seq_list)
    padded = [s + [pad_val] * (max_len - len(s)) for s in seq_list]
    return np.array(padded, dtype=np.int32)

# split params
# compute log_eta pred: A + B / (T + C)
def get_A(x): return tf.expand_dims(x[:, 0], -1)
def get_B(x): return tf.expand_dims(x[:, 1], -1)
def get_C(x): return tf.expand_dims(x[:, 2], -1)
def softplus_C(x): return tf.nn.softplus(x) + 1e-6
def compute_log_eta(inputs): 
    A, B, T, C_pos = inputs
    return A + tf.divide(B, T + C_pos)

def preprocess_edges_and_bonds(edge_list, bond_list, max_edges=None):
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

    atom_embedding = Embedding(atom_vocab_size, atom_dim, mask_zero=True, name='atom_embedding')
    bond_embedding = Embedding(bond_vocab_size, bond_dim, mask_zero=True, name='bond_embedding')

    # cation encoder
    cat_atom_emb = atom_embedding(cat_atom_in)
    cat_bond_emb = bond_embedding(cat_bond_in)
    atom_state = cat_atom_emb
    for i in range(num_steps):
        messages = BondMatrixMessage(atom_dim, bond_dim, name=f'cat_bmm_{i}')([atom_state, cat_bond_emb, cat_conn_in])
        aggregated = Reduce(name=f'cat_reduce_{i}')([messages, cat_conn_in[:,:,1], atom_state])
        atom_state = GatedUpdate(atom_dim, dropout_rate=dropout_rate, name=f'cat_update_{i}')([atom_state, aggregated])
    fp_cat = GlobalSumPool()([atom_state, cat_atom_in])
    fp_cat = Dense(fp_size, activation='relu', kernel_regularizer=l2(1e-5))(fp_cat)
    fp_cat = Dropout(dropout_rate)(fp_cat)

    # anion encoder
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

    cat_proj = Dense(mixing_size, activation='relu', name='cat_proj')(fp_cat)
    an_proj = Dense(mixing_size, activation='relu', name='an_proj')(fp_an)

    # combined = Lambda(lambda x: x[0] + x[1], name='combine')([cat_proj, an_proj])
    combined = Lambda(combine_proj, name='combine')([cat_proj, an_proj])


    visc_params = Dense(3, name='visc_params')(combined)


    A = Lambda(get_A, name='A')(visc_params)
    B = Lambda(get_B, name='B')(visc_params)
    C = Lambda(get_C, name='C')(visc_params)
    C_pos = Lambda(softplus_C, name='C_pos')(C)
    log_eta_pred = Lambda(compute_log_eta, name='log_eta')([A, B, T_input, C_pos])


    model = Model(inputs=[cat_atom_in, cat_bond_in, cat_conn_in,
                          an_atom_in, an_bond_in, an_conn_in,
                          T_input],
                  outputs=log_eta_pred)
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    def r2_metric(y_true, y_pred):
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - ss_res / (ss_tot + EPS)
    model.compile(optimizer=opt, loss='mse', metrics=[r2_metric])
    return model

def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def visualize_pred_vs_true(y_train, y_pred_train, y_dev, y_pred_dev, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(y_train, y_pred_train, s=8, alpha=0.3, label='Train')
    plt.scatter(y_dev, y_pred_dev, s=15, alpha=0.6, label='Dev')
    low = min(np.min(y_train), np.min(y_dev))
    high = max(np.max(y_train), np.max(y_dev))
    plt.plot([low, high], [low, high], 'k--')
    plt.xlabel("True Log10(η)")
    plt.ylabel("Predicted Log10(η)")
    plt.legend()
    plt.title("Viscosity: Predicted vs True")
    save_plot(outpath)

def main():
    DATA_VISC = 'data/viscosity_id_data.pkl'
    VOCAB = 'data/vocab.pkl'

    with open(DATA_VISC, 'rb') as f:
        data = pickle.load(f)
    with open(VOCAB, 'rb') as f:
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

    print(f"Train / Dev sizes: {len(y_train)} / {len(y_dev)}")

    model = build_model(atom_vocab_size+1, bond_vocab_size+1,
                        atom_dim=32, bond_dim=8, fp_size=32, mixing_size=20, num_steps=4, dropout_rate=0.2)
    model.summary()

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
    print("Saved viscosity model to models/viscosity_final.keras")

    # compute R2s
    y_pred_dev = model.predict(x_dev, batch_size=32).flatten()
    y_pred_train = model.predict(x_train, batch_size=32).flatten()
    R2_dev = r2_numpy(y_dev.flatten(), y_pred_dev)
    R2_train = r2_numpy(y_train.flatten(), y_pred_train)
    mae_dev = np.mean(np.abs(y_dev.flatten() - y_pred_dev))
    mae_train = np.mean(np.abs(y_train.flatten() - y_pred_train))
    print(f"R2_train: {R2_train:.4f}, MAE_train(log10 cP): {mae_train:.4f}")
    print(f"R2_dev:   {R2_dev:.4f}, MAE_dev(log10 cP):   {mae_dev:.4f}")

    # save Figure 2a style plot
    visualize_pred_vs_true(y_train.flatten(), y_pred_train, y_dev.flatten(), y_pred_dev, 'results/figure2_a_viscosity.png')

if __name__ == '__main__':
    main()