# train_mp_from_pkl.py
# 训练熔点预测模型：基于离子对 SMILES 预测熔点（K）
# 依据论文 "Predicting Ionic Liquid Materials Properties from Chemical Structure"

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from models.layers import MessagePassing, GlobalSumPool
from tensorflow.keras.layers import Input, Embedding, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

def r2_metric(y_true, y_pred):
    """训练过程监控用 R²（batch 级别，仅参考）"""
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + 1e-8)

def pad_sequences_1d(seq_list, max_len=None, pad_val=0):
    if max_len is None:
        max_len = max(len(s) for s in seq_list)
    padded = [s + [pad_val] * (max_len - len(s)) for s in seq_list]
    return np.array(padded, dtype=np.int32)

def pad_edge_indices(edge_list, max_edges=None, pad_val=0):
    if max_edges is None:
        max_edges = max(len(e) for e in edge_list)
    padded = []
    for edges in edge_list:
        pad_len = max_edges - len(edges)
        if pad_len > 0:
            edges = edges + [(pad_val, pad_val)] * pad_len
        padded.append(edges[:max_edges])
    return np.array(padded, dtype=np.int32)

def main():
    print("Loading preprocessed melting point data...")
    with open('data/mp_id_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    atom_vocab_size = vocab['atom_vocab_size']
    bond_vocab_size = vocab['bond_vocab_size']

    # 提取数据
    pair_ids = [rec['pair_id'] for rec in data]
    cat_atom_ids = [rec['cation']['atom_ids'] for rec in data]
    cat_bond_ids = [rec['cation']['bond_ids'] for rec in data]
    cat_edges    = [rec['cation']['edge_indices'] for rec in data]
    an_atom_ids  = [rec['anion']['atom_ids'] for rec in data]
    an_bond_ids  = [rec['anion']['bond_ids'] for rec in data]
    an_edges     = [rec['anion']['edge_indices'] for rec in data]
    labels_raw = np.array([rec['mp'] for rec in data], dtype=np.float32)

    # 归一化到 [-1, 1]（仅用于训练）
    mp_min, mp_max = 200.0, 500.0
    labels = (labels_raw - mp_min) / (mp_max - mp_min) * 2.0 - 1.0

    # 按离子对划分（严格无重叠）
    unique_pairs = list(dict.fromkeys(pair_ids))
    train_pairs, dev_pairs = train_test_split(unique_pairs, test_size=0.2, random_state=42)
    train_mask = np.array([pid in train_pairs for pid in pair_ids])
    dev_mask = np.array([pid in dev_pairs for pid in pair_ids])

    # 确定填充长度
    max_atoms = max(max(len(x) for x in cat_atom_ids), max(len(x) for x in an_atom_ids))
    max_bonds = max(max(len(x) for x in cat_bond_ids), max(len(x) for x in an_bond_ids))
    max_edges = max(max(len(x) for x in cat_edges),    max(len(x) for x in an_edges))

    # 填充训练数据
    cat_atom_train = pad_sequences_1d([cat_atom_ids[i] for i in range(len(cat_atom_ids)) if train_mask[i]], max_atoms)
    cat_bond_train = pad_sequences_1d([cat_bond_ids[i] for i in range(len(cat_bond_ids)) if train_mask[i]], max_bonds)
    cat_edge_train = pad_edge_indices([cat_edges[i] for i in range(len(cat_edges))    if train_mask[i]], max_edges)
    an_atom_train = pad_sequences_1d([an_atom_ids[i] for i in range(len(an_atom_ids)) if train_mask[i]], max_atoms)
    an_bond_train = pad_sequences_1d([an_bond_ids[i] for i in range(len(an_bond_ids)) if train_mask[i]], max_bonds)
    an_edge_train = pad_edge_indices([an_edges[i] for i in range(len(an_edges))       if train_mask[i]], max_edges)
    y_train = labels[train_mask]

    # 填充验证数据
    cat_atom_dev = pad_sequences_1d([cat_atom_ids[i] for i in range(len(cat_atom_ids)) if dev_mask[i]], max_atoms)
    cat_bond_dev = pad_sequences_1d([cat_bond_ids[i] for i in range(len(cat_bond_ids)) if dev_mask[i]], max_bonds)
    cat_edge_dev = pad_edge_indices([cat_edges[i] for i in range(len(cat_edges))    if dev_mask[i]], max_edges)
    an_atom_dev = pad_sequences_1d([an_atom_ids[i] for i in range(len(an_atom_ids)) if dev_mask[i]], max_atoms)
    an_bond_dev = pad_sequences_1d([an_bond_ids[i] for i in range(len(an_bond_ids)) if dev_mask[i]], max_bonds)
    an_edge_dev = pad_edge_indices([an_edges[i] for i in range(len(an_edges))       if dev_mask[i]], max_edges)
    y_dev = labels[dev_mask]
    y_raw_dev = labels_raw[dev_mask]

    print(f"Train: {len(y_train)}, Val: {len(y_dev)}")

    # ----------------------------
    # 构建 MPNN 模型（论文 Section 4）
    # ----------------------------
    atom_dim = 32
    bond_dim = 8
    fp_size = 32
    mixing_size = 20
    num_steps = 4

    # 输入层（无温度）
    cat_atom_in = Input(shape=(None,), dtype=tf.int32, name='cat_atom')
    cat_bond_in = Input(shape=(None,), dtype=tf.int32, name='cat_bond')
    cat_conn_in = Input(shape=(None, 2), dtype=tf.int32, name='cat_connectivity')
    an_atom_in = Input(shape=(None,), dtype=tf.int32, name='an_atom')
    an_bond_in = Input(shape=(None,), dtype=tf.int32, name='an_bond')
    an_conn_in = Input(shape=(None, 2), dtype=tf.int32, name='an_connectivity')

    # 共享嵌入层（论文：共享权重）
    atom_embedding = Embedding(atom_vocab_size, atom_dim, mask_zero=False, name='atom_embedding')
    bond_embedding = Embedding(bond_vocab_size, bond_dim, mask_zero=False, name='bond_embedding')

    # 阳离子编码
    cat_atom_emb = atom_embedding(cat_atom_in)
    cat_bond_emb = bond_embedding(cat_bond_in)
    h_cat = cat_atom_emb
    for _ in range(num_steps):
        h_cat = MessagePassing(atom_dim, bond_dim)([h_cat, cat_bond_emb, cat_conn_in])
    fp_cat = GlobalSumPool()([h_cat, cat_atom_in])
    fp_cat = Dense(fp_size, activation='relu')(fp_cat)

    # 阴离子编码（共享权重）
    an_atom_emb = atom_embedding(an_atom_in)
    an_bond_emb = bond_embedding(an_bond_in)
    h_an = an_atom_emb
    for _ in range(num_steps):
        h_an = MessagePassing(atom_dim, bond_dim)([h_an, an_bond_emb, an_conn_in])
    fp_an = GlobalSumPool()([h_an, an_atom_in])
    fp_an = Dense(fp_size, activation='relu')(fp_an)

    # 融合（独立 dense + element-wise sum）
    cat_proj = Dense(mixing_size, activation='relu')(fp_cat)
    an_proj = Dense(mixing_size, activation='relu')(fp_an)
    combined = Lambda(lambda x: x[0] + x[1])([cat_proj, an_proj])

    # 熔点头：单输出（归一化空间）
    mp_pred = Dense(1, name='mp_prediction')(combined)

    model = Model(
        inputs=[
            cat_atom_in, cat_bond_in, cat_conn_in,
            an_atom_in, an_bond_in, an_conn_in
        ],
        outputs=mp_pred
    )

    # 编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005,clipnorm=1.0),
        loss='mse',
        metrics=[r2_metric]
    )

    # 回调函数
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 100 == 0:
            return lr * 0.55
        return lr

    lr_callback = LearningRateScheduler(lr_schedule)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    # 准备输入
    x_train = {
        'cat_atom': cat_atom_train,
        'cat_bond': cat_bond_train,
        'cat_connectivity': cat_edge_train,
        'an_atom': an_atom_train,
        'an_bond': an_bond_train,
        'an_connectivity': an_edge_train
    }
    x_dev = {
        'cat_atom': cat_atom_dev,
        'cat_bond': cat_bond_dev,
        'cat_connectivity': cat_edge_dev,
        'an_atom': an_atom_dev,
        'an_bond': an_bond_dev,
        'an_connectivity': an_edge_dev
    }

    # 训练
    print("Starting melting point training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_dev, y_dev),
        epochs=1000,
        batch_size=32,
        callbacks=[lr_callback, early_stop],
        verbose=1
    )

    # 保存模型
    model.save('models/mp_final.keras')
    norm_params = {'mp_min': mp_min, 'mp_max': mp_max}
    with open('models/mp_norm_params.pkl', 'wb') as f:
        pickle.dump(norm_params, f)
    print("Model saved to models/mp_final.keras")
    print("Normalization parameters saved to models/mp_norm_params.pkl")

    # ----------------------------
    # 精确评估（全集 R²）
    # ----------------------------
    print("\nCalculating exact R_dev^2...")

    # 预测归一化值
    y_pred_dev = model.predict(x_dev, batch_size=32).flatten()
    y_true_dev = y_dev

    # 在归一化空间计算 R²（与论文一致）
    mean_y_dev = np.mean(y_true_dev)
    ss_tot = np.sum(np.square(y_true_dev - mean_y_dev))
    ss_res = np.sum(np.square(y_true_dev - y_pred_dev))
    R2_dev = 1 - (ss_res / ss_tot)

    # 反归一化预测值
    y_pred_raw = (y_pred_dev + 1.0) * (mp_max - mp_min) / 2.0 + mp_min
    mae_raw = np.mean(np.abs(y_raw_dev - y_pred_raw))

    print(f"--- Final Validation Set Performance ---")
    print(f"R_dev^2 (scaled): {R2_dev:.4f}")
    print(f"MAE (K): {mae_raw:.2f}")

    # ----------------------------
    # 可视化（仅 Val Set，与论文 Figure 2b 一致）
    # ----------------------------
    import os
    os.makedirs('./results', exist_ok=True)
    print("Generating visualizations...")

    # 1. 预测 vs 真实值散点图（原始单位 K）
    plt.figure(figsize=(6, 6))
    plt.scatter(y_raw_dev, y_pred_raw, alpha=0.5, s=10, color='#ff7f0e')
    min_val = min(y_raw_dev.min(), y_pred_raw.min())
    max_val = max(y_raw_dev.max(), y_pred_raw.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1, label='Ideal')

    plt.xlabel('True Melting Point [K]')
    plt.ylabel('Predicted Melting Point [K]')
    plt.title('Melting Point Prediction')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 添加 R²_dev 标注（左上角）
    plt.text(0.05, 0.95, f"$R^2_{{dev}} = {R2_dev:.2f}$",
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig('./results/mp_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 残差图（原始单位 K）
    residuals = y_pred_raw - y_raw_dev
    plt.figure(figsize=(8, 4))
    plt.scatter(y_raw_dev, residuals, alpha=0.5, s=10, color='#1f77b4')
    plt.axhline(0, color='r', linestyle='--', lw=1)
    plt.xlabel('True Melting Point [K]')
    plt.ylabel('Residual [K]')
    plt.title('Residuals vs True Melting Point')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./results/mp_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualizations saved:")
    print("  - ./results/mp_scatter.png")
    print("  - ./results/mp_residuals.png")

if __name__ == '__main__':
    main()