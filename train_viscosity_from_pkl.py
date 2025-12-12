# train_viscosity_from_pkl.py
# 训练粘度预测模型（NREL/nfp 风格实现）
# 依据论文 "Predicting Ionic Liquid Materials Properties from Chemical Structure"

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import models
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from models.layers import Reduce, GlobalSumPool, BondMatrixMessage, GRUUpdate

# 配置 TensorFlow 日志级别（抑制冗余信息）
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =============== 辅助函数 ===============
def r2_metric(y_true, y_pred):
    """
    自定义 R² 指标（用于监控，但实际评估使用全集计算）
    注意：该指标在 batch 级别计算，仅作训练过程参考。
    """
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + 1e-8)

def pad_sequences_1d(seq_list, max_len=None, pad_val=0):
    """将变长序列列表填充至统一长度"""
    if max_len is None:
        max_len = max(len(s) for s in seq_list)
    padded = [s + [pad_val] * (max_len - len(s)) for s in seq_list]
    return np.array(padded, dtype=np.int32)

def preprocess_edges_and_bonds(edge_list, bond_list, max_edges=None):
    """
    同时处理边索引和键类型，为每条无向边生成两条有向边及其对应的 bond ID。
    """
    processed_edges = []
    processed_bonds = []
    for edges, bonds in zip(edge_list, bond_list):
        directed_edges = []
        directed_bonds = []
        for (src, tgt), bond_id in zip(edges, bonds):
            # 正向边
            directed_edges.append([src, tgt])
            directed_bonds.append(bond_id)
            # 反向边（使用相同的 bond ID，或可学习反向映射）
            directed_edges.append([tgt, src])
            directed_bonds.append(bond_id)  # 通常正反向共享 bond type
        
        if max_edges is not None:
            total_len = max_edges * 2
            pad_len = total_len - len(directed_edges)
            if pad_len > 0:
                directed_edges += [[0, 0]] * pad_len
                directed_bonds += [0] * pad_len
            processed_edges.append(directed_edges[:total_len])
            processed_bonds.append(directed_bonds[:total_len])
        else:
            processed_edges.append(directed_edges)
            processed_bonds.append(directed_bonds)
    
    # 统一长度
    if max_edges is None:
        max_len = max(len(e) for e in processed_edges)
        processed_edges = [e + [[0,0]] * (max_len - len(e)) for e in processed_edges]
        processed_bonds = [b + [0] * (max_len - len(b)) for b in processed_bonds]
    
    return np.array(processed_edges, dtype=np.int32), np.array(processed_bonds, dtype=np.int32)

# =============== 主函数 ===============
def main():
    print("Loading preprocessed data...")
    # 加载预处理后的整数 ID 序列数据
    with open('data/viscosity_id_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    atom_vocab_size = vocab['atom_vocab_size']
    bond_vocab_size = vocab['bond_vocab_size']

    # 提取所有字段
    pair_ids = [rec['pair_id'] for rec in data]
    cat_atom_ids = [rec['cation']['atom_ids'] for rec in data]
    cat_bond_ids = [rec['cation']['bond_ids'] for rec in data]
    cat_edges    = [rec['cation']['edge_indices'] for rec in data]
    an_atom_ids  = [rec['anion']['atom_ids'] for rec in data]
    an_bond_ids  = [rec['anion']['bond_ids'] for rec in data]
    an_edges     = [rec['anion']['edge_indices'] for rec in data]
    temps = np.array([rec['T'] for rec in data], dtype=np.float32)[:, None]
    labels = np.array([rec['log_eta'] for rec in data], dtype=np.float32)

    print(f"Label stats: mean={np.mean(labels):.3f}, std={np.std(labels):.3f}")

    # 按离子对 ID 划分训练/验证集（严格无重叠）
    unique_pairs = list(dict.fromkeys(pair_ids))
    train_pairs, dev_pairs = train_test_split(unique_pairs, test_size=0.2, random_state=42)
    train_mask = np.array([pid in train_pairs for pid in pair_ids])
    dev_mask = np.array([pid in dev_pairs for pid in pair_ids])

    # 确定填充最大长度
    max_atoms = max(max(len(x) for x in cat_atom_ids), max(len(x) for x in an_atom_ids))
    max_bonds = max(max(len(x) for x in cat_bond_ids), max(len(x) for x in an_bond_ids))
    max_edges = max(max(len(x) for x in cat_edges),    max(len(x) for x in an_edges))

    # 填充训练数据
    cat_atom_train = pad_sequences_1d([cat_atom_ids[i] for i in range(len(cat_atom_ids)) if train_mask[i]], max_atoms)
    cat_edge_train, cat_bond_train = preprocess_edges_and_bonds(
        [cat_edges[i] for i in range(len(cat_edges)) if train_mask[i]],
        [cat_bond_ids[i] for i in range(len(cat_bond_ids)) if train_mask[i]],
        max_edges
    )

    an_atom_train = pad_sequences_1d([an_atom_ids[i] for i in range(len(an_atom_ids)) if train_mask[i]], max_atoms)
    an_edge_train, an_bond_train = preprocess_edges_and_bonds(
        [an_edges[i] for i in range(len(an_edges)) if train_mask[i]],
        [an_bond_ids[i] for i in range(len(an_bond_ids)) if train_mask[i]],
        max_edges
    )
    T_train = temps[train_mask]
    y_train = labels[train_mask]

    # 填充验证数据
    cat_atom_dev = pad_sequences_1d([cat_atom_ids[i] for i in range(len(cat_atom_ids)) if dev_mask[i]], max_atoms)
    cat_edge_dev, cat_bond_dev = preprocess_edges_and_bonds(
        [cat_edges[i] for i in range(len(cat_edges)) if dev_mask[i]],
        [cat_bond_ids[i] for i in range(len(cat_bond_ids)) if dev_mask[i]],
        max_edges
    )

    an_atom_dev = pad_sequences_1d([an_atom_ids[i] for i in range(len(an_atom_ids)) if dev_mask[i]], max_atoms)
    an_edge_dev, an_bond_dev = preprocess_edges_and_bonds(
        [an_edges[i] for i in range(len(an_edges)) if dev_mask[i]],
        [an_bond_ids[i] for i in range(len(an_bond_ids)) if dev_mask[i]],
        max_edges
    )
    T_dev = temps[dev_mask]
    y_dev = labels[dev_mask]

    print(f"Train: {len(y_train)}, Val: {len(y_dev)}")

    # ----------------------------
    # 构建 MPNN 模型（NREL/nfp 风格）
    # ----------------------------
    atom_dim = 32      # 原子嵌入维度
    bond_dim = 8       # 键嵌入维度
    fp_size = 32       # 分子指纹维度
    mixing_size = 20   # 融合前投影维度
    num_steps = 4      # 消息传递步数

    # 定义输入层
    cat_atom_in = Input(shape=(None,), dtype=tf.int32, name='cat_atom')
    cat_bond_in = Input(shape=(None,), dtype=tf.int32, name='cat_bond')
    cat_conn_in = Input(shape=(None, 2), dtype=tf.int32, name='cat_connectivity')
    an_atom_in = Input(shape=(None,), dtype=tf.int32, name='an_atom')
    an_bond_in = Input(shape=(None,), dtype=tf.int32, name='an_bond')
    an_conn_in = Input(shape=(None, 2), dtype=tf.int32, name='an_connectivity')
    T_input = Input(shape=(1,), dtype=tf.float32, name='temperature')
    recip_T = Lambda(lambda t: 1000.0 / (t + 1e-6))(T_input)
    
    # 共享嵌入层
    atom_embedding = Embedding(atom_vocab_size, atom_dim, mask_zero=False, name='atom_embedding')
    bond_embedding = Embedding(bond_vocab_size, bond_dim, mask_zero=False, name='bond_embedding')

    # 阳离子编码
    cat_atom_emb = atom_embedding(cat_atom_in)  # (B, N, D_a)
    cat_bond_emb = bond_embedding(cat_bond_in)  # (B, E, D_b)
    
    # 消息传递（使用新重写的层）
    atom_state = cat_atom_emb
    for _ in range(num_steps):
        messages = BondMatrixMessage(atom_dim, bond_dim)(
            [atom_state, cat_bond_emb, cat_conn_in]
        )
        atom_state = GRUUpdate(atom_dim)(
            [atom_state, messages, cat_conn_in]   # 阳离子
        )
    
    # 生成分子指纹
    fp_cat = GlobalSumPool()([atom_state, cat_atom_in])  # 使用 nfp 的 GlobalSumPool
    fp_cat = Dense(fp_size, activation='relu', kernel_regularizer=l2(1e-4))(fp_cat)
    fp_cat = Dropout(0.2)(fp_cat)

    # 阴离子编码（共享权重）
    an_atom_emb = atom_embedding(an_atom_in)
    an_bond_emb = bond_embedding(an_bond_in)
    
    atom_state = an_atom_emb
    for _ in range(num_steps):
        messages = BondMatrixMessage(atom_dim, bond_dim)(
            [atom_state, an_bond_emb, an_conn_in]
        )
        atom_state = GRUUpdate(atom_dim)(
            [atom_state, messages, an_conn_in]    # 阴离子
        )
    
    fp_an = GlobalSumPool()([atom_state, an_atom_in])
    fp_an = Dense(fp_size, activation='relu', kernel_regularizer=l2(1e-4))(fp_an)
    fp_an = Dropout(0.2)(fp_an)

    # 融合离子对信息（独立 dense + element-wise sum）
    cat_proj = Dense(mixing_size, activation='relu')(fp_cat)
    an_proj = Dense(mixing_size, activation='relu')(fp_an)
    combined = Lambda(lambda x: x[0] + x[1])([cat_proj, an_proj])

    # 粘度预测头：A + B / T（Arrhenius-like）
    # 1. 预测 Arrhenius 参数 A 和 B (输出维度改为 2)
    # A 是无穷高温粘度 (截距), B 是活化能相关项 (斜率)
    visc_params = Dense(2, name='visc_params', kernel_initializer='normal')(combined)
    A = visc_params[:, 0:1]
    B = visc_params[:, 1:2]

    log_eta_pred = Lambda(lambda x: x[0] + x[1] * x[2])([A, B, recip_T])
    # log_eta_pred = Lambda(lambda x: x[0] + tf.divide(x[1], x[2] + 1e-6))(
    #     [A, B, T_input]
    # )

    model = Model(
        inputs=[
            cat_atom_in, cat_bond_in, cat_conn_in,
            an_atom_in, an_bond_in, an_conn_in,
            T_input
        ],
        outputs=log_eta_pred
    )

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0),
        loss='mse',
        metrics=[r2_metric]  # 仅作训练过程参考
    )

    # 定义回调函数
    def lr_schedule(epoch, lr):
        """学习率调度：每100 epoch 衰减为 0.55 倍"""
        if epoch > 0 and epoch % 100 == 0:
            return lr * 0.55
        return lr

    lr_callback = LearningRateScheduler(lr_schedule)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,                # 验证损失20个epoch无改善则停止
        restore_best_weights=True   # 恢复最佳权重
    )

    # 准备输入字典
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

    # 开始训练
    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_dev, y_dev),
        epochs=100,
        batch_size=32,
        callbacks=[lr_callback, early_stop],
        verbose=2
    )

    # 保存模型
    model.save('models/viscosity_final.keras')
    print("Model saved to models/viscosity_final.keras")

    # ----------------------------
    # 在整个验证集上计算精确 R²
    # ----------------------------
    print("\nCalculating exact R_dev^2 on full dev set...")

    # 获取验证集预测值
    y_pred_dev = model.predict(x_dev, batch_size=32)
    y_true_dev = y_dev

    # 计算总平方和 (SS_tot)
    mean_y_dev = np.mean(y_true_dev)
    ss_tot = np.sum(np.square(y_true_dev - mean_y_dev))

    # 计算残差平方和 (SS_res)
    ss_res = np.sum(np.square(y_true_dev - y_pred_dev.flatten()))

    # 计算 R²
    R2_dev = 1 - (ss_res / ss_tot)
    mae_dev = np.mean(np.abs(y_true_dev - y_pred_dev.flatten()))

    print(f"--- Final Validation Set Performance ---")
    print(f"Total data points in Val Set: {len(y_dev)}")
    print(f"R_dev^2 (Exact): {R2_dev:.4f}")
    print(f"MAE (log(cP)): {mae_dev:.4f}")

    # ----------------------------
    # 可视化结果（Train + Val）
    # ----------------------------
    print("Generating visualizations...")

    # 预测训练集（用于可视化）
    y_pred_train = model.predict(x_train, batch_size=32).flatten()
    eta_true_train = 10 ** y_train
    eta_pred_train = 10 ** y_pred_train

    # 预测验证集
    eta_true_dev = 10 ** y_true_dev
    eta_pred_dev = 10 ** y_pred_dev

    # 1. 预测 vs 真实值散点图（对数刻度）
    plt.figure(figsize=(6, 6))
    plt.scatter(eta_true_train, eta_pred_train, alpha=0.3, s=8, color="#e1b20964", label='Train')
    plt.scatter(eta_true_dev,   eta_pred_dev,   alpha=0.6, s=12, color="#f77605", label='Val')

    min_val = min(eta_true_train.min(), eta_true_dev.min(), eta_pred_train.min(), eta_pred_dev.min())
    max_val = max(eta_true_train.max(), eta_true_dev.max(), eta_pred_train.max(), eta_pred_dev.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='Ideal')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Viscosity [cP]')
    plt.ylabel('Predicted Viscosity [cP]')
    plt.title('Viscosity Prediction')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 计算并标注 Val R²（论文报告的指标）
    plt.text(0.05, 0.90, f"$R^2_{{dev}} = {R2_dev:.2f}$", 
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig('./results/viscosity_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 残差图（仅 Val，因 Train 残差无实际意义）
    residuals_dev = y_pred_dev.flatten() - y_true_dev
    plt.figure(figsize=(8, 4))
    plt.scatter(y_true_dev, residuals_dev, alpha=0.6, s=12, color='#ff7f0e')
    plt.axhline(0, color='k', linestyle='--', lw=1)
    plt.xlabel('True $\log_{10}(\\eta)$ [cP]')
    plt.ylabel('Residual [$\log_{10}$(cP)]')
    plt.title('Residuals vs True Value (Val Set)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./results/viscosity_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualizations saved:")
    print("  - viscosity_scatter.png (Train + Val)")
    print("  - viscosity_residuals.png (Val only)")

if __name__ == '__main__':
    main()