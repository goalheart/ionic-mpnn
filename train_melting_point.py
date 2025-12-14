import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 = INFO, WARNING 都不显示
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        # 可选：关闭 oneDNN
os.environ["ABSL_LOG_LEVEL"] = "2"               # ← 关键！3 = ERROR 及以上，2 = WARNING 及以上
# train_melting_point.py
"""
独立训练离子液体熔点 (Melting Point) 预测模型。
模型基于 Message Passing Neural Network (MPNN) 架构，
使用 70%/15%/15% 划分，并在训练后生成预测散点图 (Figure 2b)。
已对目标标签进行标准化处理，以优化训练 Loss，并修复了 GRUCell 的调用问题。
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# --- 自定义回调：仅在指定 epoch 输出日志 ---
class SelectiveVerboseCallback(keras.callbacks.Callback):
    def __init__(self, total_epochs, verbose_epochs=None):
        super().__init__()
        self.total_epochs = total_epochs
        if verbose_epochs is None:
            base = [1, 2, 3, 4, 5, 50, 100, 150, 200]
            last_five = list(range(total_epochs - 4, total_epochs + 1))
            self.verbose_epochs = set(base + last_five)
        else:
            self.verbose_epochs = set(verbose_epochs)

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1  # epoch 是 0-indexed
        if current_epoch in self.verbose_epochs:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            r2 = logs.get('r2_score', None)
            val_r2 = logs.get('val_r2_score', None)
            msg = f"Epoch {current_epoch}/{self.total_epochs} - loss: {loss:.6f} - val_loss: {val_loss:.6f}"
            if r2 is not None:
                msg += f" - r2: {r2:.4f} - val_r2: {val_r2:.4f}"
            print(msg)

# --- 1. 辅助函数 ---

def r2_numpy(y_true, y_pred):
    """计算 R^2 分数 (适用于 NumPy 数组)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # 避免除以零
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def plot_loss(history, save_path):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss') 
    plt.title('Melting Point Model Loss (Scaled)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss plot to {save_path}")

# --- 2. 配置和数据路径 ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'ionic-mpnn/data' 
RESULTS_DIR = BASE_DIR / 'ionic-mpnn/results'
MP_DATA_PATH = DATA_DIR / 'mp_id_data.pkl'
VOCAB_PATH = DATA_DIR / 'vocab.pkl'

# 创建目录
RESULTS_DIR.mkdir(exist_ok=True)
os.makedirs(BASE_DIR / "models", exist_ok=True)

HYPERPARAMS = {
    'atom_features_size': 32, 
    'fingerprint_size': 32,   
    'mixing_size': 20,        
    'message_passing_steps': 4, 
    'learning_rate_initial': 0.01,
    'learning_rate_decay_factor': 0.55, 
    'decay_step': 100,
    'epochs': 1000,
    'batch_size': 32, 
}

# --- 3. 核心模型组件 (GRUCell 调用修复) ---

def get_vocab_sizes(vocab_path):
    """加载词汇表并返回原子和键特征的大小"""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab['atom_vocab_size'], vocab['bond_vocab_size']


class MessagePassingLayer(layers.Layer):
    """GRU 更新 h_v"""
    def __init__(self, fp_size, **kwargs):
        super(MessagePassingLayer, self).__init__(**kwargs)
        self.fp_size = fp_size
        self.gru = layers.GRUCell(fp_size, name='GRU')

    def call(self, inputs):
        h_t, m_t = inputs  # shape: (batch, num_nodes, fp_size)

        batch_size = tf.shape(h_t)[0]
        num_nodes = tf.shape(h_t)[1]

        # flatten batch 和节点维度
        h_flat = tf.reshape(h_t, (-1, self.fp_size))  # (batch*num_nodes, fp_size)
        m_flat = tf.reshape(m_t, (-1, self.fp_size))

        # GRUCell 更新：
        # GRUCell 返回 [new_output, new_state]，我们只需要 new_output/state (即第一个元素)
        h_new_flat = self.gru(m_flat, [h_flat])[0] 

        # reshape 回原来的 shape
        h_new = tf.reshape(h_new_flat, (batch_size, num_nodes, self.fp_size))

        return h_new


class GatherNeighborFeatureLayer(layers.Layer):
    """提取邻居节点特征 h_w (tf.gather 封装)"""
    def call(self, inputs):
        h_v, edge_input = inputs
        target_indices = edge_input[:, :, 1]
        h_w = tf.gather(h_v, target_indices, batch_dims=1)
        return h_w


class GraphAggregationLayer(layers.Layer):
    """消息聚合层 (tf.scatter_nd 封装)"""
    def __init__(self, feature_size, **kwargs):
        super(GraphAggregationLayer, self).__init__(**kwargs)
        self.feature_size = feature_size
    
    def call(self, inputs):
        msg_vectors, start_indices, h_v = inputs
        
        batch_size = tf.shape(start_indices)[0]
        num_nodes = tf.shape(h_v)[1] 
        
        # 1. 构造扁平化索引 
        batch_indices = tf.range(batch_size)
        batch_indices = tf.expand_dims(batch_indices, axis=1)
        batch_indices = tf.tile(batch_indices, [1, tf.shape(start_indices)[1]])

        flat_indices = tf.stack([tf.reshape(batch_indices, [-1]), 
                                 tf.reshape(start_indices, [-1])], axis=1)
        
        # 2. 扁平化更新值 
        flat_updates = tf.reshape(msg_vectors, [-1, self.feature_size]) 
        
        # 3. 执行稀疏聚合
        output_shape = (batch_size, num_nodes, self.feature_size)
        m_agg = tf.scatter_nd(
            indices=flat_indices, 
            updates=flat_updates, 
            shape=output_shape
        )
        return m_agg


def create_mpnn_base(atom_vocab_size, bond_vocab_size, params):
    """创建共享的 Message Passing Neural Network 基础架构。"""
    
    # --- 输入定义 ---
    atom_input = keras.Input(shape=(None,), dtype='int32', name='atom_ids')
    bond_input = keras.Input(shape=(None,), dtype='int32', name='bond_ids')
    edge_input = keras.Input(shape=(None, 2), dtype='int32', name='edge_indices')
    
    # --- 嵌入层 ---
    atom_embedding = layers.Embedding(
        input_dim=atom_vocab_size, output_dim=params['atom_features_size'],
        name='atom_embedding', embeddings_initializer='glorot_uniform'
    )
    bond_embedding = layers.Embedding(
        input_dim=bond_vocab_size, output_dim=params['atom_features_size'] * params['atom_features_size'], 
        name='bond_embedding', embeddings_initializer='glorot_uniform'
    )

    h_v = atom_embedding(atom_input)

    # Message Passing Steps (M=4)
    for t in range(params['message_passing_steps']):
        # 1. 提取键特征
        A_e = bond_embedding(bond_input)
        
        # 1.1 提取相邻原子的特征 h_w
        gather_layer = GatherNeighborFeatureLayer(name=f'Gather_step_{t}')
        h_w = gather_layer([h_v, edge_input]) 

        # 2. 消息向量计算封装成 Lambda 层
        def compute_msg_vectors(inputs):
            A_e_inner, h_w_inner = inputs
            A_e_reshaped = tf.reshape(A_e_inner, (-1, tf.shape(A_e_inner)[1], params['atom_features_size'], params['atom_features_size']))
            h_w_reshaped = tf.expand_dims(h_w_inner, axis=-1)
            msg_vectors_inner = tf.matmul(A_e_reshaped, h_w_reshaped)
            return tf.squeeze(msg_vectors_inner, axis=-1)

        msg_vectors = layers.Lambda(compute_msg_vectors, name=f'Msg_step_{t}')([A_e, h_w])

        # 3. 聚合消息
        agg_layer = GraphAggregationLayer(
            feature_size=params['atom_features_size'], 
            name=f'Agg_step_{t}'
        )
        m_t_plus_1 = agg_layer([msg_vectors, edge_input[:, :, 0], h_v])
        
        # 4. GRU 更新
        mp_layer = MessagePassingLayer(params['atom_features_size'], name=f'MP_step_{t}')
        h_v = mp_layer([h_v, m_t_plus_1])
        
    # 5. 分子指纹生成
    fp_regression_layer = layers.Dense(params['fingerprint_size'], activation='relu', name='fp_regression')
    fp_vector = fp_regression_layer(h_v)
    mol_fingerprint = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='molecular_fingerprint')(fp_vector)
    
    return keras.Model(inputs=[atom_input, bond_input, edge_input], outputs=mol_fingerprint, name='MPNN_Base')


def create_melting_point_model(atom_vocab_size, bond_vocab_size, params):
    """创建完整的熔点预测模型。"""
    
    mpnn_base = create_mpnn_base(atom_vocab_size, bond_vocab_size, params)
    
    # --- 输入定义（两个离子）---
    cat_inputs = [keras.Input(shape=(None,), dtype='int32', name='cat_atom_ids'),
                  keras.Input(shape=(None,), dtype='int32', name='cat_bond_ids'),
                  keras.Input(shape=(None, 2), dtype='int32', name='cat_edge_indices')]
    an_inputs = [keras.Input(shape=(None,), dtype='int32', name='an_atom_ids'),
                 keras.Input(shape=(None,), dtype='int32', name='an_bond_ids'),
                 keras.Input(shape=(None, 2), dtype='int32', name='an_edge_indices')]
    
    # 2. 通过共享的 MPNN Base 获取分子指纹
    cat_fp = mpnn_base(cat_inputs)
    an_fp = mpnn_base(an_inputs)
    
    # 3. 独立密集层
    cat_dense = layers.Dense(params['mixing_size'], activation='relu', name='cation_pre_mix_dense')(cat_fp)
    an_dense = layers.Dense(params['mixing_size'], activation='relu', name='anion_pre_mix_dense')(an_fp)
    
    # 4. 组合（Sum）
    combined_fp = layers.Add(name='mixing_layer')([cat_dense, an_dense])
    
    # 5. 预测头
    head = layers.Dense(params['mixing_size'], activation='relu', name='mp_head_1')(combined_fp)
    output = layers.Dense(1, activation='linear', name='mp_output')(head)
    
    model = keras.Model(inputs=cat_inputs + an_inputs, outputs=output, name='Melting_Point_MPNN')
    
    return model

# --- 4. 数据加载和预处理 (70/15/15 划分 & 标准化) ---

def load_and_prepare_data(data_path):
    """加载 mp_id_data.pkl 并进行 70%/15%/15% 的训练/验证/测试分割, 并标准化 Y 标签。"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"原始数据集大小: {len(data)} 条记录")
        
    np.random.shuffle(data)
    
    # 提取输入和标签 (Y 是原始的未标准化熔点值)
    X_cat_atom = [np.array(r['cation']['atom_ids']) for r in data]
    X_cat_bond = [np.array(r['cation']['bond_ids']) for r in data]
    X_cat_edge = [np.array(r['cation']['edge_indices']) for r in data]
    X_an_atom = [np.array(r['anion']['atom_ids']) for r in data]
    X_an_bond = [np.array(r['anion']['bond_ids']) for r in data]
    X_an_edge = [np.array(r['anion']['edge_indices']) for r in data]
    Y_unscaled_all = np.array([r['mp'] for r in data]) 
    
    # 对齐形状 (Padding)
    X_cat_atom = keras.preprocessing.sequence.pad_sequences(X_cat_atom, padding='post', value=0)
    X_cat_bond = keras.preprocessing.sequence.pad_sequences(X_cat_bond, padding='post', value=0)
    X_cat_edge = keras.preprocessing.sequence.pad_sequences(X_cat_edge, padding='post', value=[0, 0])
    X_an_atom = keras.preprocessing.sequence.pad_sequences(X_an_atom, padding='post', value=0)
    X_an_bond = keras.preprocessing.sequence.pad_sequences(X_an_bond, padding='post', value=0)
    X_an_edge = keras.preprocessing.sequence.pad_sequences(X_an_edge, padding='post', value=[0, 0])

    X_all = [X_cat_atom, X_cat_bond, X_cat_edge, X_an_atom, X_an_bond, X_an_edge]
    
    # 70%/15%/15% 划分逻辑
    total_size = len(data)
    train_end_idx = int(total_size * 0.70)
    val_end_idx = int(total_size * 0.85) 
    
    # --- Y 标签标准化 (基于训练集统计量) ---
    Y_train_unscaled = Y_unscaled_all[:train_end_idx]
    
    # 计算训练集的均值和标准差
    Y_mean = np.mean(Y_train_unscaled)
    Y_std = np.std(Y_train_unscaled)
    # 确保标准差非零，避免除以零
    if Y_std == 0: Y_std = 1.0 

    # 缩放所有集合
    Y_scaled_all = (Y_unscaled_all - Y_mean) / Y_std
    
    # --- 集合划分 ---
    
    # 训练集 (Train)
    X_train = [X_all[i][:train_end_idx] for i in range(len(X_all))]
    Y_train_scaled = Y_scaled_all[:train_end_idx]
    Y_train_unscaled = Y_unscaled_all[:train_end_idx]

    # 验证集 (Validation/Dev)
    X_dev = [X_all[i][train_end_idx:val_end_idx] for i in range(len(X_all))]
    Y_dev_scaled = Y_scaled_all[train_end_idx:val_end_idx]
    Y_dev_unscaled = Y_unscaled_all[train_end_idx:val_end_idx]

    # 测试集 (Test)
    X_test = [X_all[i][val_end_idx:] for i in range(len(X_all))]
    Y_test_scaled = Y_scaled_all[val_end_idx:]
    Y_test_unscaled = Y_unscaled_all[val_end_idx:]

    print(f"数据划分完成: Train={len(Y_train_scaled)}, Dev={len(Y_dev_scaled)}, Test={len(Y_test_scaled)}")
    print(f"标签标准化参数: Mean={Y_mean:.4f}, Std={Y_std:.4f}")

    # 返回 X, 缩放后的 Y, 原始 Y, 和缩放参数
    return (X_train, Y_train_scaled, Y_train_unscaled), \
           (X_dev, Y_dev_scaled, Y_dev_unscaled), \
           (X_test, Y_test_scaled, Y_test_unscaled), \
           {'mean': Y_mean, 'std': Y_std}


# --- 5. 绘图函数 ---

def plot_melting_point_prediction(model, X_train, Y_train_unscaled, X_dev, Y_dev_unscaled, scaling_params):
    """
    生成熔点预测的散点图 (Figure 2b)，仅包含 Train 和 Dev 数据。
    使用反标准化后的预测值和原始 Y 值进行绘制。
    """
    Y_mean = scaling_params['mean']
    Y_std = scaling_params['std']
    
    # 预测 (模型返回 scaled values)
    y_train_pred_scaled = model.predict(X_train).flatten()
    y_dev_pred_scaled   = model.predict(X_dev).flatten()
    
    # 反标准化预测结果
    y_train_pred_unscaled = y_train_pred_scaled * Y_std + Y_mean
    y_dev_pred_unscaled   = y_dev_pred_scaled * Y_std + Y_mean
    
    # 使用反标准化后的数据进行绘图
    Y_train = Y_train_unscaled
    Y_dev = Y_dev_unscaled
    y_train_pred = y_train_pred_unscaled
    y_dev_pred = y_dev_pred_unscaled
    
    plt.figure(figsize=(5, 5))

    # Train: 深蓝色 
    plt.scatter(
        Y_train,
        y_train_pred,
        s=10,
        alpha=0.6,
        color="#0000FF", 
        label="Train"
    )

    # Dev (Validation): 浅蓝色 
    plt.scatter(
        Y_dev,
        y_dev_pred,
        s=18,
        alpha=0.6,
        color="#A4C1F7", 
        label="Validation"
    )

    # y = x reference line
    all_y = np.concatenate([Y_train, Y_dev, y_train_pred, y_dev_pred])
    low = all_y.min() - 0.1
    high = all_y.max() + 0.1
    plt.plot([low, high], [low, high], "k--", linewidth=1)

    plt.xlabel("Experimental Melting Point")
    plt.ylabel("Predicted Melting Point")
    plt.title("Melting Point prediction (Figure 2b)")

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "figure2_b_melting_point.png", dpi=300)
    plt.close()
    print(f"\n熔点预测散点图 (使用原始尺度值) 已保存到: {RESULTS_DIR / 'figure2_b_melting_point.png'}")


# --- 6. 训练主函数 ---

def train_model():
    # 1. 加载词汇表
    atom_size, bond_size = get_vocab_sizes(VOCAB_PATH)
    
    # 2. 创建模型
    model = create_melting_point_model(atom_size, bond_size, HYPERPARAMS)
    
    # 3. 加载数据 (70/15/15 划分 & 标准化)
    (X_train, Y_train_s, Y_train_u), \
    (X_dev, Y_dev_s, Y_dev_u), \
    (X_test, Y_test_s, Y_test_u), \
    scaling_params = load_and_prepare_data(MP_DATA_PATH)
    
    Y_mean = scaling_params['mean']
    Y_std = scaling_params['std']

    # 4. 定义学习率调度器
    decay_steps = len(X_train[0]) / HYPERPARAMS['batch_size'] * HYPERPARAMS['decay_step']
    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=HYPERPARAMS['learning_rate_initial'],
        decay_steps=decay_steps,
        decay_rate=1 - HYPERPARAMS['learning_rate_decay_factor']
    )
    
    # 5. 编译模型 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error',
        metrics=[keras.metrics.R2Score(name='r2_score')]
    )

    # 6. 训练
    print("\n" + "="*50)
    print("--- 熔点模型训练开始 (Loss 在标准化尺度上) ---")
    
    total_epochs = HYPERPARAMS['epochs']
    history = model.fit(
        X_train, Y_train_s, # 使用标准化的 Y 进行训练
        validation_data=(X_dev, Y_dev_s),
        epochs=total_epochs,
        batch_size=HYPERPARAMS['batch_size'],
        callbacks=[
            keras.callbacks.EarlyStopping(patience=50, monitor='val_loss', restore_best_weights=True),
            SelectiveVerboseCallback(total_epochs=total_epochs)
        ],
        verbose=0  # 关闭默认日志
    )
    
    # ========================================================
    # Save final melting point model (for transfer learning)
    # ========================================================
    models_dir = BASE_DIR / "models"
    model.save(models_dir / "melting_point_final.keras", save_format="keras_v3")
    print(f"\nSaved melting point model to {models_dir / 'melting_point_final.keras'}")

    plot_loss(history, RESULTS_DIR / "loss_melting_point.png")

    # 7. 评估并打印 R2/MAE (使用反标准化后的结果报告)
    print("\n--- 熔点模型最终评估 (R2 和 MAE 在原始尺度上报告) ---")
    Y_sets = {
        "Train": (X_train, Y_train_u),
        "Dev":   (X_dev, Y_dev_u),
        "Test":  (X_test, Y_test_u)
    }
    
    for name, (X_, Y_unscaled) in Y_sets.items():
        # 1. 预测标准化值
        pred_scaled = model.predict(X_).flatten()
        
        # 2. 反标准化预测值
        pred_unscaled = pred_scaled * Y_std + Y_mean
        
        # 3. 使用原始未标准化值计算 R2 和 MAE
        print(
            f"{name}: R2={r2_numpy(Y_unscaled, pred_unscaled):.4f}, "
            f"MAE={np.mean(np.abs(Y_unscaled - pred_unscaled)):.4f}"
        )
    
    # 8. 绘制结果图 (仅 Train + Dev)
    plot_melting_point_prediction(model, X_train, Y_train_u, X_dev, Y_dev_u, scaling_params)


if __name__ == '__main__':
    try:
        train_model()
    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到。请检查路径配置 ({MP_DATA_PATH} 或 {VOCAB_PATH}) 或确保数据文件已生成。")
    except Exception as e:
        print(f"\n模型训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()