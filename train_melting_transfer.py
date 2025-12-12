# train_melting_transfer.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =========================
# 导入自定义函数和层
# =========================
from train_viscosity import (
    combine_proj, get_A, get_B, get_C, softplus_C, compute_log_eta,
    BondMatrixMessage, Reduce, GatedUpdate, GlobalSumPool,
    pad_sequences_1d, preprocess_edges_and_bonds
)

# =========================
# 设置随机种子
# =========================
np.random.seed(42)
tf.random.set_seed(42)

# =========================
# 绘图函数
# =========================
def plot_pred_vs_true(y_true_train, y_pred_train, y_true_dev, y_pred_dev, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_train, y_pred_train, s=8, alpha=0.3, label='Train')
    plt.scatter(y_true_dev, y_pred_dev, s=15, alpha=0.6, label='Dev')
    low = min(np.min(y_true_train), np.min(y_true_dev))
    high = max(np.max(y_true_train), np.max(y_true_dev))
    plt.plot([low, high], [low, high], 'k--')
    plt.xlabel("True Melting Point")
    plt.ylabel("Predicted Melting Point")
    plt.legend()
    plt.title("Melting Point: Predicted vs True")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

# =========================
# 读取熔点数据
# =========================
DATA_MP = 'data/melting_point_id_data.pkl'
with open(DATA_MP, 'rb') as f:
    data = pickle.load(f)

# assume melting values are pre-scaled to -1..1 as paper mentions; if not, scale here
# labels = np.array([rec['melting_scaled'] for rec in mp_data], dtype=np.float32).reshape(-1,1)
# 先提取原始熔点
raw_mps = np.array([rec['mp'] for rec in data], dtype=np.float32)
# 可选：进行归一化（例如 Min-Max 或 StandardScaler）
# 这里以简单的减均值除标准差为例（也可以改用 MinMaxScaler）
mean_mp = raw_mps.mean()
std_mp = raw_mps.std() + 1e-8  # 防止除零
scaled_mps = (raw_mps - mean_mp) / std_mp
labels = scaled_mps.reshape(-1, 1)

pair_ids = [rec['pair_id'] for rec in data]

cat_atom_ids = [rec['cation']['atom_ids'] for rec in data]
cat_bond_ids = [rec['cation']['bond_ids'] for rec in data]
cat_edges    = [rec['cation']['edge_indices'] for rec in data]

an_atom_ids  = [rec['anion']['atom_ids'] for rec in data]
an_bond_ids  = [rec['anion']['bond_ids'] for rec in data]
an_edges     = [rec['anion']['edge_indices'] for rec in data]

# =========================
# 数据集划分
# =========================
unique_pairs = list(dict.fromkeys(pair_ids))
train_pairs, dev_pairs = train_test_split(unique_pairs, test_size=0.2, random_state=42)
train_mask = np.array([pid in train_pairs for pid in pair_ids])
dev_mask = np.array([pid in dev_pairs for pid in pair_ids])

max_atoms = max(max(len(x) for x in cat_atom_ids), max(len(x) for x in an_atom_ids))
max_edges = max(max(len(x) for x in cat_edges), max(len(x) for x in an_edges))

def prepare_inputs(cat_atom_ids, cat_bond_ids, cat_edges,
                   an_atom_ids, an_bond_ids, an_edges, mask):
    cat_atom = pad_sequences_1d([cat_atom_ids[i] for i in range(len(cat_atom_ids)) if mask[i]], max_atoms, pad_val=0)
    cat_edge, cat_bond = preprocess_edges_and_bonds(
        [cat_edges[i] for i in range(len(cat_edges)) if mask[i]],
        [cat_bond_ids[i] for i in range(len(cat_bond_ids)) if mask[i]],
        max_edges
    )
    an_atom = pad_sequences_1d([an_atom_ids[i] for i in range(len(an_atom_ids)) if mask[i]], max_atoms, pad_val=0)
    an_edge, an_bond = preprocess_edges_and_bonds(
        [an_edges[i] for i in range(len(an_edges)) if mask[i]],
        [an_bond_ids[i] for i in range(len(an_bond_ids)) if mask[i]],
        max_edges
    )
    num_samples = np.sum(mask)
    # 创建一个形状为 (样本数, 1) 的全零温度输入
    dummy_temperature = np.zeros((num_samples, 1), dtype=np.float32)
    return {
            'mp_cat_atom': cat_atom, # 对应 Input(name='mp_cat_atom')
            'mp_cat_bond': cat_bond,
            'mp_cat_connectivity': cat_edge,
            'mp_an_atom': an_atom,
            'mp_an_bond': an_bond,
            'mp_an_connectivity': an_edge,
            'mp_temperature': dummy_temperature 
        }

x_train = prepare_inputs(cat_atom_ids, cat_bond_ids, cat_edges,
                         an_atom_ids, an_bond_ids, an_edges, train_mask)
x_dev = prepare_inputs(cat_atom_ids, cat_bond_ids, cat_edges,
                       an_atom_ids, an_bond_ids, an_edges, dev_mask)

y_train = labels[train_mask]
y_dev = labels[dev_mask]

# =========================
# 加载粘度模型
# =========================
VIS_MODEL = 'models/viscosity_final.keras'
vis_model = tf.keras.models.load_model(VIS_MODEL, compile=False,
    custom_objects={
        'combine_proj': combine_proj,
        'get_A': get_A,
        'get_B': get_B,
        'get_C': get_C,
        'softplus_C': softplus_C,
        'compute_log_eta': compute_log_eta,
        'BondMatrixMessage': BondMatrixMessage,
        'Reduce': Reduce,
        'GatedUpdate': GatedUpdate,
        'GlobalSumPool': GlobalSumPool
    })
print("Viscosity model loaded!")

# =========================
# 构建熔点迁移学习模型
# =========================
def build_melting_model(vis_model, freeze_gnn_layers=2):
    """
    基于粘度模型迁移学习构建熔点预测模型
    freeze_gnn_layers: 冻结前几层 GNN（BondMatrixMessage / GatedUpdate）
    """
    # 1️⃣ 克隆粘度模型结构
    vis_model.trainable = True
    for i, layer in enumerate(vis_model.layers[:freeze_gnn_layers]):
        layer.trainable = False

    # 2️⃣ 新建输入，名字加前缀避免重复
    cat_atom_in = Input(shape=vis_model.input[0].shape[1:], dtype=tf.int32, name='mp_cat_atom')
    cat_bond_in = Input(shape=vis_model.input[1].shape[1:], dtype=tf.int32, name='mp_cat_bond')
    cat_conn_in = Input(shape=vis_model.input[2].shape[1:], dtype=tf.int32, name='mp_cat_connectivity')

    an_atom_in = Input(shape=vis_model.input[3].shape[1:], dtype=tf.int32, name='mp_an_atom')
    an_bond_in = Input(shape=vis_model.input[4].shape[1:], dtype=tf.int32, name='mp_an_bond')
    an_conn_in = Input(shape=vis_model.input[5].shape[1:], dtype=tf.int32, name='mp_an_connectivity')

    T_input = Input(shape=(1,), dtype=tf.float32, name='mp_temperature')

    # 3️⃣ 用粘度模型共享的中间表示
    # 注意这里用同样的层，但输入层换名字
    vis_output = vis_model([cat_atom_in, cat_bond_in, cat_conn_in,
                            an_atom_in, an_bond_in, an_conn_in,
                            T_input])

    # 4️⃣ 熔点预测头
    x = tf.keras.layers.Dense(64, activation='relu')(vis_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    mp_output = tf.keras.layers.Dense(1, activation='linear', name='melting_point')(x)

    model = tf.keras.models.Model(
        inputs=[cat_atom_in, cat_bond_in, cat_conn_in,
                an_atom_in, an_bond_in, an_conn_in,
                T_input],
        outputs=mp_output
    )
    return model

model = build_melting_model(vis_model, freeze_gnn_layers=2)
model.summary()

# =========================
# 学习率调度
# =========================
def lr_schedule(epoch, lr):
    if epoch > 0 and epoch % 100 == 0:
        return lr * 0.5
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # 选择优化器和学习率
    loss='mse', # 选择损失函数 (均方误差 Mean Squared Error，适用于回归任务)
    metrics=['mae'] # 选择评估指标 (平均绝对误差 Mean Absolute Error)
)
print("Melting point transfer model compiled successfully!")

# =========================
# 训练
# =========================
history = model.fit(x_train, y_train,
                    validation_data=(x_dev, y_dev),
                    epochs=500,
                    batch_size=32,
                    callbacks=[lr_callback, early_stop],
                    verbose=1)

# =========================
# 保存模型
# =========================
os.makedirs('models', exist_ok=True)
model.save('models/melting_transfer_final.keras')
print("Saved melting point transfer model!")

# =========================
# 绘制训练曲线和预测
# =========================
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Dev Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.savefig('results/melting_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

y_pred_train = model.predict(x_train).flatten()
y_pred_dev = model.predict(x_dev).flatten()
plot_pred_vs_true(y_train.flatten(), y_pred_train, y_dev.flatten(), y_pred_dev,
                  'results/melting_pred_vs_true.png')
print("Saved prediction figure and loss curve!")

# 计算训练集指标
R2_train = r2_score(y_train.flatten(), y_pred_train)
MAE_train = mean_absolute_error(y_train.flatten(), y_pred_train)

# 计算开发集指标
R2_dev = r2_score(y_dev.flatten(), y_pred_dev)
MAE_dev = mean_absolute_error(y_dev.flatten(), y_pred_dev)

print("\n--- Final Model Performance ---")
print(f"R2_train: {R2_train:.4f}, MAE_train: {MAE_train:.4f}")
print(f"R2_dev:   {R2_dev:.4f}, MAE_dev:   {MAE_dev:.4f}")
print("-------------------------------")
