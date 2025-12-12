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
# 导入自定义函数和层 (确保路径和名称正确)
# 假设您将 train_viscosity.py 中的辅助函数放在了这里或 models/layers.py 中
# 为了代码独立性，这里直接将需要的辅助函数复制过来
# =========================
EPS = 1e-6

def combine_proj(inputs): 
    # 对应于 viscosity 模型中的 'combine' 层
    return inputs[0] + inputs[1]

def pad_sequences_1d(seq_list, max_len=None, pad_val=0):
    if max_len is None:
        max_len = max(len(s) for s in seq_list)
    padded = [s + [pad_val] * (max_len - len(s)) for s in seq_list]
    return np.array(padded, dtype=np.int32)

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


# 粘度模型中用于计算 log_eta 的辅助函数，在迁移学习中不需要，但必须保留在 custom_objects 中
# 否则加载模型会失败
def get_A(x): return tf.expand_dims(x[:, 0], -1)
def get_B(x): return tf.expand_dims(x[:, 1], -1)
def get_C(x): return tf.expand_dims(x[:, 2], -1)
def softplus_C(x): return tf.nn.softplus(x) + 1e-6
def compute_log_eta(inputs): 
    A, B, T, C_pos = inputs
    return A + tf.divide(B, T + C_pos)

# 确保 GNN 的自定义层可以被加载
# 注意：您需要确保 'models/layers.py' 存在且包含 BondMatrixMessage, Reduce, GatedUpdate, GlobalSumPool
from models.layers import BondMatrixMessage, Reduce, GatedUpdate, GlobalSumPool


# =========================
# 设置随机种子
# =========================
np.random.seed(42)
tf.random.set_seed(42)

# =========================
# 绘图函数
# =========================
def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_pred_vs_true(y_true_train, y_pred_train, y_true_dev, y_pred_dev, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_train, y_pred_train, s=8, alpha=0.3, label='Train')
    plt.scatter(y_true_dev, y_pred_dev, s=15, alpha=0.6, label='Dev')
    low = min(np.min(y_true_train), np.min(y_true_dev))
    high = max(np.max(y_true_train), np.max(y_true_dev))
    plt.plot([low, high], [low, high], 'k--')
    plt.xlabel("True Scaled Melting Point")
    plt.ylabel("Predicted Scaled Melting Point")
    plt.legend()
    plt.title("Melting Point (Transfer): Predicted vs True")
    save_plot(outpath)

# =========================
# 读取熔点数据
# =========================
DATA_MP = 'data/melting_point_id_data.pkl'
with open(DATA_MP, 'rb') as f:
    data = pickle.load(f)

pair_ids = [rec['pair_id'] for rec in data]
raw_mps = np.array([rec['mp'] for rec in data], dtype=np.float32)

cat_atom_ids = [rec['cation']['atom_ids'] for rec in data]
cat_bond_ids = [rec['cation']['bond_ids'] for rec in data]
cat_edges    = [rec['cation']['edge_indices'] for rec in data]

an_atom_ids  = [rec['anion']['atom_ids'] for rec in data]
an_bond_ids  = [rec['anion']['bond_ids'] for rec in data]
an_edges     = [rec['anion']['edge_indices'] for rec in data]

# 熔点标签缩放 (Min-Max 缩放到 -1 到 1，以匹配论文描述)
min_mp = raw_mps.min()
max_mp = raw_mps.max()
# 论文中提到 y-values scaled between -1 and 1
# 使用 Min-Max Scaler 缩放到 [0, 1]
normalized_mps = (raw_mps - min_mp) / (max_mp - min_mp + EPS)
# 再缩放到 [-1, 1]
scaled_mps = 2 * normalized_mps - 1
labels = scaled_mps.reshape(-1, 1)


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
    # 熔点预测不需要温度，但作为占位符必须传入
    dummy_temperature = np.zeros((num_samples, 1), dtype=np.float32) 
    return {
            'cat_atom': cat_atom, # 注意：这里使用 viscosity 模型中的输入名
            'cat_bond': cat_bond,
            'cat_connectivity': cat_edge,
            'an_atom': an_atom,
            'an_bond': an_bond,
            'an_connectivity': an_edge,
            'temperature': dummy_temperature 
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
# 必须提供所有自定义对象
custom_objects_vis = {
    'combine_proj': combine_proj,
    'get_A': get_A, 'get_B': get_B, 'get_C': get_C, 
    'softplus_C': softplus_C,
    'compute_log_eta': compute_log_eta,
    'BondMatrixMessage': BondMatrixMessage,
    'Reduce': Reduce,
    'GatedUpdate': GatedUpdate,
    'GlobalSumPool': GlobalSumPool
}
# 加载完整粘度模型
try:
    vis_model = tf.keras.models.load_model(VIS_MODEL, compile=False, custom_objects=custom_objects_vis)
    print("Viscosity model loaded successfully!")
except Exception as e:
    print(f"Error loading viscosity model: {e}")
    print("Please ensure 'models/viscosity_final.keras' exists and models/layers.py is correctly set up.")
    exit()

# =========================
# 构建熔点迁移学习模型
# =========================
def build_transfer_model(vis_model):
    """
    基于粘度模型提取特征层，构建熔点预测模型。
    冻结从输入到 'combine' 层（Addition Layer）之前的所有层。
    """
    
    # 1. 找到 'combine' 层的输出
    try:
        combine_layer_output = vis_model.get_layer('combine').output
    except ValueError:
        print("Error: Could not find 'combine' layer in the viscosity model.")
        return None

    # 2. 创建一个新模型，只包含 GNN 和特征提取部分 (Base Model)
    # 输入层与 vis_model 的输入层完全相同
    base_model = Model(inputs=vis_model.inputs, outputs=combine_layer_output)
    
    # 3. 冻结 Base Model 的所有权重
    # 论文要求 "all weights from the basic model up until the addition layer were saved and set to non-trainable"
    base_model.trainable = False
    print("Base GNN/MPNN weights frozen.")

    # 4. 熔点预测头
    # 输入是 base_model 的输出，即 combined feature vector (mixing_size 维度)
    x = base_model.output
    # 可以在这里添加新的Dense层，作为新的可训练预测头
    x = Dense(64, activation='relu', name='mp_dense_1')(x)
    x = Dropout(0.2, name='mp_dropout')(x)
    
    # 最终输出层 (输出维度 1，activation='linear' 因为 y 值是缩放过的)
    mp_output = Dense(1, activation='linear', name='melting_point')(x)

    # 5. 组建最终模型
    transfer_model = Model(inputs=base_model.inputs, outputs=mp_output)
    return transfer_model

model = build_transfer_model(vis_model)
if model is None:
    exit()
model.summary()

# =========================
# 学习率调度
# =========================
# 论文提到 learning rate step decay with 0.55 factor drop every 100 epochs
def lr_schedule(epoch, lr):
    initial_lr = 0.01 # 论文中提到的初始学习率
    drop_factor = 0.55
    drop_epoch = 100
    
    # 使用自定义的初始学习率（在 Adam 优化器中设置）
    # 这里我们只负责衰减
    if epoch > 0 and epoch % drop_epoch == 0:
        return lr * drop_factor
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
# 论文中训练了 1000 个 epoch
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# 使用论文中提到的初始学习率 0.01
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
    loss='mse', 
    metrics=['mae'] 
)
print("Melting point transfer model compiled successfully!")

# =========================
# 训练
# =========================
history = model.fit(x_train, y_train,
                    validation_data=(x_dev, y_dev),
                    epochs=1000, # 使用论文中的 1000 epochs
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
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Dev Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Transfer Learning Training Loss Curve')
save_plot('results/melting_transfer_loss_curve.png')

y_pred_train = model.predict(x_train).flatten()
y_pred_dev = model.predict(x_dev).flatten()
plot_pred_vs_true(y_train.flatten(), y_pred_train, y_dev.flatten(), y_pred_dev,
                  'results/figure2_c_melting_transfer.png')
print("Saved prediction figure and loss curve!")

# 计算训练集指标
R2_train = r2_score(y_train.flatten(), y_pred_train)
MAE_train = mean_absolute_error(y_train.flatten(), y_pred_train)

# 计算开发集指标
R2_dev = r2_score(y_dev.flatten(), y_pred_dev)
MAE_dev = mean_absolute_error(y_dev.flatten(), y_pred_dev)

print("\n--- Final Transfer Model Performance ---")
print(f"R2_train: {R2_train:.4f}, MAE_train (Scaled MP): {MAE_train:.4f}")
print(f"R2_dev:   {R2_dev:.4f}, MAE_dev (Scaled MP):   {MAE_dev:.4f}")
print("----------------------------------------")