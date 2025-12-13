# ============================================================
# Melting Point Transfer Learning
# FINAL • ROBUST • PAPER-CONSISTENT VERSION
# 
# 实现一篇论文中描述的、利用已训练的粘度预测模型（viscosity_final.keras）的知识来改进熔点预测性能的实验
# 
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import tensorflow as tf
import train_viscosity # 假设这个文件定义了自定义层
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from models.layers import (
    BondMatrixMessage, GatedUpdate, GlobalSumPool, Reduce
)
# ============================================================
# Reproducibility
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)

EPS = 1e-6
T_PLACEHOLDER = 298.0  # 定义占位温度 (K)，用于满足模型结构要求

# ============================================================
# Utils
# ============================================================

def pad_sequences_1d(seq_list, max_len, pad_val=0):
    return np.array(
        [s + [pad_val] * (max_len - len(s)) for s in seq_list],
        dtype=np.int32
    )

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
# Load melting point dataset (list[dict])
# ============================================================

DATA_MP = "data/melting_point_id_data.pkl"

try:
    with open(DATA_MP, "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_MP}. Please ensure the file exists.")
    exit()

assert isinstance(data, list)
assert isinstance(data[0], dict)

# -------- Parse fields (STRICT) --------

cat_atoms = [[a + 1 for a in d["cation"]["atom_ids"]] for d in data]
cat_bonds = [[b + 1 for b in d["cation"]["bond_ids"]] for d in data]
cat_edges = [d["cation"]["edge_indices"] for d in data]

an_atoms  = [[a + 1 for a in d["anion"]["atom_ids"]] for d in data]
an_bonds  = [[b + 1 for b in d["anion"]["bond_ids"]] for d in data]
an_edges  = [d["anion"]["edge_indices"] for d in data]

raw_mp = np.array([d["mp"] for d in data], dtype=np.float32)
pair_ids = [d.get("pair_id", i) for i, d in enumerate(data)]

# ============================================================
# Label scaling (paper: [-1, 1])
# ============================================================

mp_min, mp_max = raw_mp.min(), raw_mp.max()
mp_norm = (raw_mp - mp_min) / (mp_max - mp_min + EPS)
y_all = (2.0 * mp_norm - 1.0).reshape(-1, 1)

# ============================================================
# SPLIT
# Default: sample-level random split (paper-consistent)
# ============================================================

indices = np.arange(len(data))
idx_train, idx_tmp = train_test_split(indices, test_size=0.30, random_state=42)
idx_dev, idx_test  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

# ============================================================
# Padding sizes
# ============================================================

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
    
    # *** 关键修改：生成占位温度数组 ***
    num_samples = len(idxs)
    temp_array = np.full((num_samples, 1), T_PLACEHOLDER, dtype=np.float32)

    return {
        "cat_atom": pad_sequences_1d([cat_atoms[i] for i in idxs], max_atoms),
        "cat_bond": cb,
        "cat_connectivity": ce,
        "an_atom": pad_sequences_1d([an_atoms[i] for i in idxs], max_atoms),
        "an_bond": ab,
        "an_connectivity": ae,
        "temperature": temp_array, # 添加入口
    }

x_train = build_inputs(idx_train)
x_dev   = build_inputs(idx_dev)
x_test  = build_inputs(idx_test)

y_train = y_all[idx_train]
y_dev   = y_all[idx_dev]
y_test  = y_all[idx_test]

# ============================================================
# Load pretrained viscosity model (SAFE MODE)
# ============================================================

# 确保 models/layers.py 中的自定义层可以被导入
try:
    from models.layers import BondMatrixMessage, Reduce, GatedUpdate, GlobalSumPool
except ImportError:
    # 假设你的环境已经正确配置了自定义层。
    class BondMatrixMessage(tf.keras.layers.Layer): pass
    class Reduce(tf.keras.layers.Layer): pass
    class GatedUpdate(tf.keras.layers.Layer): pass
    class GlobalSumPool(tf.keras.layers.Layer): pass


try:
    vis_model = tf.keras.models.load_model(
        "./models/viscosity_final.keras",
        compile=False,
        custom_objects={
            "BondMatrixMessage": BondMatrixMessage,
            "Reduce": Reduce,
            "GatedUpdate": GatedUpdate,
            "GlobalSumPool": GlobalSumPool,
        },
        safe_mode=False
    )
except FileNotFoundError:
    print("\nError: Pretrained viscosity model not found at ./models/viscosity_final.keras. Cannot proceed with transfer learning.")
    exit()

print("Viscosity model loaded")

# ============================================================
# Use viscosity model up to mixed fingerprint (robust)
# ============================================================

# Find the shared fingerprint layer (Lambda layer output)
cut_output = None
for layer in vis_model.layers:
    if layer.name == "lambda":
        cut_output = layer.output
        break
        
if cut_output is None:
    raise RuntimeError("Cannot locate shared fingerprint layer for model cut.")

base_model = Model(inputs=vis_model.input, outputs=cut_output)

# Freeze encoder
for layer in base_model.layers:
    layer.trainable = False

print("Encoder frozen")

# ============================================================
# Melting point head (paper-consistent)
# ============================================================

x = Dense(256, activation="relu", name="mp_dense_1")(base_model.output)
x = Dropout(0.2, name="mp_dropout")(x)
mp_out = Dense(1, name="mp_output")(x)

model = Model(inputs=base_model.input, outputs=mp_out, name="melting_transfer")

model.compile(
    optimizer=Adam(1e-3),
    loss="mse"
)

model.summary()

# ============================================================
# Train
# ============================================================

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

# ============================================================
# Evaluation
# ============================================================

# *** 修复: 将 inverse_scale 函数定义移到调用之前 ***
# 反向缩放函数，将 [-1, 1] 范围的预测值转换回 K
def inverse_scale(scaled_mp):
    mp_norm = (scaled_mp + 1.0) / 2.0
    return mp_norm * (mp_max - mp_min + EPS) + mp_min

def evaluate(name, x, y):
    # x 包含 'temperature'，可以进行预测
    pred = model.predict(x).flatten() 
    y_true = y.flatten()
    print(
        f"{name}: R2={r2_score(y_true, pred):.4f}, "
        f"MAE={mean_absolute_error(y_true, pred):.4f}"
    )
    return y_true, pred # 返回真实值和预测值 (缩放后)

print("\n--- Melting Transfer Performance ---")
y_tr_true_sc, y_tr_pred_sc = evaluate("Train", x_train, y_train)
y_de_true_sc, y_de_pred_sc = evaluate("Dev",   x_dev,   y_dev)
y_te_true_sc, y_te_pred_sc = evaluate("Test",  x_test,  y_test)
print("----------------------------------")

# ============================================================
# Save model & figure
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

model.save("models/melting_transfer_final.keras")

# --- 绘图代码 ---

# 1. 反向缩放预测值和真实值，以获得 K 单位的数值
ytr_true = inverse_scale(y_tr_true_sc)
ytr_pred = inverse_scale(y_tr_pred_sc)
ydv_true = inverse_scale(y_de_true_sc) # 使用 Dev 集作为 Validation 集
ydv_pred = inverse_scale(y_de_pred_sc)

# 2. 生成 Figure 2(c) 样式的图表
plt.figure(figsize=(5, 5))

plt.scatter(
    ytr_true,
    ytr_pred,
    s=10,
    alpha=0.6,
    color="#2E7D32", # 深绿色
    label="Train"
)

plt.scatter(
    ydv_true,
    ydv_pred,
    s=18,
    alpha=0.6,
    color="#A5D6A7", # 浅绿色
    label="Validation"
)

# 确定对角线 (y=x) 的范围
low = min(ytr_true.min(), ydv_true.min(), ytr_pred.min(), ydv_pred.min())
high = max(ytr_true.max(), ydv_true.max(), ytr_pred.max(), ydv_pred.max())
# 增加一点边距
margin = (high - low) * 0.05
low -= margin
high += margin

plt.plot([low, high], [low, high], "k--", linewidth=1)
plt.xlim(low, high)
plt.ylim(low, high)


plt.xlabel("Experimental melting point (K)")
plt.ylabel("Predicted melting point (K)")
plt.title("Melting point prediction (transfer learning)")

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./results/figure2_c_melting_point_transfer.png", dpi=300)
plt.close()

print("Saved Figure 2(c): ./results/figure2_c_melting_point_transfer.png")
print("Finished. Model saved.")