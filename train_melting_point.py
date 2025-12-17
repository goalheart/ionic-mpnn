# train_melting_point.py
# 训练熔点（Melting Point, MP）预测模型：基于离子对 SMILES 预测熔点
# ============================================================

import os
# 抑制 TensorFlow 的 INFO/WARNING 日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 禁用 oneDNN 优化（避免某些环境下的兼容性警告）
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 抑制 absl 日志（如 WARNING）
os.environ["ABSL_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from pathlib import Path

# 导入自定义图神经网络层（来自 models.layers）
from models.layers import BondMatrixMessage, GatedUpdate, GlobalSumPool, Reduce

# 极小值，防止除零
EPS = 1e-6

# ============================================================
# 路径配置（使用 pathlib 确保跨平台兼容性）
# ============================================================

BASE_DIR = Path(__file__).parent               # 项目根目录
DATA_DIR = BASE_DIR / "data"        # 数据目录
RESULTS_DIR = BASE_DIR / "results"  # 结果输出目录
MODELS_DIR = BASE_DIR / "models"    # 模型保存目录

MP_DATA_PATH = DATA_DIR / "mp_id_data.pkl"     # 熔点数据路径
VOCAB_PATH = DATA_DIR / "vocab.pkl"            # 原子/键词表路径

# 自动创建输出目录
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================
# 工具函数 Utils（与 train_viscosity.py 保持一致）
# ============================================================

def r2_numpy(y_true, y_pred):
    """计算 R² 决定系数（用于评估回归性能）"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + EPS)

def pad_sequences_1d(seq_list, max_len, pad_val=0):
    """对一维整数序列列表进行填充，使其长度统一"""
    return np.array(
        [s + [pad_val] * (max_len - len(s)) for s in seq_list],
        dtype=np.int32
    )

def preprocess_edges_and_bonds(edge_list, bond_list, max_edges):
    """
    对边和键类型进行预处理：
      - 添加反向边（使图为无向图）
      - 统一填充至 max_edges * 2（正向+反向）
    """
    processed_edges, processed_bonds = [], []

    for edges, bonds in zip(edge_list, bond_list):
        e2, b2 = [], []
        for (src, tgt), bond_id in zip(edges, bonds):
            e2.append([src, tgt])   # 正向
            b2.append(bond_id)
            e2.append([tgt, src])   # 反向（键类型相同）
            b2.append(bond_id)
        processed_edges.append(e2)
        processed_bonds.append(b2)

    max_len = max_edges * 2

    # 填充或截断
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

def plot_loss(history, out_path):
    """绘制训练/验证损失曲线并保存"""
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (scaled)")
    plt.title("Training curve (Melting Point)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

class SelectiveVerboseCallback(keras.callbacks.Callback):
    """
    自定义回调：仅在指定 epoch 打印日志（如 1~5, 50, 100... 和最后5个）
    """
    def __init__(self, total_epochs):
        super().__init__()
        base = [1, 2, 3, 4, 5, 50, 100, 150, 200]
        last = list(range(total_epochs - 4, total_epochs + 1))
        self.verbose_epochs = set(base + last)
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        e = epoch + 1  # Keras epoch 从 0 开始
        if e in self.verbose_epochs:
            print(
                f"Epoch {e}/{self.total_epochs} - "
                f"loss: {logs['loss']:.6f} - "
                f"val_loss: {logs['val_loss']:.6f}"
            )

# ============================================================
# 模型构建：用于熔点预测的 MPNN（Message Passing Neural Network）
# 注意：与粘度模型不同，此处为纯数据驱动，无物理公式约束
# ============================================================

def build_model(
    atom_vocab_size,      # 原子类型词表大小（+1 用于 padding）
    bond_vocab_size,      # 键类型词表大小
    atom_dim=32,          # 原子嵌入维度
    fp_size=32,           # 分子指纹维度
    mixing_size=20,       # 阴/阳离子混合投影维度
    num_steps=4           # 图消息传递步数
):
    # 键嵌入维度设为 atom_dim × atom_dim（因 BondMatrixMessage 需要矩阵形式）
    bond_matrix_dim = atom_dim * atom_dim

    # 嵌入层
    atom_emb = Embedding(atom_vocab_size, atom_dim, mask_zero=False)
    bond_emb = Embedding(bond_vocab_size, bond_matrix_dim, mask_zero=False)

    def encode(atom_ids, bond_ids, conn, prefix):
        """
        对单个离子（阴或阳）进行图消息传递编码
        返回：分子级表示（指纹）
        """
        h = atom_emb(atom_ids)  # 节点初始表示
        b = bond_emb(bond_ids)  # 键嵌入（矩阵形式）

        for i in range(num_steps):
            # 1. 消息生成：使用 BondMatrixMessage 生成消息矩阵
            m = BondMatrixMessage(
                atom_dim, bond_matrix_dim, name=f"{prefix}_bmm_{i}"
            )([h, b, conn])
            # 2. 聚合：按目标节点聚合消息
            agg = Reduce(name=f"{prefix}_reduce_{i}")([m, conn[:, :, 1], h])
            # 3. 更新：门控机制更新节点表示
            h = GatedUpdate(atom_dim)([h, agg])

        # 全局池化：对所有原子求和得到分子指纹
        fp = GlobalSumPool()([h, atom_ids])
        # 非线性 + L2 正则
        fp = Dense(fp_size, activation="relu", kernel_regularizer=l2(1e-5))(fp)
        return fp

    # ========== 输入层 ==========
    # 阳离子
    cat_atom = Input(shape=(None,), dtype=tf.int32, name="cat_atom")
    cat_bond = Input(shape=(None,), dtype=tf.int32, name="cat_bond")
    cat_conn = Input(shape=(None, 2), dtype=tf.int32, name="cat_connectivity")
    # 阴离子
    an_atom = Input(shape=(None,), dtype=tf.int32, name="an_atom")
    an_bond = Input(shape=(None,), dtype=tf.int32, name="an_bond")
    an_conn = Input(shape=(None, 2), dtype=tf.int32, name="an_connectivity")

    # 分别编码阴/阳离子
    fp_cat = encode(cat_atom, cat_bond, cat_conn, "cat")
    fp_an  = encode(an_atom,  an_bond,  an_conn,  "an")

    # 投影到混合空间并相加（与论文一致）
    mixed = Add()([
        Dense(mixing_size, activation="relu")(fp_cat),
        Dense(mixing_size, activation="relu")(fp_an)
    ])

    # 最终预测头：两层 MLP
    x = Dense(fp_size, activation="relu", kernel_regularizer=l2(1e-5))(mixed)
    out = Dense(1)(x)  # 单输出：熔点（标准化后）

    # 构建模型
    model = Model(
        inputs=[
            cat_atom, cat_bond, cat_conn,
            an_atom, an_bond, an_conn
        ],
        outputs=out,
        name="MeltingPoint_MPNN"
    )

    # 编译：Adam + 梯度裁剪 + MSE 损失
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss="mse"
    )
    return model

# ============================================================
# 主函数
# ============================================================

def main():
    # 加载数据和词表
    with open(MP_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    atom_vocab_size = vocab["atom_vocab_size"] + 1  # +1 用于 padding（ID=0）
    bond_vocab_size = vocab["bond_vocab_size"] + 1

    # 提取离子对 ID（可用于防泄漏分割）
    pair_ids = [d["pair_id"] for d in data]

    # 提取阳离子特征
    cat_atoms = [[a + 1 for a in d["cation"]["atom_ids"]] for d in data]  # 避免 0（padding）
    cat_bonds = [[b + 1 for b in d["cation"]["bond_ids"]] for d in data]
    cat_edges = [d["cation"]["edge_indices"] for d in data]

    # 提取阴离子特征
    an_atoms  = [[a + 1 for a in d["anion"]["atom_ids"]] for d in data]
    an_bonds  = [[b + 1 for b in d["anion"]["bond_ids"]] for d in data]
    an_edges  = [d["anion"]["edge_indices"] for d in data]

    # 目标值：熔点（mp）
    Y_all = np.array([d["mp"] for d in data], np.float32)

    # 随机划分索引（可替换为按 pair_id 的严格划分）
    indices = np.arange(len(data))
    idx_train, idx_tmp = train_test_split(indices, test_size=0.20, random_state=42)
    idx_dev, idx_test  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    print(f"数据划分完成: Train={len(idx_train)}, Dev={len(idx_dev)}, Test={len(idx_test)}")

    # ========== 标准化目标值（仅基于训练集）==========
    Y_mean = Y_all[idx_train].mean()
    Y_std  = Y_all[idx_train].std() or 1.0  # 防止 std=0

    Y_scaled = (Y_all - Y_mean) / Y_std  # 标准化

    # 计算最大长度用于 padding
    max_atoms = max(max(map(len, cat_atoms)), max(map(len, an_atoms)))
    max_edges = max(max(map(len, cat_edges)), max(map(len, an_edges)))

    def build_inputs(idxs):
        """根据索引构建模型输入字典（含边预处理和 padding）"""
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
        }

    # 构建训练/验证/测试输入
    x_train = build_inputs(idx_train)
    x_dev   = build_inputs(idx_dev)
    x_test  = build_inputs(idx_test)

    # 构建模型
    model = build_model(atom_vocab_size, bond_vocab_size)

    # 训练配置
    total_epochs = 1000
    history = model.fit(
        x_train, Y_scaled[idx_train],
        validation_data=(x_dev, Y_scaled[idx_dev]),
        epochs=total_epochs,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=50, restore_best_weights=True),
            SelectiveVerboseCallback(total_epochs)
        ],
        verbose=0  # 关闭默认日志（由自定义回调控制）
    )

    # ========================================================
    # 保存训练 history（用于 Notebook 可视化）
    # ========================================================
    os.makedirs("results", exist_ok=True)

    with open("./results/history_melting_point.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print("Saved training history to results/history_melting_point.pkl")

    # 保存模型
    model.save(MODELS_DIR / "melting_point_final.keras")
    
    # 保存损失曲线
    plot_loss(history, RESULTS_DIR / "loss_melting_point.png")

    # ========== 在原始尺度上评估 ==========
    print("\n--- Final Evaluation (original scale) ---")
    for name, x_, y_true in [
        ("Train", x_train, Y_all[idx_train]),
        ("Dev",   x_dev,   Y_all[idx_dev]),
        ("Test",  x_test,  Y_all[idx_test]),
    ]:
        # 预测 → 逆标准化
        pred = model.predict(x_).flatten() * Y_std + Y_mean
        print(
            f"{name}: R2={r2_numpy(y_true, pred):.4f}, "
            f"MAE={np.mean(np.abs(y_true - pred)):.4f}"
        )

if __name__ == "__main__":
    main()