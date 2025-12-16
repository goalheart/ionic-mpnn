# train_viscosity.py
# 训练粘度预测模型：基于离子对 SMILES 预测粘度
# 依据论文 "Predicting Ionic Liquid Materials Properties from Chemical Structure"
# ============================================================
# Viscosity prediction – FINAL STABLE PAPER-CONSISTENT VERSION
# ============================================================

import os
# 屏蔽 TensorFlow 的 INFO 和 WARNING 日志（只显示 ERROR）
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# 导入自定义图神经网络层（来自 models.layers）
from models.layers import (
    BondMatrixMessage, 
    GatedUpdate, 
    GlobalSumPool,
    Reduce, 
    AddTwoTensors, 
    SliceParamA, 
    SliceParamB, 
    SliceParamC,
    ScaleTemperature,
    ComputeLogEta
)

# 极小值，防止除零
EPS = 1e-6

# ============================================================
# 工具函数（Utils）
# ============================================================

def r2_numpy(y_true, y_pred):
    """
    手动计算 R² 决定系数（用于评估模型性能）
    """
    ss_res = np.sum((y_true - y_pred) ** 2)             # 残差平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)    # 总平方和
    return 1.0 - ss_res / (ss_tot + EPS)

def pad_sequences_1d(seq_list, max_len, pad_val=0):
    """
    对一维整数序列列表进行填充，使其长度统一为 max_len
    """
    return np.array(
        [s + [pad_val] * (max_len - len(s)) for s in seq_list],
        dtype=np.int32
    )

def plot_loss(history, out_path="loss_curve_viscosity.png"):
    """
    绘制训练/验证损失曲线并保存为图片
    """
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training curve (viscosity)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def preprocess_edges_and_bonds(edge_list, bond_list, max_edges):
    """
    对边和键类型列表进行预处理：
      - 为每条边添加反向边（使图无向化）
      - 统一填充或截断至 max_edges * 2 条边（正向+反向）
    返回：处理后的边索引数组和键ID数组
    """
    processed_edges, processed_bonds = [], []

    for edges, bonds in zip(edge_list, bond_list):
        e2, b2 = [], []
        for (src, tgt), bond_id in zip(edges, bonds):
            e2.append([src, tgt])   # 正向边
            b2.append(bond_id)
            e2.append([tgt, src])   # 反向边（复制键类型）
            b2.append(bond_id)
        processed_edges.append(e2)
        processed_bonds.append(b2)

    max_len = max_edges * 2  # 最大边数（含反向）

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

class SelectiveVerboseCallback(keras.callbacks.Callback):
    """
    自定义回调：仅在指定 epoch 打印训练日志（减少冗余输出）
    """
    def __init__(self, total_epochs, verbose_epochs=None):
        super().__init__()
        self.total_epochs = total_epochs
        if verbose_epochs is None:
            # 默认打印 epoch: 1~5, 50, 100, 150, 200 以及最后5个
            base = [1, 2, 3, 4, 5, 50, 100, 150, 200]
            last_five = list(range(total_epochs - 4, total_epochs + 1))
            self.verbose_epochs = set(base + last_five)
        else:
            self.verbose_epochs = set(verbose_epochs)

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1  # Keras 的 epoch 是从 0 开始的
        if current_epoch in self.verbose_epochs:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            print(f"Epoch {current_epoch}/{self.total_epochs} - loss: {loss:.6f} - val_loss: {val_loss:.6f}")


# ============================================================
# 模型构建（与论文一致的设置）
# ============================================================

def build_model(
    atom_vocab_size,      # 原子类型词表大小
    bond_vocab_size,      # 键类型词表大小
    atom_dim=32,          # 原子嵌入维度
    bond_dim=8,           # 键嵌入维度
    fp_size=32,           # 最终分子指纹维度
    mixing_size=20,       # 阴/阳离子混合层维度
    num_steps=4           # 图消息传递步数
):
    # ---------- 输入层定义 ----------
    # 阳离子（cation）输入
    cat_atom = Input(shape=(None,), dtype=tf.int32, name="cat_atom")
    cat_bond = Input(shape=(None,), dtype=tf.int32, name="cat_bond")
    cat_conn = Input(shape=(None, 2), dtype=tf.int32, name="cat_connectivity")

    # 阴离子（anion）输入
    an_atom = Input(shape=(None,), dtype=tf.int32, name="an_atom")
    an_bond = Input(shape=(None,), dtype=tf.int32, name="an_bond")
    an_conn = Input(shape=(None, 2), dtype=tf.int32, name="an_connectivity")

    # 温度输入（标量）
    T_input = Input(shape=(1,), dtype=tf.float32, name="temperature")

    # ---------- 嵌入层 ----------
    atom_emb_layer = Embedding(atom_vocab_size, atom_dim, mask_zero=False)
    bond_emb_layer = Embedding(bond_vocab_size, bond_dim, mask_zero=False)

    def encode(atom_ids, bond_ids, conn, prefix):
        """
        对单个离子（阴或阳）进行图神经网络编码
        返回：分子级嵌入（指纹）
        """
        atom_emb = atom_emb_layer(atom_ids)  # (batch, num_atoms, atom_dim)
        bond_emb = bond_emb_layer(bond_ids)  # (batch, num_bonds, bond_dim)
        h = atom_emb  # 初始节点表示

        # 多轮消息传递
        for i in range(num_steps):
            # 1. 消息生成：BondMatrixMessage 聚合邻居信息
            m = BondMatrixMessage(atom_dim, bond_dim, name=f"{prefix}_bmm_{i}")(
                [h, bond_emb, conn]
            )
            # 2. 聚合：将消息按目标节点聚合（Reduce 层）
            agg = Reduce(name=f"{prefix}_reduce_{i}")([m, conn[:, :, 1], h])
            # 3. 更新：门控更新节点表示
            h = GatedUpdate(atom_dim)([h, agg])

        # 全局池化：对所有原子求和得到分子指纹
        fp = GlobalSumPool()([h, atom_ids])
        # 非线性变换 + L2 正则
        fp = Dense(fp_size, activation="relu", kernel_regularizer=l2(1e-4))(fp)
        return fp

    # 分别编码阴、阳离子
    fp_cat = encode(cat_atom, cat_bond, cat_conn, "cat")
    fp_an  = encode(an_atom,  an_bond,  an_conn,  "an")

    # 投影到同一混合空间
    cat_proj = Dense(mixing_size, activation="relu")(fp_cat)
    an_proj  = Dense(mixing_size, activation="relu")(fp_an)

    # 论文使用 element-wise sum 混合离子表示
    mixed = AddTwoTensors(name="mix_cat_an")([cat_proj, an_proj])

    # 输出三个粘度参数 (A, B, C)
    visc_params = Dense(3)(mixed)
    A = SliceParamA(name="param_A")(visc_params)
    B = SliceParamB(name="param_B")(visc_params)
    C = SliceParamC(name="param_C")(visc_params)


    # 温度归一化（除以 100，使数值更稳定）
    T_scaled = ScaleTemperature(name="scale_T")(T_input)

    # 物理公式计算 log(η)
    log_eta = ComputeLogEta(name="log_eta")([A, B, T_scaled, C])

    # 构建完整模型
    model = Model(
        inputs=[
            cat_atom, cat_bond, cat_conn,
            an_atom, an_bond, an_conn,
            T_input
        ],
        outputs=log_eta
    )

    # 编译：使用 Adam 优化器 + 梯度裁剪 + MSE 损失
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss="mse"
    )
    return model

# ============================================================
# 主函数
# ============================================================

def main():
    # 数据路径
    DATA = "data/viscosity_id_data.pkl"
    VOCAB = "data/vocab.pkl"

    # 加载数据和词表
    with open(DATA, "rb") as f:
        data = pickle.load(f)
    with open(VOCAB, "rb") as f:
        vocab = pickle.load(f)

    atom_vocab_size = vocab["atom_vocab_size"] + 1  # +1 用于 padding (ID=0)
    bond_vocab_size = vocab["bond_vocab_size"] + 1

    # 提取数据字段
    pair_ids = [d["pair_id"] for d in data]  # 离子对唯一ID（用于防泄漏分割）

    # 阳离子原子、键、连接关系
    cat_atoms = [[a + 1 for a in d["cation"]["atom_ids"]] for d in data]  # +1 避免0（padding）
    cat_bonds = [[b + 1 for b in d["cation"]["bond_ids"]] for d in data]
    cat_edges = [d["cation"]["edge_indices"] for d in data]

    # 阴离子
    an_atoms  = [[a + 1 for a in d["anion"]["atom_ids"]] for d in data]
    an_bonds  = [[b + 1 for b in d["anion"]["bond_ids"]] for d in data]
    an_edges  = [d["anion"]["edge_indices"] for d in data]

    # 温度（T）和目标（log(粘度)）
    T = np.array([d["T"] for d in data], np.float32)[:, None]  # (N, 1)
    y = np.array([d["log_eta"] for d in data], np.float32)     # (N,)

    indices = np.arange(len(data))

    # ========================================================
    # 数据划分（默认：随机划分，可能有离子对泄漏）
    # ========================================================
    idx_train, idx_tmp = train_test_split(indices, test_size=0.20, random_state=42)
    idx_dev, idx_test  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    # --------- 严格无泄漏划分（按离子对ID划分）----------
    # 取消注释以下代码可启用
    # unique_pairs = np.unique(pair_ids)
    # p_train, p_tmp = train_test_split(unique_pairs, test_size=0.30, random_state=42)
    # p_dev, p_test  = train_test_split(p_tmp, test_size=0.50, random_state=42)
    # idx_train = [i for i, p in enumerate(pair_ids) if p in p_train]
    # idx_dev   = [i for i, p in enumerate(pair_ids) if p in p_dev]
    # idx_test  = [i for i, p in enumerate(pair_ids) if p in p_test]

    print(f"数据划分完成: Train={len(idx_train)}, Dev={len(idx_dev)}, Test={len(idx_test)}")

    # 计算最大原子数和边数（用于统一 padding）
    max_atoms = max(max(map(len, cat_atoms)), max(map(len, an_atoms)))
    max_edges = max(max(map(len, cat_edges)), max(map(len, an_edges)))

    def build_inputs(idxs):
        """
        根据索引构建模型输入字典
        """
        # 预处理边和键（添加反向边 + padding）
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
            "temperature": T[idxs]
        }

    # 构建训练/验证/测试集
    x_train = build_inputs(idx_train)
    x_dev   = build_inputs(idx_dev)
    x_test  = build_inputs(idx_test)

    y_train, y_dev, y_test = y[idx_train], y[idx_dev], y[idx_test]

    # 构建模型
    model = build_model(atom_vocab_size, bond_vocab_size)

    total_epochs = 1000
    # 训练模型
    history = model.fit(
        x_train, y_train,
        validation_data=(x_dev, y_dev),
        epochs=total_epochs,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
            SelectiveVerboseCallback(total_epochs=total_epochs)
        ],
        verbose=0  # 关闭 Keras 默认日志（由自定义回调控制）
    )
    
    # ========================================================
    # 保存训练 history（用于 Notebook 可视化）
    # ========================================================
    os.makedirs("results", exist_ok=True)

    with open("./results/history_viscosity.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print("Saved training history to results/history_viscosity.pkl")

    # ========================================================
    # 保存最终粘度模型（可用于迁移学习）
    # ========================================================
    os.makedirs("models", exist_ok=True)
    model.save("models/viscosity_final.keras", save_format="keras_v3")
    print("Saved viscosity model to models/viscosity_final.keras")

    # 绘制并保存损失曲线
    plot_loss(history, "./results/loss_viscosity.png")

    # 打印各集 R² 和 MAE
    for name, x_, y_ in [
        ("Train", x_train, y_train),
        ("Dev",   x_dev,   y_dev),
        ("Test",  x_test,  y_test)
    ]:
        pred = model.predict(x_).flatten()
        print(
            f"{name}: R2={r2_numpy(y_, pred):.4f}, "
            f"MAE={np.mean(np.abs(y_ - pred)):.4f}"
        )

    # ===============================
    # 绘制 Figure 2(a)：粘度预测 vs 实验值（训练+验证）
    # ===============================

    y_train_pred = model.predict(x_train).flatten()
    y_dev_pred   = model.predict(x_dev).flatten()

    plt.figure(figsize=(5, 5))

    # 训练点：深橘色
    plt.scatter(
        y_train,
        y_train_pred,
        s=10,
        alpha=0.6,
        color="#FF8B32",
        label="Train"
    )

    # 验证点：浅橘色（带透明度）
    plt.scatter(
        y_dev,
        y_dev_pred,
        s=18,
        alpha=0.6,
        color="#FFD582BE",
        label="Validation"
    )

    # y = x 参考线
    low = min(y_train.min(), y_dev.min(), y_train_pred.min(), y_dev_pred.min())
    high = max(y_train.max(), y_dev.max(), y_train_pred.max(), y_dev_pred.max())
    plt.plot([low, high], [low, high], "k--", linewidth=1)

    plt.xlabel("Experimental log(viscosity)")
    plt.ylabel("Predicted log(viscosity)")
    plt.title("Viscosity prediction (Figure 2a)")

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("./results/figure2_a_viscosity.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()