# ============================================================
# train_melting_point_transfer.py
# 熔点预测（迁移学习）：冻结粘度模型 MPNN，复用结构指纹
# 对齐 train_viscosity.py 的工具函数、日志与输出风格
# ============================================================

import os
import sys  # 添加 sys 模块导入
# 屏蔽 TensorFlow 的 INFO 和 WARNING 日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from absl import flags 

# 定义你需要的命令行参数
FLAGS = flags.FLAGS
flags.DEFINE_string('viscosity_model_path', 'models/viscosity_final.keras', 'Path to the pre-trained viscosity model.')
flags.DEFINE_float('lr', 1e-3, 'Initial learning rate.')
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from pathlib import Path

# 导入自定义图神经网络层
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

BASE_DIR = Path(__file__).parent.parent        # 项目根目录
MODELS_DIR = BASE_DIR / "models" 

# 从粘度训练脚本中复用工具函数和构建逻辑（保持一致性）
from train_viscosity import (
    r2_numpy,
    pad_sequences_1d,
    preprocess_edges_and_bonds,
    SelectiveVerboseCallback,
    plot_loss
)

# 极小值，防除零
EPS = 1e-6

# ============================================================
# 迁移学习模型构建器
# ============================================================

def build_transfer_model(
    viscosity_model_path,
    freeze_base=True,
    dropout_rate=0.2  # 移除未使用的 mixing_dim 参数
):
    # 1. 加载预训练的粘度模型
    # 必须提供 custom_objects 字典，否则无法识别自定义层
    temp_model = keras.models.load_model(
        viscosity_model_path,
        custom_objects={
            "BondMatrixMessage": BondMatrixMessage,
            "GatedUpdate": GatedUpdate,
            "GlobalSumPool": GlobalSumPool,
            "Reduce": Reduce,
            "AddTwoTensors": AddTwoTensors,
            "SliceParamA": SliceParamA,
            "SliceParamB": SliceParamB,
            "SliceParamC": SliceParamC,
            "ScaleTemperature": ScaleTemperature,
            "ComputeLogEta": ComputeLogEta
        }
    )

    # 2. 提取特征混合层 (mix_cat_an)
    # 这一层输出了阴阳离子的混合指纹
    mixed_fp = temp_model.get_layer("mix_cat_an").output

    # 3. 获取正确的输入张量
    # 粘度模型有 7 个输入: [cat_atom, cat_bond, cat_conn, an_atom, an_bond, an_conn, temperature]
    # 但 'mix_cat_an' 层只用到了前 6 个（与结构相关的输入）。
    # 我们直接复用这些输入张量，千万不要自己新建 Input(...)。
    transfer_inputs = temp_model.inputs[:6]

    # 4. 构建特征提取器 (Base Model)
    # 这部分仅仅是截取了旧模型的一部分
    feature_extractor = Model(inputs=transfer_inputs, outputs=mixed_fp, name="feature_extractor")

    # 5. 冻结权重 (可选)
    if freeze_base:
        feature_extractor.trainable = False
    
    # 6. 添加熔点预测头 (New Head)
    # 使用截取出来的 output 作为新网络的起点
    x = feature_extractor.output
    
    # 添加新的全连接层用于熔点预测
    x = Dense(64, activation="relu", name="mp_dense_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = Dense(32, activation="relu", name="mp_dense_2")(x)
    
    # 输出层：熔点通常是正数，可以用 softplus 或直接 linear
    # 这里假设预测 Tm (K)，直接输出标量
    mp_output = Dense(1, name="melting_point")(x)

    # 7. 构建最终模型
    # 注意：inputs 必须是 feature_extractor.inputs (即 temp_model.inputs[:6])
    final_model = Model(inputs=feature_extractor.inputs, outputs=mp_output)

    return final_model

# ============================================================
# 主函数
# ============================================================

def main():
    # 数据与模型路径
    MP_DATA = "data/melting_point_id_data.pkl"
    VOCAB = "data/vocab.pkl"
    VISC_MODEL = FLAGS.viscosity_model_path  # 使用命令行参数指定的路径

    # 加载数据
    with open(MP_DATA, "rb") as f:
        data = pickle.load(f)
    with open(VOCAB, "rb") as f:
        vocab = pickle.load(f)

    atom_vocab_size = vocab["atom_vocab_size"] + 1  # +1 for padding (ID=0)
    bond_vocab_size = vocab["bond_vocab_size"] + 1

    # ========================================================
    # 提取输入特征（与粘度模型完全一致）
    # ========================================================

    cat_atoms = [[a + 1 for a in d["cation"]["atom_ids"]] for d in data]  # 避免 0
    cat_bonds = [[b + 1 for b in d["cation"]["bond_ids"]] for d in data]
    cat_edges = [d["cation"]["edge_indices"] for d in data]

    an_atoms  = [[a + 1 for a in d["anion"]["atom_ids"]] for d in data]
    an_bonds  = [[b + 1 for b in d["anion"]["bond_ids"]] for d in data]
    an_edges  = [d["anion"]["edge_indices"] for d in data]

    y = np.array([d["mp"] for d in data], np.float32)  # 目标：熔点（单位 K）

    # 划分数据集（随机划分，可替换为按 pair_id 的严格划分）
    indices = np.arange(len(data))
    idx_train, idx_tmp = train_test_split(indices, test_size=0.20, random_state=42)
    idx_dev, idx_test  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    print(
        f"数据划分完成: "
        f"Train={len(idx_train)}, Dev={len(idx_dev)}, Test={len(idx_test)}"
    )

    # 计算 padding 所需最大长度
    max_atoms = max(max(map(len, cat_atoms)), max(map(len, an_atoms)))
    max_edges = max(max(map(len, cat_edges)), max(map(len, an_edges)))

    def build_inputs(idxs):
        """构建模型输入字典（与粘度训练完全一致）"""
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

    x_train = build_inputs(idx_train)
    x_dev   = build_inputs(idx_dev)
    x_test  = build_inputs(idx_test)

    y_train, y_dev, y_test = y[idx_train], y[idx_dev], y[idx_test]

    # ========================================================
    # 标签缩放（论文要求：映射到 [-1, 1] 区间）
    # 注意：仅基于训练集计算 min/max
    # ========================================================

    y_min, y_max = y_train.min(), y_train.max()

    def scale(y):
        """线性缩放到 [-1, 1]"""
        return 2 * (y - y_min) / (y_max - y_min + EPS) - 1  # 添加 EPS 防除零

    def inverse_scale(y):
        """逆变换回原始尺度"""
        return (y + 1) * 0.5 * (y_max - y_min) + y_min

    y_train_s = scale(y_train)
    y_dev_s   = scale(y_dev)

    # ========================================================
    # 训练迁移模型
    # ========================================================

    # 修复：只传递必需的参数（移除了 atom_vocab_size 和 bond_vocab_size）
    model = build_transfer_model(VISC_MODEL)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.lr),
        loss=tf.keras.losses.MeanSquaredError(), # 熔点预测是回归任务，通常使用 MSE
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    # 保存模型（Keras v3 格式）
    model.save(MODELS_DIR / "melting_point_transferfinal.keras", save_format="keras_v3")

    total_epochs = 1000
    history = model.fit(
        x_train, y_train_s,
        validation_data=(x_dev, y_dev_s),
        epochs=total_epochs,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
            SelectiveVerboseCallback(total_epochs=total_epochs)  # 仅关键 epoch 打印
        ],
        verbose=0  # 关闭默认日志
    )

    # ========================================================
    # 保存训练 history（用于 Notebook 可视化）
    # ========================================================
    os.makedirs("results", exist_ok=True)

    with open("./results/history_melting_point_transfer.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print("Saved training history to results/history_melting_point_transfer.pkl")

    # 保存损失曲线
    plot_loss(history, "./results/loss_melting_point_transfer.png")

    # ========================================================
    # 评估（在原始尺度上，格式对齐粘度模型输出）
    # ========================================================

    for name, x_, y_ in [
        ("Train", x_train, y_train),
        ("Dev",   x_dev,   y_dev),
        ("Test",  x_test,  y_test)
    ]:
        # 预测 → 逆缩放 → 评估
        pred = inverse_scale(model.predict(x_, verbose=0).flatten())  # 添加 verbose=0
        print(
            f"{name}: R2={r2_numpy(y_, pred):.4f}, "
            f"MAE={np.mean(np.abs(y_ - pred)):.4f}"
        )

    # ========================================================
    # 绘制 Figure 2(c)：迁移学习熔点预测图
    # ========================================================

    y_train_pred = inverse_scale(model.predict(x_train, verbose=0).flatten())
    y_dev_pred   = inverse_scale(model.predict(x_dev, verbose=0).flatten())

    plt.figure(figsize=(5, 5))

    # 训练集：深绿色
    plt.scatter(
        y_train, y_train_pred,
        s=10, alpha=0.6,
        color="#2E7D32",
        label="Train"
    )

    # 验证集：浅绿色
    plt.scatter(
        y_dev, y_dev_pred,
        s=18, alpha=0.6,
        color="#A5D6A7",
        label="Validation"
    )

    # y = x 参考线
    low = min(y_train.min(), y_dev.min(), y_train_pred.min(), y_dev_pred.min())
    high = max(y_train.max(), y_dev.max(), y_train_pred.max(), y_dev_pred.max())
    plt.plot([low, high], [low, high], "k--", linewidth=1)

    plt.xlabel("Experimental melting point (K)")
    plt.ylabel("Predicted melting point (K)")
    plt.title("Melting point prediction (transfer learning)")

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("./results/figure2_c_melting_point_transfer.png", dpi=300)
    plt.close()

# ============================================================

if __name__ == "__main__":
    # 解析命令行参数
    FLAGS(sys.argv)  # 关键修复：在访问标志前解析参数
    main()