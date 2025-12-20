# ============================================================
# train_melting_point_transfer.py
# Transfer learning for ionic liquid melting point prediction
# ============================================================

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from pathlib import Path
from absl import flags

# ============================================================
# 命令行参数
# ============================================================

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "viscosity_model_path",
    "models/viscosity_final.keras",
    "Path to pretrained viscosity model"
)
flags.DEFINE_float("lr_stage1", 1e-3, "LR for head training")
flags.DEFINE_float("lr_stage2", 1e-4, "LR for fine-tuning")

# ============================================================
# 自定义层
# ============================================================

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
    ComputeLogEta,
)

# ============================================================
# 工具函数
# ============================================================

from train_viscosity import (
    r2_numpy,
    pad_sequences_1d,
    preprocess_edges_and_bonds,
    SelectiveVerboseCallback,
)

# ============================================================
# 路径
# ============================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPS = 1e-6

# ============================================================
# 构建迁移学习模型（不训练）
# ============================================================

def build_transfer_model(viscosity_model_path):

    base_model = tf.keras.models.load_model(
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
            "ComputeLogEta": ComputeLogEta,
        },
        compile=False,
    )

    mixed_fp = base_model.get_layer("mix_cat_an").output
    inputs = base_model.inputs[:6]

    x = Dense(256, activation="relu", name="mp_dense_1")(mixed_fp)
    x = BatchNormalization(name="mp_bn_1")(x)
    x = Dense(128, activation="relu", name="mp_dense_2")(x)
    x = Dropout(0.3, name="mp_dropout")(x)
    x = Dense(64, activation="relu", name="mp_dense_3")(x)
    output = Dense(1, name="melting_point")(x)

    model = Model(inputs=inputs, outputs=output)
    return model

# ============================================================
# 主流程
# ============================================================

def main():

    # ---------------------------
    # 1. 数据加载
    # ---------------------------
    with open("data/melting_point_id_data.pkl", "rb") as f:
        data = pickle.load(f)

    cat_atoms = [[a + 1 for a in d["cation"]["atom_ids"]] for d in data]
    cat_bonds = [[b + 1 for b in d["cation"]["bond_ids"]] for d in data]
    cat_edges = [d["cation"]["edge_indices"] for d in data]

    an_atoms  = [[a + 1 for a in d["anion"]["atom_ids"]] for d in data]
    an_bonds  = [[b + 1 for b in d["anion"]["bond_ids"]] for d in data]
    an_edges  = [d["anion"]["edge_indices"] for d in data]

    y = np.array([d["mp"] for d in data], np.float32)

    # ---------------------------
    # 2. 数据划分
    # ---------------------------
    idx = np.arange(len(data))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.20, random_state=42)
    idx_dev, idx_test  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    # ---------------------------
    # 3. Padding
    # ---------------------------
    max_atoms = max(max(map(len, cat_atoms)), max(map(len, an_atoms)))
    max_edges = max(max(map(len, cat_edges)), max(map(len, an_edges)))

    def build_inputs(idxs):
        ce, cb = preprocess_edges_and_bonds(
            [cat_edges[i] for i in idxs],
            [cat_bonds[i] for i in idxs],
            max_edges,
        )
        ae, ab = preprocess_edges_and_bonds(
            [an_edges[i] for i in idxs],
            [an_bonds[i] for i in idxs],
            max_edges,
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

    y_train = y[idx_train]
    y_dev   = y[idx_dev]
    y_test  = y[idx_test]

    # ---------------------------
    # 4. Z-score 标准化
    # ---------------------------
    y_mean = y_train.mean()
    y_std  = y_train.std() + EPS

    scale = lambda y: (y - y_mean) / y_std
    inverse_scale = lambda y: y * y_std + y_mean

    y_train_s = scale(y_train)
    y_dev_s   = scale(y_dev)

    # ========================================================
    # Stage 1：冻结 base，仅训练 head
    # ========================================================

    model = build_transfer_model(FLAGS.viscosity_model_path)

    for layer in model.layers:
        if not layer.name.startswith("mp_") and layer.name != "melting_point":
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FLAGS.lr_stage1),
        loss=tf.keras.losses.Huber(delta=1.0),
    )

    history_stage1 = model.fit(
        x_train, y_train_s,
        validation_data=(x_dev, y_dev_s),
        epochs=1000,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=50, restore_best_weights=True),
            SelectiveVerboseCallback(1000),
        ],
        verbose=0,
    )

    # ========================================================
    # Stage 2：解冻部分 GNN 层
    # ========================================================

    UNFREEZE_KEYS = [
        "cat_bmm_2", "cat_bmm_3",
        "an_bmm_2", "an_bmm_3",
        "gated_update_2", "gated_update_3",
        "gated_update_6", "gated_update_7",
        "mix_cat_an",
    ]

    for layer in model.layers:
        if any(k in layer.name for k in UNFREEZE_KEYS):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FLAGS.lr_stage2),
        loss=tf.keras.losses.Huber(delta=1.0),
    )

    history_stage2 = model.fit(
        x_train, y_train_s,
        validation_data=(x_dev, y_dev_s),
        epochs=1000,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=50, restore_best_weights=True),
            SelectiveVerboseCallback(1000),
        ],
        verbose=0,
    )

    # ========================================================
    # 保存 history（用于 Notebook & 论文）
    # ========================================================

    with open(RESULTS_DIR / "melting_point_transfer_history.pkl", "wb") as f:
        pickle.dump(
            {
                "stage1": history_stage1.history,
                "stage2": history_stage2.history,
            },
            f,
        )

    # ========================================================
    # 评估
    # ========================================================

    print("\nFinal evaluation:")
    for name, x_, y_ in [
        ("Train", x_train, y_train),
        ("Dev",   x_dev,   y_dev),
        ("Test",  x_test,  y_test),
    ]:
        pred = inverse_scale(model.predict(x_, verbose=0).flatten())
        print(
            f"{name}: R2={r2_numpy(y_, pred):.4f}, "
            f"MAE={np.mean(np.abs(y_ - pred)):.2f}"
        )

    # ========================================================
    # 保存模型 + scaler
    # ========================================================

    model.save(MODELS_DIR / "melting_point_transfer_final.keras")

    with open(RESULTS_DIR / "melting_point_transfer_scaler.pkl", "wb") as f:
        pickle.dump(
            {
                "y_mean": y_mean,
                "y_std": y_std,
                "max_atoms": max_atoms,
                "max_edges": max_edges,
            },
            f,
        )

# ============================================================

if __name__ == "__main__":
    FLAGS(sys.argv)
    main()
