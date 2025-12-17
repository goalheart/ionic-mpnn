import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras.saving import register_keras_serializable

# ============================================================
# 粘度预测头（物理约束形式）
# log(η) = A + B / (T + C)
# ============================================================

@register_keras_serializable()
class ComputeLogEta(Layer):
    def call(self, inputs):
        A, B, T, C = inputs
        return A + B / (T + C + 1e-6)


@register_keras_serializable()
class ScaleTemperature(Layer):
    def call(self, t):
        return t / 100.0


@register_keras_serializable()
class SliceParamA(Layer):
    def call(self, x):
        return x[:, 0:1]


@register_keras_serializable()
class SliceParamB(Layer):
    def call(self, x):
        b = x[:, 1:2]
        b = tf.nn.softplus(b)
        return tf.clip_by_value(b, 0.0, 20.0)


@register_keras_serializable()
class SliceParamC(Layer):
    def call(self, x):
        c = x[:, 2:3]
        c = tf.nn.softplus(c)
        return tf.clip_by_value(c, 0.1, 50.0)


@register_keras_serializable()
class AddTwoTensors(Layer):
    def call(self, inputs):
        a, b = inputs
        return a + b


@register_keras_serializable()
class Reduce(Layer):
    """
    Aggregate messages to target atoms, ignoring tgt_idx == 0 (padding).
    """
    def call(self, inputs):
        messages, tgt_idx, atom_ref = inputs

        batch_size = tf.shape(messages)[0]
        num_edges = tf.shape(messages)[1]
        atom_dim = tf.shape(messages)[2]
        num_atoms = tf.shape(atom_ref)[1]

        batch_idx = tf.range(batch_size, dtype=tf.int32)[:, None]
        batch_idx = tf.tile(batch_idx, [1, num_edges])

        full_idx = tf.stack([batch_idx, tgt_idx], axis=-1)
        full_idx_flat = tf.reshape(full_idx, [-1, 2])

        messages_flat = tf.reshape(messages, [-1, atom_dim])
        tgt_flat = tf.reshape(tgt_idx, [-1])

        valid_mask = tgt_flat > 0
        valid_indices = tf.boolean_mask(full_idx_flat, valid_mask)
        valid_updates = tf.boolean_mask(messages_flat, valid_mask)

        aggregated = tf.scatter_nd(
            valid_indices,
            valid_updates,
            shape=(batch_size, num_atoms, atom_dim)
        )
        return aggregated


@register_keras_serializable()
class BondMatrixMessage(Layer):
    def __init__(self, atom_dim, bond_dim, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

    def build(self, input_shape):
        self.bond_transform = self.add_weight(
            shape=(self.bond_dim, self.atom_dim, self.atom_dim),
            initializer="glorot_uniform",
            name="bond_transform"
        )

    def call(self, inputs):
        atom_state, bond_state, connectivity = inputs

        src_idx = connectivity[:, :, 0]
        tgt_idx = connectivity[:, :, 1]

        src_atoms = tf.gather(atom_state, src_idx, batch_dims=1)

        bond_mats = tf.tensordot(bond_state, self.bond_transform, axes=[[2], [0]])

        src_exp = tf.expand_dims(src_atoms, -1)
        messages = tf.matmul(bond_mats, src_exp)
        messages = tf.squeeze(messages, axis=-1)

        valid = tf.logical_and(src_idx > 0, tgt_idx > 0)
        messages *= tf.cast(valid[..., None], tf.float32)

        return messages

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "atom_dim": self.atom_dim,
            "bond_dim": self.bond_dim
        })
        return cfg


@register_keras_serializable()
class GatedUpdate(Layer):
    def __init__(self, atom_dim, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense_z = Dense(self.atom_dim)
        self.dense_r = Dense(self.atom_dim)
        self.dense_h = Dense(self.atom_dim)
        self.layernorm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        atom_state, agg = inputs
        concat = tf.concat([atom_state, agg], axis=-1)

        z = tf.nn.sigmoid(self.dense_z(concat))
        r = tf.nn.sigmoid(self.dense_r(concat))

        r_state = r * atom_state
        h_input = tf.concat([r_state, agg], axis=-1)
        h_tilde = tf.nn.tanh(self.dense_h(h_input))

        new_state = (1 - z) * atom_state + z * h_tilde
        new_state = self.layernorm(new_state)
        new_state = new_state + atom_state
        return self.dropout(new_state, training=training)


@register_keras_serializable()
class GlobalSumPool(Layer):
    def call(self, inputs):
        atom_features, atom_ids = inputs
        mask = tf.cast(atom_ids > 0, tf.float32)[..., None]
        return tf.reduce_sum(atom_features * mask, axis=1)
