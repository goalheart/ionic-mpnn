# models/layers.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, GRU

tf_layers = tf.keras.layers

class Gather(tf_layers.Layer):
    def call(self, inputs, mask=None, **kwargs):
        reference, indices = inputs
        return tf.gather(reference, indices, batch_dims=1)


class Reduce(Layer):
    """
    根据 connectivity 中的目标索引（tgt_idx）将消息聚合到原子上。
    输入: [messages, tgt_indices, atom_features_ref]
      - messages: (B, E, D)        每条边的消息
      - tgt_indices: (B, E)        每条边的目标原子索引（从 0 开始，0 可能是 padding）
      - atom_features_ref: (B, N, D) 用于推断原子数量 N 和特征维度 D
    输出: (B, N, D) 聚合后的原子特征（按目标索引求和）
    """
    def __init__(self, reduction='sum', **kwargs):
        super().__init__(**kwargs)
        if reduction != 'sum':
            raise NotImplementedError("Only 'sum' reduction is supported.")
        self.reduction = reduction

    def call(self, inputs, mask=None):
        messages, tgt_indices, atom_features_ref = inputs
        batch_size = tf.shape(messages)[0]
        num_edges = tf.shape(messages)[1]
        atom_dim = tf.shape(messages)[2]
        num_atoms = tf.shape(atom_features_ref)[1]  # 从 reference 推断 N

        # 构造 scatter_nd 所需的索引: (B*E, 2)
        batch_idx = tf.range(batch_size, dtype=tf.int32)[:, None]  # (B, 1)
        batch_idx = tf.tile(batch_idx, [1, num_edges])             # (B, E)
        full_indices = tf.stack([batch_idx, tgt_indices], axis=-1) # (B, E, 2)
        full_indices = tf.reshape(full_indices, [-1, 2])           # (B*E, 2)

        # 展平消息: (B*E, D)
        messages_flat = tf.reshape(messages, [-1, atom_dim])

        # 使用 scatter_nd 聚合（自动对相同索引求和）
        aggregated = tf.scatter_nd(
            indices=full_indices,
            updates=messages_flat,
            shape=(batch_size, num_atoms, atom_dim)
        )

        return aggregated

    def compute_output_shape(self, input_shape):
        # input_shape: [msg_shape, idx_shape, ref_shape]
        msg_shape, _, ref_shape = input_shape
        # 输出形状与 atom_features_ref 相同
        return ref_shape

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config


class BondMatrixMessage(Layer):
    """
    实现论文中的消息计算：
        m_v^{t+1} = sum_{w in N(v)} A_{e_vw} h_w^t
    其中 A_e 由 bond embedding 通过可学习张量映射得到。
    """
    def __init__(self, atom_dim, bond_dim, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

    def build(self, input_shape):
        self.bond_transform = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer='glorot_uniform',
            name='bond_transform'
        )
        self.gather = Gather()
        super().build(input_shape)

    def call(self, inputs, mask=None):
        atom_state, bond_state, connectivity = inputs
        src_idx = connectivity[:, :, 0]  # (B, E)
        src_atoms = self.gather([atom_state, src_idx])  # (B, E, atom_dim)

        # (B, E, bond_dim) @ (bond_dim, atom_dim*atom_dim) -> (B, E, atom_dim*atom_dim)
        bond_weights = tf.matmul(bond_state, tf.reshape(self.bond_transform, [self.bond_dim, -1]))
        bond_weights = tf.reshape(bond_weights, [-1, tf.shape(bond_state)[1], self.atom_dim, self.atom_dim])

        src_exp = tf.expand_dims(src_atoms, -1)  # (B, E, atom_dim, 1)
        messages = tf.matmul(bond_weights, src_exp)  # (B, E, atom_dim, 1)
        messages = tf.squeeze(messages, axis=-1)  # (B, E, atom_dim)

        return messages

    def get_config(self):
        config = super().get_config()
        config.update({
            "atom_dim": self.atom_dim,
            "bond_dim": self.bond_dim,
        })
        return config


class GRUUpdate(Layer):
    """
    实现： h_v^{t+1} = GRU(h_v^t, m_v^{t+1})
    输入: [atom_state, messages, connectivity]
    输出: updated_atom_state
    """
    def __init__(self, atom_dim, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim

    def build(self, input_shape):
        self.gru = GRU(self.atom_dim, return_sequences=True, name='atom_gru')
        self.reduce = Reduce(reduction='sum')  # 用于聚合消息到节点
        super().build(input_shape)

    def call(self, inputs, mask=None):
        atom_state, messages, connectivity = inputs
        tgt_idx = connectivity[:, :, 1]  # (B, E)

        # 聚合消息到每个原子
        aggregated = self.reduce([messages, tgt_idx, atom_state], mask=mask)  # (B, N, atom_dim)

        # GRU 更新：注意 GRU 期望 (B*N, 1, D) 的输入
        batch_size = tf.shape(atom_state)[0]
        num_atoms = tf.shape(atom_state)[1]

        atom_flat = tf.reshape(atom_state, [batch_size * num_atoms, self.atom_dim])  # (B*N, D)
        msg_flat = tf.reshape(aggregated, [batch_size * num_atoms, 1, self.atom_dim])  # (B*N, 1, D)

        updated_flat = self.gru(msg_flat, initial_state=atom_flat)  # (B*N, 1, D)
        updated = tf.reshape(updated_flat, [batch_size, num_atoms, self.atom_dim])  # (B, N, D)

        return updated

    def get_config(self):
        config = super().get_config()
        config.update({"atom_dim": self.atom_dim})
        return config

class GlobalSumPool(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            atom_features, atom_ids = inputs
            mask = tf.cast(atom_ids > 0, tf.float32)  # 0 是 pad
            mask = tf.expand_dims(mask, -1)
            return tf.reduce_sum(atom_features * mask, axis=1)
        else:
            return tf.reduce_sum(inputs, axis=1)


class MessagePassing(Layer):
    def __init__(self, atom_dim, bond_dim, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

    def build(self, input_shape):
        self.bond_transform = self.add_weight(
            shape=(self.bond_dim, self.atom_dim, self.atom_dim),
            initializer='glorot_uniform',
            name='bond_transform'
        )
        self.gru = GRU(self.atom_dim, return_sequences=True)
        super().build(input_shape)

    def call(self, inputs):
        atom_features, bond_features, connectivity = inputs
        # atom_features: (B, N, D_a)
        # bond_features: (B, E, D_b)
        # connectivity: (B, E, 2)  [src, tgt]

        # Transform bonds to (D_a, D_a) matrices
        bond_weights = tf.einsum('bed,dlm->b elm', bond_features, self.bond_transform)  # (B, E, D_a, D_a)

        # Gather source atom features
        src_idx = connectivity[:, :, 0]  # (B, E)
        src_atoms = tf.gather(atom_features, src_idx, batch_dims=1)  # (B, E, D_a)

        # Compute messages: A_e @ h_w
        messages = tf.einsum('b elm,b ek->b el', bond_weights, src_atoms)  # (B, E, D_a)

        # Aggregate to target atoms
        tgt_idx = connectivity[:, :, 1]  # (B, E)
        num_atoms = tf.shape(atom_features)[1]
        batch_size = tf.shape(atom_features)[0]

        batch_indices = tf.range(batch_size, dtype=tf.int32)[:, None]  # (B, 1)
        batch_indices = tf.tile(batch_indices, [1, tf.shape(tgt_idx)[1]])  # (B, E)
        full_indices = tf.stack([batch_indices, tgt_idx], axis=-1)  # (B, E, 2)
        full_indices = tf.reshape(full_indices, [-1, 2])  # (B*E, 2)
        messages_flat = tf.reshape(messages, [-1, self.atom_dim])  # (B*E, D_a)

        aggregated = tf.scatter_nd(
            full_indices,
            messages_flat,
            (batch_size, num_atoms, self.atom_dim)
        )

        # Update with GRU
        updated = self.gru(tf.concat([atom_features, aggregated], axis=-1))
        return updated