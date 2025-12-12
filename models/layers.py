# models/layers.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, GRU
from nfp.layers import Gather, Reduce

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