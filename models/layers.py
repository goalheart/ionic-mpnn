# models/layers.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, GRU, GRUCell
tf_layers = tf.keras.layers

class Gather(tf_layers.Layer):
    def call(self, inputs, mask=None, **kwargs):
        reference, indices = inputs
        return tf.gather(reference, indices, batch_dims=1)


class Reduce(Layer):
    def __init__(self, reduction='sum', **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def call(self, inputs, mask=None):
        messages, tgt_indices, atom_features_ref = inputs
        # messages: (B, E, D)
        # tgt_indices: (B, E)
        
        batch_size = tf.shape(messages)[0]
        num_atoms = tf.shape(atom_features_ref)[1]
        atom_dim = tf.shape(messages)[2]

        # 展平 batch 维度以进行 segment sum
        # 我们需要为每个 batch 生成唯一的 segment id
        # segment_id = batch_index * max_atoms + atom_index
        
        # 这种偏移方法需要知道 max_atoms，动态获取：
        offset = tf.range(batch_size, dtype=tf.int32) * num_atoms
        offset = tf.expand_dims(offset, -1) # (B, 1)
        
        # 调整 indices: (B, E) + (B, 1) -> (B, E)
        # 此时 indices 中的值在 [0, B*N) 范围内全局唯一
        flat_indices = tgt_indices + offset
        
        flat_messages = tf.reshape(messages, [-1, atom_dim])
        flat_indices = tf.reshape(flat_indices, [-1])
        
        # 聚合
        total_atoms = batch_size * num_atoms
        aggregated_flat = tf.math.unsorted_segment_sum(
            flat_messages, flat_indices, num_segments=total_atoms
        )
        
        # 恢复形状 (B, N, D)
        aggregated = tf.reshape(aggregated_flat, [batch_size, num_atoms, atom_dim])
        
        return aggregated

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
    def __init__(self, atom_dim, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        # state_size 是 GRUCell 必需的属性
        self.state_size = atom_dim 

    def build(self, input_shape):
        self.gru_cell = GRUCell(self.atom_dim)
        self.reduce = Reduce(reduction='sum')
        super().build(input_shape)

    def call(self, inputs, mask=None):
        atom_state, messages, connectivity = inputs
        tgt_idx = connectivity[:, :, 1]

        # 1. 聚合消息 (B, N, D)
        aggregated_messages = self.reduce([messages, tgt_idx, atom_state], mask=mask)

        # 2. 展平以通过 Cell 处理 (B*N, D)
        batch_size = tf.shape(atom_state)[0]
        num_atoms = tf.shape(atom_state)[1]
        
        flat_inputs = tf.reshape(aggregated_messages, [-1, self.atom_dim])
        flat_states = tf.reshape(atom_state, [-1, self.atom_dim])
        
        # GRUCell 返回 (output, [new_state])
        # 对于 GRU，output == new_state
        new_flat_state, _ = self.gru_cell(flat_inputs, states=[flat_states])
        
        # 3. 恢复形状 (B, N, D)
        updated_state = tf.reshape(new_flat_state, [batch_size, num_atoms, self.atom_dim])
        
        return updated_state


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