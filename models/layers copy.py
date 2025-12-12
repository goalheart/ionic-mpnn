# models/layers.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, GRU

class GlobalSumPool(Layer):
    """全局求和池化层，用于将原子特征聚合为分子指纹"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            atom_features, atom_ids = inputs
            # atom_ids 用于掩码填充（ID=0 为 pad）
            mask = tf.cast(atom_ids > 0, tf.float32)
            mask = tf.expand_dims(mask, -1)
            return tf.reduce_sum(atom_features * mask, axis=1)
        else:
            return tf.reduce_sum(inputs, axis=1)

class MessagePassing(Layer):
    """
    消息传递层（论文 Section 4 公式）
    m_v^{t+1} = sum_{w in N(v)} A_{e_vw} h_w^t
    h_v^{t+1} = GRU(h_v^t, m_v^{t+1})
    """
    def __init__(self, atom_dim, bond_dim, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

    def build(self, input_shape):
        # 键变换矩阵：将 bond embedding 映射为 (atom_dim, atom_dim) 矩阵
        self.bond_transform = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer='glorot_uniform',
            name='bond_transform'
        )
        # GRU 用于更新原子状态
        self.gru = GRU(self.atom_dim, return_sequences=True, name='atom_gru')
        super().build(input_shape)

    def call(self, inputs):
        atom_features, bond_features, connectivity = inputs
        # atom_features: (B, N, D_a)
        # bond_features: (B, E, D_b)
        # connectivity: (B, E, 2)  [source, target]
        
        batch_size = tf.shape(atom_features)[0]
        num_atoms = tf.shape(atom_features)[1]
        
        # 1. 将 bond features 转换为权重矩阵
        bond_transform_reshaped = tf.reshape(self.bond_transform, 
                                            [self.bond_dim, self.atom_dim, self.atom_dim])
        bond_weights = tf.einsum('bed,dlm->belm', bond_features, bond_transform_reshaped)
        
        # 2. 获取源原子特征
        src_idx = connectivity[:, :, 0]  # (B, E)
        src_atoms = tf.gather(atom_features, src_idx, batch_dims=1)  # (B, E, D_a)
        
        # 3. 计算消息 m = A_e @ h_w
        messages = tf.einsum('belm,bel->bem', bond_weights, src_atoms)  # (B, E, D_a)
        
        # 4. 聚合到目标原子
        tgt_idx = connectivity[:, :, 1]  # (B, E)
        
        # 创建索引张量用于散射操作
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        batch_indices = tf.expand_dims(batch_indices, axis=-1)  # (B, 1)
        batch_indices = tf.tile(batch_indices, [1, tf.shape(tgt_idx)[1]])  # (B, E)
        
        # 准备散射所需索引
        indices = tf.stack([batch_indices, tgt_idx], axis=-1)  # (B, E, 2)
        indices = tf.reshape(indices, [-1, 2])  # (B*E, 2)
        
        # 准备散射所需消息
        messages_flat = tf.reshape(messages, [-1, self.atom_dim])  # (B*E, D_a)
        
        # 聚合消息
        aggregated = tf.zeros([batch_size, num_atoms, self.atom_dim], dtype=tf.float32)
        aggregated = tf.tensor_scatter_nd_add(aggregated, indices, messages_flat)
        
        # 5. 用 GRU 更新原子特征
        # 将所有原子特征和消息展平以便批量处理
        atom_flat = tf.reshape(atom_features, [-1, self.atom_dim])  # (B*N, D_a)
        msg_flat = tf.reshape(aggregated, [-1, 1, self.atom_dim])   # (B*N, 1, D_a)
        
        # GRU 更新 (需要扩展维度以符合GRU的时序要求)
        updated_flat = self.gru(msg_flat, initial_state=atom_flat)  # (B*N, 1, D_a)
        
        # 重塑回原始形状
        updated = tf.reshape(updated_flat, [batch_size, num_atoms, self.atom_dim])  # (B, N, D_a)
        
        return updated
    
    def compute_output_shape(self, input_shape):
        """显式定义输出形状，帮助Keras进行形状推断"""
        atom_features_shape = input_shape[0]
        return (atom_features_shape[0], atom_features_shape[1], self.atom_dim)