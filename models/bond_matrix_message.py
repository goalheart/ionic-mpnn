# models/bond_matrix_message.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from nfp.layers import Gather, Reduce

class BondMatrixMessage(Layer):
    """
    实现论文 Section 4 中的消息传递：
        m_v^{t+1} = sum_{w in N(v)} A_{e_vw} h_w^t
    其中 A_e 是由 bond embedding 通过可学习张量映射得到的 (atom_dim, atom_dim) 矩阵。

    输入:
        [atom_state, bond_state, connectivity]
        - atom_state: (B, N, atom_dim)
        - bond_state: (B, E, bond_dim)
        - connectivity: (B, E, 2)  # [src, tgt]

    输出:
        messages_aggregated: (B, N, atom_dim)
    """
    def __init__(self, atom_dim, bond_dim, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

    def build(self, input_shape):
        # 可学习张量：将 bond_dim 向量映射为 (atom_dim, atom_dim) 矩阵
        self.bond_transform = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer='glorot_uniform',
            name='bond_transform'
        )
        self.gather = Gather()
        self.reduce = Reduce(reduction='sum')
        super().build(input_shape)

    def call(self, inputs, mask=None):
        atom_state, bond_state, connectivity = inputs
        # 1. 获取源原子特征
        src_idx = connectivity[:, :, 0]  # (B, E)
        src_atoms = self.gather([atom_state, src_idx])  # (B, E, atom_dim)

        # 2. 将 bond_state 映射为 (B, E, atom_dim, atom_dim) 矩阵
        bond_mat_flat = tf.einsum('be,dlm->bedlm', bond_state, 
                                  tf.reshape(self.bond_transform, 
                                            [self.bond_dim, self.atom_dim, self.atom_dim]))
        # -> 但 einsum 不支持 5D，改用 reshape + matmul

        # 更高效实现：
        bond_weights = tf.matmul(
            bond_state,  # (B, E, bond_dim)
            tf.reshape(self.bond_transform, [self.bond_dim, -1])  # (bond_dim, atom_dim*atom_dim)
        )  # (B, E, atom_dim*atom_dim)
        bond_weights = tf.reshape(bond_weights, [-1, tf.shape(bond_state)[1], self.atom_dim, self.atom_dim])

        # 3. 计算消息: m = A_e @ h_w
        src_atoms_exp = tf.expand_dims(src_atoms, -1)  # (B, E, atom_dim, 1)
        messages = tf.matmul(bond_weights, src_atoms_exp)  # (B, E, atom_dim, 1)
        messages = tf.squeeze(messages, -1)  # (B, E, atom_dim)

        # 4. 聚合到目标原子
        tgt_idx = connectivity[:, :, 1]  # (B, E)
        # Reduce 自动处理 batch + mask，target_shape = atom_state.shape[1]
        aggregated = self.reduce([messages, tgt_idx, atom_state], mask=mask)
        return aggregated

    def get_config(self):
        config = super().get_config()
        config.update({
            "atom_dim": self.atom_dim,
            "bond_dim": self.bond_dim,
        })
        return config