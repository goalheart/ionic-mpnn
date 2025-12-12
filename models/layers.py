# models/layers.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from tensorflow.keras.utils import register_keras_serializable
import tensorflow.keras.backend as K

class Reduce(Layer):
    """
    Aggregate messages to target atoms, ignoring tgt_idx == 0 (padding).
    Inputs:
      - messages: (B, E, D)
      - tgt_idx: (B, E)  (0 means padding)
      - atom_ref: (B, N, D)  (used to infer N)
    Output:
      - aggregated: (B, N, D)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        messages, tgt_idx, atom_ref = inputs
        batch_size = tf.shape(messages)[0]
        num_edges = tf.shape(messages)[1]
        atom_dim = tf.shape(messages)[2]
        num_atoms = tf.shape(atom_ref)[1]

        # flatten
        batch_idx = tf.range(batch_size, dtype=tf.int32)[:, None]  # (B,1)
        batch_idx = tf.tile(batch_idx, [1, num_edges])            # (B,E)
        full_idx = tf.stack([batch_idx, tgt_idx], axis=-1)        # (B,E,2)
        full_idx_flat = tf.reshape(full_idx, [-1, 2])             # (B*E, 2)

        messages_flat = tf.reshape(messages, [-1, atom_dim])     # (B*E, D)

        # mask out tgt == 0 (padding)
        tgt_flat = tf.reshape(tgt_idx, [-1])
        valid_mask = tf.where(tgt_flat > 0, True, False)

        valid_indices = tf.boolean_mask(full_idx_flat, valid_mask)
        valid_updates = tf.boolean_mask(messages_flat, valid_mask)

        aggregated = tf.scatter_nd(
            valid_indices,
            valid_updates,
            shape=(batch_size, num_atoms, atom_dim)
        )

        return aggregated

    def compute_output_shape(self, input_shape):
        _, _, ref_shape = input_shape
        return ref_shape
    
@register_keras_serializable()
class BondMatrixMessage(Layer):
    """
    Compute messages: for each edge, message = A_e @ h_src
    where A_e is obtained from bond embedding via a learned tensor.
    """
    def __init__(self, atom_dim, bond_dim, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

    def build(self, input_shape):
        # bond_transform: (bond_dim, atom_dim, atom_dim)
        self.bond_transform = self.add_weight(
            shape=(self.bond_dim, self.atom_dim, self.atom_dim),
            initializer='glorot_uniform',
            name='bond_transform'
        )
        super().build(input_shape)

    def call(self, inputs):
        atom_state, bond_state, connectivity = inputs
        # atom_state: (B, N, D_a)
        # bond_state: (B, E, D_b)
        # connectivity: (B, E, 2)  src,tgt ; src==0 or tgt==0 is padding

        src_idx = connectivity[:, :, 0]  # (B,E)
        tgt_idx = connectivity[:, :, 1]  # (B,E)

        # gather source atom features (src==0 -> gather atom embedding 0 -> zeros if embedding mask_zero=True)
        src_atoms = tf.gather(atom_state, src_idx, batch_dims=1)  # (B,E,D_a)

        # convert bond_state -> (B,E,D_a,D_a) using tensordot
        # bond_state: (B,E,D_b), bond_transform: (D_b, D_a, D_a)
        # result: (B,E,D_a,D_a)
        bond_mats = tf.tensordot(bond_state, self.bond_transform, axes=[[2], [0]])

        # messages = bond_mats @ src_atoms[...,None] -> (B,E,D_a,1) -> squeeze -> (B,E,D_a)
        src_exp = tf.expand_dims(src_atoms, -1)
        messages = tf.matmul(bond_mats, src_exp)
        messages = tf.squeeze(messages, axis=-1)

        # mask invalid edges (src==0 or tgt==0)
        valid = tf.logical_and(tf.greater(src_idx, 0), tf.greater(tgt_idx, 0))
        valid = tf.cast(valid, tf.float32)
        messages = messages * tf.expand_dims(valid, -1)

        return messages

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"atom_dim": self.atom_dim, "bond_dim": self.bond_dim})
        return cfg

class GatedUpdate(Layer):
    """
    Node update with gating similar to GRU but implemented in dense ops for efficiency.
    Inputs:
       - atom_state (B,N,D)
       - aggregated_messages (B,N,D)
    Output:
       - updated_state (B,N,D) with residual connection and LayerNorm
    """
    def __init__(self, atom_dim, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.atom_dim = atom_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # We'll use Dense layers for gates and candidate
        self.dense_z = Dense(self.atom_dim)
        self.dense_r = Dense(self.atom_dim)
        self.dense_h = Dense(self.atom_dim)
        self.layernorm = LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        atom_state, agg = inputs  # both (B,N,D)
        # compute gates
        concat = tf.concat([atom_state, agg], axis=-1)  # (B,N,2D)
        z = tf.nn.sigmoid(self.dense_z(concat))
        r = tf.nn.sigmoid(self.dense_r(concat))
        # candidate
        r_state = r * atom_state
        h_input = tf.concat([r_state, agg], axis=-1)
        h_tilde = tf.nn.tanh(self.dense_h(h_input))
        # update
        new_state = (1.0 - z) * atom_state + z * h_tilde
        new_state = self.layernorm(new_state)
        new_state = new_state + atom_state  # residual
        new_state = self.dropout(new_state, training=training)
        return new_state

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"atom_dim": self.atom_dim, "dropout_rate": self.dropout_rate})
        return cfg

class GlobalSumPool(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        # inputs can be [atom_features, atom_ids]
        if isinstance(inputs, (list, tuple)):
            atom_features, atom_ids = inputs
            mask = tf.cast(tf.greater(atom_ids, 0), tf.float32)  # (B,N)
            mask = tf.expand_dims(mask, -1)
            return tf.reduce_sum(atom_features * mask, axis=1)
        else:
            return tf.reduce_sum(inputs, axis=1)
