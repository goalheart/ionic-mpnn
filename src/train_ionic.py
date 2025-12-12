# src/train_ionic.py
import pickle
import tensorflow as tf
from rdkit import Chem
from nfp import MessagePassing, GlobalSumPool
from nfp.preprocessing import atom_features, bond_features

# === 1. SMILES → 图（整数 ID 列表）===
def smiles_to_inputs(smiles, atom_vocab, bond_vocab):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid: {smiles}")
    mol = Chem.AddHs(mol)

    atom_ids = [atom_vocab.get(atom_features(a), 0) for a in mol.GetAtoms()]
    bond_ids, edges = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bfeat = bond_features(bond)
        bid = bond_vocab.get(bfeat, 0)
        bond_ids.extend([bid, bid])
        edges.extend([[i, j], [j, i]])
    return atom_ids, bond_ids, edges

# === 2. 构建模型 ===
def build_model(atom_vocab_size, bond_vocab_size):
    # 输入（阳离子）
    cat_atom = tf.keras.Input(shape=(None,), dtype=tf.int32, name='cat_atom')
    cat_bond = tf.keras.Input(shape=(None,), dtype=tf.int32, name='cat_bond')
    cat_conn = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='cat_connectivity')
    # 输入（阴离子）
    an_atom = tf.keras.Input(shape=(None,), dtype=tf.int32, name='an_atom')
    an_bond = tf.keras.Input(shape=(None,), dtype=tf.int32, name='an_bond')
    an_conn = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='an_connectivity')
    T_input = tf.keras.Input(shape=(1,), name='temperature')

    # 共享嵌入
    atom_embed = tf.keras.layers.Embedding(atom_vocab_size, 32)
    bond_embed = tf.keras.layers.Embedding(bond_vocab_size, 8)

    # 共享 MPNN
    def mpnn(atom_in, bond_in, conn_in):
        atom_vec = atom_embed(atom_in)
        bond_vec = bond_embed(bond_in)
        for _ in range(4):
            atom_vec = MessagePassing()([atom_vec, bond_vec, conn_in])
        return GlobalSumPool()(atom_vec)

    cat_fp = mpnn(cat_atom, cat_bond, cat_conn)
    an_fp = mpnn(an_atom, an_bond, an_conn)

    # 融合
    cat_proj = tf.keras.layers.Dense(20, activation='relu')(cat_fp)
    an_proj = tf.keras.layers.Dense(20, activation='relu')(an_fp)
    combined = tf.keras.layers.Add()([cat_proj, an_proj])

    # 粘度头：log(η) = A + B / T
    params = tf.keras.layers.Dense(2)(combined)
    A, B = params[:, 0:1], params[:, 1:2]
    log_eta = A + B / T_input

    return tf.keras.Model(
        inputs=[cat_atom, cat_bond, cat_conn, an_atom, an_bond, an_conn, T_input],
        outputs=log_eta
    )

# === 3. 训练 ===
if __name__ == '__main__':
    # 加载数据
    with open('data/viscosity.pkl', 'rb') as f:
        data = pickle.load(f)

    # === 构建词汇表 ===
    atom_set, bond_set = set(), set()
    for rec in data:
        # 阳离子
        mol_cat = Chem.AddHs(Chem.MolFromSmiles(rec['cation_smiles']))
        for a in mol_cat.GetAtoms():
            atom_set.add(atom_features(a))
        for b in mol_cat.GetBonds():
            bond_set.add(bond_features(b))

        # 阴离子
        mol_an = Chem.AddHs(Chem.MolFromSmiles(rec['anion_smiles']))
        for a in mol_an.GetAtoms():
            atom_set.add(atom_features(a))
        for b in mol_an.GetBonds():
            bond_set.add(bond_features(b))

    atom_vocab = {f: i for i, f in enumerate(sorted(atom_set))}
    bond_vocab = {f: i for i, f in enumerate(sorted(bond_set))}

    # 构建模型
    model = build_model(len(atom_vocab), len(bond_vocab))
    model.compile(optimizer='adam', loss='mse')

    # === 准备 batch ===
    batch = [data[i] for i in range(min(4, len(data)))]  # 防止越界
    inputs = {
        'cat_atom': [], 'cat_bond': [], 'cat_connectivity': [],
        'an_atom': [], 'an_bond': [], 'an_connectivity': [],
        'temperature': [rec['T'] for rec in batch]
    }
    targets = [rec['log_eta'] for rec in batch]

    for rec in batch:
        # 阳离子
        a, b, e = smiles_to_inputs(rec['cation_smiles'], atom_vocab, bond_vocab)
        inputs['cat_atom'].append(a)
        inputs['cat_bond'].append(b)
        inputs['cat_connectivity'].append(e)
        # 阴离子
        a2, b2, e2 = smiles_to_inputs(rec['anion_smiles'], atom_vocab, bond_vocab)
        inputs['an_atom'].append(a2)
        inputs['an_bond'].append(b2)
        inputs['an_connectivity'].append(e2)

    # 用 nfp 的 pad_batch 处理变长序列
    from nfp.preprocessing import pad_batch
    inputs = pad_batch(inputs)

    # 训练
    model.fit(inputs, targets, epochs=10)