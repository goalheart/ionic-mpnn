# src/build_vocab.py
"""
1. 代码加载了两个 Pickle 文件：data/viscosity_graph_data.pkl（粘度相关的图数据）和 data/mp_graph_data.pkl（熔点相关的图数据）。

2. 最终生成的 vocab.pkl 文件是一个包含以下键的字典：
    'atom_vocab'： {原子特征元组: 整数ID}
    'bond_vocab'： {化学键特征元组: 整数ID}
    'atom_vocab_size'： 唯一原子特征的数量（词汇表大小）
    'bond_vocab_size'： 唯一化学键特征的数量（词汇表大小）

3. 目的分子的图结构数据转换为 GNN 的模型输入
"""

import pickle
from pathlib import Path
def build_vocab_from_graph_data(vis_file='data/viscosity_graph_data.pkl',
                                mp_file='data/mp_graph_data.pkl'):
    """
    从图数据中构建原子和化学键的词汇表（feature tuple → integer ID）
    
    Returns:
        dict: {
            'atom_vocab': {('C', 0, ...): 0, ...},
            'bond_vocab': {('SINGLE', ...): 0, ...},
            'atom_vocab_size': int,
            'bond_vocab_size': int
        }
    """
    # 收集所有唯一特征
    atom_set = set()
    bond_set = set()

    # 加载粘度数据
    with open(vis_file, 'rb') as f:
        vis_data = pickle.load(f)
    for rec in vis_data:
        atom_set.update(rec['cation_graph']['atom_features'])
        atom_set.update(rec['anion_graph']['atom_features'])
        bond_set.update(rec['cation_graph']['bond_features'])
        bond_set.update(rec['anion_graph']['bond_features'])

    # 加载熔点数据
    with open(mp_file, 'rb') as f:
        mp_data = pickle.load(f)
    for rec in mp_data:
        atom_set.update(rec['cation_graph']['atom_features'])
        atom_set.update(rec['anion_graph']['atom_features'])
        bond_set.update(rec['cation_graph']['bond_features'])
        bond_set.update(rec['anion_graph']['bond_features'])

    # 构建映射字典（按排序保证可复现）
    atom_vocab = {feat: idx for idx, feat in enumerate(sorted(atom_set))}
    bond_vocab = {feat: idx for idx, feat in enumerate(sorted(bond_set))}

    print(f"Vocabulary built:")
    print(f"   - Atom types: {len(atom_vocab)}")
    print(f"   - Bond types: {len(bond_vocab)}")

    vocab = {
        'atom_vocab': atom_vocab,
        'bond_vocab': bond_vocab,
        'atom_vocab_size': len(atom_vocab),
        'bond_vocab_size': len(bond_vocab)
    }

    # 保存词汇表
    vocab_path = Path('data/vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary to: {vocab_path}")

    return vocab


if __name__ == '__main__':
    build_vocab_from_graph_data()