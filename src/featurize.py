# src/featurize.py
"""
分子特征化模块：将SMILES字符串转换为图结构表示
"""
from rdkit import Chem


def get_atom_features(atom):
    """
    提取原子特征
    """
    return (
        atom.GetSymbol(),            # 原子符号，如 'C', 'N', 'O' 等
        atom.GetFormalCharge(),      # 形式电荷，对离子重要
        atom.GetTotalNumHs(),        # 连接的氢原子数 (AddHs 后)
        int(atom.GetIsAromatic()),   # 是否为芳香原子
        str(atom.GetHybridization()) # 杂化类型
    )


def get_bond_features(bond):
    """
    提取化学键特征
    """
    return (
        str(bond.GetBondType()),  # 键类型，如 'SINGLE', 'DOUBLE', 'AROMATIC' 等
        bond.GetIsConjugated(),   # 是否为共轭键
        bond.IsInRing()           # 是否在环中
    )


def smiles_to_graph(smiles):
    """
    将SMILES字符串转换为图结构表示
    
    异常:
        ValueError: 当SMILES字符串无效时抛出
    """
    # 1. 从SMILES字符串创建RDKit分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES字符串: {smiles}")
    
    # 2. 添加氢原子
    mol = Chem.AddHs(mol)
    
    # 3. 提取所有原子特征
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    
    # 4. 提取键特征并构建边列表
    bond_features = []
    edge_indices = []
    
    for bond in mol.GetBonds():
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_feature = get_bond_features(bond)
        
        # 为无向图添加双向边和特征
        edge_indices.append((start_atom_idx, end_atom_idx))
        edge_indices.append((end_atom_idx, start_atom_idx))
        bond_features.append(bond_feature)
        bond_features.append(bond_feature)
    
    # 5. 构建返回的图结构字典
    graph_data = {
        'smiles': smiles,              
        'atom_features': atom_features,
        'bond_features': bond_features,
        'edge_indices': edge_indices,
        'num_atoms': len(atom_features)
    }
    
    return graph_data


if __name__ == '__main__':
    # 示例测试
    test_smiles = ["C[N+](C)(C)C", "CC(=O)[O-]", "c1ccccc1"]
    for smi in test_smiles:
        try:
            print(f"\n处理SMILES: {smi}")
            graph = smiles_to_graph(smi)
            print(f"  原子数: {graph['num_atoms']}")
        except ValueError as e:
            print(f"  错误: {e}")