# src/featurize.py
"""
分子特征化模块
将SMILES字符串转换为图结构表示，用于机器学习模型输入
"""

from rdkit import Chem


def get_atom_features(atom):
    """
    提取原子特征
    
    Args:
        atom: RDKit原子对象
        
    Returns:
        tuple: 包含原子特征的元组，包括：
            - 原子符号 (str)
            - 形式电荷 (int)
            - 连接的氢原子数 (int),(可选)
            - 是否为芳香原子 (int, 0或1)
            - 杂化类型 (str)
    """
    return (
        atom.GetSymbol(),            # 原子符号，如 'C', 'N', 'O' 等
        atom.GetFormalCharge(),      # 形式电荷，对离子重要
        atom.GetTotalNumHs(),        # 连接的氢原子数
        int(atom.GetIsAromatic()),   # 是否为芳香原子，转换为0或1
        str(atom.GetHybridization()) # 杂化类型，如 'SP3', 'SP2' 等
    )


def get_bond_features(bond):
    """
    提取化学键特征
    
    Args:
        bond: RDKit键对象
        
    Returns:
        tuple: 包含键特征的元组，包括：
            - 键类型 (str)
            - 是否为共轭键 (bool)
            - 是否在环中 (bool)
    """
    return (
        str(bond.GetBondType()),  # 键类型，如 'SINGLE', 'DOUBLE', 'AROMATIC' 等
        bond.GetIsConjugated(),   # 是否为共轭键
        bond.IsInRing()           # 是否在环中
    )


def smiles_to_graph(smiles):
    """
    将SMILES字符串转换为图结构表示
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        dict: 包含图结构信息的字典，包含以下键：
            - 'smiles': 原始SMILES字符串
            - 'atom_features': 原子特征列表，每个元素是get_atom_features的返回结果
            - 'bond_features': 键特征列表，对应每个边的特征
            - 'edge_indices': 边索引列表，每个元素是(起始原子索引, 终止原子索引)的元组
            - 'num_atoms': 原子总数
            
    异常:
        ValueError: 当SMILES字符串无效时抛出
    """
    # 1. 从SMILES字符串创建RDKit分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES字符串: {smiles}")
    
    # 2. 添加氢原子（对离子液体特别重要）
    #    注：对于离子，氢原子的存在可能影响电荷分布
    # 可选地，可以根据需要决定是否添加氢原子
    mol = Chem.AddHs(mol)
    
    # 3. 提取所有原子特征
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    
    # 4. 提取键特征并构建边列表
    bond_features = []
    edge_indices = []
    
    for bond in mol.GetBonds():
        # 获取键连接的两个原子的索引
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        
        # 获取键的特征
        bond_feature = get_bond_features(bond)
        
        # 为无向图添加双向边（i->j 和 j->i）
        # 这样在后续处理时可以更方便地进行消息传递
        edge_indices.append((start_atom_idx, end_atom_idx))
        edge_indices.append((end_atom_idx, start_atom_idx))
        
        # 对应的键特征也添加两次
        bond_features.append(bond_feature)
        bond_features.append(bond_feature)
    
    # 5. 构建返回的图结构字典
    graph_data = {
        'smiles': smiles,              # 原始SMILES字符串
        'atom_features': atom_features, # 原子特征列表
        'bond_features': bond_features, # 键特征列表
        'edge_indices': edge_indices,   # 边索引列表
        'num_atoms': len(atom_features) # 原子总数
    }
    
    return graph_data


# 示例使用和测试代码（可选）
if __name__ == '__main__':
    # 测试示例SMILES
    test_smiles = [
        "C[N+](C)(C)C",           # 四甲基铵阳离子
        "CC(=O)[O-]",             # 乙酸根阴离子
        "c1ccccc1",               # 苯（测试芳香性）
    ]
    
    for smi in test_smiles:
        try:
            print(f"\n处理SMILES: {smi}")
            graph = smiles_to_graph(smi)
            print(f"  原子数: {graph['num_atoms']}")
            print(f"  边数: {len(graph['edge_indices'])}")
            print(f"  第一个原子特征: {graph['atom_features'][0]}")
            if graph['bond_features']:
                print(f"  第一个键特征: {graph['bond_features'][0]}")
        except ValueError as e:
            print(f"  错误: {e}")