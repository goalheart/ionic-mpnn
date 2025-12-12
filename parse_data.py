# parse_data.py
"""
数据解析模块
将离子液体的SMILES字符串转换为图结构数据，并保存为pickle文件
包含粘度数据和熔点数据的处理
"""

import pickle
import os
import sys
from pathlib import Path

# 将 src/ 目录加入 Python 路径，确保可以导入 featurize 模块
sys.path.append(str(Path(__file__).parent / 'src'))

from featurize import smiles_to_graph


def load_ca_smiles(ca_file='data/CA.smi'):
    """
    CA.sim:   <SMILES>   <CID or AID>
    以 C 开头的是阳离子，A 开头的是阴离子；
    cation_smiles = {}  # e.g., 'C0582' -> 'CCN1C=C[N+]...'
    anion_smiles  = {}  # e.g., 'A0033' -> '[N-](=O)(=O)...'
    
    Args:
        ca_file (str): CA.smi文件路径
        
    Returns:
        tuple: (cation_smiles字典, anion_smiles字典)
               键为离子ID（如'C001'），值为对应的SMILES字符串
    """
    cation_smiles = {}
    anion_smiles = {}
    
    with open(ca_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: 跳过无效行 {line_num}: {line}")
                continue
                
            # SMILES可能包含空格，所以用除最后一个元素外的所有部分
            smi = ' '.join(parts[:-1])
            ion_id = parts[-1]
            
            # 根据ID前缀分类为阳离子或阴离子
            if ion_id.startswith('C'):
                cation_smiles[ion_id] = smi
            elif ion_id.startswith('A'):
                anion_smiles[ion_id] = smi
            else:
                print(f"Warning: 未知的离子ID在第{line_num}行: {ion_id}")
                
    return cation_smiles, anion_smiles


def parse_viscosity(vis_file='data/VISCOSITY.txt', cation_smiles=None, anion_smiles=None):
    """
    解析粘度数据文件
    将离子对 ID（如 C0979_A0048）转换为真实的 SMILES 字符串。

    Args:
        vis_file (str): 粘度数据文件路径
        cation_smiles (dict): 阳离子SMILES字典
        anion_smiles (dict): 阴离子SMILES字典
        
    Returns:
        list: 粘度数据记录列表，每个记录是包含以下字段的字典：
            - pair_id: 离子对ID（如'C001_A001'）
            - cation_smiles: 阳离子SMILES
            - anion_smiles: 阴离子SMILES  
            - T: 温度（单位：K）
            - log_eta: 粘度对数（log10(η/Pa·s)）
    """
    records = []
    
    with open(vis_file, 'r') as f:
        # 跳过标题行
        header = f.readline()
        
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
                
            # 解析数据行
            pair_id = parts[0]
            pressure = float(parts[1])      # 压力 (P)
            temperature = float(parts[2])   # 温度 (T)
            log_viscosity = float(parts[3]) # 粘度对数值 (log η)
            
            # 检查离子对ID格式
            if '_' not in pair_id:
                continue
                
            # 分割阳离子和阴离子ID
            cation_id, anion_id = pair_id.split('_', 1)
            
            # 验证离子ID是否存在于SMILES字典中
            if cation_id not in cation_smiles or anion_id not in anion_smiles:
                continue
                
            # 构建记录字典
            records.append({
                'pair_id': pair_id,
                'cation_smiles': cation_smiles[cation_id],
                'anion_smiles': anion_smiles[anion_id],
                'T': temperature,           # 温度
                'log_eta': log_viscosity    # 粘度对数值
            })
            
    return records


def parse_melting_point(mp_file='data/MP.txt', cation_smiles=None, anion_smiles=None):
    """
    解析熔点数据文件
    将离子对 ID（如 C0979_A0048）转换为真实的 SMILES 字符串。

    Args:
        mp_file (str): 熔点数据文件路径
        cation_smiles (dict): 阳离子SMILES字典
        anion_smiles (dict): 阴离子SMILES字典
        
    Returns:
        list: 熔点数据记录列表，每个记录是包含以下字段的字典：
            - pair_id: 离子对ID
            - cation_smiles: 阳离子SMILES
            - anion_smiles: 阴离子SMILES
            - mp: 熔点（单位：K）
    """
    records = []
    
    with open(mp_file, 'r') as f:
        # 跳过标题行
        header = f.readline()
        
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            # 解析数据行
            pair_id = parts[0]
            melting_point = float(parts[1])  # 熔点温度
            
            # 检查离子对ID格式
            if '_' not in pair_id:
                continue
                
            # 分割阳离子和阴离子ID
            cation_id, anion_id = pair_id.split('_', 1)
            
            # 验证离子ID是否存在于SMILES字典中
            if cation_id not in cation_smiles or anion_id not in anion_smiles:
                continue
                
            # 构建记录字典
            records.append({
                'pair_id': pair_id,
                'cation_smiles': cation_smiles[cation_id],
                'anion_smiles': anion_smiles[anion_id],
                'mp': melting_point  # 熔点
            })
            
    return records


def main():
    """
    主函数：执行数据解析和转换流程
    """
    print("开始加载CA.smi文件...")
    cation_smiles, anion_smiles = load_ca_smiles()
    print(f"加载完成: {len(cation_smiles)} 个阳离子, {len(anion_smiles)} 个阴离子")
    
    print("开始解析VISCOSITY.txt文件...")
    vis_data = parse_viscosity(cation_smiles=cation_smiles, anion_smiles=anion_smiles)
    print(f"解析完成: {len(vis_data)} 条粘度记录")
    
    print("开始解析MP.txt文件...")
    mp_data = parse_melting_point(cation_smiles=cation_smiles, anion_smiles=anion_smiles)
    print(f"解析完成: {len(mp_data)} 条熔点记录")
    
    # 将SMILES字符串转换为图结构数据
    print("开始将SMILES转换为图结构数据 (可能需要几分钟)...")
    
    # 处理粘度数据的图转换
    vis_graph_data = []
    for record in vis_data:
        try:
            # 转换阳离子和阴离子的SMILES为图结构
            cation_graph = smiles_to_graph(record['cation_smiles'])
            anion_graph = smiles_to_graph(record['anion_smiles'])
            
            vis_graph_data.append({
                'pair_id': record['pair_id'],
                'cation_graph': cation_graph,
                'anion_graph': anion_graph,
                'T': record['T'],           # 温度
                'log_eta': record['log_eta'] # 粘度对数值
            })
        except Exception as e:
            print(f"警告: 跳过无效的SMILES (粘度数据): {record['pair_id']} - 错误信息: {e}")
    
    # 处理熔点数据的图转换
    mp_graph_data = []
    for record in mp_data:
        try:
            # 转换阳离子和阴离子的SMILES为图结构
            cation_graph = smiles_to_graph(record['cation_smiles'])
            anion_graph = smiles_to_graph(record['anion_smiles'])
            
            mp_graph_data.append({
                'pair_id': record['pair_id'],
                'cation_graph': cation_graph,
                'anion_graph': anion_graph,
                'mp': record['mp']  # 熔点
            })
        except Exception as e:
            print(f"警告: 跳过无效的SMILES (熔点数据): {record['pair_id']} - 错误信息: {e}")
    
    # 保存转换后的图数据到pickle文件
    with open('data/viscosity_graph_data.pkl', 'wb') as f:
        pickle.dump(vis_graph_data, f)
    
    with open('data/mp_graph_data.pkl', 'wb') as f:
        pickle.dump(mp_graph_data, f)
    
    print("图结构数据已保存到以下文件:")
    print("   - data/viscosity_graph_data.pkl")
    print("   - data/mp_graph_data.pkl")
    print("数据处理完成!")


if __name__ == '__main__':
    main()