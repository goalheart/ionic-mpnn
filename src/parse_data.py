# src/parse_data.py
"""
数据解析模块
将离子液体的SMILES字符串转换为图结构数据，并保存为pickle文件
"""
import pickle
import sys
from pathlib import Path

# 确保可以导入 featurize 模块
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / 'src'))

from featurize import smiles_to_graph

# 定义数据路径
DATA_DIR = BASE_DIR / 'data'
CA_SMI_PATH = DATA_DIR / 'CA.smi'
VISCOSITY_PATH = DATA_DIR / 'VISCOSITY.txt'
MP_PATH = DATA_DIR / 'MP.txt'
VIS_GRAPH_OUTPUT = DATA_DIR / 'viscosity_graph_data.pkl'
MP_GRAPH_OUTPUT = DATA_DIR / 'mp_graph_data.pkl'


def load_ca_smiles(ca_file=CA_SMI_PATH):
    """
    加载阳离子和阴离子的 SMILES 字典
    """
    cation_smiles = {}
    anion_smiles = {}
    
    # ... (与 prepare_pairs.py 中的 load_ca_smiles 逻辑相同) ...
    try:
        with open(ca_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 2:
                    continue
                    
                smi = ' '.join(parts[:-1])
                ion_id = parts[-1]
                
                if ion_id.startswith('C'):
                    cation_smiles[ion_id] = smi
                elif ion_id.startswith('A'):
                    anion_smiles[ion_id] = smi
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到原始数据文件: {ca_file}")
        
    return cation_smiles, anion_smiles


def parse_viscosity(vis_file=VISCOSITY_PATH, cation_smiles=None, anion_smiles=None):
    """
    解析粘度数据文件，将 ID 转换为 SMILES
    """
    records = []
    
    # ... (与 prepare_pairs.py 中的 parse_viscosity 逻辑相同) ...
    try:
        with open(vis_file, 'r') as f:
            f.readline() # 跳过标题行
            
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                    
                pair_id = parts[0]
                # pressure = float(parts[1]) # 压力未被保留
                temperature = float(parts[2])
                log_viscosity = float(parts[3])
                
                if '_' not in pair_id:
                    continue
                    
                cation_id, anion_id = pair_id.split('_', 1)
                
                if cation_id not in cation_smiles or anion_id not in anion_smiles:
                    continue
                    
                records.append({
                    'pair_id': pair_id,
                    'cation_smiles': cation_smiles[cation_id],
                    'anion_smiles': anion_smiles[anion_id],
                    'T': temperature,
                    'log_eta': log_viscosity
                })
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到原始数据文件: {vis_file}")
        
    return records


def parse_melting_point(mp_file=MP_PATH, cation_smiles=None, anion_smiles=None):
    """
    解析熔点数据文件，将 ID 转换为 SMILES
    """
    records = []
    
    # ... (与 prepare_pairs.py 中的 parse_melting_point 逻辑相同) ...
    try:
        with open(mp_file, 'r') as f:
            f.readline() # 跳过标题行
            
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                pair_id = parts[0]
                melting_point = float(parts[1])
                
                if '_' not in pair_id:
                    continue
                    
                cation_id, anion_id = pair_id.split('_', 1)
                
                if cation_id not in cation_smiles or anion_id not in anion_smiles:
                    continue
                    
                records.append({
                    'pair_id': pair_id,
                    'cation_smiles': cation_smiles[cation_id],
                    'anion_smiles': anion_smiles[anion_id],
                    'mp': melting_point
                })
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到原始数据文件: {mp_file}")

    return records


def process_graph_conversion(data, task_label):
    """ 通用图转换函数，记录并返回被跳过的记录 """
    graph_data = []
    skipped_records = []

    for record in data:
        pair_id = record['pair_id']
        try:
            # 核心步骤：调用 featurize.py 转换图结构
            cation_graph = smiles_to_graph(record['cation_smiles'])
            anion_graph = smiles_to_graph(record['anion_smiles'])
            
            new_rec = {
                'pair_id': pair_id,
                'cation_graph': cation_graph,
                'anion_graph': anion_graph,
            }
            # 保留任务特定的标签
            if task_label == 'viscosity':
                new_rec.update({'T': record['T'], 'log_eta': record['log_eta']})
            elif task_label == 'mp':
                new_rec.update({'mp': record['mp']})
                
            graph_data.append(new_rec)
            
        except Exception as e:
            # RDKit 解析失败，记录并跳过
            skipped_records.append({
                'pair_id': pair_id,
                'cation_smi': record['cation_smiles'],
                'anion_smi': record['anion_smiles'],
                'error': str(e)
            })
            print(f"警告: 跳过无效的SMILES ({task_label}数据): {pair_id} - 错误信息: {e}")
    
    return graph_data, skipped_records


def print_skipped_summary(task_label, skipped_records):
    """ 打印跳过记录的详细总结 """
    print("\n" + "="*50)
    print(f"--- {task_label.upper()} 数据跳过记录总结 ---")
    if skipped_records:
        print(f"总计跳过 {len(skipped_records)} 条 {task_label} 记录。详细信息:")
        for rec in skipped_records:
            print(f"\n  ID: {rec['pair_id']}")
            print(f"    Cation SMILES: {rec['cation_smi']}")
            print(f"    Anion SMILES: {rec['anion_smi']}")
            print(f"    RDKit Error: {rec['error']}")
    else:
        print(f"{task_label} 数据未跳过任何记录。")
    print("="*50)


def main():
    try:
        print("开始加载CA.smi文件...")
        cation_smiles, anion_smiles = load_ca_smiles()
        print(f"加载完成: {len(cation_smiles)} 个阳离子, {len(anion_smiles)} 个阴离子")
        
        print("开始解析VISCOSITY.txt文件...")
        vis_data = parse_viscosity(cation_smiles=cation_smiles, anion_smiles=anion_smiles)
        print(f"解析完成: {len(vis_data)} 条粘度记录 (初始数量)") # 初始数量 7666
        
        print("开始解析MP.txt文件...")
        mp_data = parse_melting_point(cation_smiles=cation_smiles, anion_smiles=anion_smiles)
        print(f"解析完成: {len(mp_data)} 条熔点记录")
        
    except FileNotFoundError as e:
        print(f"致命错误: {e}")
        return

    # 粘度数据图转换
    print("\n开始将SMILES转换为图结构数据 (粘度)...")
    vis_graph_data, skipped_vis_records = process_graph_conversion(vis_data, 'viscosity')
    print_skipped_summary('粘度', skipped_vis_records)

    # 熔点数据图转换
    print("开始将SMILES转换为图结构数据 (熔点)...")
    mp_graph_data, skipped_mp_records = process_graph_conversion(mp_data, 'mp')
    print_skipped_summary('熔点', skipped_mp_records)
    
    # 保存转换后的图数据
    with open(VIS_GRAPH_OUTPUT, 'wb') as f:
        pickle.dump(vis_graph_data, f)
    
    with open(MP_GRAPH_OUTPUT, 'wb') as f:
        pickle.dump(mp_graph_data, f)
    
    print("\n--- 最终输出总结 ---")
    print(f"图结构数据已保存到 {VIS_GRAPH_OUTPUT}，有效记录数: {len(vis_graph_data)}")
    print(f"图结构数据已保存到 {MP_GRAPH_OUTPUT}，有效记录数: {len(mp_graph_data)}")
    print("----------------------")


if __name__ == '__main__':
    main()