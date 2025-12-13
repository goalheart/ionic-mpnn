import pickle
from pathlib import Path

def convert_graph_to_ids(graph, atom_vocab, bond_vocab):
    """
    将单个分子图的原子/键特征转换为整数 ID 序列。
    """
    # 尝试将原子特征转换为 ID，如果特征不在字典中，则抛出 KeyError
    atom_ids = [atom_vocab[feat] for feat in graph['atom_features']]
    # 尝试将键特征转换为 ID，如果特征不在字典中，则抛出 KeyError
    bond_ids = [bond_vocab[feat] for feat in graph['bond_features']]
    
    edge_indices = graph['edge_indices']

    return {
        'atom_ids': atom_ids,
        'bond_ids': bond_ids,
        'edge_indices': edge_indices,
        'num_atoms': len(atom_ids)
    }


def process_dataset(data_file, vocab_file, output_file):
    """
    将整个数据集（包含阳离子/阴离子图）转换为整数 ID 形式并保存。
    同时记录被跳过的记录 ID。
    """
    # 加载数据
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    atom_vocab = vocab['atom_vocab']
    bond_vocab = vocab['bond_vocab']

    # 用于存储被跳过的记录信息
    skipped_records = []

    # 转换每条记录
    processed_data = []
    for rec in data:
        pair_id = rec['pair_id']
        try:
            # 尝试转换阳离子
            cat_id_graph = convert_graph_to_ids(rec['cation_graph'], atom_vocab, bond_vocab)
            # 尝试转换阴离子
            an_id_graph = convert_graph_to_ids(rec['anion_graph'], atom_vocab, bond_vocab)

            new_rec = {
                'pair_id': pair_id,
                'cation': cat_id_graph,
                'anion': an_id_graph
            }

            # 保留任务特定标签
            if 'log_eta' in rec:
                new_rec['T'] = rec['T']
                new_rec['log_eta'] = rec['log_eta']
            if 'mp' in rec:
                new_rec['mp'] = rec['mp']

            processed_data.append(new_rec)

        except KeyError as e:
            # 记录被跳过的记录 ID 和导致错误的特征
            skipped_records.append({
                'pair_id': pair_id,
                'missing_feature': str(e)
            })
            print(f"Warning: Feature not in vocab, skipping record {pair_id}: {e}")
            continue

    # 保存
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    print("-" * 50)
    print(f"Processed {len(processed_data)} records and saved to {output_file}")
    print(f"Total skipped records: {len(skipped_records)}")
    
    if skipped_records:
        print("\n--- SKIPPED RECORDS DETAILS ---")
        for record in skipped_records:
            print(f"  ID: {record['pair_id']}, Missing Feature: {record['missing_feature']}")
        print("-------------------------------")
        
    return skipped_records


def main():
    # 处理粘度数据
    print("--- Processing Viscosity Data ---")
    skipped_vis = process_dataset(
        data_file='./data/viscosity_graph_data.pkl',
        vocab_file='./data/vocab.pkl',
        output_file='./data/viscosity_id_data.pkl'
    )
    
    # 处理熔点数据
    print("\n--- Processing Melting Point Data ---")
    skipped_mp = process_dataset(
        data_file='./data/mp_graph_data.pkl',
        vocab_file='./data/vocab.pkl',
        output_file='./data/mp_id_data.pkl'
    )
    
    # 最终总结
    print("\n--- FINAL SUMMARY ---")
    print(f"Total Viscosity records skipped: {len(skipped_vis)}")
    print(f"Total Melting Point records skipped: {len(skipped_mp)}")
    print("---------------------")


if __name__ == '__main__':
    main()