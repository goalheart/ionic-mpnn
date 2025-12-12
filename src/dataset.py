# src/dataset.py
import pickle
from pathlib import Path

def convert_graph_to_ids(graph, atom_vocab, bond_vocab):
    """
    将单个分子图的原子/键特征转换为整数 ID 序列。

    Args:
        graph (dict): 包含 'atom_features', 'bond_features', 'edge_indices'
        atom_vocab (dict): {atom_feature_tuple: id}
        bond_vocab (dict): {bond_feature_tuple: id}

    Returns:
        dict: {
            'atom_ids': list[int],
            'bond_ids': list[int],
            'edge_indices': list[(int, int)],
            'num_atoms': int
        }
    """
    atom_ids = [atom_vocab[feat] for feat in graph['atom_features']]
    bond_ids = [bond_vocab[feat] for feat in graph['bond_features']]
    edge_indices = graph['edge_indices']  # already list of (i, j)

    return {
        'atom_ids': atom_ids,
        'bond_ids': bond_ids,
        'edge_indices': edge_indices,
        'num_atoms': len(atom_ids)
    }


def process_dataset(data_file, vocab_file, output_file):
    """
    将整个数据集（包含阳离子/阴离子图）转换为整数 ID 形式并保存。

    Args:
        data_file (str): 输入图数据路径（如 'data/viscosity_graph_data.pkl'）
        vocab_file (str): 词汇表路径（'data/vocab.pkl'）
        output_file (str): 输出 ID 数据路径
    """
    # 加载数据
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    atom_vocab = vocab['atom_vocab']
    bond_vocab = vocab['bond_vocab']

    # 转换每条记录
    processed_data = []
    for rec in data:
        try:
            cat_id_graph = convert_graph_to_ids(rec['cation_graph'], atom_vocab, bond_vocab)
            an_id_graph = convert_graph_to_ids(rec['anion_graph'], atom_vocab, bond_vocab)

            new_rec = {
                'pair_id': rec['pair_id'],
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
            print(f"Warning: Feature not in vocab, skipping record {rec['pair_id']}: {e}")
            continue

    # 保存
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Processed {len(processed_data)} records and saved to {output_file}")


def main():
    # 处理粘度数据
    process_dataset(
        data_file='data/viscosity_graph_data.pkl',
        vocab_file='data/vocab.pkl',
        output_file='data/viscosity_id_data.pkl'
    )

    # 处理熔点数据
    process_dataset(
        data_file='data/mp_graph_data.pkl',
        vocab_file='data/vocab.pkl',
        output_file='data/mp_id_data.pkl'
    )


if __name__ == '__main__':
    main()