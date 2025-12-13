# 生成一个包含离子液体（Ionic Liquids, ILs）化学结构、粘度数据和熔点数据的统一数据集合 pairs.csv
# scripts/prepare_pairs.py
# scripts一般存放数据处理、数据生成、实验运行和清理等辅助性脚本。
# pairs.csv：包含 cation (SMILES), anion (SMILES), T(温度), log_eta(log_{10}VISCOSITY), 熔点 (MP) 五个关键列

import pandas as pd

# 1. 加载和解析离子结构数据（CA.smi）
cation, anion = {}, {}
with open('./data/CA.smi') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            smi, iid = ' '.join(parts[:-1]), parts[-1]
            if iid.startswith('C'):
                cation[iid] = smi
            elif iid.startswith('A'):
                anion[iid] = smi

# 2. 加载和解析粘度数据（VISCOSITY.txt）
vis = []
with open('./data/VISCOSITY.txt') as f:
    next(f)
    for line in f:
        p_id, _, T, log_eta = line.strip().split()[:4]
        c, a = p_id.split('_')
        if c in cation and a in anion:
            vis.append([cation[c], anion[a], float(T), float(log_eta), None])

# 3. 加载和解析熔点数据（MP.txt）
mp_dict = {}
with open('./data/MP.txt') as f:
    next(f)
    for line in f:
        p_id, mp = line.strip().split()
        mp_dict[p_id] = float(mp)

# 4. 合并（以粘度为主，补充熔点）
records = []
for cat, an, T, log_eta, _ in vis:
    pair_id = None
    for k, v in cation.items():
        if v == cat:
            c_id = k
            break
    for k, v in anion.items():
        if v == an:
            a_id = k
            break
    pair_id = f"{c_id}_{a_id}"
    mp_val = mp_dict.get(pair_id, None)
    records.append([cat, an, T, log_eta, mp_val])

# 5. 保存
df = pd.DataFrame(records, columns=['cation', 'anion', 'T', 'log_eta', 'mp'])
df.to_csv('./data/pairs.csv', index=False)
print(f"Saved {len(df)} records to ./data/pairs.csv")