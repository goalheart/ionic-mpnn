### 环境配置

直接使用conda，而不用docker

```
# 在 shell 中正确设置代理export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export no_proxy=localhost,127.0.0.1,.local,.internal
# 创建环境（自动从官方源下载，走你的代理）
conda env create -f environment.yml

# 在 conda中设置代理（仅在当前终端会话有效）
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export no_proxy=localhost,127.0.0.1,.local,.internal
```

------

###  **ionic-mpnn 项目构建全流程（完整步骤）**

------

#### 项目结构：

```
ionic-mpnn/
│
├── data/                                  # 所有数据相关文件
│   ├── viscosity_id_data.pkl        			 # 训练粘度的 ID 数据
│   ├── mp_id_data.pkl               			 # 训练熔点的 ID 数据
│   └── vocab.pkl                    			 # 原子/键 词典
│
├── results/                               # 训练结果、可视化与指标
│   └── pred_vs_true                       # 判断模型对单个样本预测准不准
│
├── models/                                # 模型定义、保存的权重与参数
│   ├── layers.py                          # 自定义层（GRUUpdate, GlobalSumPool 等）
│   ├── bond_matrix_message.py             # 消息传递的 BondMatrix 实现
│   ├── mp_final.keras                     # 最终训练好的熔点模型
│   ├── viscosity_final.keras              # 最终训练好的粘度模型
│   └──  mp_norm_params.pkl                 # 熔点预测需要的标准化参数
│
├── src/                                   # 源代码（数据构建、特征工程）
│   ├── dataset.py                         # 数据加载、Dataset 构建（Numpy/Tensor 格式）
│   ├── featurize.py                       # SMILES → 图数据的特征提取
│   └── build_vocab.py                     # 构建 atom/bond vocab（可选）
│
├── scripts/                               # 工具脚本（预处理、生成文件等）
│   └── prepare_pairs.py                   # 将阳离子/阴离子组合成 training pairs
│
├── train_viscosity_from_pkl.py            # 主训练脚本：训练粘度预测 MPNN
├── train_mp_from_pkl.py                   # 主训练脚本：训练熔点预测 MPNN
├── parse_data.py                          # 原始数据 → pkl 格式的解析脚本
│
├── environment.yml                        # Conda 环境配置（依赖版本）
├── Dockerfile                             # Docker 部署环境
└── README.md                              # 项目说明文档（运行、训练、结构说明）

```

#### 1. **解析 `CA.smi` → 构建阳离子/阴离子 SMILES 字典**

**目的：**
 从 SMILES 文件中读取所有离子结构，并为每个离子分配唯一 ID。

输入文件：

- `CA.smi`（Cation-Anion）阳离子-阴离子对

------

#### 2. **对齐实验数据 (`VISCOSITY.txt`, `MP.txt`) → 得到最终样本对**

**目的：**
 将实验粘度/熔点数据中的 cation_id 与 anion_id 匹配到 SMILES。

最终构造训练样本：

- 粘度：`(smiles_c, smiles_a, T, log_eta)`
- 熔点：`(smiles_c, smiles_a, mp)`

关键步骤：

1. 根据离子-ID 匹配 SMILES
2. 温度转 float
3. 粘度取 log10（论文做法）
4. 移除缺失、重复、异常值

------

#### 3. **SMILES → Graph：用 RDKit 生成图结构特征（MPNN 输入）**

核心操作：

给定 SMILES，例如 `"C[N+](C)(C)CC"`, 输出：

##### **原子特征（atom_features）：**

每个原子 → 一个特征 tuple，例如：

```
('C', degree=3, valence=4, formal_charge=0, hybrid=SP3)
```

##### **键特征（bond_features）：**

每条键 → 一个特征 tuple，例如：

```
('SINGLE', conjugated=False, in_ring=True)
```

##### **边索引（edge_index）：**

用于 MPNN 消息传递：

```
[(0,1), (1,0), (1,2), (2,1), ...]
```

映射到最终输入：

```
atom_ids: [3, 6, 4, 12, ...]
bond_ids: [1, 2, 1, 3, ...]
edge_index: [(0,1), (1,0), ...]
```

保存到：

```
processed/viscosity_id_data.pkl
processed/melting_point_id_data.pkl
```

------

#### 4. **构建 Vocabulary（原子/键词典）**

**目的：**
 将可变长的特征 tuple 映射为固定整数 ID，以便进入 Keras Embedding 层。

例子：

原子元组：

```
('C', 3, 4, 0, 'SP3')
```

→ 转为整数 ID，例如：

```
12
```

键元组：

```
('SINGLE', False, True)
```

→ 转为：

```
4
```

输出：

```
processed/vocab.pkl
{
    'atom_vocab_size': 123,
    'bond_vocab_size': 71,
    'atom_to_id': {...},
    'bond_to_id': {...}
}
```

------

#### 5. **构建共享 MPNN 模型（NREL/nfp 风格）**

核心思想：

- 阳离子与阴离子共享嵌入层与 MPNN 层
- 对每个分子进行消息传递 → 生成 fingerprint
- cation_fp 与 anion_fp → 投影后相加
- 粘度预测采用 Arrhenius-like form：
  **log(η) = A + B / T**

模型组件：

1. Embedding(atom_vocab_size, atom_dim)
2. Embedding(bond_vocab_size, bond_dim)
3. BondMatrixMessage（消息计算）
4. GRUUpdate（节点状态更新）
5. GlobalSumPool（分子指纹）
6. Dense mixing
7. Head（粘度或熔点回归）

最终输出：

- 粘度：`log_eta_pred`
- 熔点：`mp_pred`

------

#### 6. **训练两个独立模型（或多任务）**

目前实现：

##### ✔ 粘度模型

运行：

```
python train_viscosity_from_pkl.py
```

输出：

- `models/viscosity_final.keras`
- 可视化：`results/viscosity/*.png`

##### ✔ 熔点模型

运行：

```
python train_mp_from_pkl.py
```

输出：

- `models/mp_final.keras`
- `models/mp_norm_params.pkl`
- 可视化：`results/mp/*.png`

训练技巧：

- Pair-level split（防止数据泄漏）
- 学习率调度
- EarlyStopping
- Dropout、L2 正则、梯度裁剪
- 数据 standardization（熔点）

------

#### 7. **评估 R² / MAE + 可视化性能图**

模型训练结束后：

##### ✔ 计算 R²（全验证集）

```
R² = 1 - sum((y_true - y_pred)²) / sum((y_true - mean)²)
```

##### ✔ 保存图到 results/：

| 图                  | Vis  | MP   |                                                              |
| ------------------- | :--: | ---- | ------------------------------------------------------------ |
| 1. Loss Curve       |      |      | 模型训练是否稳定？有没有过拟合？<br />**理想情况**：两条曲线都下降并趋于稳定，且差距不大。 |
| **2. Pred vs True** |      |      | 模型对单个样本预测准不准？<br />点越靠近 **对角线（y = x）**，说明预测越准； |
| 3. Residual Plot    |      |      | 模型有没有系统性误差？对大/小粘度是否表现不同？<br />**理想情况**：残差点在 y=0 附近 **均匀、无规律地随机分布** |
| 4. Residual Hist    |      |      | 误差是否随机？有没有异常模式？<br />**理想情况**：对称的钟形分布，峰值在 0 附近。 |
| 5. Dist Compare     |      |      | 模型是否还原了整体数据分布？<br />**理想情况**：两条分布曲线 **形状、位置、宽度高度重合** |

------

### **完整数据流概览（超清晰版）**

```
        raw SMILES                    raw experiments
          (CA.smi)          (VISCOSITY.txt / MP.txt)
             │                            │
             └──────────┬─────────────────┘
                        ↓
           构建阳离子/阴离子 SMILES 字典
                        ↓
              对齐实验数据（pair-level）
                        ↓
           生成 (cation_smiles, anion_smiles, T, y)
                        ↓
             RDKit → 图结构（atom/bond/edge）
                        ↓
           词典映射 → atom_ids / bond_ids
                        ↓
                保存 pkl（processed/）
                        ↓
            MPNN 输入（padding、embedding）
                        ↓
            MPNN + GRU 更新（共享权重）
                        ↓
             fingerprint_cat / fp_an
                        ↓
                多层融合（Dense）
                        ↓
       输出：log_eta_pred 或 mp_pred
                        ↓
         性能评估（R²、MAE）+ 可视化
                        ↓
                保存模型与结果
```

------



------

##  **Predicting Ionic Liquid Materials Properties from Chemical Structure**

------

## 1. 论文目的

**准确预测离子液体（ILs）的理化性质（**熔点**和**粘度**），基于其组成**阴离子和阳离子的化学结构**。

**重要性：** 离子液体是一类室温熔盐，在催化、能源存储（如电池电解质）等各种材料应用中具有巨大潜力。然而，合成上可行的离子液体超过 100 万种，但只有一小部分被合成和表征。

**目标：** 通过高通量筛选（high throughput screening）来预测尚未生产或表征的离子液体的性质，以指导材料选择并加速其在实际技术中的应用。

## 2. 创新点

- **改编神经指纹算法 (Neural-Fingerprinting Algorithm)：** 论文将一个已发布的神经指纹算法进行改编，使其能够接收**阴离子和阳离子的化学结构对**作为输入，而非单一有机分子。

- **基于 SMILES 字符串的端到端学习：** 算法输入是阴离子和阳离子结构的 SMILES 字符串（一种将分子结构编码为 ASCII 字符的标准方法），并使用**消息传递神经网络 (Message Passing Neural Networks, MPNNs)** 和后续的密集层来进行预测，实现了从化学结构到性质的**端到端（end-to-end）学习**。
  - **共享权重和嵌入：** 阴离子和阳离子共享相同的原子和键嵌入，并经过相同的消息传递算法层，且权重共享。这有助于在数据集有限的情况下限制可训练参数的数量，并确保特征提取的普适性。

- **预测粘度的温度依赖性：** 粘度预测模型使用一个更高级的网络头部，其最终输出是一个维度为 3 的向量，结合经验关系和内置 Keras 函数来**映射粘度随温度的变化**，从而充分利用整个数据集进行参数预测。

- **尝试迁移学习：** 研究尝试将粘度模型中学到的权重（直到加法层之前）转移到熔点预测模型中，以对抗熔点数据集不足的弱点。

## 3. 核心模型架构：**神经指纹架构(NFP)**

该论文的核心是其改编的**神经指纹架构**(NFP)，它属于**消息传递神经网络 (Message Passing Neural Networks, MPNNs)** 的范畴。

- **输入编码 (Input Encoding):**

  - 模型输入是阴离子和阳离子的 SMILES 字符串。

  - 每个字符串被转换为原子、键和连接信息，构建出分子图。

  - **原子状态**基于原子符号、电荷态、键合环境和芳香性等特征来区分。

  - **键状态**由键类型（单、双、三）、组成原子、共轭和是否在环中等决定。

  - 原子和键的特征向量通过 Xavier 初始化生成**嵌入 (embeddings)**。

- **共享与限制 (Shared Weights):**

  - 阴离子和阳离子**共享相同的原子和键嵌入**（即一个原子如果在阴离子和阳离子中都出现，它引用相同的嵌入向量）。
  - 两者也通过**共享权重**的相同消息传递算法层。
  - **目的：** 这种共享机制是为了在数据集有限的情况下限制可训练参数的数量，并假设预测相关的特征在阴离子和阳离子中应该是通用的。

- **消息传递过程 (The Message Passing Step):**

  - 在每个时间步 $t$，针对每个原子 $\nu$，通过以下公式计算消息 $m$：

    

    $$m _ {\nu} ^ {t + 1} = \sum_ {w \in N (\nu)} A _ {e _ {\nu w}} h _ {w} ^ {t}$$

    - $N(\nu)$ 是原子 $\nu$ 的所有相邻键合原子。
    - $A_{e_{\nu w}}$ 是对应于 $\nu$ 和其邻居 $w$ 之间键类型的**键嵌入矩阵**。
    - $h_{w}^t$ 是前一时间步计算出的原子 $w$ 的**原子特征嵌入**。

  - 随后，该消息 $m_{\nu}^{t+1}$ 被传递到一个门控循环层 (GRU) 中，以更新原子特征嵌入 $h_{\nu}^{t+1}$：

    

    $$h _ {\nu} ^ {t + 1} = G R U (h _ {\nu} ^ {t}, m _ {\nu} ^ {t + 1})$$

  - 论文中设置的消息传递步数 $M=4$。

- **分子指纹生成 (Molecular Fingerprint):**

  - 经过 $M$ 步消息传递后，得到的原子嵌入向量通过一个**密集回归层**，然后在**每个分子（阴离子和阳离子）的基础上求和**，生成一个可调大小的**分子指纹 (molecular fingerprint)**。

## 4. 预测头 (Prediction Heads)

模型在生成阴离子指纹 ($mf^{AN}$) 和阳离子指纹 ($mf^{CAT}$) 后，将它们通过独立的密集层，然后进行**元素求和 (element-wise summed)**，并将结果输入到特定性质的预测头。

##### A. 熔点预测头 (Melting Point)

- **结构：** 如图 1 中青绿色所示，最终输出维度为 1，直接作为预测值 $\hat{y}$。
- **损失函数：** 对已知值 $y$ 使用**均方误差 (Mean Squared Error, MSE)** 损失函数。
- **数据缩放：** 熔点的 $y$ 值被缩放到 -1 和 1 之间，以提高网络性能。

##### B. 粘度预测头 (Viscosity)

- **结构：** 如图 1 中金色所示，最终输出是一个**维度为 3 的向量**。
- **温度依赖性：** 该 3 维向量通过内置的 Keras 函数映射到**粘度** $\hat{\eta}$，从而引入了粘度与温度之间的**经验关系**。
- **数据缩放：** 由于粘度数据跨越数个数量级（1.34 到 7,079,457.8 cP），模型使用**粘度的对数** $\log_{10}(\eta)$ 作为带标签的 $y$ 值，以使大小数据点对损失函数的贡献更均衡。

## 5. 迁移学习的局限性

研究尝试将粘度模型中训练好的权重冻结后，用于预训练熔点模型，但性能反而降低。

- **原因：** 熔点数据集中约有 $1/6$ 的特征（原子和键类型）未出现在粘度数据集中。
- **影响：** 由于嵌入层在迁移学习中被设置为**不可训练**，这些未训练过的特征（仍是初始值）导致分子指纹计算不正确，从而损害了模型性能。主要结果



## 6. 结果讨论

- **粘度模型：** 展现了令人印象深刻的 $R^2$ 分数和紧密的预测分布，跨越了数据集的 8 个数量级。
- **熔点模型：** 尽管 $R^2$ 较低且存在显著方差，但已超过了其他在相同数据集上尝试的网络性能。作者认为方差可能是由于**熔点数据集仍然太小**，不足以进行适当的学习和泛化。
- **迁移学习：** 性能下降是由于熔点数据集中约 $\frac{1}{6}$ 的原子和键特征在粘度数据集中**不存在**，导致这些特征的嵌入向量从未被训练而保留了初始值。由于嵌入层在迁移学习中被设置为不可训练，分子指纹的计算受到影响。

该研究最终的预测模型取得了以下性能

| **性质** | **模型** | **$$R^2_{dev}$$** | **误差分析**                                                 |
| -------- | -------- | :---------------- | ------------------------------------------------------------ |
| **粘度** | 独立网络 | **0.89**          | 均方误差为 $$0.155 \log(\text{cP})$$，转化回原始单位，开发集误差为 $(-30.0\%, +42.9\%)$ |
| **熔点** | 独立网络 | **0.64**          | 结果显著优于其他在相同数据集上尝试的网络。                   |
| 熔点     | 迁移学习 | 性能更低          | **表观方差有所降低**，但性能下降是由于熔点数据集中约 $\frac{1}{6}$ 的特征在粘度数据集中**不存在**，且嵌入层被冻结。 |

但论文自己在 **Section 7** 明确承认：

> *“the full dataset was shuffled into train and dev sets, several of the anion/cation pairs were likely in both sets, albeit at differing temperatures.”*

也就是说：

- **同一个离子对 (cation, anion)**
- 在 **298K 在 train**
- 在 **350K 在 dev**
- 模型其实“见过这个分子结构”

这会 **显著抬高 R²_dev**。