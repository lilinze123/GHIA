from deeprobust.graph.utils import accuracy
# from deeprobust.graph.defense import GCN
from torch_sparse import SparseTensor
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph import utils as _utils
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, KarateClub
from torch_geometric.utils import subgraph
# from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.data import Data
import torch
import numpy as np
import logging
import random
import os
import json


# def asr(time_, features, adj, modified_features, modified_adj, labels, idx_test, idx_train=None, idx_val=None, retrain=False):
#     gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
#               nhid=16, dropout=0.5, with_relu=False, with_bias=True, device='cpu').to('cpu')
#     gcn.load_state_dict(torch.load('ori_model/ori_sgc'+time_+'.pkl'))
#     output = gcn.predict(features,adj)
#     acc1 = np.float(accuracy(output[idx_test],labels[idx_test]))
#     if retrain:
#         gcn.fit(modified_features, modified_adj, labels, idx_train, idx_val, patience=30)
#     modified_output = gcn.predict(modified_features, modified_adj)
#     acc2 = np.float(accuracy(modified_output[idx_test],labels[idx_test]))
#
#     print('The accuracy before the attacks:', acc1)
#     print('The accuracy after the attacks:', acc2)
#     return acc1, acc2

# def evasion_test(node_injection, time_, features, adj, idx_train, idx_val, idx_test):
#     modified_features = node_injection.modified_features
#     modified_adj = node_injection.modified_adj
#     n_added_labels = node_injection.n_added_labels
#     acc1, acc2 = asr(time_, features, adj, modified_features, modified_adj, n_added_labels, idx_test, idx_train, idx_val)
#     return acc1, acc2
#
# def poisoning_test(node_injection, time_, features, adj, idx_train, idx_val, idx_test):
#     modified_features = node_injection.modified_features
#     modified_adj = node_injection.modified_adj
#     n_added_labels = node_injection.n_added_labels
#     acc1, acc2 = asr(time_, features, adj, modified_features, modified_adj, n_added_labels, idx_test,
#                      idx_train, idx_val, retrain=True)
#     return acc1, acc2


def scipy_sparse_to_edge_index(scipy_sparse):
    """将SciPy稀疏矩阵转换为COO格式的边索引"""
    coo = scipy_sparse.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)  # 形状 [2, num_edges]
    return edge_index

def scipy_sparse_to_pytorch_sparse(scipy_sparse):
    """将SciPy稀疏矩阵转换为PyTorch稀疏张量"""
    # 转换为COO格式并合并重复项
    scipy_sparse_coo = scipy_sparse.tocoo()
    scipy_sparse_coo.sum_duplicates()

    # 提取行、列索引和数据，并确保类型正确
    row = torch.from_numpy(scipy_sparse_coo.row.astype(np.int64))
    col = torch.from_numpy(scipy_sparse_coo.col.astype(np.int64))
    value = torch.from_numpy(scipy_sparse_coo.data.astype(np.float32))
    sparse_tensor = SparseTensor(
        row=row, col=col, value=value,
        sparse_sizes=scipy_sparse.shape
    )
    return sparse_tensor

def scipy_sparse_to_pytorch_dense(scipy_sparse):
    """将SciPy稀疏矩阵转换为PyTorch普通张量"""
    # 转换为稠密数组并调整数据类型
    dense_array = scipy_sparse.toarray().astype(np.float32)  # 确保类型一致
    # 转换为PyTorch张量
    dense_tensor = torch.from_numpy(dense_array)
    return dense_tensor

def normalize_feature_tensor(x):        #归一化特征张量
    x = _utils.to_scipy(x)
    x = _utils.normalize_feature(x)
    x = torch.FloatTensor(np.array(x.todense()))#归一化后，都为浮点数，所以转化为浮点数张量
    return x

def random_walk_sampling(data, num_nodes=8000, walk_length=40, walks_per_node=5):
    # 生成随机游走序列
    all_nodes = []
    for _ in range(walks_per_node):
        start_nodes = torch.randint(0, data.num_nodes, (data.num_nodes // 10,))  # 10%节点作为起点
        walks = []
        for start in start_nodes:
            walk = [start.item()]
            current = start
            for _ in range(walk_length - 1):
                neighbors = data.edge_index[1, data.edge_index[0] == current]
                if len(neighbors) > 0:
                    current = neighbors[torch.randint(0, len(neighbors), (1,))]
                    walk.append(current.item())
            walks.extend(walk)
        all_nodes.extend(list(set(walks)))  # 去重

    # 取前num_nodes个节点
    sampled_nodes = torch.tensor(list(set(all_nodes))[:num_nodes])

    # 构建子图
    sampled_edge_index, _ = subgraph(sampled_nodes, data.edge_index, relabel_nodes=True)
    sampled_data = Data(x=data.x[sampled_nodes],
                        edge_index=sampled_edge_index,
                        y=data.y[sampled_nodes])
    # 按类别选择训练节点
    train_mask = torch.zeros(sampled_data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    sampled_data.num_classes = data.num_classes
    y = sampled_data.y
    for c in range(data.num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx))[:20]]  # 随机选20
        train_mask[idx] = True

    # 剩余节点池（排除训练节点）
    remaining = (~train_mask).nonzero(as_tuple=True)[0]
    remaining = remaining[torch.randperm(len(remaining))]  # 随机打乱

    # 划分验证集和测试集
    val_mask[remaining[:500]] = True
    test_mask[remaining[500:1500]] = True
    sampled_data.train_mask = train_mask
    sampled_data.val_mask = val_mask
    sampled_data.test_mask = test_mask
    sampled_data.adj_t = T.ToSparseTensor()(sampled_data).adj_t
    sampled_data.edge_index = sampled_edge_index
    return sampled_data

def load_data(name='cora', path='D:/pycode/node_injection_with_camouflage/datasets/', seed=15, x_normalize=True):#path为数据集存放路径
    assert name in ['cora', 'citeseer', 'pubmed', 'cora_ml']
    # x_normalize = False if name == 'polblogs' else True,polblogs的数据特征不能归一化
    # freeze_seed(seed)
    if name in ['cora_ml']:
        dataset = Dataset(root=path, name=name, setting='gcn')#按照gcn论文的数据集设置
        dataset = Dpr2Pyg(dataset, transform=T.ToSparseTensor(remove_edge_index=False))
    elif name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name, transform=T.ToSparseTensor(remove_edge_index=False))
    elif name == 'karate_club':
        dataset = KarateClub(transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    data.num_classes = dataset.num_classes
    # if name == 'pubmed':
    #     data = random_walk_sampling(data)
    if name == 'karate_club':#该数据集没有验证集
        data.test_mask = ~(data.train_mask)
        data.val_mask = torch.zeros_like(data.test_mask, dtype=torch.bool)
    if x_normalize:
        data.x = normalize_feature_tensor(data.x)
    return data

def get_device(gpu_id):         #获取装置信息
    if torch.cuda.is_available() and gpu_id >= 0:
        device = f'cuda:{gpu_id}'
    else:
        device = 'cpu'
    return device

def freeze_seed(seed):          #设置种子数，以便还原实验结果
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pretrained_model(dataset, model, path):        #加载模型，比如代理模型和受害模型
    assert dataset in ['cora', 'citeseer', 'pubmed', 'cora_ml']#断言判断数据集是否这几个
    # assert model in ['gcn', 'gat', 'sgc', 'rgcn', 'graph-sage', 'median-gcn', 'gcnsvd', 'gcn-jaccard', 'grand', 'gnn-guard', 'simp-gcn', 'dense-gcn']
    assert os.path.exists(path)#断言判断路径是否存在
    filename = model + '-' + dataset + '.pth'#获取权重文件名
    filename = os.path.join(path, filename)#权重文件完整路径
    assert os.path.exists(filename)#再次判断
    return torch.load(filename)#返回参数和状态信息

def get_logger(filename, level=1, name=None):           #创建logger
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[level])

    fh = logging.FileHandler(filename, "w")     #文件处理器对象
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()            #流处理器对象
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def calc_modified_rate(clean_adj_t, mod_adj_t, n_edges):        #计算扰动率
    sp_clean = clean_adj_t.to_scipy(layout='coo')       #转换为稀疏矩阵，coo表示为坐标形式的稀疏矩阵，包括非零元素的行列索引和值
    sp_modified = mod_adj_t.to_scipy(layout='coo')
    diff = sp_clean - sp_modified
    n_diff = diff.getnnz()          #获取两矩阵差值矩阵的非零元素个数
    return float(n_diff) / n_edges

def evaluate_attack_performance(victim, x, mod_adj, labels, mask):      #计算准确率
    victim.eval()#转换为推理模式
    device = victim.device
    logit_detach = victim.predict(x.to(device), mod_adj.to(device))
    return _utils.accuracy(logit_detach[mask], labels[mask])

def save_result_to_json(attack, dataset, victim, ptb_rate, attacked_acc, attack_type):
    assert attack_type in ['evasion', 'poisoning']
    # Data to be written
    dictionary = {
        "attack": attack,
        "dataset": dataset,
        "victim": victim,
        "ptb_rate": ptb_rate,
        "attacked_acc": attacked_acc,
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    save_name = f"{attack}-{dataset}-{victim}-{ptb_rate}.json"
    # Writing to sample.json
    path = f"D:/pycode/node_injection_with_camouflage/results/{attack_type}"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f"{path}/{save_name}", "w") as outfile:
        outfile.write(json_object)