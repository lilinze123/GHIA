import random
import numpy as np
import torch
import logging
import os
from copy import deepcopy
import json
import networkx as nx

import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid, CitationFull, KarateClub
from torch_geometric.utils import convert
import torch_geometric.transforms as T

from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph import utils as _utils


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


def load_perturbed_adj(dataset, attack, ptb_rate, path):        #加载扰动图
    assert dataset in ['cora', 'citeseer', 'pubmed', 'cora_ml']
    # assert attack in ['prbcd', 'greedy-rbcd', 'pga', 'apga', 'mettack', 'minmax', 'pgdattack', 'graD', 'greedy']
    assert os.path.exists(path)
    filename = attack + '-' + dataset + '-' + f'{ptb_rate}' + '.pth'
    filename = os.path.join(path, filename)
    return torch.load(filename)


def load_pretrained_model(dataset, model, path):        #加载模型，比如代理模型和受害模型
    assert dataset in ['cora', 'citeseer', 'pubmed', 'cora_ml']#断言判断数据集是否这几个
    # assert model in ['gcn', 'gat', 'sgc', 'rgcn', 'graph-sage', 'median-gcn', 'gcnsvd', 'gcn-jaccard', 'grand', 'gnn-guard', 'simp-gcn', 'dense-gcn']
    assert os.path.exists(path)#断言判断路径是否存在
    filename = model + '-' + dataset + '.pth'#获取权重文件名
    filename = os.path.join(path, filename)#权重文件完整路径
    assert os.path.exists(filename)#再次判断
    return torch.load(filename)#返回参数和状态信息


def gen_pseudo_label(model, labels, mask):         #获取伪标签
    device = labels.device
    model.eval()
    logit = model.predict()#模型输出的原始预测值或得分
    pred = logit.argmax(dim=1)#获取每一个样本的预测logits中最大值的索引，即预测分类
    labels[mask] = pred[mask].to(device)
    return labels


def evaluate_attack_performance(victim, x, mod_adj, labels, mask):      #计算准确率
    victim.eval()#转换为推理模式
    device = victim.device
    logit_detach = victim.predict(x.to(device), mod_adj.to(device))
    return _utils.accuracy(logit_detach[mask], labels[mask])


def load_data(name='cora', path='F:/pycode/node_injection_with_camouflage/datasets/', seed=15, x_normalize=True):#path为数据集存放路径
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
    data = dataset[0]       #dataset里面内容是怎么存放的
    data.num_classes = dataset.num_classes
    if name == 'karate_club':#该数据集没有验证集
        data.test_mask = ~(data.train_mask)
        data.val_mask = torch.zeros_like(data.test_mask, dtype=torch.bool)
    if x_normalize:
        data.x = normalize_feature_tensor(data.x)
    return data


def calc_modified_rate(clean_adj_t, mod_adj_t, n_edges):        #计算扰动率
    sp_clean = clean_adj_t.to_scipy(layout='coo')       #转换为稀疏矩阵，coo表示为坐标形式的稀疏矩阵，包括非零元素的行列索引和值
    sp_modified = mod_adj_t.to_scipy(layout='coo')
    diff = sp_clean - sp_modified
    n_diff = diff.getnnz()          #获取两矩阵差值矩阵的非零元素个数
    return float(n_diff) / n_edges


def check_undirected(mod_adj_t):        #检查是否无向图
    adj = mod_adj_t.to_dense()          #稀疏张量转换为密集张量
    adj_T = mod_adj_t.t().to_dense()
    is_symmetric = bool(torch.all(adj == adj_T))#判断是否对称
    assert is_symmetric is True


def normalize_feature_tensor(x):        #归一化特征张量
    x = _utils.to_scipy(x)
    x = _utils.normalize_feature(x)
    x = torch.FloatTensor(np.array(x.todense()))#归一化后，都为浮点数，所以转化为浮点数张量
    return x


def classification_margin(logits: torch.Tensor, labels: torch.Tensor):      #计算分类间隙
    probs = torch.exp(logits).cpu()#通过指数函数将logit转化为概率值
    label = labels.cpu()
    fill_zeros = torch.zeros_like(probs)
    true_probs = probs.gather(1, label.view(-1, 1)).flatten()#从probs获取每个样本的概率值最大的标签(或者真实标签)的预测概率
    probs.scatter_(1, label.view(-1, 1), fill_zeros)#将probs中每个样本的预测值最大标签(或者真实标签)的预测概率置为0
    best_wrong_probs = probs.max(dim=1)[0]#找到每个样本除了预测值最大标签(真实标签)之外的最大的标签预测概率，由于返回元组(value,index)所以取[0]
    return true_probs - best_wrong_probs

def calculate_entropy(logits: torch.Tensor):        #计算交叉熵损失
    return -(logits * logits.log()).sum(1)


def calculate_degree(adj_t):        #计算度
    assert type(adj_t) is torch.Tensor or type(adj_t) is SparseTensor
    if type(adj_t) is SparseTensor:
        return torch_sparse.sum(adj_t, dim=1).to_dense().cpu()
    # TODO: edge_index和dense_adj计算度


def kth_best_wrong_label(logits, labels, k=1):          #获得除真实标签之外预测概率最大的标签，且可以迭代获取第二大、第三大等等的标签
    logits = logits.exp()
    prev = deepcopy(labels).detach().cpu() #prev初始化为各样本的真实标签
    best_wrong_label = None
    while k > 0:
        fill_zeros = torch.zeros_like(logits) #创建一个形状与logits相同的零张量
        logits.scatter_(1, prev.view(-1, 1), fill_zeros)        #该函数将prev中所包含的标签作为下标，在logits的每一行进行索引，然后替换为0，view是扩展prev维度跟logit一样
        best_wrong_label = logits.argmax(1)
        prev = best_wrong_label #迭代寻找真实标签之外，置信度第二大、第三大等等的标签
        k = k - 1
    return best_wrong_label


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


###################### analysis
"""
计算graph中各个结点的统计量：
    degree
    degree_centrality
    pagerank
    clustering_coefficient
    eigenvector_centrality
    # betweenness_centrality 计算复杂度过大, 所以舍弃
"""
def calc_statistic_data(pyg_data, logits):      #计算各节点的统计特征

    G = convert.to_networkx(pyg_data)

    degrees = calculate_degree(pyg_data.adj_t)
    degree_centrality = torch.as_tensor(list(nx.degree_centrality(G).values()))
    pagerank = torch.as_tensor(list(nx.pagerank(G).values()))
    clustering_coefficient = torch.as_tensor(list(nx.clustering(G).values()))
    eigenvector_centrality = torch.as_tensor(list(nx.eigenvector_centrality(G).values()))
    cls_margin = classification_margin(logits, logits.argmax(1))

    return degrees, degree_centrality, pagerank, clustering_coefficient, eigenvector_centrality, cls_margin



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
    path = f"F:/pycode/PGA-main/my_results/{attack_type}"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f"{path}/{save_name}", "w") as outfile:
        outfile.write(json_object)