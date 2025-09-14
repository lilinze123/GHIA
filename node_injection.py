import torch
import torch_sparse
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import coalesce
from torch_geometric.utils import k_hop_subgraph, coalesce as pyg_coalesce
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from copy import deepcopy
import scipy.sparse as sp
from utils import *
import os
import random

# from utils import evasion_test


class NI:
    def __init__(self, features, adj, labels, num_nodes, edge_index, surrogate, loss_type, zeroone_features):
        self.device = 'cuda:0'
        self.surrogate = surrogate
        self.surrogate.eval()
        self.sur_logits = self.surrogate.predict()
        self.features = features.clone().detach().to(self.device)
        self.modified_features = deepcopy(features)
        self.adj = adj.clone()
        self.modified_adj = deepcopy(adj)
        self.edge_index = deepcopy(edge_index).to(self.device)
        self.labels = labels.clone()
        self.n_added_labels = [] #该参数为添加了注入节点类别的类别集合的初始化
        self.classes = torch.unique(labels).tolist() #先转换为set消除重复元素，随后转换为列表
        self.injected_nodes_classes = []
        self.features_dim = features.shape[1] #获取特征维度
        self.n_nodes = num_nodes
        # self.mean_degree = int(adj.sum() / self.n_nodes) + 1
        # feature-attack budget
        # 计算平均非零特征数量
        self.average_features_nums = np.diff(csr_matrix(features.cpu().numpy()).indptr).mean() #先将feature从稀疏矩阵转换为csr格式，
        # indptr包含了每一行非零元素的列索引在非零元素列索引数组里面的索引，对indptr做差值能够获得每一行的非零元素个数
        # Construct the statistical information of features 计算特征统计信息
        self.get_sorted_features(zeroone_features) #将特征按照频率排序，获取出现频率最高的特征索引，以及平均特征向量
        self.major_features_nums = int(self.average_features_nums)
        self.major_features_candidates = self.features_id[:, :self.major_features_nums] #从特征索引列表选择每个类别top平均非零特征个数的索引
        # degree sampling related 度采样
        # sample_array1 = np.zeros(self.n_nodes, dtype='int') #用于存放每个节点的度
        # for i in range(self.n_nodes):
        #     current_degree1 = int(adj[i].sum())
        #     # maximal link-attack budget: min(current degree, 2 * mean degree) 预算
        #     sample_array1[i] = current_degree1 #小于2倍平均度才能保留
        self.sample_array1 = torch_sparse.sum(self.adj, dim=1).numpy().astype('int')
        self.loss_type = loss_type

    def compute_laplacian(self, adj_matrix): #计算图的拉普拉斯矩阵
        # 获取图的拉普拉斯矩阵
        # L = nx.laplacian_matrix(graph).toarray()
        degree = torch.diag(torch_sparse.sum(adj_matrix, dim=1).cpu())
        laplacian = degree.to(self.device) - adj_matrix.to_dense()
        return laplacian

    def compute_heat_kernel_matrix(self, adj_matrix, dataset, t=1, method="expm"): #计算热核矩阵
        """
            t (float): 时间参数，控制热扩散的速率
            method (str): 计算方法，"eig" 使用特征值分解，"expm" 使用矩阵指数
            terms: 泰勒级数展开项数
        """
        if method == "eig":
            return self.compute_heat_kernel_eig(t, adj_matrix)
        elif method == "expm":
            if dataset == 'pubmed':
                terms = 3
            else:
                terms = 4
            return self.compute_heat_kernel_expm(t, terms, adj_matrix)

    def compute_heat_kernel_eig(self, t, adj_matrix):  # 使用特征值分解计算热核矩阵
        # 计算拉普拉斯矩阵的特征值和特征向量
        L = self.compute_laplacian(adj_matrix)
        eigenvalues, eigenvectors = torch.linalg.eig(L)

        # 确保特征值是实数
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # 计算热核矩阵
        heat_kernel = eigenvectors @ torch.diag(torch.exp(-t * eigenvalues)) @ eigenvectors.T
        return heat_kernel

    def compute_heat_kernel_expm(self, t, terms, adj_matrix): #使用矩阵指数计算热核矩阵
        L = self.compute_laplacian(adj_matrix)
        # 矩阵指数显式计算
        # heat_kernel = expm(-t * L)
        # 泰勒级数展开近似计算
        # 初始化热核矩阵
        heat_kernel = torch.eye(adj_matrix.size(0), device=self.device)
        # 初始化拉普拉斯矩阵的幂
        laplacian_power = torch.eye(adj_matrix.size(0), device=self.device)
        # 计算泰勒级数展开
        for k in range(1, terms + 1):
            laplacian_power = laplacian_power @ L
            term = (-t) ** k / np.math.factorial(k) * laplacian_power
            heat_kernel += term
        return heat_kernel

    def classification_margin(self, logits: torch.Tensor, labels: torch.Tensor):  # 计算分类间隙
        probs = torch.exp(logits).cpu()  # 通过指数函数将logit转化为概率值
        label = labels.cpu()
        fill_zeros = torch.zeros_like(probs)
        true_probs = probs.gather(1, label.view(-1, 1)).flatten()  # 从probs获取每个样本的概率值最大的标签(或者真实标签)的预测概率
        probs.scatter_(1, label.view(-1, 1), fill_zeros)  # 将probs中每个样本的预测值最大标签(或者真实标签)的预测概率置为0
        best_wrong_probs = probs.max(dim=1)[0]  # 找到每个样本除了预测值最大标签(真实标签)之外的最大的标签预测概率，由于返回元组(value,index)所以取[0]
        best_wrong_labels = probs.max(dim=1)[1]
        margins = true_probs - best_wrong_probs
        return margins, best_wrong_labels

    def calculate_degree(self, adj_t):  # 计算度
        assert type(adj_t) is torch.Tensor or type(adj_t) is SparseTensor
        if type(adj_t) is SparseTensor:
            return torch_sparse.sum(adj_t, dim=1).cpu() # 可在cpu()前加to_dense()
        # TODO: edge_index和dense_adj计算度

    def select_attacking_targets(self, sur_logits, labels, idx_attack, adj, pre_ratio, select_ratio):
        #此处sur_logits为代理模型预测结果，label为真实标签，idx_attack为攻击目标索引，初始化为测试集索引idx_test，adj为原始邻接矩阵，
        #policy为选择策略(已删除)，默认以degree和margin为准，
        #pre_ratio, select_ratio为后面所用的比例参数

        correct_mask = sur_logits.argmax(1).cpu() == labels.cpu()  # 获取正确预测的样本
        index_attack = torch.nonzero(
            correct_mask & idx_attack.cpu()).flatten().detach().cpu()  # 获取正确预测样本中的初步攻击目标，在第一步初始化时，所有正确样本即为攻击目标，flatten展开为一维，detach与计算图分离避免梯度传播

        margins, best_wrong_labels = self.classification_margin(sur_logits, labels)  # 计算初步攻击目标的分类间隔
        margins = margins[index_attack]
        best_wrong_labels = best_wrong_labels[index_attack]
        degrees = self.calculate_degree(adj)[index_attack]  # 计算初步攻击目标的度
        # degree_cty = stat_data['degree_centrality'][index_attack]
        # pagerank = stat_data['pagerank'][index_attack]
        # cluster_coeff = stat_data['clustering_coefficient'][index_attack]
        # eigen_cty = stat_data['eigenvector_centrality'][index_attack]
        # surr_logit = stat_data['sur_logits']
        # if type(surr_logit) is list:
        #     surr_logit = surr_logit[0]
        # surr_logit = surr_logit[index_attack]
        # entropy = -(surr_logit * surr_logit.log()).sum(1)

        node_ranks_zip = zip(margins, degrees, index_attack)
        # edges_ranks[:, 1]获取了每条候选边的注入节点索引，edges_ranks_zip里面有包含了三个属性迭代器的一个个元组
        node_ranks_zip = sorted(node_ranks_zip,  # 升序
                                 key=lambda edges_ranks_zip: (-edges_ranks_zip[0], -edges_ranks_zip[1]))
        # 根据rank分数和负同质下降分数来进行排序，先后顺序同上
        node_ranks_list = list(zip(*node_ranks_zip))
        # 将元组解包然后转换为列表
        node_ranks = np.array(node_ranks_list[2])
        robust_nodes = node_ranks[0]
        # 获得了根据前两个分数排序后的edge rank

        n_nodes = index_attack.size(0)

        sorted_margins, _ = margins.sort()  # 分别接收margin值排序和索引排序，_丢弃索引排序
        margin_threshold = sorted_margins[int(pre_ratio * n_nodes)]  # 通过预定义比率，寻找一个margin值，作为阈值，保证固定数量的节点进入下一筛选阶段
        # 此处的阈值选择，如果太大，会导致筛选出的节点不易攻击，如果太小，会浪费一些攻击预算
        pre_mask = torch.where(margins > margin_threshold, True, False)  # 寻找正确预测样本里分类置信度足够高的样本

        margins = margins[pre_mask]
        degrees = degrees[pre_mask]
        best_wrong_labels = best_wrong_labels[pre_mask]
        # degree_cty = degree_cty[pre_mask]
        # pagerank = pagerank[pre_mask]
        # cluster_coeff = cluster_coeff[pre_mask]
        # eigen_cty = eigen_cty[pre_mask]
        # entropy = entropy[pre_mask]

        index_attack = index_attack[pre_mask]  # 更新攻击目标索引

        n_nodes = index_attack.size(0)  # 更新
        selected_idx = torch.arange(0, n_nodes)
        n_selected = int(select_ratio * n_nodes)  # 确定阈值，获得阈值索引

        indicators = [
            # entropy,
            degrees,
            # degree_cty,
            # pagerank,
            # eigen_cty,
            # cluster_coeff,
            margins,
        ]

        for item in indicators:
            _, sorted_index = item.sort()  # 将进入第二筛选阶段的样本的度进行排序，获得索引排序
            item_selected = sorted_index[:n_selected]  # 进行筛选，由于sort是从小到大排序，直接获得阈值索引之前的元素
            selected_idx = torch.tensor(np.intersect1d(item_selected, selected_idx), dtype=torch.long) #一维交集

        # selected_idx = torch.randperm(n_nodes)[:n_selected]

        return index_attack[selected_idx], best_wrong_labels[selected_idx], robust_nodes

    # Statistical information of features of each class
    def get_sorted_features(self, zeroone_features): #获取出现频率最高的特征索引以及平均特征向量
        MAX_N = 100
        features_id = []
        feature_avg = []
        for label in self.classes:
            label_features = self.features[self.labels == label].cpu().numpy() #按真实标签来归纳每一类的节点特征(连续)
            zeroone_label_features = zeroone_features[self.labels == label].numpy() #按预测标签来归纳每一类的节点特征(离散)
            count = zeroone_label_features.sum(0) #对离散特征向量进行按列求和，即第一个维度
            real_count = label_features.sum(0) #对连续(原始)特征向量进行按列求和
            count[count == 0] = 1 #将某些值全为0的特征的总和赋值为1，方便下一步计算平均特征向量
            current_avg = real_count / count #将每个特征的原始值总和除以每个特征的出现次数
            df = pd.DataFrame(columns=['count', 'features_id'],
                              data={'count': count[count.nonzero()[0]], 'features_id': count.nonzero()[0]})
            #获取每个特征的索引和该特征出现次数
            df = df.sort_values('count', ascending=False) #按count值 降序排列
            df.name = 'Label ' + str(label)
            features_id.append(df['features_id'].values[:MAX_N]) #选取每个类别的top100的特征索引
            feature_avg.append(current_avg) #获取每个类别的平均特征向量
        self.features_id = np.array(features_id)
        self.feature_avg = np.array(feature_avg)

     # 因为是顺序生成，所以n_added只能为1，一次生成一个
    def make_statistic_features(self, n_added, added_node_labels=None, rand=True): #默认n_added=1
        labels = np.random.choice(self.classes, n_added) if rand else added_node_labels
        added_node_features = np.zeros((n_added, self.features_dim)) #生成注入节点特征向量列表
        added_node_features[0][self.major_features_candidates[labels[0]]] = self.feature_avg[labels[0]][ #n_added_features[0]的0应为i，labels[]与上面对应，feature_avg[0]的0应该为labels[]
                self.major_features_candidates[labels[0]]] #在选择好的特征上用平均特征值赋值
        return added_node_features

    def gen_edges(self, injected_node, attacking_targets, sub_nodes, sub_edges):
        row = np.repeat(attacking_targets, 1) #将攻击目标节点索引重复锚点索引长度的次数，即每个锚点对应一个攻击目标节点集合
        col = np.tile(injected_node, attacking_targets.size(0)) #将锚点索引复制攻击目标节点长度的次数，即每个攻击目标节点对应一个锚点集合

        non_edges = np.row_stack([row, col]) #将每个锚点与每个攻击目标节点一一对应，形成非边集合
        unique_nodes = np.union1d(sub_nodes.tolist(), injected_node) #计算并集

        non_edges = torch.as_tensor(non_edges, device=self.device)
        unique_nodes = torch.as_tensor(unique_nodes, dtype=torch.long, device=self.device)

        self_loop = unique_nodes.repeat((2, 1)) # 复制两次，按y维度拼接，自环边
        edges_all = torch.cat([ #按y维度拼接在一起
            sub_edges, sub_edges[[1, 0]], #子图的边和反向边
            non_edges, non_edges[[1, 0]], #非边和反向非边
            self_loop,
        ], dim=1)

        edge_weight = torch.ones(sub_edges.size(1), device=self.device).requires_grad_(True) #该权重可被优化
        non_edge_weight = torch.zeros(non_edges.size(1), device=self.device).requires_grad_(True) #该权重可被优化
        self_loop_weight = torch.ones(self_loop.size(1), device=self.device)

        return (
            sub_edges, non_edges, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight
        )

    def compute_gradient(self, edges_all, weights, injected_node, robust_nodes, inputs, attacking_targets, targets_best_wrong_label, target_weight=None):
        inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs) #将inputs转换为一个元组
        if target_weight is not None:
            inputs = inputs + (target_weight, )
        logit, mod_adj_t = self.calc_logit(edges_all, weights) #计算预测置信度
        loss = self.calc_loss(logit, mod_adj_t, injected_node, robust_nodes, attacking_targets, targets_best_wrong_label, target_weight=target_weight) #计算损失
        gradients = torch.autograd.grad(loss, inputs, create_graph=False) #利用torch库计算loss对于inputs的梯度，此处inputs应为non_edge
        return gradients[0], mod_adj_t

    def calc_logit(self, edges_all, weights_all):
        edge_index = torch.cat([self.edge_index, edges_all], dim=-1) #将攻击目标节点的所有边索引按y维度连接
        edge_weight = torch.cat([torch.ones(self.edge_index.size(1), dtype=torch.float32, device=self.device), weights_all], dim=-1) #将权重连接
        edge_index, edge_weight = coalesce(edge_index, edge_weight, m=self.n_nodes + 1, n=self.n_nodes + 1, op='sum') #合并重复索引 合并重复权重
        mod_adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, (self.n_nodes + 1, self.n_nodes + 1)) #构建新邻接矩阵，包括了那些非边的连接
        logit = self.surrogate.forward(self.modified_features, mod_adj_t, edge_weight) #进行前向传播 计算预测置信度
        return logit, mod_adj_t

    def compute_hks_loss(self, injected_nodes, robust_nodes, adj, times=[0.1, 0.5, 1.0]):# 计算HKS特征及误差

        num_eigenvalues = 3

        num_nodes = self.n_nodes + 1
        num_times = len(times)
        hks_features = torch.zeros((num_nodes, num_times))

        L = self.compute_laplacian(adj)
        # eigenvalues, eigenvectors = torch.linalg.eig(L)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)  # 直接返回实数，无需转换
        # 确保特征值是实数
        # eigenvalues = eigenvalues.real
        # eigenvectors = eigenvectors.real

        # 如果指定了num_eigenvalues，则只使用前num_eigenvalues个特征值和特征向量
        # if num_eigenvalues is not None:
        #     # 按照特征值从小到大排序
        #     sorted_indices = torch.argsort(eigenvalues)
        #     eigenvalues = eigenvalues[sorted_indices]
        #     eigenvectors = eigenvectors[:, sorted_indices]
        #
        #     # 选择前num_eigenvalues个特征值和特征向量
        #     eigenvalues = eigenvalues[:num_eigenvalues]
        #     eigenvectors = eigenvectors[:, :num_eigenvalues]
        # for i, t in enumerate(times):
        #     hks = torch.sum(eigenvectors ** 2 * torch.exp(-eigenvalues * t).reshape(1, -1), axis=1)
        #     hks_features[:, i] = hks
        sorted_indices = torch.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices][:num_eigenvalues]
        eigenvectors = eigenvectors[:, sorted_indices][:, :num_eigenvalues]

        times_tensor = torch.tensor(times, device=eigenvalues.device)
        exp_terms = torch.exp(-eigenvalues.unsqueeze(1) * times_tensor.unsqueeze(0))  # (num_eigenvalues, num_times)
        hks_features = (eigenvectors ** 2) @ exp_terms  # (num_nodes, num_times)

        robust_mean = hks_features[robust_nodes].mean(dim=0)
        loss_hks = torch.sum((hks_features[injected_nodes] - robust_mean) ** 2)
        # loss_hks = torch.sum((hks_features[injected_nodes] - hks_features[robust_nodes].mean(dim=0)) ** 2)
        return loss_hks

    def calc_loss(self, logit, mod_adj_t, injected_nodes, robust_nodes, attacking_targets, targets_best_wrong_label, eps=5.0, target_weight=None):

        loss = None
        targets = attacking_targets
        labels = self.labels[targets]
        if self.loss_type == 'CW':
            logit = F.log_softmax(logit / eps, dim=1) #先除去eps，随后计算对数
            loss = 1.0*(F.nll_loss(logit[targets], labels.to(self.device)) - \
                   F.nll_loss(logit[targets], targets_best_wrong_label.to(self.device))) #攻击目标节点预测置信度与真实标签和最易误分类标签的负对数似然损失相减
        elif self.loss_type == 'CE':
            loss = F.cross_entropy(logit[targets], labels)
        elif self.loss_type == 'MCE': #Modified...
            target_preds = logit[targets].argmax(1) #攻击目标节点预测标签
            not_flipped = target_preds == labels #判断是否相同
            loss = F.cross_entropy(logit[targets][not_flipped], labels[not_flipped]) #只关注正确预测的攻击目标节点的预测置信度和真实标签
        elif self.loss_type == 'MCW': #Modified...
            target_preds = logit[targets].argmax(1)
            logit = F.log_softmax(logit, dim=1) #计算对数
            not_flipped = target_preds == labels
            loss = F.nll_loss(logit[targets][not_flipped], labels[not_flipped]) - \
                   F.nll_loss(logit[targets][not_flipped], targets_best_wrong_label[not_flipped]) #相比cw，mcw只关注正确预测的节点
        elif self.loss_type == 'CW_HKS':
            logit = F.log_softmax(logit / eps, dim=1)  # 先除去eps，随后计算对数
            loss = 1 * (F.nll_loss(logit[targets], labels.to(self.device)) - \
                   F.nll_loss(logit[targets], targets_best_wrong_label.to(self.device))) + 1 * self.compute_hks_loss(injected_nodes, robust_nodes, mod_adj_t)
        elif self.loss_type == 'CE_HKS':
            loss = F.cross_entropy(logit[targets], labels) + 0.01 * self.compute_hks_loss(injected_nodes, robust_nodes, mod_adj_t)
        assert loss is not None, f"Not support loss type: {self.loss_type}"

        return loss

    def construct_modified_edges(self, attacking_targets, injected_node, robust_nodes, targets_best_wrong_label):

        sub_nodes, sub_edges, *_ = k_hop_subgraph(attacking_targets, 2, self.edge_index) #提取攻击目标节点的子图，获取节点和边
        sub_edges = sub_edges[:, sub_edges[0] < sub_edges[1]].to(self.device) #，二维张量，行数即边的条数，列数为2，即边的起始点和结束点，对提取的子图的边进行筛选，避免出现重复的边，如边(i, j)和(j, i)表示相同的边

        (
            edge_index, non_edge_index, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight
        ) = self.gen_edges(injected_node, attacking_targets, sub_nodes, sub_edges) #获取了攻击节点子图边及对应权重、锚点和攻击目标节点的非边及对应权重

        weights = torch.cat([edge_weight, edge_weight, #之所以这里权重会重复一次是因为其中一个代表反向边的权重
                             non_edge_weight, non_edge_weight,
                             self_loop_weight], dim=-1) #按最后一个维度拼接

        add_gradient, mod_adj_t = self.compute_gradient(edges_all, weights, injected_node, robust_nodes, (non_edge_weight, ), attacking_targets, targets_best_wrong_label) #计算增加非边后，其对应的梯度
        return (
            edge_index, non_edge_index, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight, add_gradient,
            mod_adj_t
        )

    # Main attack function
    def attack_edges(self, n_added, mean_degree, dataset, pre_ratio, select_ratio, idx_test, verbose=True):

        self.dataset = dataset

        # 随机度采样
        selected_degree_distrubution = random.sample(list(self.sample_array1), n_added) #从每个节点的度中选择注入节点的边预算

        # 筛选出攻击目标
        attacking_targets, targets_best_wrong_label, robust_nodes = self.select_attacking_targets(self.sur_logits, self.labels, idx_test, self.adj, pre_ratio, select_ratio)

        num_nodes = self.n_nodes
        # 顺序注入节点
        tmp = 0
        for added_node in range(num_nodes, num_nodes + n_added):  # 将注入节点的序号进行调整
            # maximal link-attack budget: min(current degree, 2 * mean degree) 边预算
            if selected_degree_distrubution[added_node - num_nodes] > 2 * mean_degree:
                selected_degree_distrubution[added_node - num_nodes] = int(2 * mean_degree)
            if selected_degree_distrubution[added_node - num_nodes] == 0:
                selected_degree_distrubution[added_node - num_nodes] = 1
            if verbose:
                print("\n\n----- injecting ID {} node -----".format(added_node - num_nodes))
                print("----- injecting {} edges -----".format(selected_degree_distrubution[added_node - num_nodes]))

            # 随机分配一个最佳错误类标签给注入节点
            added_node_label = np.random.choice(list(set(targets_best_wrong_label)), 1)[0]

            # 记录注入节点标签
            self.injected_nodes_classes.append(added_node_label) #记录标签

            # 为注入节点进行当前所分配类的节点平均采样
            added_node_feature = self.make_statistic_features(1, [added_node_label], False) #创建特征,rand参数控制随机分配特征

            # 将注入节点的特征加入
            modified_features = sp.vstack((self.modified_features.cpu(), added_node_feature)).tocsr().toarray()# 特征矩阵加入新节点特征
            self.modified_features = torch.from_numpy(modified_features).to(torch.float32).to(self.device)

            # 为注入节点和目标攻击节点创建非边，并计算非边梯度
            current_attacking_targets = attacking_targets[torch.where(targets_best_wrong_label == torch.tensor(added_node_label, dtype=torch.long), True, False)]
            current_targets_bestwrong_label = targets_best_wrong_label[targets_best_wrong_label == torch.tensor(added_node_label, dtype=torch.long)]
            (
                edge_index, non_edge_index, self_loop, edges_all,
                edge_weight, non_edge_weight, self_loop_weight, add_gradient,
                mod_adj_t
            ) = self.construct_modified_edges(current_attacking_targets, added_node, robust_nodes, current_targets_bestwrong_label)

            # 从上面获取将所有非边连接起来的邻接矩阵，用于计算其热核矩阵
            heat_kernel_matrix = self.compute_heat_kernel_matrix(mod_adj_t,self.dataset)
            # heat_pob_matrix = torch.zeros_like(heat_kernel_matrix)
            # for i in range(heat_kernel_matrix.shape[0]):
            #     for j in range(heat_kernel_matrix.shape[1]):
            #         heat_pob_matrix[i,j] = heat_kernel_matrix[i,j] / torch.max(heat_kernel_matrix)
            heat_pob_matrix = heat_kernel_matrix / heat_kernel_matrix.max()
            # print(heat_pob_matrix)
            #提取非边索引和非边所对应的热核概率的可迭代对象
            # heat_kernel_probability = []
            # item_edge_index = []
            # for i in range(non_edge_index.shape[1]):
            #     heat_kernel_probability.append(heat_pob_matrix(non_edge_index[0, i], non_edge_index[1, i]).item())
            #     item_edge_index.append((non_edge_index[0, i], non_edge_index[1, i]))
            heat_kernel_probability = heat_pob_matrix[non_edge_index[0], non_edge_index[1]].detach().cpu().numpy()
            item_edge_index = non_edge_index.t().cpu().numpy().tolist()
            item_add_gradient = add_gradient.tolist()
            # 显式释放较大矩阵
            del heat_kernel_matrix, heat_pob_matrix

            # if selected_degree_distrubution[added_node - num_nodes] != 1:
            #     while i < selected_degree_distrubution[added_node - num_nodes] :
                    # min_non_edge_grad, min_non_edge_idx = torch.min(add_gradient, dim=0) #寻找非边梯度...
                    # add_gradient.data[min_non_edge_idx] = float('inf') #防止重复选择
                    # best_edge = non_edge_index[:, min_non_edge_idx]
                    # best_edge_reverse = best_edge[[1, 0]]
                    # non_edge_weight.data[min_non_edge_idx] = 1.0 #权重置为1，下次梯度重新计算便会计算它
            rand_link = False
            num_add_edges = selected_degree_distrubution[added_node - num_nodes]
            if rand_link:
                np.random.shuffle(item_edge_index)
                final_potential_edges = item_edge_index[0: num_add_edges]
            else:
                edges_ranks_zip = zip(heat_kernel_probability, item_add_gradient, item_edge_index)
                # edges_ranks_zip里面有包含了三个属性迭代器的一个个元组
                edges_ranks_zip = sorted(edges_ranks_zip, #升序
                                    key=lambda edges_ranks_zip: (-edges_ranks_zip[1], -edges_ranks_zip[0]))
                edges_ranks_list = list(zip(*edges_ranks_zip))
                # 将元组解包然后转换为列表
                edges_ranks = np.array(edges_ranks_list[2])
                # 获得了根据前两个分数排序后的edge rank
                final_potential_edges = edges_ranks[0: num_add_edges]

            final_edges = torch.tensor(final_potential_edges).T.to(self.device)
            reverse_edges = final_edges.flip(0)  # 生成反向边 [2, N]
            all_new_edges = torch.cat([final_edges, reverse_edges], dim=1)
            self.edge_index = pyg_coalesce(
                torch.cat([self.edge_index, all_new_edges], dim=1)
            )

            # 重构造邻接矩阵
            modified_adj = SparseTensor.from_edge_index(edge_index=self.edge_index, sparse_sizes=(self.n_nodes + 1, self.n_nodes + 1)).coalesce().detach()
            # inject the current node to the original adj
            self.modified_adj = modified_adj
            self.n_nodes = self.n_nodes + 1
            if verbose:
                print("----- ID {} node is injected -----".format(added_node - num_nodes))
            tmp += selected_degree_distrubution[added_node - num_nodes]
            # 一个节点注入成功
        add_label = []
        add_label.extend(self.labels)
        add_label.extend(self.injected_nodes_classes)
        add_label = np.array(add_label)
        self.n_added_labels = add_label
        self.tmp = tmp
    # Begin the attacks and final evaluation
    def train(self, n_added, mean_degree, features, adj, dataset, pre_ratio, select_ratio, idx_train, idx_val, idx_test, verbose=True):
        # self.train_init()
        self.attack_edges(n_added, mean_degree, dataset, pre_ratio, select_ratio, idx_test, verbose)
        print('\nFinish attack\n')

        # evasion_test(self, time_, features, adj, idx_train, idx_val, idx_test)
        # modified_features = self.modified_features
        # modified_adj = self.modified_adj
        # # save corresponding matrices
        # sp.save_npz(file1, modified_adj)
        # sp.save_npz(file2, modified_features)



