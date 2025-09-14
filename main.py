import torch
import numpy as np

import argparse
import yaml
from yaml import SafeLoader
from copy import deepcopy
import time
from utils import *
import os.path as osp
import os
import sys
# from attackers import attacker_map
from models import model_map, choice_map
sys.path.insert(0, 'F:\\pycode\\node_injection_with_camouflage')
from node_injection import NI
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



def calculate_degree_distribution(adj_matrix):
    degrees = adj_matrix.sum(dim=1).cpu().numpy()
    return degrees

def visualize_degree_distribution(original_degrees, perturbed_degrees, dataset):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.weight'] = 'bold'
    # 两个度分布位于同一张图
    # plt.figure(figsize=(10, 7))
    #
    # plt.hist(original_degrees, bins=30, color='blue', alpha=0.7, label='Original Graph', density=True)
    # kde = gaussian_kde(original_degrees)
    # x = np.linspace(min(original_degrees), max(original_degrees), 1000)
    # plt.plot(x, kde(x), color='blue', linewidth=2)
    #
    # plt.hist(perturbed_degrees, bins=30, color='red', alpha=0.5, label='Perturbed Graph', density=True)
    # kde = gaussian_kde(perturbed_degrees)
    # x = np.linspace(min(perturbed_degrees), max(perturbed_degrees), 1000)
    # plt.plot(x, kde(x), color='red', linewidth=2)
    #
    # # plt.title('Degree Distribution Comparison')
    # plt.xlabel('Degree', fontsize=24, fontweight='bold')
    # plt.ylabel('Frequency', fontsize=24, fontweight='bold')
    # plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.legend(fontsize=16)
    # 创建一个包含两个子图的图形，上下排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))

    # 绘制原始图的度分布
    ax1.hist(original_degrees, bins=30, color='blue', alpha=0.7, label='Original Graph', density=True)
    kde = gaussian_kde(original_degrees)
    x = np.linspace(min(original_degrees), max(original_degrees), 1000)
    ax1.plot(x, kde(x), color='blue', linewidth=2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.set_xlabel('Degree', fontsize=24, fontweight='bold')
    # ax1.set_ylabel('Frequency', fontsize=24, fontweight='bold')
    # ax1.tick_params(axis='both', which='major', labelsize=16)
    # ax1.legend(fontsize=16)

    # 绘制扰动图的度分布
    ax2.hist(perturbed_degrees, bins=30, color='red', alpha=0.7, label='Perturbed Graph', density=True)
    kde = gaussian_kde(perturbed_degrees)
    x = np.linspace(min(perturbed_degrees), max(perturbed_degrees), 1000)
    ax2.plot(x, kde(x), color='red', linewidth=2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    # ax2.set_xlabel('Degree', fontsize=24, fontweight='bold')
    # ax2.set_ylabel('Frequency', fontsize=24, fontweight='bold')
    # ax2.tick_params(axis='both', which='major', labelsize=16)
    # ax2.legend(fontsize=16)

    plt.tight_layout()
    save_path = 'D:/pycode/node_injection_with_camouflage/image'
    plt.savefig(os.path.join(save_path, f'{dataset}_degree_sep.jpg'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def visualize_features_with_tsne(perturbed_features, labels, n_perturbs, dataset):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.weight'] = 'bold'
    # 合并原始特征和扰动特征
    features =  perturbed_features.cpu().numpy()
    # 创建标签，0 表示原始节点，1 表示注入节点
    node_types = np.zeros(features.shape[0])
    node_types[-n_perturbs:] = 1  # 最后 n_perturbs 个节点是注入节点

    # 进行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 7))
    # unique_labels = np.unique(labels)
    # for label in unique_labels:
    #     mask = (labels == label)
    #     plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
    #                 label=f'Class {int(label)}', alpha=0.6)
    plt.scatter(features_tsne[:-n_perturbs, 0], features_tsne[:-n_perturbs, 1],
                c='blue', alpha=0.6, label='Original Node')
    plt.scatter(features_tsne[-n_perturbs:, 0], features_tsne[-n_perturbs:, 1],
                c='red', alpha=0.6, label='Injection Node')
    # plt.title('t-SNE Visualization of Features')
    plt.xlabel('Dim-1', fontsize=24, fontweight='bold')
    plt.ylabel('Dim-2', fontsize=24, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)
    save_path = 'D:/pycode/node_injection_with_camouflage/image'
    plt.savefig(os.path.join(save_path, f'{dataset}_feature.jpg'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--logger_level', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'citeseer', 'pubmed', 'cora_ml'])
    parser.add_argument('--attack', type=str, default='ni')
    parser.add_argument('--victim', type=str, default='normal')
    parser.add_argument('--ptb_rate', type=float, default=0.05)
    parser.add_argument('--n_running', type=int, default=1)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--save_prefix', type=str, default="")
    args = parser.parse_args()
    assert args.gpu_id in range(0, 3)  # 显卡编号，最好为0
    assert args.logger_level in [0, 1, 2]  # 分别代表从低到高的优先级

    # get gpu device
    device = get_device(args.gpu_id)
    # device = 'cpu'

    # load attack config
    config_file = osp.join(osp.expanduser('./atk_configs'), args.attack + '.yaml')
    attack_config = yaml.load(open(config_file), Loader=SafeLoader)[args.dataset]

    # set logger and print information
    logger_filename = f"./atk_logs/{args.attack}-{args.dataset}-{args.victim}-{args.ptb_rate}.log"
    logger_name = 'attack_train'
    logger = get_logger(logger_filename, level=args.logger_level, name=logger_name)
    logger.info(args)
    logger.info(f"Attacker's NAME: {args.attack}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")

    # load surrogate states and config
    surrogate_type = attack_config['surrogate']
    surrogate_pretrained_model = load_pretrained_model(args.dataset, surrogate_type, path='./sur_models/')  # 返回一个字典对象
    surrogate_state_dicts = surrogate_pretrained_model['state_dicts']  # 获取状态、权重信息
    surrogate_config = surrogate_pretrained_model['config']  # 获取配置信息

    # config random number seed
    init_seed = surrogate_config['seed']
    freeze_seed(init_seed)

    # load dataset
    data = load_data(name=args.dataset)
    logger.info(f"Dataset Information: ")
    logger.info(f"\t\t The Number of Nodes: {data.num_nodes}")
    logger.info(f"\t\t The Number of Edges: {data.num_edges}")
    logger.info(f"\t\t The Dimension of Features: {data.num_features}")
    logger.info(f"\t\t The Number of Classes: {data.num_classes}")
    logger.info(f"\t\t Split(train/val/test): "
                f"{data.train_mask.sum().item()}, "
                f"{data.val_mask.sum().item()}, "
                f"{data.test_mask.sum().item()}")
    adj, features, labels = data.adj_t, data.x, data.y
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # mapping matrix to handle continuous feature
    zeroone_features = deepcopy(features)
    zeroone_features[zeroone_features > 0] = 1  # 连续特征转化为离散特征

    # load surrogate model to device
    surrogate_performance = surrogate_pretrained_model['performance'] # 获取模型性能表现
    surrogate = model_map[surrogate_type](config=surrogate_config, pyg_data=data, device=device, logger=logger)
    surrogate = surrogate.to(device)
    logger.info("\n\n")
    logger.info(f"Loaded pretrained surrogate: {surrogate_type}")
    logger.info(f"Surrogate config: {surrogate_config}")
    logger.info(f"Surrogate performance: {surrogate_performance}\n\n")

    # get victim models name
    victim_types = []
    choices = choice_map[args.victim]
    victim_types.extend(choices)

    # load victim models
    victims = []
    attack_acc_dict = dict()  # 存储对应目标模型的攻击后分类准确率
    clean_acc_dict = dict()  # 存储对应目标模型的攻击前分类准确率
    for victim_name in victim_types:
        victim_pretrained_model = load_pretrained_model(args.dataset, victim_name, path='./tar_models/')
        victim_state_dicts = victim_pretrained_model['state_dicts']
        victim_config = victim_pretrained_model['config']
        victim = model_map[victim_name](config=victim_config, pyg_data=data, device=device, logger=logger)
        victim = victim.to(device)
        victims.append({  # 目标模型信息
            'name': victim_name,
            'model': victim,
            'state_dicts': victim_state_dicts,
            'configs': victim_config,
            'performance': victim_pretrained_model['performance']
        })
        attack_acc_dict[victim_name] = list()  # 不止一个准确率，最后将其求平均
        clean_acc_dict[victim_name] = list()  # 不止一个准确率，最后将其求平均

    # perturbed budget
    # n_added = 固定值
    # n_added = int(adj.shape[0] * args.ptb_rate) * 5
    n_perturbs = int(args.ptb_rate * num_nodes)
    n_pre_num=num_nodes+n_perturbs
    logger.info(f"Rate of perturbation: {args.ptb_rate}")
    logger.info(f"The number of perturbations: {n_perturbs}")

    # calculate the mean degree
    mean_degree = int(adj.sum() / num_nodes)  # 算平均度，所有边被算两次
    modified_adj_list = []
    time_cost_list = []

    n_running = min(len(victims[0]['state_dicts']), args.n_running)  # 这里state_dicts里面存放的是经过训练的模型的状态信息字典，训练了几次就有几个状态信息
    # len()获取了状态个数，保证攻击的次数不超过此数值

    # implement attack
    for i in range(n_running):  # 注意这里每次生成一个新的attacker, 额外开销其实不大；如果每个running修改全局attacker数据，会让代码稍乱一点
        freeze_seed(init_seed + i)
        # victim.load_state_dict(victim_state_dicts[i])
        surrogate.load_state_dict(surrogate_state_dicts[i])  # 每一次加载的模型不一样
        if args.attack == 'ni':
            pre_ratio = attack_config['pre_ratio']
            select_ratio = attack_config['select_ratio']
            loss_type = attack_config['loss_type']
            dataset = args.dataset
            attacker = NI(features, adj, labels, num_nodes, edge_index, surrogate, loss_type, zeroone_features)
            start = time.time()
            attacker.train(n_perturbs, mean_degree, features, adj, dataset,
                                 pre_ratio, select_ratio, idx_train, idx_val, idx_test)
            time_cost_list.append(time.time() - start)  # 获得执行攻击生成扰动图的花费时间
            mod_adj, mod_ftr, mod_lb = attacker.modified_adj, attacker.modified_features, attacker.n_added_labels
            mod_edge = attacker.edge_index
            mod_rate = attacker.tmp / data.num_edges
            logger.debug(f"Modified rate: {mod_rate:.2f}")

        modified_adj_list.append(deepcopy(mod_adj.detach().cpu()))

        logger.info(f"Running[{i + 1:03d}], time cost= {time_cost_list[-1]:.3f}")
        performance = dict()
        for victim in victims:
            name = victim['name']
            model = victim['model']
            model.load_state_dict(victim['state_dicts'][i])
            if args.attack == 'ni':
                add_mask = torch.zeros(n_perturbs, dtype=torch.bool)
                ni_idx_test = torch.cat((idx_test, add_mask), dim=0)
                attack_acc = evaluate_attack_performance(model, mod_ftr, mod_adj, mod_lb, ni_idx_test)
            else:
                attack_acc = evaluate_attack_performance(model, features, mod_adj, labels, idx_test)
            attack_acc_dict[name].append(attack_acc)

            # clean_acc = model.test(test_mask=idx_test, verbose=False)
            # clean_acc_dict[name].append(clean_acc)

            logger.info(
                f"name= {name:15s} clean accuracy= {victim['performance']}, attacked accuracy= {attack_acc * 100:.2f}")
            performance[name] = victim['performance']
        # prt_feat = mod_ftr.cpu()
        # visu_label = mod_lb
        # visualize_features_with_tsne(prt_feat, visu_label, n_perturbs, args.dataset)
        ori_deg = calculate_degree_distribution(adj)
        prt_deg = calculate_degree_distribution(mod_adj)
        visualize_degree_distribution(ori_deg, prt_deg, args.dataset)
        # row, col = edge_index[0], edge_index[1]
        # edges = torch.stack((row, col), dim=0)
        # edges = edges.cpu().numpy().T
        # np.savetxt(f'D:/pycode/node_injection_with_camouflage/score/{args.dataset}_edges.txt', edges, delimiter=' ', fmt='%d')
        # row1, col1 = attacker.edge_index[0], attacker.edge_index[1]
        # pur_edges = torch.stack((row1, col1), dim=0)
        # pur_edges = pur_edges.cpu().numpy().T  # 转置为形状 [E, 2]
        # np.savetxt(f'D:/pycode/node_injection_with_camouflage/score/{args.dataset}_pur_edges.txt', pur_edges, delimiter=' ', fmt='%d')
                                      # logger.info(
            #     f"name= {name:15s}  attacked accuracy= {attack_acc * 100:.2f}")
            # logger.info(
            #     f"name= {name:15s}  clean accuracy= {clean_acc * 100:.2f}")
            # 在整个过程中，代理模型的第i个state对应每个模型的第i个state

    total_mean = []
    total_std = []

    for k in attack_acc_dict.keys():
        attack_accs = attack_acc_dict[k]
        clean_accs = clean_acc_dict[k]
        total_mean.append(float(f"{np.mean(attack_accs) * 100:.2f}"))  # 计算每一个目标模型被攻击后的平均预测准确率
        total_std.append(float(f"{np.std(attack_accs) * 100:.2f}"))  # 计算方差
        logger.info(
            f"victim= {k:15s} clean acc= {performance[k]}, "  # chr(177)表示加减号
            f"attacked acc= {np.mean(attack_accs) * 100:.2f}{chr(177)}{np.std(attack_accs) * 100:.2f}"
            f"\t#ptb_rate= {args.ptb_rate}")
        if args.save:
            attack_name = args.attack
            attack_name = args.save_prefix + attack_name
            save_result_to_json(  # 将攻击结果以及数据集、目标模型的信息存入json文件
                attack=attack_name,
                dataset=args.dataset,
                victim=k,
                ptb_rate=args.ptb_rate,
                attacked_acc=f"{np.mean(attack_accs) * 100:.2f}{chr(177)}{np.std(attack_accs) * 100:.2f}",
                attack_type='evasion',
            )

    if args.victim == 'all':
        if args.save:
            attack_name = args.attack
            attack_name = args.save_prefix + attack_name
            save_result_to_json(
                attack=attack_name,
                dataset=args.dataset,
                victim='all',
                ptb_rate=args.ptb_rate,
                attacked_acc=f"{np.mean(total_mean):.2f}{chr(177)}{np.mean(total_std):.2f}",
                attack_type='evasion',
            )
        logger.info(
            f"Averaged Attack Performance= {np.mean(total_mean):.2f}{chr(177)}{np.mean(total_std):.2f}")  # 将所有目标模型的平均准确率求和平均，即该数据集上的整体表现

    if args.save:
        save_path = "./perturbed_adjs/"
        if not osp.exists(save_path):
            os.makedirs(save_path)
        attack_name = args.attack
        if attack_name == 'pgdattack':
            loss_name = attack_config['loss_type']
            if loss_name != 'tanhMargin':
                attack_name = attack_name + "-" + loss_name
        attack_name = args.save_prefix + attack_name
        save_path += f"{attack_name}-{args.dataset}-{args.ptb_rate}.pth"
        torch.save(obj={
            'modified_adj_list': modified_adj_list,
            'attack_config': attack_config,
        }, f=save_path)



    # The adversarial adjacency matrix and feature matrix are saved in '\data\xxx'.