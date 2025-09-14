import os.path as osp
import os
import sys
sys.path.insert(0, 'F:\\pycode\\node_injection_with_camouflage')

import argparse
import yaml
from yaml import SafeLoader
from copy import deepcopy
from utils import *
from models import model_map
import time



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--logger_level', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'citeseer', 'pubmed', 'cora_ml'])
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()

    assert args.model in model_map
    assert args.gpu_id in range(0, 4)
    assert args.logger_level in [0, 1, 2]

    #load models config
    config_file = osp.join(osp.expanduser('./mdl_configs'), args.model+'.yaml')
    config = yaml.load(open(config_file), Loader=SafeLoader)[args.dataset]

    #set logger
    logger_filename = "./mdl_logs/" + args.model + '-' + args.dataset + '.log'
    logger_name = 'model_train'
    logger = get_logger(logger_filename, level=args.logger_level, name=logger_name)
    logger.info(config)

    #get gpu device
    device = get_device(args.gpu_id)
    # device = 'cpu'

    #print information
    logger.info(f"MODEL NAME: {args.model}")
    logger.info(f"DATASET: {args.dataset}")
    logger.info(f"DEVICE: {device}")
    seed = config['seed']#获取种子
    freeze_seed(seed)
    logger.info(f"INITIAL SEED: {seed}")

    #load dataset
    pyg_data = load_data(name=args.dataset)

    #print dataset info
    logger.info("DATASET INFOs: ")
    logger.info(f"n_features: {pyg_data.num_features}, n_nodes: {pyg_data.num_nodes}, n_classes: {pyg_data.num_classes}")
    logger.info(f"train shape: {pyg_data.train_mask.shape}, val shape: {pyg_data.val_mask.shape}, test shape: {pyg_data.test_mask.shape}")

    #get model instantiate function
    model_func = model_map[args.model]

    #load model
    model = model_func(config=config, pyg_data=pyg_data, device=device, logger=logger)
    model = model.to(device)
    logger.info(f"Model: \n{model}")

    # train model
    state_dicts = []
    statistics = []
    time_costs = []
    for i in range(5):
        freeze_seed(seed + i)
        logger.debug(f"Run {i+1:02d} ============================================================")
        start = time.time()
        model.fit(pyg_data, verbose=True)
        time_costs.append(time.time() - start)
        statistics.append(model.test(pyg_data.test_mask))#记录每一个训练round得到的模型的测试准确率
        state_dicts.append(deepcopy(model.cpu().state_dict()))#深度拷贝model状态信息，且将
        model.to(device)
        logger.info(f"accuracy={statistics[-1]*100:.3f}")#保证总是将最新获得的准确率打印出来
        logger.debug(f"============================================================ Run {i+1:02d} \n\n")
    logger.info(f"statistics: {np.mean(statistics)*100:.2f}{chr(177)}{np.std(statistics)*100:.2f}")#计算并打印所有训练round的平均准确率和方差
    logger.info(f"time cost: {np.mean(time_costs):.2f}")#平均时耗

    #save model weight file
    save_path = None
    if args.save:
        save_path = "./tar_models/"
        if not osp.exists(save_path):
            os.makedirs(save_path)
        save_path += f"{args.model+'-'+args.dataset}.pth"
        torch.save(obj={
            'state_dicts': state_dicts,
            'config': config,
            'performance': f"{np.mean(statistics)*100:.2f}{chr(177)}{np.std(statistics)*100:.2f}",
        }, f=save_path)
        logger.info(f"[{save_path}] Saved\n\n")
        print("save success！")


    # 验证一下保存的数据
    if save_path is not None:
        savings = torch.load(save_path)
        config = savings['config']
        states = savings['state_dicts']
        performance = savings['performance']

        logger.debug(f"Performance: {performance}")
        logger.debug(f"Config: {config}")
        init_seed = config['seed']
        n_running = len(states)
        for i in range(n_running):
            freeze_seed(init_seed+i)
            model.load_state_dict(states[i])
            model.test(pyg_data.test_mask)


if __name__ == '__main__':
    main()