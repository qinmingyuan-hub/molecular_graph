from argparse import Namespace
from logging import Logger
import numpy as np
import os, torch

from model.train import fold_train
from model.tool import set_log, set_train_argument, get_task_name, mkdir
from model.train import training
import time

if __name__ == '__main__':
    args = set_train_argument()
    log = set_log('', args.log_path)

    args.num_folds = 10
    args.epochs = 20
    args.dataset_type = 'regression'  # 'classification'
    args.model_type = 'moleculeformer_mini'  #  'moleculeformer', 'FPonly', 'FPGNN'
    args.metric = 'rmse'  # 'auc', 'auc_prcauc', 'prc-auc', 'rmse_mse_mae_mape_r2'
    args.split_type = 'random'  # 'ratio_random'
    args.dropout = 0
    args.hidden_size = 100
    args.seed = 0

    path = 'LOG HLM_CLint (mL_min_kg).csv'
    args.save_path = 'model_save_' + args.model_type + "_" + args.split_type + "_" + args.metric + "_" + str(args.seed) + "_" + args.force_field
    args.log_path = 'log_' + args.model_type + "_" + args.split_type + "_" + args.metric + "_" + str(args.seed) + "_" + args.force_field
    args.data_path = 'Data/ADME/split_results/' + path

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    log.info(f"Training starts at {start_time_str, path}")
    training(args, log)

    end_time = time.time()
    elapsed_time = end_time - start_time
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    log.info(f"Training ends at {end_time_str, path}")
    log.info(f"Training for dataset {path} took {elapsed_time:.2f} seconds.")

    torch.cuda.empty_cache()

# nohup python train_main.py > molecuformer_breast_nomini.log 2>&1 &