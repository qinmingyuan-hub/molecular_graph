from argparse import Namespace
import os
import time
import csv
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import random
from model.tool.tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, NoamLR, load_model, choose_model
from model.data import MoleDataSet
from tqdm import tqdm
import logging

def epoch_train(model, data, loss_f, optimizer, scheduler, args):
    model.train()
    data.random_data(args.seed)
    loss_sum = 0
    data_used = 0
    iter_step = args.batch_size
    train_label = []
    train_pred = []
    for i in tqdm(range(0, len(data), iter_step), desc="epoch_train"):
        if data_used + iter_step > len(data):
            break

        data_now = MoleDataSet(data[i:i+iter_step])
        smile = data_now.smile()
        label = data_now.label()
        train_label.extend(label)
        mask = torch.Tensor([[x is not None for x in tb] for tb in label])
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])

        if next(model.parameters()).is_cuda:
            mask, target = mask.cuda(), target.cuda()

        weight = torch.ones(target.shape)
        if args.cuda:
            weight = weight.cuda()

        model.zero_grad()
        pred = model(smile)
        train_pred.extend(pred.cpu().tolist())
        loss = loss_f(pred, target) * weight * mask
        loss = loss.sum() / mask.sum()
        loss_sum += loss.item()
        data_used += len(smile)
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
    if isinstance(scheduler, ExponentialLR):
        scheduler.step()
    return train_pred, train_label

def predict(model, data, batch_size, scaler, loss_f, args):
    model.eval()
    pred = []
    data_total = len(data)

    for i in tqdm(range(0, data_total, batch_size), desc="predict"):
        data_now = MoleDataSet(data[i:i+batch_size])
        smile = data_now.smile()
        label = data_now.label()

        with torch.no_grad():
            pred_now = model(smile)

            mask = torch.Tensor([[x is not None for x in tb] for tb in label])
            target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])
            weight = torch.ones(target.shape)

            if next(model.parameters()).is_cuda:
                mask, target = mask.cuda(), target.cuda()

            if loss_f and args:
                weight = torch.ones(target.shape)
                if args.cuda:
                    weight = weight.cuda()

                loss = loss_f(pred_now, target) * weight * mask
                loss = loss.sum() / mask.sum()

        pred_now = pred_now.data.cpu().numpy()

        if scaler is not None:
            ave = scaler[0]
            std = scaler[1]
            pred_now = np.array(pred_now).astype(float)
            change_1 = pred_now * std + ave
            pred_now = np.where(np.isnan(change_1), None, change_1)

        pred_now = pred_now.tolist()
        pred.extend(pred_now)

    return pred

def compute_score(pred, label, metric_f, args, log):  # Dimensions: n, m. n is the number of tasks, m is the number of metrics
    info = log.info

    batch_size = args.batch_size
    task_num = args.task_num
    data_type = args.dataset_type

    if len(pred) == 0:
        return [float('nan')] * task_num

    pred_val = []
    label_val = []
    for i in range(task_num):
        pred_val_i = []
        label_val_i = []
        for j in range(len(pred)):
            if label[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)

    result = []  # Outer layer: tasks, inner layer: metrics

    for i in range(task_num):  # Loop through different prediction tasks
        if data_type == 'classification':
            if all(one == 0 for one in label_val[i]) or all(one == 1 for one in label_val[i]):
                info('Warning: All labels are 1 or 0.')
                result.append(float('nan'))
                continue
            if all(one == 0 for one in pred_val[i]) or all(one == 1 for one in pred_val[i]):
                info('Warning: All predictions are 1 or 0.')
                result.append(float('nan'))
                continue

        re = metric_f(label_val[i], pred_val[i])  # Returns multiple metrics for a single task

        result.append(re)

    return result

def compute_ave_result(train_score):  # Returns the average prediction value for different tasks
    train_score = np.array(train_score)
    col_mean = np.mean(train_score, axis=0)
    if train_score.ndim == 1:
        return [col_mean]
    if train_score.ndim == 2:
        return col_mean

def info_ave_result(score, args, score_type, log):
    info = log.info
    train_score = np.array(score)
    col_mean = np.mean(train_score, axis=0)
    info(f'{score_type}:{args.metric} = {col_mean}')

def fold_train(args, log):
    info = log.info
    debug = log.debug

    debug('Start loading data')

    args.task_names = get_task_name(args.data_path)
    data = load_data(args.data_path, args)
    args.task_num = data.task_num()
    data_type = args.dataset_type
    if args.task_num > 1:
        args.is_multitask = 1

    debug(f'Splitting dataset with Seed = {args.seed}.')
    if args.val_path:
        val_data = load_data(args.val_path, args)
    if args.test_path:
        test_data = load_data(args.test_path, args)
    if args.val_path and args.test_path:
        train_data = data
    elif args.val_path:
        split_ratio = (args.split_ratio[0], 0, args.split_ratio[2])
        train_data, _, test_data = split_data(data, args.split_type, split_ratio, args.seed, log)
    elif args.test_path:
        split_ratio = (args.split_ratio[0], args.split_ratio[1], 0)
        train_data, val_data, _ = split_data(data, args.split_type, split_ratio, args.seed, log)
    else:
        train_data, val_data, test_data = split_data(data, args.split_type, args.split_ratio, args.seed, log)  # Split data
    debug(f'Dataset size: {len(data)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')
    if args.noise_rate != 0:  # Add noise
        train_to_modify = int(len(train_data) * args.noise_rate)
        val_to_modify = int(len(val_data) * args.noise_rate)
        train_indices_to_modify = random.sample(range(len(train_data)), train_to_modify)
        val_indices_to_modify = random.sample(range(len(val_data)), val_to_modify)

        # Modify training data labels
        for index in train_indices_to_modify:
            labels = train_data.data[index].label
            for i in range(len(labels)):
                if labels[i] == 0:
                    labels[i] = 1
                elif labels[i] == 1:
                    labels[i] = 0

        # Modify validation data labels
        for index in val_indices_to_modify:
            labels = val_data.data[index].label
            for i in range(len(labels)):
                if labels[i] == 0:
                    labels[i] = 1
                elif labels[i] == 1:
                    labels[i] = 0

    if data_type == 'regression':
        label_scaler = get_label_scaler(train_data)
    else:
        label_scaler = None
    args.train_data_size = len(train_data)

    loss_f = get_loss(data_type)
    metric_f = get_metric(args.metric)

    debug('Training Model')

    model = choose_model(args)  # Choose model

    debug(model)
    if args.cuda:
        model = model.to(torch.device("cuda"))
    save_model(os.path.join(args.save_path, 'model.pt'), model, label_scaler, args)
    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=0.01)  # L2 regularization
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[args.warmup_epochs], total_epochs=None or [args.epochs] * args.num_lrs,
                       steps_per_epoch=args.train_data_size // args.batch_size, init_lr=[args.init_lr], max_lr=[args.max_lr],
                       final_lr=[args.final_lr])
    if data_type == 'classification':
        best_score = -float('inf')
    else:
        best_score = float('inf')
    best_epoch = 0
    n_iter = 0

    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')
        # Start training
        train_start_time = time.time()

        train_pred, train_label = epoch_train(model, train_data, loss_f, optimizer, scheduler, args)
        train_score = compute_score(train_pred, train_label, metric_f, args, log)

        train_end_time = time.time()
        # Start validation
        val_start_time = time.time()
        val_pred = predict(model, val_data, args.batch_size, label_scaler, loss_f, args)  # Validation predict
        val_label = val_data.label()
        val_score = compute_score(val_pred, val_label, metric_f, args, log)

        val_end_time = time.time()

        info_ave_result(val_score, args, "val", log)
        info_ave_result(train_score, args, "train", log)

        ave_val_score = compute_ave_result(val_score)
        if args.task_num > 1:
            for one_name, one_score in zip(args.task_names, val_score):
                info(f'Validation {one_name} {args.metric} ={str(one_score)}')

        if data_type == 'classification' and ave_val_score[0] > best_score:
            best_score = ave_val_score[0]
            best_epoch = epoch
            save_model(os.path.join(args.save_path, 'model.pt'), model, label_scaler, args)  # Save model if validation AUC improves
        elif data_type == 'regression' and ave_val_score[0] < best_score:
            best_score = ave_val_score[0]
            best_epoch = epoch
            save_model(os.path.join(args.save_path, 'model.pt'), model, label_scaler, args)

    info(f'Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

    model = load_model(os.path.join(args.save_path, 'model.pt'), args.cuda, log)  # Test with the best model
    test_smile = test_data.smile()
    test_label = test_data.label()

    test_pred = predict(model, test_data, args.batch_size, label_scaler, loss_f, args)
    test_score = compute_score(test_pred, test_label, metric_f, args, log)

    info_ave_result(test_score, args, "test", log)

    if args.task_num > 1:
        for one_name, one_score in zip(args.task_names, test_score):
            info(f'Test {one_name} {args.metric} ={str(one_score)}')

    return test_score

def training(args, log):
    info = log.info

    seed_first = args.seed
    data_path = args.data_path
    save_path = args.save_path

    score = []

    for num_fold in range(args.num_folds):
        info(f'Seed {args.seed}')
        args.seed = seed_first + num_fold
        args.save_path = os.path.join(save_path, f'Seed_{args.seed}')
        mkdir(args.save_path)

        fold_score = fold_train(args, log)

        score.append(fold_score)
    score = np.array(score)  # Dimensions: fold, specific task, number of metrics

    if score.ndim == 2:  # If the number of metrics is 1, add a dimension
        score = np.expand_dims(score, axis=-1)
    info(f'Running {args.num_folds} folds in total.')

    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(score):  # score: fold, number of tasks, metrics
            info_ave_result(fold_score, args, f'Seed {seed_first + num_fold},test', log)

            if args.task_num > 1:
                for one_name, one_score in zip(args.task_names, fold_score):
                    info(f' Task {one_name} {args.metric} ={str(one_score)}')

    means = np.nanmean(score, axis=0)
    stds = np.nanstd(score, axis=0)

    # Iterate through all tasks
    for task_idx, task_name in enumerate(args.task_names):
        # Collect all metric results for the current task
        metric_results = []

        # Iterate through all metrics
        for metric_idx, metric_name in enumerate(args.metric.split("_")):
            mean_val = means[task_idx, metric_idx]
            std_val = stds[task_idx, metric_idx]

            # Format output (handle NaN cases)
            if np.isnan(mean_val) or np.isnan(std_val):
                formatted = "NaN"
            else:
                formatted = f"{mean_val:.3f}+/-{std_val:.3f}"

            metric_results.append(f"{metric_name} {formatted}")

        # Join all metric results for the current task into a string
        metrics_str = ", ".join(metric_results)
        info(f"{task_name}: {metrics_str}")

    return means, stds

if __name__ == "__main__":
    train_score = []
    for i in range(10):
        addin = []
        for j in range(3):
            addin.append(0.3)
        train_score.append(addin)