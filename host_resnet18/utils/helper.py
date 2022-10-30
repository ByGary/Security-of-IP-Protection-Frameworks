import logging
import shutil
import torch
import os
import time
import csv
import random
import warnings
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from config.defaults import get_default_config


def load_config():
    config = get_default_config()
    config.merge_from_file(r"")  # Fill in the absolute path of YAML file.
    if torch.cuda.is_available():
        config.system.device = 'cuda'
        config.train.dataloader.pin_memory = True
    else:
        config.system.device = 'cpu'
        config.validation.dataloader.pin_memory = False
    config.freeze()
    return config


cfg = load_config()

if torch.cuda.is_available():
    cudnn.benchmark = False
    if cfg.train.seed is not None:
        np.random.seed(cfg.train.seed)  # Numpy module.
        random.seed(cfg.train.seed)  # Python random module.
        torch.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers.
        torch.cuda.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed_all(cfg.train.seed)  # Sets the seed for generating random numbers on all GPUs.
        cudnn.deterministic = True


def create_folder(runs_folder):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)
    start_time = f'{time.strftime("%Y.%m.%d--%H-%M-%S")}'
    this_run_folder = os.path.join(runs_folder, start_time)

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoint'))
    os.makedirs(os.path.join(this_run_folder, 'writer/scalar'))
    os.makedirs(os.path.join(this_run_folder, 'writer/cm'))
    os.makedirs(os.path.join(this_run_folder, 'code'))
    return this_run_folder


def write_scalars(epoch, file_name, metrics, effective_duration, total_duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [name for name in metrics.keys()] + ['effective_duration'] + ['total_duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(value) for value in metrics.values()] + \
                       ['{:.2f}'.format(effective_duration)] + ['{:.2f}'.format(total_duration)]
        writer.writerow(row_to_write)


def plot_scalars(epoch, run_folder, train_metrics, val_metrics):
    writer = SummaryWriter(os.path.join(run_folder, 'writer/scalar'))
    # Plot the curves of training and validation in the same figure.
    for train_tag, val_tag in zip(train_metrics.keys(), val_metrics.keys()):
        writer.add_scalars(train_tag, {'train': train_metrics[train_tag],
                                       'val': val_metrics[val_tag]}, global_step=epoch)
    # Plot the curves of losses and positive metrics jointly during training.
    for tag in train_metrics.keys():
        writer.add_scalar('train_scalars/' + tag, train_metrics[tag], global_step=epoch)
    # Plot the curves of losses and positive metrics jointly during validation.
    for tag in val_metrics.keys():
        writer.add_scalar('val_scalars/' + tag, val_metrics[tag], global_step=epoch)
    writer.close()


def plot_confusion_matrix(mode, step, run_folder, y_true, y_pred, data_loader):
    writer = SummaryWriter(os.path.join(run_folder, 'writer/cm'))
    # constant for classes
    classes = list(data_loader.dataset.name2label.values())
    # build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(25, 20))

    writer.add_figure(mode + "_confusion_matrix", sn.heatmap(df_cm, annot=True).get_figure(), global_step=step)
    writer.close()


def save_checkpoint(epoch, model, optimizer, val_metrics, run_folder):

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['Loss'],
        'acc': val_metrics['Acc'],
        'prec': val_metrics['Prec'],
        'recall': val_metrics['Recall'],
        'f1': val_metrics['F1'],
        'epoch': epoch
    }

    if(run_folder!=None):
        checkpoint_folder = os.path.join(run_folder, 'checkpoint')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        checkpoint_filename = f'best_acc_' + str(epoch) + 'epoch_' + str(
            round(val_metrics['Acc'].item(), 6)) + '.pt'
        checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)

        shutil.rmtree(checkpoint_folder)
        os.makedirs(checkpoint_folder)
        torch.save(checkpoint, checkpoint_filename)
        logging.info('According to best accuracy：Save checkpoint to {}.'.format(checkpoint_filename))
    else:  # Save checkpoints generated by early stopping mechanism.
        torch.save(checkpoint, os.path.join('earlystop_checkpoint', 'es_ckp_' + str(epoch) + 'epoch' + '.pt'))
        logging.info('Early Stopping：Save checkpoint to the root directory.')
