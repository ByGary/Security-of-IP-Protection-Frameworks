import csv
import os
import random
import shutil
import time
import torch
import logging
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.backends import cudnn
from torch.utils.data import DataLoader
from thop import profile, clever_format
from torchsummary import summary
from matplotlib import pyplot as plt
from collections import defaultdict
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from config.defaults import get_default_config


def load_config():
    config = get_default_config()
    config.merge_from_file(r"")  # Fill in the absolute path of YAML file.
    if torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'
    config.freeze()
    return config

cfg = load_config()

if torch.cuda.is_available():
    cudnn.benchmark = True
    if cfg.seed is not None:
        np.random.seed(cfg.seed)  # Numpy module.
        random.seed(cfg.seed)  # Python random module.
        torch.manual_seed(cfg.seed)  # Sets the seed for generating random numbers.
        torch.cuda.manual_seed(cfg.seed)  # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed_all(cfg.seed)  # Sets the seed for generating random numbers on all GPUs.
        cudnn.deterministic = True


def create_folder(runs_folder):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoint'))
    os.makedirs(os.path.join(this_run_folder, 'writer/scalar'))

    return this_run_folder


def plot_scalars(epoch, run_folder, train_losses_dict, train_acc_dict, val_losses_dict, val_acc_dict):
    writer = SummaryWriter(os.path.join(run_folder, 'writer/scalar'))
    # Plot the curves of losses and positive metrics jointly during training.
    for loss_tag in train_losses_dict.keys():
        writer.add_scalar('train_scalars/'+loss_tag, train_losses_dict[loss_tag].avg, global_step=epoch)
    for metirc_tag in train_acc_dict.keys():
        writer.add_scalars('train_scalars/'+metirc_tag, {metirc_tag: train_acc_dict[metirc_tag].avg}, global_step=epoch)
    # Plot the curves of losses and positive metrics jointly during validation.
    if val_losses_dict is not None and val_acc_dict is not None:
        for loss_tag in val_losses_dict.keys():
            writer.add_scalar('val_scalars/' + loss_tag, val_losses_dict[loss_tag].avg, global_step=epoch)
        for metirc_tag in val_acc_dict.keys():
            writer.add_scalars('val_scalars/' + metirc_tag, {metirc_tag: val_acc_dict[metirc_tag].avg}, global_step=epoch)

    writer.close()


def save_checkpoint(epoch, run_folder, ana_model, optimizer, val_losses_dict, val_acc_dict):
    checkpoint_folder = os.path.join(run_folder, 'checkpoint')
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    model = {
        'epoch': epoch,
        'train_loss': val_losses_dict['ana_loss'].avg,
        'val_loss': val_losses_dict['ana_loss'].avg,
        'train_accuracy': val_acc_dict['accuracy'].avg,
        'val_accuracy': val_acc_dict['accuracy'].avg,
        'model_state_dict': ana_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr']
    }
    # It depends on whether you would like to preserve every 'best' checkpoint.
    # shutil.rmtree(save_path)
    # os.makedirs(save_path)
    torch.save(model, checkpoint_folder + '/ana_model_' + str(epoch) + 'epoch_' + str(round(val_losses_dict['ana_loss'].avg, 4)) + '.pt')

    logging.info('Save checkpoint to {}'.format(checkpoint_folder))
