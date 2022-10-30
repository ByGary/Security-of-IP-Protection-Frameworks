import logging
import os
import random
import torch
import torchvision.datasets
import numpy as np
import torchvision.transforms as transforms
from torch.backends import cudnn
from yacs.config import CfgNode
from PIL import Image
from torch.utils.data import DataLoader

from load_data.dataset import ImageNet
from utils.helper import load_config, denormalize

cfg = load_config()
if torch.cuda.is_available():
    cudnn.benchmark = True
    if cfg.train.seed is not None:
        np.random.seed(cfg.train.seed)  # Numpy module.
        random.seed(cfg.train.seed)  # Python random module.
        torch.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers.
        torch.cuda.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed_all(cfg.train.seed)  # Sets the seed for generating random numbers on all GPUs.
        cudnn.deterministic = True


def get_loader(cfg, mode):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tf = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
        transforms.ToTensor(),
        normalize
    ])

    cover_data = ImageNet(cfg.train_on_mini.cover_data, cfg.train_on_mini.cover_csv, mode, tf)
    secret_data = ImageNet(cfg.train_on_mini.secret_data, cfg.train_on_mini.secret_csv, mode, tf)

    if mode == 'cover_train' or mode == 'trigger_cover':
        cover_train_loader = DataLoader(
            cover_data,
            batch_size=cfg.train.batchsize,
            shuffle=cfg.train.dataloader.shuffle,
            drop_last=cfg.train.dataloader.drop_last,
        )
        return cover_train_loader

    elif mode == 'cover_val' or mode == 'cover_test':
        cover_val_loader = DataLoader(
            cover_data,
            batch_size=cfg.train.batchsize,
            shuffle=False,
            drop_last=False,
        )
        return cover_val_loader

    elif mode == 'secret_train':
        secret_train_loader = DataLoader(
            secret_data,
            batch_size=cfg.train.batchsize,
            shuffle=cfg.train.dataloader.shuffle,
            drop_last=cfg.train.dataloader.drop_last,
        )
        return secret_train_loader

    elif mode == 'secret_val' or mode == 'secret_test':
        secret_val_loader = DataLoader(
            secret_data,
            batch_size=cfg.train.batchsize,
            shuffle=False,
            drop_last=False,
        )
        return secret_val_loader

    elif mode == 'trigger_secret':
        secret_loader = DataLoader(
            secret_data,
            batch_size=1,  # Only one secret image is loaded every batch.
            shuffle=cfg.train.dataloader.shuffle,
            drop_last=cfg.train.dataloader.drop_last,
        )
    

def load_secret(cfg):

    device = torch.device(cfg.system.device)
    secret_loader, cover_loader = get_loader(cfg, 'trigger_secret'), get_loader(cfg, 'trigger_cover')
    all_wm, all_wm_label = {}, {}

    for wm, wm_label in secret_loader:
        wm, wm_label = wm.to(device), wm_label.to(device)
        all_wm[wm_label.item()] = wm
        all_wm_label[wm_label.item()] = wm_label
    cover_imgs, cover_img_labels, wms, wm_labels = [], [], [], []

    for cover_img, cover_img_label in cover_loader:
        cover_img = cover_img.to(device)
        cover_imgs.append(cover_img)
        cover_img_labels.append(cover_img_label)

        temp_wm = torch.LongTensor().to(device)
        temp_wm_label = torch.LongTensor().to(device)
        # Build a random mapping between the labels of cover images and that of secret images respectively.
        for a_cover_label in cover_img_label:
            if a_cover_label == 0:
                temp_wm = torch.cat((temp_wm, all_wm[98]))
                temp_wm_label = torch.cat((temp_wm_label, all_wm_label[98]))
            elif a_cover_label == 51:
                temp_wm = torch.cat((temp_wm, all_wm[7]))
                temp_wm_label = torch.cat((temp_wm_label, all_wm_label[7]))
            elif a_cover_label == 72:
                temp_wm = torch.cat((temp_wm, all_wm[83]))
                temp_wm_label = torch.cat((temp_wm_label, all_wm_label[83]))
            elif a_cover_label == 89:
                temp_wm = torch.cat((temp_wm, all_wm[56]))
                temp_wm_label = torch.cat((temp_wm_label, all_wm_label[56]))

        wms.append(temp_wm)
        wm_labels.append(temp_wm_label)

    return wms, wm_labels

