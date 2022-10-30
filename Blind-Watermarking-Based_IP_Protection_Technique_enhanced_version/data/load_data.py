import logging
import os
import random
import torch
import torchvision.datasets
import numpy as np
import torchvision.transforms as transforms

from torch.backends import cudnn
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from PIL import Image
from data.dataset import ImageNet
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


def get_loader(cfg, mode, trigger_floder=None):

    dataset_dir = cfg.dataset.dataroot
    dataset_csv = cfg.dataset.dataset_csv
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if mode == 'train':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_data = ImageNet(dataset_dir, dataset_csv, mode, tf)
        train_loader = DataLoader(
            train_data,
            batch_size=cfg.train.batchsize,
            shuffle=cfg.train.dataloader.shuffle,
            drop_last=cfg.train.dataloader.drop_last,
            pin_memory=cfg.train.dataloader.pin_memory,
            num_workers=cfg.train.dataloader.num_workers,
            prefetch_factor=cfg.train.dataloader.prefetch_factor
        )
        return train_loader

    elif mode == 'val':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize
        ])

        val_data = ImageNet(dataset_dir, dataset_csv, mode, tf)
        val_loader = DataLoader(
            val_data,
            batch_size=cfg.train.batchsize,
            shuffle=False,
            drop_last=False,
            pin_memory=cfg.train.dataloader.pin_memory,
            num_workers=cfg.train.dataloader.num_workers,
            prefetch_factor=cfg.train.dataloader.prefetch_factor
        )
        return val_loader

    elif mode == 'test':
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize
        ])

        test_data = ImageNet(dataset_dir, dataset_csv, mode, tf)
        test_loader = DataLoader(
            test_data,
            batch_size=cfg.train.batchsize,
            shuffle=False,
            drop_last=False
        )
        return test_loader

    elif mode == 'cover_train' or mode == 'cover_val' or mode == 'cover_test':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize
        ])

        cover_set = ImageNet(r'', r'', mode, tf)
        cover_loader = DataLoader(
            cover_set,
            batch_size=cfg.watermark.wm_batchsize,
            shuffle=True,
            drop_last=False
        )
        return cover_loader

    elif mode == 'watermark':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize,
            transforms.Grayscale(num_output_channels=1)  # Note that En2D receives gray watermarks as inputs, while GglNet can process three-channel ones.
        ])

        wm_set = ImageNet(cfg.watermark.wm_root, cfg.watermark.wm_csv, mode, tf)
        wm_loader = DataLoader(
            wm_set,
            batch_size=1,  # Only one watermark is loaded every batch.
            shuffle=False,
            drop_last=False
        )
        return wm_loader

    elif mode == 'trigger_train' or mode == 'trigger_val' or mode == 'trigger_test':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize
        ])

        trigger_data = ImageNet(cfg.trigger.data, cfg.trigger.csv, mode, tf)
        trigger_loader = DataLoader(
            trigger_data,
            batch_size=cfg.watermark.wm_batchsize,
            shuffle=False,
            drop_last=False
        )
        return trigger_loader

    elif mode == 'original_cover':  # Load cover images as ACSAC19 does.
        cover_set = ImageNet(dataset_dir, '', mode, transform_test)  # Fill in the path of CSV file for cover data.
        cover_loader = DataLoader(
            cover_set,
            batch_size=cfg.wm_batchsize,
            # DO NOT set to True, since every trigger is assigned a predefine label.
            # If cover images are shuffled every epoch, their labels are required to be matched strictly.
            shuffle=False,
            drop_last=False
        )
    return cover_loader


def load_cover_and_wm(cfg, mode):
    device = torch.device(cfg.system.device)
    wm_loader = get_loader(cfg, 'watermark')
    all_wm, all_wm_label = {}, {}

    for wm, wm_label in wm_loader:
        wm, wm_label = wm.to(device), wm_label.to(device)
        all_wm[wm_label.item()] = wm
        all_wm_label[wm_label.item()] = wm_label

    cover_loader = get_loader(cfg, mode)
#     logging.info("wm_loader:{} cover_loader:{}".format(len(wm_loader.dataset), len(cover_loader.dataset)))
    cover_imgs, cover_img_labels, wms, wm_labels = [], [], [], []

    for cover_img, cover_img_label in cover_loader:

        cover_img, cover_img_label = cover_img.to(device), cover_img_label.to(device)
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

    return cover_imgs, cover_img_labels, wms, wm_labels