import random
import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from torch.backends import cudnn

from data.dataset import ImageNet
from utils.helper import load_config


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


def get_loader(cfg, mode):
    dataset_dir = cfg.dataset.dataroot
    dataset_csv = cfg.dataset.dataset_csv
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
            batch_size=cfg.train.batch_size,
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
            batch_size=cfg.train.batch_size,
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
            batch_size=cfg.train.batch_size,
            shuffle=False,
            drop_last=False
        )
        return test_loader

    elif mode == 'trigger':

        tf = transforms.Compose([normalize])

        trigger_data = ImageNet(cfg.attack.trigger_data, cfg.attack.trigger_csv, mode, tf)
        trigger_loader = DataLoader(
            trigger_data,
            batch_size=cfg.attack.wm_batchsize,
            shuffle=False,
            drop_last=False
        )
        return trigger_loader

