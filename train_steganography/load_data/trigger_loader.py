import logging
import random
import torch
import numpy as np
import torchvision.transforms as transforms

from torch.backends import cudnn
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from PIL import Image

from load_data.dataset import ImageNet
from utils.helper import load_config

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


def get_loader(cfg: CfgNode, mode: str):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if mode == 'trigger':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize
        ])

        trigger_data = ImageNet(cfg.attack.trigger_data, cfg.attack.trigger_csv, mode, tf)
        trigger_loader = DataLoader(
            trigger_data,
            batch_size=cfg.attack.wm_batchsize,
            shuffle=False,
            drop_last=False
        )
        return trigger_loader

    elif mode == 'watermark':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize,
            transforms.Grayscale(num_output_channels=1)  # Note that En2D receives gray watermarks as inputs, while GglNet can process three-channel ones.
        ])

        wm_set = ImageNet(cfg.attack.wm_root, cfg.attack.wm_csv, mode, tf)
        wm_loader = DataLoader(
            wm_set,
            batch_size=1,  # Note that only one watermark is loaded every batch.
            shuffle=False,
            drop_last=False
        )
        return wm_loader