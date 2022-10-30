import torch
import random
import torch.nn as nn
import numpy as np
from torch.backends import cudnn
from utils.helper import load_config

cfg = load_config()

if torch.cuda.is_available():
    cudnn.benchmark = True
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        cudnn.deterministic = True


class DiscriminatorNet(nn.Module):
    def __init__(self, nc=3, nhf=8, output_function=nn.Sigmoid):
        super(DiscriminatorNet, self).__init__()
        #self.key_pre = RevealPreKey()
        # input is (3+1) x 256 x 256
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 4, 2, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 4, 2, 1),
            nn.BatchNorm2d(nc),
            nn.ReLU(True)
            #output_function()
        )
        self.linear = nn.Sequential(
            nn.Linear(12288, 1),
            output_function()
        )

        # nn.Sigmoid()
        #self.reveal_Message = UnetRevealMessage()

    def forward(self, input):
        #pkey = pkey.view(-1, 1, 32, 32)
        #pkey_feature = self.key_pre(pkey)

        #input_key = torch.cat([input, pkey_feature], dim=1)
        ste_feature = self.main(input)
        out = ste_feature.view(ste_feature.shape[0], -1)
        out = self.linear(out)
        return out


class DiscriminatorNet_mnist(nn.Module):
    def __init__(self):#3072):
        super(DiscriminatorNet_mnist, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
