import os
import imageio
import torch
import random
import numbers
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from PIL import Image
from scipy import misc
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from utils.helper import *


cfg = load_config()
device = torch.device(cfg.device)

# to be reproducible
if torch.cuda.is_available():
    cudnn.benchmark = True
    if cfg.seed is not None:
        np.random.seed(cfg.seed)  # Numpy module.
        random.seed(cfg.seed)  # Python random module.
        torch.manual_seed(cfg.seed)  # Sets the seed for generating random numbers.
        torch.cuda.manual_seed(cfg.seed)  # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed_all(cfg.seed)  # Sets the seed for generating random numbers on all GPUs.
        cudnn.deterministic = True


class RandomRotation(object):

    def __init__(self, resample=False, expand=False, center=None, p=cfg.p_rot ):

        self.degrees = 90
        self.resample = resample
        self.center = center
        self.expand = expand
        self.p = p

    @staticmethod
    def get_params(degrees):
        angle = degrees
        return angle

    def __call__(self, sample):
        angle = self.get_params(self.degrees)

        if random.random() < self.p:
            return F.rotate(sample, angle, self.resample, self.expand, self.center)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)

        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'

        return format_string


class RandomHorizontalFlip(object):

    def __init__(self, p=cfg.p_hflip):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return F.hflip(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToPILImage(object):

    def __call__(self, sample):
        return F.to_pil_image(sample)


class ToGrayImage(object):

    def __call__(self, sample):
        return F.rgb_to_grayscale(Dataset_Load_trigger)


class ToTensor(object):

    def __call__(self, sample):
        sample = np.asarray(sample).reshape((cfg.resize, cfg.resize, 1)).astype(np.float32)
        sample = torch.from_numpy(sample)
        sample = torch.transpose(torch.transpose(sample, 2, 0), 1, 2)
        return sample


class Dataset_Load(Dataset):

    def __init__(self, cover_path, stego_path, size, transform=None):
        self.cover = cover_path
        self.stego = stego_path
        self.transforms = transform
        self.data_size = size

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        index += 1
        img_name = str(index) + ".jpg"

        cover_img = Image.open(os.path.join(self.cover, img_name))
        stego_img = Image.open(os.path.join(self.stego, img_name))

        label1 = torch.tensor(0, dtype=torch.long).to(device)
        label2 = torch.tensor(1, dtype=torch.long).to(device)

        sample = {'cover': cover_img, 'stego': stego_img}
        if self.transforms is not None:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)

            sample = {'cover': cover_img, 'stego': stego_img}

        sample['label'] = [label1, label2]

        return sample


class Dataset_Load_trigger(Dataset):

    def __init__(self, stego_path, size, transform=None):
        self.stego = stego_path
        self.transforms = transform
        self.data_size = size

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        index += 1
        img_name = str(index) + ".jpg"
        
        stego_img = imageio.imread(os.path.join(self.stego, img_name))
        label = torch.tensor(1, dtype=torch.long).to(device)

        sample = stego_img
        sample = {'stego': sample}

        if self.transforms is not None:
            sample = self.transforms(sample)
            
        sample['label'] = [label]

        return sample




