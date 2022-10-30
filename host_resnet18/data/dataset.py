import torchvision
import os
import glob
import csv
import torch
import random
import numpy as np

from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.backends import cudnn
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


class ImageNet(Dataset):
    def __init__(self, dataset_dir, dataset_csv, mode, transform: torchvision.transforms):
        super(ImageNet, self).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_csv = dataset_csv
        self.csv_file = os.path.join(dataset_csv, 'mini-ImageNet.csv')
        self.mode = mode
        self.transform = transform
        self.name2label = {}

        for className in sorted(os.listdir(self.dataset_dir)):
            if not os.path.isdir(self.dataset_dir):
                continue
            self.name2label[className] = len(self.name2label.keys())
        self.images, self.labels = self.load_csv()

        if mode == 'train':
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.8 * len(self.images)):int(0.9 * len(self.images))]
            self.labels = self.labels[int(0.8 * len(self.labels)):int(0.9 * len(self.labels))]
        elif mode == 'test':
            self.images = self.images[int(0.9 * len(self.images)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]
        elif mode == 'trigger':
            self.images = self.images[:]
            self.labels = self.labels[:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        # mat_index = img.split(os.sep)[-1].split('.')[0]
        # img = torch.from_numpy(loadmat(os.path.join(img))[str(mat_index)])
        img = self.transform(img)
        label = torch.tensor(label)
        return img, label

    def load_csv(self):

        if not os.path.exists(os.path.join(self.csv_file)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.dataset_dir, name, '*.mat'))
                images += glob.glob(os.path.join(self.dataset_dir, name, '*.png'))
                images += glob.glob(os.path.join(self.dataset_dir, name, '*.jpg'))
                images += glob.glob(os.path.join(self.dataset_dir, name, '*.jpeg'))

            random.shuffle(images)
            with open(os.path.join(self.csv_file), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    className = img.split(os.sep)[-2]
                    label = self.name2label[className]
                    writer.writerow([img, label])

        images, labels = [], []
        with open(os.path.join(self.csv_file)) as f:
            reader = csv.reader(f)
#             next(reader)
            for row in reader:
                img, label = row
                images.append(img)
                label = int(label)
                labels.append(label)
            assert len(images) == len(labels)
            return images, labels
