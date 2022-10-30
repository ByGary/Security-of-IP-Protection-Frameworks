import torchvision
import os
import glob
import csv
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
from torch.backends import cudnn

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


class ImageNet(Dataset):
    def __init__(self, dataset_dir, dataset_csv, mode, transform: torchvision.transforms):
        super(ImageNet, self).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_csv = dataset_csv
        self.csv_file = os.path.join(dataset_csv, '')  # Fill in the filename of CSV.
        self.mode = mode
        self.transform = transform
        self.name2label = {}

        for className in sorted(os.listdir(self.dataset_dir)):
            if not os.path.isdir(self.dataset_dir):
                continue
            self.name2label[className] = len(self.name2label.keys())
        self.images, self.labels = self.load_csv()

        if mode == 'cover_train' or mode == 'secret_train':
            self.images = self.images[:int(0.9 * len(self.images))]
            self.labels = self.labels[:int(0.9 * len(self.labels))]
        elif mode == 'cover_val' or mode == 'secret_val':
            self.images = self.images[int(0.9 * len(self.images)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]
        elif mode == 'trigger' or mode == 'watermark' or mode == 'cover_test' or mode == 'secret_test' or mode == 'trigger_cover' or mode == 'trigger_secret':
            self.images = self.images[:]
            self.labels = self.labels[:]

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        # Convert arrays to tensors first when given data in mat format.
        if img.split('.')[-1] == 'mat':
            mat_img = loadmat(img)
            np_img = mat_img[str(img.split('/')[-1].split('.')[0])]
            img = torch.from_numpy(np_img)
            # Only normalization is required in this case.
            mat_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            img = mat_tf(img)
            label = torch.tensor(label)
            return img, label

        img = self.transform(img)
        label = torch.tensor(label)
        return img, label

    def load_csv(self):
        # Create a CSV file when it is not offered.
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
                    
        # Directly read the CSV next time.
        images, labels = [], []
        with open(os.path.join(self.csv_file)) as f:
            reader = csv.reader(f)
            # next(reader)  # Pay attention to the header.
            for row in reader:
                img, label = row
                images.append(img)
                label = int(label)
                labels.append(label)
            assert len(images) == len(labels)
            return images, labels


