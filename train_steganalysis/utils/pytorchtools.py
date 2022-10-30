import numpy as np
import torch
import random
from torch.backends import cudnn
from utils.helper import load_config, save_checkpoint

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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, epoch, run_folder, ste_model, optimizer, val_losses_dict, val_img_quality_dict):
        val_loss = val_losses_dict['loss_ste'].avg
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_best_checkpoint(epoch, run_folder, ste_model, optimizer, val_losses_dict, val_img_quality_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func("early stopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_best_checkpoint(epoch, run_folder, ste_model, optimizer, val_losses_dict, val_img_quality_dict)
            self.counter = 0

    def save_best_checkpoint(self, epoch, run_folder, ste_model, optimizer, val_losses_dict, val_img_quality_dict):
        val_loss = val_losses_dict['loss_ste'].avg
        if self.verbose:
            self.trace_func("loss_ste decreased ({:.6f} --> {:.6f}). Saving model ...".format(self.val_loss_min, val_loss))

        save_checkpoint(epoch, run_folder, ste_model, optimizer, val_losses_dict, val_img_quality_dict)

        self.val_loss_min = val_loss
