import numpy as np
import torch
import random
from torch.backends import cudnn
from utils.helper import *

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


class E2E_EarlyStopping:
    """Early stop the training if validation loss doesn't improve after a given patience."""

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

    def __call__(self, val_loss,
                 epoch, run_folder,
                 ste_model, Dnnet,
                 optimizerH, optimizerN,
                 val_losses_dict, val_metrics_dict,
                 img_quality_dict, criteria):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_E2E_checkpoint(val_loss, epoch, run_folder,
                                      ste_model, Dnnet,
                                      optimizerH, optimizerN,
                                      val_losses_dict, val_metrics_dict,
                                      img_quality_dict, criteria)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func("{} earlyStopping counter: {} out of {}".format(criteria, self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_E2E_checkpoint(val_loss, epoch, run_folder,
                                      ste_model, Dnnet,
                                      optimizerH, optimizerN,
                                      val_losses_dict, val_metrics_dict,
                                      img_quality_dict, criteria)
            self.counter = 0

    def save_E2E_checkpoint(self, val_loss, epoch, run_folder,
                                      ste_model, Dnnet,
                                      optimizerH, optimizerN,
                                      val_losses_dict, val_metrics_dict,
                                      img_quality_dict, criteria):
        '''Save models when loss decrease.'''
        if self.verbose:
            self.trace_func("{} decreased ({:.6f} --> {:.6f}). Saving model ...".format(criteria, self.val_loss_min, val_loss))

#         save_all_models(epoch, run_folder,
#                         ste_model, Dnnet,
#                         optimizerH, optimizerN,
#                         val_losses_dict, val_metrics_dict, img_quality_dict,
#                         criteria, val_loss)

        self.val_loss_min = val_loss


class tuned_EarlyStopping:
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

    def __call__(self, val_loss, epoch, run_folder, Dnnet, optimizerN, val_losses_dict, val_metrics_dict, criteria):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_tuned_host(val_loss, epoch, run_folder, Dnnet, optimizerN, val_losses_dict, val_metrics_dict, criteria)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func("{} earlyStopping counter: {} out of {}".format(criteria, self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_tuned_host(val_loss, epoch, run_folder, Dnnet, optimizerN, val_losses_dict, val_metrics_dict, criteria)
            self.counter = 0

    def save_tuned_host(self, val_loss, epoch, run_folder, Dnnet, optimizerN, val_losses_dict, val_metrics_dict, criteria):
        if self.verbose:
            self.trace_func("{} decreased ({:.6f} --> {:.6f}). Saving model ...".format(criteria, self.val_loss_min, val_loss))
#         save_host(epoch, run_folder, Dnnet, optimizerN, val_losses_dict, val_metrics_dict)
        
        self.val_loss_min = val_loss