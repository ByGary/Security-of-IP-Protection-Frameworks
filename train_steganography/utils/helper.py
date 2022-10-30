import csv
import os
import random
import shutil
import time
import torch
import logging
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.backends import cudnn
from torch.utils.data import DataLoader
from thop import profile, clever_format
from torchsummary import summary
from matplotlib import pyplot as plt
from collections import defaultdict
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from config.defaults import get_default_config


def load_config():
    config = get_default_config()
    config.merge_from_file(r"")  # Fill in the absolute path of YAML file.
    if torch.cuda.is_available():
        config.system.device = 'cuda'
    else:
        config.system.device = 'cpu'
    config.freeze()
    return config

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


def create_folder(runs_folder):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoint'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/ste_img/train'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/ste_img/val'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/ex_secret/train'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/ex_secret/val'))
    os.makedirs(os.path.join(this_run_folder, 'ste_image'))
    os.makedirs(os.path.join(this_run_folder, 'rev_image'))
    os.makedirs(os.path.join(this_run_folder, 'writer/scalar'))

    return this_run_folder


def write_scalars(epoch, file_name, losses_dict, img_quality_dict, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [tag for tag in losses_dict.keys()] + [tag for tag in img_quality_dict.keys()] +  ['duration']
            writer.writerow(row_to_write)

        row_to_write = [epoch] + ['{:.4f}'.format(value.avg) for value in losses_dict.values()] + \
                       ['{:.2f}'.format(img_quality.avg) for img_quality in img_quality_dict.values()] + \
                       ['{:.2f}'.format(duration)]
        writer.writerow(row_to_write)


def plot_scalars(epoch, run_folder, train_losses_dict, train_img_quality_dict, val_losses_dict, val_img_quality_dict):
    writer = SummaryWriter(os.path.join(run_folder, 'writer/scalar'))
    # Plot the loss curves of training and validation in the same figure.
    for train_tag, val_tag in zip(train_losses_dict.keys(), val_losses_dict.keys()):
        writer.add_scalars(train_tag, {'train': train_losses_dict[train_tag].avg,
                                      'val': val_losses_dict[val_tag].avg}, global_step=epoch)
    # Plot the positive metrics of training and validation in the same figure.
    for train_tag, val_tag in zip(train_img_quality_dict.keys(), val_img_quality_dict.keys()):
        writer.add_scalars(train_tag, {'train': train_img_quality_dict[train_tag].avg,
                                       'val': val_img_quality_dict[val_tag].avg}, global_step=epoch)
    # Plot the curves of losses and positive metrics jointly during training.
    for loss_tag in train_losses_dict.keys():
        writer.add_scalar('train_scalars/'+loss_tag, train_losses_dict[loss_tag].avg, global_step=epoch)
    for metirc_tag in train_img_quality_dict.keys():
        writer.add_scalars('train_scalars/'+metirc_tag, {metirc_tag: train_img_quality_dict[metirc_tag].avg}, global_step=epoch)
    # Plot the curves of losses and positive metrics jointly during validation.
    for loss_tag in val_losses_dict.keys():
        writer.add_scalar('val_scalars/' + loss_tag, val_losses_dict[loss_tag].avg, global_step=epoch)
    for metirc_tag in val_img_quality_dict.keys():
        writer.add_scalars('val_scalars/' + metirc_tag, {metirc_tag: val_img_quality_dict[metirc_tag].avg}, global_step=epoch)

    writer.close()


def save_cat_image(cfg, epoch, run_folder, cover_img, ste_img, secret_img, ex_secret, mode):
    # denormalize first
    cover_img, ste_img, secret_img, ex_secret = \
        denormalize(cover_img), denormalize(ste_img), denormalize(secret_img), denormalize(ex_secret)
    # image difference
    resi_cover_1, resi_secret_1 = ste_img - cover_img, ex_secret - secret_img
    resi_cover_5, resi_secret_5 = (ste_img - cover_img) * 5, (ex_secret - secret_img) * 5

    # Note that color images and gray images can NOT be concatenated in the same CSS Sprites.
    result_ste_img = torch.cat([cover_img, ste_img, resi_cover_1, resi_cover_5], 0)
    torchvision.utils.save_image(result_ste_img,
                                 run_folder + '/cat_image/ste_img/' + mode + '/Epoch_' + str(epoch) + '.png',
                                 nrow=cfg.train.batchsize,
                                 padding=1, normalize=False)
    result_sec_img = torch.cat([secret_img, ex_secret, resi_secret_1, resi_secret_5], 0)
    torchvision.utils.save_image(result_sec_img,
                                 run_folder + '/cat_image/ex_secret/' + mode + '/Epoch_' + str(epoch) + '.png',
                                 nrow=cfg.train.batchsize,
                                 padding=1, normalize=False)


def save_separate_image(run_folder, cover_imgs, ste_imgs, rev_imgs, cover_test_loader):
    cover_imgs, ste_imgs, rev_imgs = denormalize(cover_imgs), denormalize(ste_imgs), denormalize(rev_imgs)

    cover_root = run_folder + '/full_image/cover_imgs/'
    trigger_root = run_folder + '/full_image/ste_imgs/'
    ex_root = run_folder + '/full_image/rev_imgs/'
    # Save images in jpg or mat format.
    for i, img in enumerate(cover_imgs):
        torchvision.utils.save_image(img,  cover_root + str(i+1) + '.jpg')
#         mdic = {str(i+1): img.detach().cpu().numpy()}
#         io.savemat(cover_root + str(i+1) + '.mat', mdic)
    for i, img in enumerate(ste_imgs):
        torchvision.utils.save_image(img, trigger_root + str(i+1) + '.jpg')
#         mdic = {str(i+1): img.detach().cpu().numpy()}
#         io.savemat(trigger_root + str(i+1) + '.mat', mdic)
    for i, img in enumerate(rev_imgs):
        torchvision.utils.save_image(img, ex_root + str(i+1) + '.jpg')


def save_trigger(run_folder, cover_imgs, ste_imgs, rev_imgs, trigger_labels, cover_test_loader):
    # Different from the function 'save_separate_image', here trigger labels are required for naming images.
    cover_imgs, ste_imgs, rev_imgs = denormalize(cover_imgs), denormalize(ste_imgs), denormalize(rev_imgs)

    resi_steg = (ste_imgs - cover_imgs) * 5
    cover_root = run_folder + '/full_image/cover_imgs/'
    trigger_root = run_folder + '/full_image/ste_imgs/'
    ex_root = run_folder + '/full_image/rev_imgs/'
    # Save images both in jpg and mat format if necessary.
    for i, img in enumerate(cover_imgs):
        torchvision.utils.save_image(img,  cover_root + str(i+1) + '.jpg')
        mdic = {str(i+1): img.detach().cpu().numpy()}
        io.savemat(cover_root + str(i+1) + '.mat', mdic)
    for i, img in enumerate(ste_imgs):
        torchvision.utils.save_image(img, trigger_root + str(i+1) + '_' + str(trigger_labels[i].item()) + '_' + '.jpg')
        torchvision.utils.save_image(resi_steg[i], trigger_root + str(i+1) + '_resi_' + '.jpg')
        mdic = {str(i+1): img.detach().cpu().numpy()}
        io.savemat(trigger_root + str(i+1) + '_' + str(trigger_labels[i].item()) +  '.mat', mdic)
    for i, img in enumerate(rev_imgs):
        torchvision.utils.save_image(img, ex_root + str(i+1) + '.jpg')
        

def save_rev(cfg, run_folder, secret_img, ex_secret):

    tf = transforms.Grayscale(num_output_channels=1)
    secret_img, ex_secret = tf(denormalize(secret_img)), tf(denormalize(ex_secret))
    resi_rev_1, resi_rev_5 = ex_secret - secret_img, (ex_secret - secret_img) * 5
    # save jointly
    result_rev_img = torch.cat([secret_img, ex_secret, resi_rev_1, resi_rev_5], 0)
    torchvision.utils.save_image(result_rev_img, os.path.join(run_folder, 'rev_image', 'example.png'),
                            nrow=cfg.attack.wm_batchsize, padding=1, normalize=False)
    # save separately
    for i, (s_img, e_img) in enumerate(zip(secret_img, ex_secret)):
        torchvision.utils.save_image(s_img, os.path.join(run_folder, 'rev_image', str(i) + 'secret_img.png'))
        torchvision.utils.save_image(e_img, os.path.join(run_folder, 'rev_image', str(i) + 'ex_secret.png'))
        

def save_checkpoint(epoch, run_folder, ste_model, optimizer, val_losses_dict, img_quality_dict):
    checkpoint_folder = os.path.join(run_folder, 'checkpoint')
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    model = {
        'model_state_dict': ste_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_hid': val_losses_dict['loss_hid'].avg,
        'loss_rev': val_losses_dict['loss_rev'].avg,
        'loss_ste': val_losses_dict['loss_ste'].avg,
        'ste_psnr': img_quality_dict['ste_psnr'].avg,
        'ex_psnr': img_quality_dict['ex_psnr'].avg,
        'epoch': epoch
    }

    # It depends on whether you would like to preserve every 'best' checkpoint.
    # shutil.rmtree(checkpoint_folder)
    # os.makedirs(checkpoint_folder)
    torch.save(model, checkpoint_folder + '/ste_model_' + str(epoch) + 'epoch' + '.pt')

    logging.info('Save checkpoint to {}'.format(checkpoint_folder))


def cal_psnr(cover_imgs, modified_imgs):

    cover_imgs, modified_imgs = denormalize(cover_imgs), denormalize(modified_imgs)
    wm_batchsize = modified_imgs.size(0)

    gary_tf = transforms.Grayscale(num_output_channels=1)
    cover_imgs = torch.as_tensor(gary_tf(cover_imgs)).cpu().numpy()
    modified_imgs = torch.as_tensor(gary_tf(modified_imgs)).cpu().numpy()
    total_psnr = 0

    for cover_img, modified_img in zip(cover_imgs, modified_imgs):
        total_psnr += psnr(cover_img, modified_img)

    return total_psnr / wm_batchsize


def RGB_to_gray(img: torch.tensor):
    tf = transforms.Grayscale(num_output_channels=1)
    return tf(img)


def denormalize(img_hat):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    img = img_hat.cpu() * std + mean

    return img
