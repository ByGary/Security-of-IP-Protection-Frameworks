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
import seaborn as sn
import torchvision.transforms as transforms

from torch.backends import cudnn
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from collections import defaultdict
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr

from config.defaults import get_default_config
# Use "defaults_ACSAC19" if you want to train ACSAC19.
# from config.defaults_ACSAC19 import get_default_config
from torch.utils.data import DataLoader


def load_config():
    config = get_default_config()
    config.merge_from_file(r"")  # Fill in the absolute path of YAML file.
    if torch.cuda.is_available():
        config.system.device = 'cuda'
        config.train.dataloader.pin_memory = True
    else:
        config.system.device = 'cpu'
        config.validation.dataloader.pin_memory = False
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
    # save checkpoints
    os.makedirs(os.path.join(this_run_folder, 'checkpoint/loss_real'))
    os.makedirs(os.path.join(this_run_folder, 'checkpoint/loss_trigger'))
    os.makedirs(os.path.join(this_run_folder, 'checkpoint/loss_cat_Dnn'))
    # save CSS Sprites
    os.makedirs(os.path.join(this_run_folder, 'cat_image/trigger/train'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/trigger/val'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/trigger/test'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/re_wm/train'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/re_wm/val'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/re_wm/test'))
    # save full images
    os.makedirs(os.path.join(this_run_folder, 'full_image/trigger/train'))
    os.makedirs(os.path.join(this_run_folder, 'full_image/trigger/val'))
    os.makedirs(os.path.join(this_run_folder, 'full_image/trigger/test'))
    os.makedirs(os.path.join(this_run_folder, 'full_image/re_wm/train'))
    os.makedirs(os.path.join(this_run_folder, 'full_image/re_wm/val'))
    os.makedirs(os.path.join(this_run_folder, 'full_image/re_wm/test'))
    # save log file of tensorboardX
    os.makedirs(os.path.join(this_run_folder, 'writer/scalar'))
    os.makedirs(os.path.join(this_run_folder, 'writer/cm/val'))
    os.makedirs(os.path.join(this_run_folder, 'writer/cm/test'))

    return this_run_folder


def create_folder_acsac(runs_folder):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    # save checkpoints
    os.makedirs(os.path.join(this_run_folder, 'checkpoint'))
    # save CSS Sprites
    os.makedirs(os.path.join(this_run_folder, 'cat_image/trigger/val'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/trigger/test'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/re_wm/val'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/re_wm/test'))
    # save full images
    os.makedirs(os.path.join(this_run_folder, 'full_image/trigger/val'))
    os.makedirs(os.path.join(this_run_folder, 'full_image/trigger/test'))
    os.makedirs(os.path.join(this_run_folder, 'full_image/re_wm/val'))
    os.makedirs(os.path.join(this_run_folder, 'full_image/re_wm/test'))
    # save cover images(selected randomly)
    os.makedirs(os.path.join(this_run_folder, 'cover_imgs'))
    # save log file of tensorboardX
    os.makedirs(os.path.join(this_run_folder, 'writer/scalar'))
    os.makedirs(os.path.join(this_run_folder, 'writer/cm/val'))
    os.makedirs(os.path.join(this_run_folder, 'writer/cm/test'))

    return this_run_folder


def write_scalars(epoch, file_name, losses_dict, metrics_dict: defaultdict, img_quality_dict, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            if img_quality_dict is None:
                row_to_write = ['epoch'] + [tag for tag in losses_dict.keys()] + [tag for tag in metrics_dict.keys()] + ['duration']
            else:
                row_to_write = ['epoch'] + [tag for tag in losses_dict.keys()] + [tag for tag in metrics_dict.keys()] + [tag for tag in img_quality_dict.keys()] + ['duration']
            writer.writerow(row_to_write)
        if img_quality_dict is None:
            row_to_write = [epoch] + ['{:.4f}'.format(value.avg) for value in losses_dict.values()] + ['{:.6f}'.format(value) for value in metrics_dict.values()] + ['{:.2f}'.format(duration)]
        else:
            row_to_write = [epoch] + ['{:.4f}'.format(value.avg) for value in losses_dict.values()] + ['{:.6f}'.format(value) for value in metrics_dict.values()] + \
                           ['{:.2f}'.format(img_quality.avg) for img_quality in img_quality_dict.values()] + \
                           ['{:.2f}'.format(duration)]

        writer.writerow(row_to_write)


def write_scalars_acsac(epoch, file_name, losses_dict, metrics_dict, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 0:
            row_to_write = ['epoch'] + [tag for tag in losses_dict.keys()] + [tag for tag in metrics_dict.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(value) for value in losses_dict.values()] + ['{:.6f}'.format(value) for value in metrics_dict.values()] + ['{:.2f}'.format(duration)]
        writer.writerow(row_to_write)
        

def plot_scalars(epoch, run_folder, train_losses_dict, train_metrics_dict, val_losses_dict, val_metrics_dict, img_quality_dict):
    writer = SummaryWriter(os.path.join(run_folder, 'writer/scalar'))
    # Plot the curves of training and validation in the same figure.
    for train_tag, val_tag in zip(train_losses_dict.keys(), val_losses_dict.keys()):
        writer.add_scalars(train_tag, {'train': train_losses_dict[train_tag].avg, 'val': val_losses_dict[val_tag].avg}, global_step=epoch)
    for train_tag, val_tag in zip(train_metrics_dict.keys(), val_metrics_dict.keys()):
        writer.add_scalars(train_tag, {'train': train_metrics_dict[train_tag], 'val': val_metrics_dict[val_tag]}, global_step=epoch)
    # Plot the curves of losses and positive metrics jointly during training.
    for loss_tag in train_losses_dict.keys():
        writer.add_scalar('train_scalars/'+loss_tag, train_losses_dict[loss_tag].avg, global_step=epoch)
    for metirc_tag in train_metrics_dict.keys():
        writer.add_scalars('train_scalars/'+metirc_tag, {metirc_tag: train_metrics_dict[metirc_tag]}, global_step=epoch)
    # Plot the curves of losses and positive metrics jointly during validation.
    for loss_tag in val_losses_dict.keys():
        writer.add_scalar('val_scalars/' + loss_tag, val_losses_dict[loss_tag].avg, global_step=epoch)
    for metirc_tag in val_metrics_dict.keys():
        writer.add_scalars('val_scalars/' + metirc_tag, {metirc_tag: val_metrics_dict[metirc_tag]}, global_step=epoch)
    # Plot the curves of PSNR during validation.
    if img_quality_dict is not None:
        for img_tag in img_quality_dict.keys():
            writer.add_scalar('val_scalars/' + img_tag, img_quality_dict[img_tag].avg, global_step=epoch)
    writer.close()
    

def plot_confusion_matrix(epoch, run_folder, tag, y_pred, y_true, data_loader: DataLoader):

    if "test" in tag:
        writer = SummaryWriter(os.path.join(run_folder, 'writer/cm/test'))
    else:
        writer = SummaryWriter(os.path.join(run_folder, 'writer/cm/val'))
    # constant for classes
    if tag == 'train_dis' or tag == 'val_dis' or tag == 'test_dis':
        classes = [0, 1]
        fig_size = 5
    elif tag == 'test_trigger':
        classes = [7, 56, 83, 93]
        fig_size = 15
    else:
        classes = list(data_loader.dataset.name2label.values())
        fig_size = 20
    # Build confusion matrix.
    cf_matrix = confusion_matrix(y_true, y_pred, classes)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(fig_size+5, fig_size))

    writer.add_figure(tag + "_confusion_matrix", sn.heatmap(df_cm, annot=True).get_figure(), global_step=epoch)
    writer.close()


def save_cat_image(cfg, epoch, run_folder, cover_img, trigger, secret_img, trigger_ext_output, mode):
    # denormalize first
    cover_img, trigger, secret_img, trigger_ext_output = \
        denormalize(cover_img), denormalize(trigger), denormalize(secret_img), denormalize(trigger_ext_output)
    # image difference
    resi_cover_1, resi_secret_1 = trigger - cover_img, trigger_ext_output - secret_img
    resi_cover_5, resi_secret_5 = (trigger - cover_img) * 5, (trigger_ext_output - secret_img) * 5
    result_ste_img = torch.cat([cover_img, trigger, resi_cover_1, resi_cover_5], 0)
    # Note that color images and gray images can NOT be concatenated in the same CSS Sprites.
    torchvision.utils.save_image(result_ste_img,
                                 run_folder + '/cat_image/trigger/' + mode + '/Epoch_' + str(epoch) + '.jpg',
                                 nrow=cfg.watermark.wm_batchsize,
                                 padding=1, normalize=False)
    result_sec_img = torch.cat([secret_img, trigger_ext_output, resi_secret_1, resi_secret_5], 0)
    torchvision.utils.save_image(result_sec_img,
                                 run_folder + '/cat_image/re_wm/' + mode + '/Epoch_' + str(epoch) + '.jpg',
                                 nrow=cfg.watermark.wm_batchsize,
                                 padding=1, normalize=False)



def save_cat_image_acsac(cfg, epoch, run_folder, wm_input, wm_img, secret_img, wm_ext, mode):
    # denormalize first
    wm_input, wm_img = denormalize(wm_input), denormalize(wm_img)
    # image difference
    resi_cover_5 = (wm_img - wm_input) * 5
    result_ste_img = torch.cat([wm_input, wm_img, resi_cover_5], 0)

    torchvision.utils.save_image(result_ste_img,
                                 run_folder + '/cat_image/trigger/' + mode + '/epoch_' + str(epoch) + '.png',
                                 nrow=cfg.wm_batchsize,
                                 padding=1, normalize=False)
    if wm_ext is not None:
        secret_img, wm_ext = denormalize(secret_img), denormalize(wm_ext)
        resi_secret_5 = (secret_img - wm_ext) * 5
        result_ext_img = torch.cat([secret_img, wm_ext, resi_secret_5], 0)
        torchvision.utils.save_image(result_ext_img,
                                 run_folder + '/cat_image/re_wm/' + mode + '/epoch_' + str(epoch) + '.png',
                                 nrow=cfg.wm_batchsize,
                                 padding=1, normalize=False)
        

def save_separate_image(epoch, run_folder, triggers, trigger_labels, trigger_ext_output, mode):

    triggers, trigger_ext_output = denormalize(triggers), denormalize(trigger_ext_output)

    trigger_root = os.path.join(run_folder, 'full_image', 'trigger', mode)
    ex_root = os.path.join(run_folder, 'full_image', 're_wm', mode)

    if os.path.getsize(trigger_root) > 0:
        shutil.rmtree(trigger_root)
        os.makedirs(trigger_root)

    if os.path.getsize(ex_root) > 0:
        shutil.rmtree(ex_root)
        os.makedirs(ex_root)

    for i, img in enumerate(triggers):
        torchvision.utils.save_image(img,  trigger_root + '/' + str(i) + '_label_' +
            str(trigger_labels[i//cfg.watermark.wm_batchsize][i % cfg.watermark.wm_batchsize].item()) + '_epoch_' + str(epoch) + '.jpg')
    for i, img in enumerate(trigger_ext_output):
        torchvision.utils.save_image(img, ex_root + '/' + 'epoch_' + str(epoch) + 're_wm' + str(i) + '.jpg')


def save_all_models(epoch, run_folder, ste_model, Dnnet, optimizerH, optimizerN,
                    val_losses_dict, val_metrics_dict, img_quality_dict, criteria, best):
    checkpoint_folder = os.path.join(run_folder, 'checkpoint')
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    Hstate = {
        'Ete': ste_model.state_dict(),
        'optimizerH': optimizerH.state_dict(),
        'loss_H': val_losses_dict['loss_H'].avg,
        'loss_hid': val_losses_dict['loss_hid'].avg,
        'loss_rev': val_losses_dict['loss_rev'].avg,
        'loss_dnn': val_losses_dict['loss_dnn'].avg,
        'ste_psnr': img_quality_dict['ste_psnr'].avg,
        'rev_psnr': img_quality_dict['rev_psnr'].avg,
        'epoch': epoch
    }
    Nstate = {
        'Dnnet': Dnnet.state_dict(),
        'optimizerN': optimizerN.state_dict(),
        'loss_cat_Dnn': val_losses_dict['loss_cat_Dnn'].avg,
        'loss_real': val_losses_dict['loss_real'].avg,
        'loss_trigger': val_losses_dict['loss_trigger'].avg,
        'real_acc': val_metrics_dict['real_acc'],
        'precision': val_metrics_dict['precision'],
        'recall': val_metrics_dict['recall'],
        'f1': val_metrics_dict['f1'],
        'trigger_acc': val_metrics_dict['trigger_acc'],
        'cover_acc': val_metrics_dict['cover_acc'],
        'epoch': epoch
    }
    if criteria is None and best is None:
        save_path = checkpoint_folder
        torch.save(Hstate, save_path + '/Stegan_' + str(epoch) + 'epoch' + '.pt')
        torch.save(Nstate, save_path + '/Dnnet_' + str(epoch) + 'epoch' + '.pt')
    else:
        save_path = os.path.join(run_folder, 'checkpoint', criteria)
        
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        torch.save(Hstate, save_path + '/Stegan_' + str(epoch) + 'epoch_' + '_' + str(round(best, 6)) + '.pt')
        torch.save(Nstate, save_path + '/Dnnet_' + str(epoch) + 'epoch_' + '_' + str(round(best, 6)) + '.pt')

    logging.info('Save checkpoints to {}'.format(checkpoint_folder))



def save_all_models_acsac(epoch, run_folder, Hidnet, Disnet, Dnnet, optimizerH, optimizerD, optimizerN, val_losses_dict, val_metrics_dict, wm_labels):
    checkpoint_folder = os.path.join(run_folder, 'checkpoint')
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    Hstate = {
        'Hidnet': Hidnet.state_dict(),
        'optimizerH': optimizerH.state_dict(),
        'loss_H': val_losses_dict['loss_H'],
        'loss_mse': val_losses_dict['loss_mse'],
        'ssim': val_losses_dict['ssim'],
        'ste_psnr': val_metrics_dict['ste_psnr'],
        'epoch': epoch
    }
    Dstate = {
        'Disnet': Disnet.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'loss_D': val_losses_dict['loss_D'],
        'dis_acc': val_metrics_dict['dis_acc'],
        'epoch': epoch
    }
    Nstate = {
        'Dnnet': Dnnet.state_dict(),
        'optimizerN': optimizerN.state_dict(),
        'loss_DNN': val_losses_dict['loss_DNN'],
        'loss_real': val_losses_dict['loss_real'],
        'real_acc': val_metrics_dict['real_acc'],
        'wm_acc': val_metrics_dict['wm_acc'],
        'wm_labels': None if wm_labels is None else wm_labels,
        'epoch': epoch
    }

    # shutil.rmtree(checkpoint_folder)
    # os.makedirs(checkpoint_folder)
    torch.save(Hstate, checkpoint_folder + '/Hidnet' + str(epoch) + 'epoch' + '.pt')
    torch.save(Dstate, checkpoint_folder + '/Disnet_' + str(epoch) + 'epoch' + '.pt')
    torch.save(Nstate, checkpoint_folder + '/Dnnet_' + str(epoch) + 'epoch' + '.pt')

    logging.info('Save checkpoint to {}'.format(checkpoint_folder))
    

def save_host(epoch, run_folder, Dnnet, optimizerN, val_losses_dict, val_metrics_dict):

    checkpoint_folder = os.path.join(run_folder, 'checkpoint')
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    Nstate = {
        'Dnnet': Dnnet.state_dict(),
        'optimizerN': optimizerN.state_dict(),
        'loss_cat_Dnn': val_losses_dict['loss_cat_Dnn'].avg,
        'loss_real': val_losses_dict['loss_real'].avg,
        'loss_trigger': val_losses_dict['loss_trigger'].avg,
        'real_acc': val_metrics_dict['real_acc'],
        'precision': val_metrics_dict['precision'],
        'recall': val_metrics_dict['recall'],
        'f1': val_metrics_dict['f1'],
        'trigger_acc': val_metrics_dict['trigger_acc'],
        'epoch': epoch
    }

    torch.save(Nstate, checkpoint_folder + '/Dnnet_' + str(epoch) + 'epoch' + '.pt')

    logging.info('Save tuned host model checkpoint to {}'.format(checkpoint_folder))


def cal_psnr(cover_imgs, triggers):

    cover_imgs, triggers = denormalize(cover_imgs), denormalize(triggers)
    wm_batchsize = triggers.size(0)

    gary_tf = transforms.Grayscale(num_output_channels=1)
    cover_imgs = torch.as_tensor(gary_tf(cover_imgs)).cpu().numpy()
    triggers = torch.as_tensor(gary_tf(triggers)).cpu().numpy()
    total_psnr = 0

    for cover_img, trigger in zip(cover_imgs, triggers):
        total_psnr += psnr(cover_img, trigger)

    return total_psnr / wm_batchsize


def RGB_to_gray(img: torch.tensor):
    tf = transforms.Grayscale(num_output_channels=1)
    return tf(img)


def denormalize(img_hat):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    img = img_hat.cpu() * std +mean

    return img

