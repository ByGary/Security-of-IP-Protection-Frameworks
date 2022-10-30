import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1
from yacs.config import CfgNode
from utils.AverageMeter import AverageMeter
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

def train(epoch, run_folder, cfg, device, train_loader, optimizer, criterion, model):
    model.train()
    step = 1
    train_duration, total_duration = 0, 0
    metrics_dict = defaultdict()
    loss_avg = AverageMeter()

    acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    precision = Precision(num_classes=cfg.dataset.num_classes, average='weighted')
    recall = Recall(num_classes=cfg.dataset.num_classes, average='weighted')
    f1 = F1(num_classes=cfg.dataset.num_classes, average='weighted')

    epoch_start_time = time.time()

    for _, (input, target) in enumerate(train_loader):

        batch_start_time = time.time()

        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_duration += time.time() - batch_start_time

        train_preds_cpu = torch.argmax(output, dim=1).cpu()
        target_cpu = target.cpu()

        loss_avg.update(loss.item(), input.size(0))
        acc.update(train_preds_cpu, target_cpu)
        precision.update(train_preds_cpu, target_cpu)
        recall.update(train_preds_cpu, target_cpu)
        f1.update(train_preds_cpu, target_cpu)

        if step % cfg.train.print_freq == 0 or step == (len(train_loader)):
            logging.info('Epoch: {}/{} Step: {}/{}'
                     'Loss: {:.4f} (average: {:.4f})\t'
                     'acc: {:.4%}\tprec: {:.4%}\trecall: {:.4%}\tf1: {:.4%}'.format(
            epoch, cfg.train.num_epochs, step, len(train_loader),
            loss_avg.val, loss_avg.avg,
            acc.compute(), precision.compute(), recall.compute(), f1.compute()))
            logging.info('-' * 120)

        step += 1

    total_duration = time.time() - epoch_start_time
    logging.info('Epoch {} training duration: {:.2f} sec'.format(epoch, train_duration))
    logging.info('Epoch {} total duration: {:.2f} sec'.format(epoch, total_duration))
    logging.info('-' * 120)

    metrics_dict['Loss'] = loss_avg.avg
    metrics_dict['Acc'] = acc.compute()
    metrics_dict['Prec'] = precision.compute()
    metrics_dict['Recall'] = recall.compute()
    metrics_dict['F1'] = f1.compute()

    write_scalars(epoch, os.path.join(run_folder, 'train.csv'), metrics_dict, train_duration, total_duration)

    return metrics_dict