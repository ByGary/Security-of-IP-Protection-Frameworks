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

        
def validation(epoch, run_folder, cfg, device, val_loader, criterion, model):

    logging.info('#' * 120)
    logging.info('Running validation for epoch {}/{}'.format(epoch, cfg.train.num_epochs))

    model.eval()
    val_duration, total_duration = 0, 0
    metrics_dict = defaultdict()
    loss_avg = AverageMeter()

    acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    precision = Precision(num_classes=cfg.dataset.num_classes, average='weighted')
    recall = Recall(num_classes=cfg.dataset.num_classes, average='weighted')
    f1 = F1(num_classes=cfg.dataset.num_classes, average='weighted')

    epoch_start_time = time.time()

    with torch.no_grad():
        for _, (input, target) in enumerate(val_loader):
            batch_start_time = time.time()

            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            val_duration += time.time() - batch_start_time

            val_preds_cpu = torch.argmax(output, dim=1).cpu()
            target_cpu = target.cpu()

            loss_avg.update(loss.item(), input.size(0))
            acc.update(val_preds_cpu, target_cpu)
            precision.update(val_preds_cpu, target_cpu)
            recall.update(val_preds_cpu, target_cpu)
            f1.update(val_preds_cpu, target_cpu)

        logging.info('Epoch: {}/{}\t'
                     'Loss: {:.4f} (average: {:.4f})\t'
                     'acc: {:.4%}\tprec: {:.4%}\trecall: {:.4%}\tf1: {:.4%}\t'.format(
            epoch, cfg.train.num_epochs, loss_avg.val, loss_avg.avg,
            acc.compute(), precision.compute(), recall.compute(), f1.compute()))

        total_duration = time.time() - epoch_start_time
        logging.info('Epoch {} validation duration: {:.2f} sec'.format(epoch, val_duration))
        logging.info('Epoch {} total duration: {:.2f} sec'.format(epoch, total_duration))
        logging.info('#' * 120)

        metrics_dict['Loss'] = loss_avg.avg
        metrics_dict['Acc'] = acc.compute()
        metrics_dict['Prec'] = precision.compute()
        metrics_dict['Recall'] = recall.compute()
        metrics_dict['F1'] = f1.compute()

        write_scalars(epoch, os.path.join(run_folder, 'val.csv'), metrics_dict, val_duration, total_duration)

    return metrics_dict