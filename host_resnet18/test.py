import logging
import torch
import time
import os
from torchmetrics import Accuracy, Precision, Recall, F1
from utils.helper import *
from utils.AverageMeter import AverageMeter

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


def test(run_folder, cfg, device, test_loader, criterion, model):

    ckp_path = ''  # Fill in the path of specific version of trained ResNet18.
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint[''])  # model weights
    logging.info("Have loaded test checkpoint from '{}'".format(ckp_path))

    model.eval()
    test_duration = 0
    test_preds, test_trues = [], []
    loss_avg = AverageMeter()

    acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    precision = Precision(num_classes=cfg.dataset.num_classes, average='weighted')
    recall = Recall(num_classes=cfg.dataset.num_classes, average='weighted')
    f1 = F1(num_classes=cfg.dataset.num_classes, average='weighted')

    with torch.no_grad():
        for _, (input, target) in enumerate(test_loader):

            batch_start_time = time.time()

            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)

            test_duration += time.time() - batch_start_time

            test_preds_cpu = torch.argmax(output, dim=1).detach().cpu()
            target_cpu = target.detach().cpu()

            test_preds.extend(test_preds_cpu.numpy())
            test_trues.extend(target_cpu.numpy())

            loss_avg.update(loss.item(), input.size(0))
            acc.update(test_preds_cpu, target_cpu)
            precision.update(test_preds_cpu, target_cpu)
            recall.update(test_preds_cpu, target_cpu)
            f1.update(test_preds_cpu, target_cpu)

        logging.info('Loss: {:.4f} (average: {:.4f})\t'
                     'accuracy_socre: {:.4%}\tprecision_score: {:.4%}\trecall_score: {:.4%}\tf1_score: {:.4%}\t'
                     'test duration: {:.2f} sec'.format(loss_avg.val, loss_avg.avg,
            acc.compute(), precision.compute(), recall.compute(), f1.compute(), test_duration))

        plot_confusion_matrix('test', 1, run_folder, test_trues, test_preds, test_loader)
