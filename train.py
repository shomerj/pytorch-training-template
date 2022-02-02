import os
import sys
import time
import torch
import argparse
import collections
import numpy as np
import pickle as pkl
import torchvision
from torch.utils import data
from datetime import datetime
from pprint import pprint
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from adabelief_pytorch import AdaBelief
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import model.loss as loss
from utils.config_parser import *
from utils.logger import *
from utils.training_utils import *
from utils.tensorboard_helpers import *
from utils.evaluation import *
from model.model import *
from dataset import WZDataset
from utils.progress.progress.bar import Bar

best_acc = 0
best_loss = 100 #just a high number
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
# fix random seeds for reproducibility
SEED = 35
torch.manual_seed(SEED)
np.random.seed(SEED)

def main(config):
    global best_acc
    global best_loss

    pprint(config.config)
    loss_acc_metrics = {'training_accs':[], 'training_losses': [], 'val_accs': [], 'val_losses':[]}
    #load data from config
    img_dir = config['data_loader']['args']['img_dir']
    save_dir = config['trainer']['save_dir']
    training_session_name = config['name']
    save_by = config['trainer']['save_by']
    #check if log directory exists
    logdir = config.log_dir
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    sys.stdout = StdOutLogger(os.path.join(logdir, 'log'))

    if torch.cuda.device_count() > 1:
        print(f'Loaded model to {torch.cuda.device_count()} gpus')
        model = build_model(config)
        model = torch.nn.DataParallel(model)
    else:
        print(f'Loaded model to {device}')
        model = build_model(config).to(device)

    weights = config.config.get('weights')
    if weights is not None:
        print(f'Loading weights from: {weights}')
        weight = torch.load(weights)
        model.load_state_dict(weight['state_dict'])

    criterion = getattr(loss, config['loss'])
    #loss fucntion and optimizer
    if config['optimizer']['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), config['optimizer']['args']['lr'], weight_decay=config['optimizer']['args']['weight_decay'])
    elif config['optimizer']['type'] == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=config['optimizer']['args']['lr'], eps=1e-10, betas=(0.9,0.999), weight_decouple=False)

    else:
        print('Optimer not available')
        assert False

    if config.resume:

        print(f"Loading Checkpoint from {config['name']}")
        model, optimizer, current_epoch, best_loss, best_acc = resume_checkpoint(config.resume, model, optimizer)
        optimizer.param_groups[0]['lr'] = config['optimizer']['args']['lr']
        if os.path.isfile(os.path.join(config.log_dir, 'log.txt')):
            logger = Logger(os.path.join(config.log_dir, 'log.txt'), title=f'Training date: {datetime.today()}', resume=True)

        else:
           logger = Logger(os.path.join(config.log_dir, 'log.txt'), title=f'Training date: {datetime.today()}' )
           logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                      'Train Acc', 'Val Acc'])
        print(f"Loaded Checkpoint {config['name']} | Epoch {current_epoch}")

    else:
        logger = Logger(os.path.join(config.log_dir, 'log.txt'), title=f'Training date: {datetime.today()}' )
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])

        current_epoch = 0

    print(' Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    train_dataset = ModelDataset(True, config)
    test_dataset = ModelDataset(False, config)

    if config['data_loader']['args']['balance']:
        print("Balancing dataset")
        sampling_weights = torch.Tensor(train_dataset.sampling_weights())
        sampler = WeightedRandomSampler(sampling_weights.type('torch.DoubleTensor'), len(sampling_weights))
        train_loader = data.DataLoader(train_dataset,
                        batch_size=config['data_loader']['args']['batch_size'],sampler=sampler,
                        num_workers=config['data_loader']['args']['num_workers'], pin_memory=True
                    )
        test_loader = data.DataLoader(test_dataset,
                        batch_size=config['data_loader']['args']['batch_size'], shuffle=False,
                        num_workers=config['data_loader']['args']['num_workers'], pin_memory=True
                    )
    else:
        train_loader = data.DataLoader(train_dataset,
                        batch_size=config['data_loader']['args']['batch_size'],shuffle=True,
                        num_workers=config['data_loader']['args']['num_workers'], pin_memory=True
                    )
        test_loader = data.DataLoader(test_dataset,
                        batch_size=config['data_loader']['args']['batch_size'], shuffle=False,
                        num_workers=config['data_loader']['args']['num_workers'], pin_memory=True
                    )

    #set up tensorboard
    if config['trainer']['tensorboard']:
        tensorboard_dir = config.tensorboard_dir
        write_freq = config['trainer']['write_freq']
        writer = SummaryWriter(tensorboard_dir)
        data_iter = iter(train_loader)
        images, label = data_iter.next()
        img_grid = torchvision.utils.make_grid(images[:8])
        writer.add_image('train_images', img_grid)
        writer.add_graph(model.to('cpu'), images[:2])

    #move model to gpu
    model = model.to(device)
    print(f'Number of train images: {len(train_loader.dataset)} | Number of test images: {len(test_loader.dataset)}')
    lr = config['optimizer']['args']['lr']

    if config['lr_scheduler']['type'] == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                            optimizer,
                                            config['lr_scheduler']['args']['step_size'],
                                            config['lr_scheduler']['args']['gamma'])
        scheduler = True
    elif config['lr_scheduler']['type'] == 'Cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                            optimizer,
                                            T_max=config['lr_scheduler']['args']['max_epochs'],
                                            eta_min=0)

        scheduler = True
    else:
        scheduler = False
        print('No learning rate scheduler')

    epochs = config['trainer']['epochs']

    for epoch in range(current_epoch, epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch+1} | LR: {optimizer.param_groups[0]['lr']}")
        #training loop
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, writer, write_freq)
        val_loss, val_acc = validate(test_loader, model, criterion, epoch, writer, write_freq)

        train_images, train_labels = train_loader.dataset.grab_random_images(4)
        #TODO: Bugged need to work on creating a graph
        # writer.add_figure(f"Train Predictions at epoch {epoch}",
        #                 plot_predictions(model, train_images.to(device), train_labels.to(device)),
        #                 global_step=epoch)

        test_images, test_labels =  test_loader.dataset.grab_random_images(4)
        # writer.add_figure(f"Test Predictions at epoch {epoch}",
        #                 plot_predictions(model, test_images.to(device), test_labels.to(device)),
        #                 global_step=epoch)

        loss_acc_metrics['training_losses'].append(train_loss)
        loss_acc_metrics['training_accs'].append(train_acc)
        loss_acc_metrics['val_losses'].append(val_loss)
        loss_acc_metrics['val_accs'].append(val_acc)

        logger.append([epoch + 1, lr, train_loss, val_loss, train_acc, val_acc])
        if save_by == 'loss':
            is_best = val_loss < best_loss
        elif save_by == 'acc':
            is_best = val_acc > best_acc
        elif save_by == 'every':
            is_best = True
        best_loss = min(val_loss, best_loss)
        best_acc = max(val_acc, best_acc)

        if scheduler:
            lr_scheduler.step()

        if is_best:
            print(f'Saving model to {config.save_dir} | Epoch: {epoch+1} | Loss {best_loss} | Acc {val_acc}')
            save_checkpoint(model, optimizer, config, epoch+1, best_acc, best_loss, config.save_dir)
            with open(os.path.join(config.log_dir, 'loss_acc_metrics.pkl'), 'wb') as f:
                pkl.dump(loss_acc_metrics,f)

    logger.close()

def train(train_loader, model, criterion, optimizer, epoch, writer, write_freq=1):
    ##TODO: Make this more elegant
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    kill_recalls = AverageMeter()
    kill_precisions = AverageMeter()
    down_recalls = AverageMeter()
    down_precisions = AverageMeter()

    model.train() #train mode
    end = time.time()
    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))

    for i, data in enumerate(train_loader):

        input, target = data
        data_time.update(time.time() - end)

        input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(input)
        loss = criterion(output, target)

        losses.update(loss.item(), input.size(0))
        acc = accuracy(output, target)
        acces.update(acc, input.size(0))
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f} '.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg,
                    )
        bar.next()

        if i % write_freq == 0:
            num_steps_per_epoch = len(train_loader.dataset) // input.size(0)
            global_step = (num_steps_per_epoch*epoch) + i
            writer.add_scalar('Loss/Train', loss, global_step )
            writer.add_scalar('Acc/Train', acces.avg, global_step)

    return  losses.avg, acces.avg


def validate(test_loader, model, criterion, epoch, writer, write_freq=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Eval', max=len(test_loader))

    accs = 0
    predictions = torch.zeros(len(test_loader.dataset), test_loader.dataset.num_classes)
    targets = torch.zeros(len(test_loader.dataset), test_loader.dataset.num_classes)
    with torch.no_grad():
        for i ,data in enumerate(test_loader):
            data_time.update(time.time()-end)
            input, target = data
            input = input.to(device,non_blocking=True)
            target = target.to(device,non_blocking=True)
            output = model(input)
            loss = criterion(output, target)

            for si, pred in enumerate(output):
                idx = si + (i*len(data))
                predictions[idx] = pred
                targets[idx] = target[si]

            acc = accuracy(output, target)
            acces.update(acc, input.size(0))

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                        batch=i + 1,
                        size=len(test_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc=acces.avg
                        )
            bar.next()
            bar.finish()

            if i % write_freq == 0:
                num_steps_per_epoch = len(test_loader.dataset) // input.size(0)
                global_step = (num_steps_per_epoch*epoch) + i
                writer.add_scalar('Loss/Test', loss, global_step )
                writer.add_scalar('Acc/Test', acces.avg, global_step)

    print("***"*20)
    print(f'End of Test Epoch {epoch} metrics\n')
    print("Accuracy by Thresholds")
    acc_thresh_dict = accuracy_by_thresh(predictions, targets)
    print("\n")
    print("Accuracy by Class")
    acc_cls_dict = accuracy_by_class(predictions, targets, test_loader.dataset.inv_class_mapping)
    print("\nRecall/Precision Scores")
    calculate_precision_recall(predictions, targets, test_loader.dataset.inv_class_mapping)
    print("***"*20)

    return losses.avg, acces.avg

if __name__ =='__main__':

    args = argparse.ArgumentParser(description='Scene Detector Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--load_weights', default=None, type=str, help='load weight from a .tar file')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
