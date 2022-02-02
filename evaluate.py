import torch
import json
import os
import shutil
from torch.utils import data

from model.model import *
import model.loss as loss
from dataset import WZDataset
from utils.evaluation import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(model_path, cfg_path, train=False, copy=True):

    config = json.load(open(cfg_path))

    model = build_model(config).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    criterion = getattr(loss, config['loss'])


    dataset = WZDataset(True if train else False, config, eval=True)

    dataloader = data.DataLoader(dataset,
                    batch_size=1,shuffle=False,
                    num_workers=config['data_loader']['args']['num_workers'], pin_memory=True)

    save_dir = config['name']
    incorrect_path = os.path.join(save_dir, 'incorrect_images')
    correct_path = os.path.join(save_dir, 'correct_images')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(incorrect_path)
        os.mkdir(correct_path)

    incorrect_losses = []
    correct_losses = []
    preds = []
    for inp in dataloader:
        input, target, img = inp
        img = img[0]
        input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(input)
        los = criterion(output, target)
        acc = accuracy(output, target)
        root, fname = os.path.split(img)

        out = torch.sigmoid(output)


        if acc < 1:
            shutil.copy(img, os.path.join(incorrect_path, fname) )
            incorrect_losses.append((img, los.item()))
            preds.append((out[0].detach().cpu(), target[0].detach().cpu(), fname))
        else:
            if copy:
                shutil.copy(img, os.path.join(correct_path, fname) )
            correct_losses.append((img, los.item()))

    with open(os.path.join(save_dir, 'incorrect_losses.txt'), 'w') as f:
        for l in incorrect_losses:
            f.write(f"{l[0]}, {round(l[1], 5)}\n")

    with open(os.path.join(save_dir, 'correct_losses.txt'), 'w') as f:
        for l in correct_losses:
            f.write(f"{l[0]}, {round(l[1], 5)}\n")

    with open(os.path.join(save_dir, 'incorrect_preds.txt'), 'w') as f:
        for pred, gt, name in preds:
            pred = list(map(lambda x: round(x.item(), 4), pred))
            gt = list(map(lambda x: int(x.item()), gt))
            f.write(f"{pred}, {gt}, {name}\n")


    with open(os.path.join(save_dir, 'pred_list.txt'), 'w') as f:
        for path, _ in incorrect_losses:
            f.write(f'{path}\n')
