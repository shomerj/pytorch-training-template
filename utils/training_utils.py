import os
import sys
import glob
import torch

def save_checkpoint(model, optimizer, config, epoch, best_acc, best_loss, save_dir):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': best_acc,
        'loss': best_loss
        }
    filename = str(save_dir / 'GameDetector_{}.pth'.format(epoch))
    torch.save(state, filename)

def resume_checkpoint(resume_path, model, optimizer):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']
        state_dict = checkpoint['state_dict']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(state_dict)
        ##TODO: Make it so it can reuse partial models
        return model, optimizer, epoch, loss, acc

#https://github.com/pytorch/examples/blob/00ea159a99f5cb3f3301a9bf0baa1a5089c7e217/imagenet/main.py#L367
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class StdOutLogger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        if not path.endswith(".out"):
            path += ".out"
        self.log = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
