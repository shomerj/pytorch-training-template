import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from sklearn.metrics import precision_score, recall_score

def plot_predictions(model, inputs, labels):
    class_map = {0:'down',1:'eliminated', 2:'none'}
    output = model(inputs)
    labels = labels.detach().cpu().numpy()
    preds = torch.sigmoid(output)
    preds = preds.detach().cpu().numpy()
    probs = preds
    preds = (preds > 0.5)*1
    fig = plt.figure(figsize=(16,16))
    for idx in range(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(inputs[idx])
        pred = preds[idx]
        prob = probs[idx]
        true = labels[idx]

        if sum(pred) == 0:
            pred = 2
            prob = str(np.round(prob, 3))
        else:
            pred = np.argmax(pred)
            prob = str(np.round(prob[pred], 3))

        if sum(true) == 0:
            true = 2
        else:
            true = np.argmax(true)

        ax.set_title(f"{class_map[pred]} | {prob}",
                    color= ("green" if pred == true else "red"))

    return fig



def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img * 255   # unnormalize
    npimg = img.detach().cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))


def return_recall_precision(predictions, gt):
    predictions = torch.sigmoid(predictions)
    down_pred = predictions[:, 0]
    kill_pred = predictions[:, 1]
    down_gt = gt[:, 0].detach().cpu().numpy()
    kill_gt = gt[:, 1].detach().cpu().numpy()
    down_pred = (down_pred>0.5)*1
    kill_pred = (down_pred>0.5)*1
    down_pred = down_pred.detach().cpu().numpy()
    kill_pred = kill_pred.detach().cpu().numpy()

    down_recall = recall_score(down_gt.astype(int), down_pred.astype(int))
    down_precision = precision_score(down_gt.astype(int), down_pred.astype(int))
    kill_recall = recall_score(kill_gt.astype(int), kill_pred.astype(int))
    kill_precision = precision_score(kill_gt.astype(int), kill_pred.astype(int))

    return down_recall, down_precision, kill_recall, kill_precision
