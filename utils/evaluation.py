import numpy
import torch
import torch.nn as nn
from prettytable import PrettyTable
from sklearn.metrics import recall_score, precision_score

def accuracy(pred, target, sigmoid=True):
    if sigmoid:
        pred = torch.sigmoid(pred)
    pred = (pred>0.5)*1
    acc = torch.sum(torch.all(pred==target, dim=1))
    return acc/pred.size(0)

def calculate_precision_recall(pred, target, class_mapping, sigmoid=True, print_pretty=True):
    precision_by_cls = {}
    recall_by_cls = {}

    if sigmoid:
        pred = torch.sigmoid(pred)

    for i in range(target.size(1)):
        gt = target[:, i].cpu().detach().numpy()
        p = pred[:, i].cpu().detach().numpy()
        p = (p>0.5)*1
        precision_by_cls[class_mapping[i]] = precision_score(gt, p, zero_division=1)
        recall_by_cls[class_mapping[i]] = recall_score(gt, p, zero_division=1)

    mask = torch.all(target==0, dim=1)
    gt = mask[mask]
    neg_out = pred[mask]>0.5
    neg_pred = torch.all(neg_out==0, dim=1)
    gt = gt.cpu().detach().numpy()
    neg_pred = neg_pred.cpu().detach().numpy()
    precision_by_cls['negative'] = precision_score(gt, neg_pred, zero_division=1)
    recall_by_cls['negative'] = recall_score(gt, neg_pred, zero_division=1)

    if print_pretty:
        columns = ["Precision/Recall"]
        columns.extend(list(precision_by_cls.keys()))
        table = PrettyTable(columns)
        precision_row = ['precision']
        precision_row.extend(map(lambda x: round(x, 5), precision_by_cls.values()))
        recall_row = ['recall']
        recall_row.extend(map(lambda x: round(x, 5), recall_by_cls.values()))
        table.add_row(precision_row)
        table.add_row(recall_row)

        print(table)

def accuracy_by_thresh(pred, target, thresholds=[0.5, 0.9, 0.95, 0.99], sigmoid=True, print_pretty=True ):
    acc_per_threshold = {}

    if sigmoid:
        pred = torch.sigmoid(pred)

    for thresh in thresholds:
        output_prediction = pred > thresh
        acc = torch.sum(torch.all(output_prediction==target, dim=1))/pred.size(0)
        acc_per_threshold[thresh] = acc

    if print_pretty:
        columns = ["Threshold", "Accuracy"]
        table = PrettyTable(columns)
        for key, val in acc_per_threshold.items():
            table.add_row([key, round(val.item(), 5)])

        print(table)

    return acc_per_threshold

def accuracy_by_class(y_pred, y_true, class_mapping, thresholds=[0.5, 0.9, 0.95, 0.99], sigmoid=True, print_pretty=True):
    """
    args:
        y_pred: torch.tensor (n, num_classes)
        y_true: torch.tensor (n, num_classes)
    Returns:
        A dictionary of class accuracies based on thresholds
    """
    acc_dict = {thresh: {} for thresh in thresholds}
    if sigmoid:
        y_pred = torch.sigmoid(y_pred)

    for thresh in thresholds:
        preds = y_pred > thresh
        thresh_dict = acc_dict[thresh]
        for i in range(y_true.size(1)):
            trues = y_true[:, i]
            predictions = preds[:, i]
            acc=torch.sum(trues==predictions)/trues.size(0)
            thresh_dict[class_mapping[i]] = acc

    if print_pretty:
        columns = ["Threshold"]
        columns.extend(list(class_mapping.values()))
        table = PrettyTable(columns)
        for key, val in acc_dict.items():
            row = [key]
            vals = map(lambda x: round(x.item(), 5), val.values())
            row.extend(vals)
            table.add_row(row)
        print(table)
    return acc_dict
