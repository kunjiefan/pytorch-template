import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def auc(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    return roc_auc_score(target, output)


def aupr(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    return average_precision_score(target, output)

def precision_at_5(output, target, top=0.05):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    sorted_index = np.argsort(-output)
    top_num = int(top * len(target))
    sorted_targets = target[sorted_index[:top_num]]
    acc = float(sorted_targets.sum())/float(top_num)
    return acc

def ppv_at_0_4(output, target, k=0.4):
    """
    PPV at 40% recall
    """
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    precision, recall, thresholds = precision_recall_curve(target, output)
    recall_0_4 = np.absolute(recall - k)
    pos = np.argmin(recall_0_4)
    return precision[pos]

def ppv(output, target):
    """
    PPV=precision, use 0.5 as the threshold
    """
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    threshold = 0.5
    predictions = (output > threshold).astype(np.int32)
    tp = np.sum(predictions * target)
    fp = np.sum(predictions) - tp
    ppv = tp/(tp+fp)
    return ppv

def f1_max(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    output = np.round(output, 2)
    target = target.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (output > threshold).astype(np.int32)
        tp = np.sum(predictions * target)
        fp = np.sum(predictions) - tp
        fn = np.sum(target) - tp
        sn = tp / (1.0 * np.sum(target))
        sp = np.sum((predictions ^ 1) * (target ^ 1))
        sp /= 1.0 * np.sum(target ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max