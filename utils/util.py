import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn import metrics
from torch import nn
import torch.nn.functional as F

plt.switch_backend('agg')


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.8, alpha=8.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_val_auc_max = -np.Inf
        self.delta = delta

    def __call__(self, val_auc, model, path):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model, path)
            self.counter = 0

    def save_checkpoint(self, val_auc, model, path):
        if self.verbose:
            print(f'Validation auc increased ({self.val_val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_val_auc_max = val_auc


def calculate_performance(y, y_score, y_pred):
    # Calculate Evaluation Metrics
    acc = accuracy_score(y_pred, y)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = precision_recall_curve(y, y_score)
        auprc = np.nan_to_num(metrics.auc(r, p))
    spec = np.nan_to_num(tn / (tn + fp))
    # acc = ((tn + tp) / total) * 100
    balacc = ((spec + sen) / 2)
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))
    F1 = 2 * (prec * recall) / (prec + recall)
    try:
        auc = roc_auc_score(y, y_score)
    except ValueError:
        auc = 0

    return acc, auc, auprc, prec, recall, F1, balacc, sen, spec


