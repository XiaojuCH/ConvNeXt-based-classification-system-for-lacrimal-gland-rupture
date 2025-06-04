# -*- coding: utf-8 -*-
# @Author :Xiaoju
# @Time : 2025/4/9 下午11:31

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def getAUC(y_true, y_score, task):
    auc = 0.0
    if task == "multi-label":
        for i in range(y_true.shape[1]):
            y_true_binary = y_true[:, i]
            y_score_binary = y_score[:, i]
            if len(np.unique(y_true_binary)) == 1:
                continue
            auc += roc_auc_score(y_true_binary, y_score_binary)
        auc /= y_true.shape[1]
    elif task == "multi-class":
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    else:
        if len(np.unique(y_true)) == 1:
            return 0.0
        auc = roc_auc_score(y_true, y_score)
    return auc


def getACC(y_true, y_score, task, threshold=0.5):
    if task in ['multi-label', 'multi-label, binary-class']:
        y_pred = (y_score >= threshold).astype(int)
        acc_list = []
        for i in range(y_true.shape[1]):
            acc_list.append(accuracy_score(y_true[:, i], y_pred[:, i]))
        return np.mean(acc_list)
    elif task == 'binary-class':
        y_pred = (y_score[:, -1] > threshold).astype(int)
        return accuracy_score(y_true, y_pred)
    elif task == 'multi-class':
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)
    else:
        raise ValueError(f"Unrecognized task type: {task}")


def getClassificationMetrics(y_true, y_score, task, threshold=0.5):
    if task == 'multi-class':
        y_pred = np.argmax(y_score, axis=1)
    elif task == 'binary-class':
        y_pred = (y_score[:, -1] > threshold).astype(int)
    elif task in ['multi-label', 'multi-label, binary-class']:
        y_pred = (y_score >= threshold).astype(int)
    else:
        raise ValueError(f"Unsupported task: {task}")

    metrics = {}

    if task == 'multi-label':
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    return metrics


def save_results(y_true, y_score, outputpath):
    if y_true.ndim == 1:
        y_true_mat = y_true.reshape(-1, 1)
    else:
        y_true_mat = y_true
    y_score_mat = y_score

    n_samples = y_score_mat.shape[0]
    n_true_cols = y_true_mat.shape[1]
    n_score_cols = y_score_mat.shape[1]

    cols = ['id'] + [f'true_{i}' for i in range(n_true_cols)] + [f'score_{i}' for i in range(n_score_cols)]
    data = []

    for idx in range(n_samples):
        row = [idx] + y_true_mat[idx].tolist() + y_score_mat[idx].tolist()
        data.append(row)

    df = pd.DataFrame(data, columns=cols)
    df.to_csv(outputpath, sep=',', index=False, encoding="utf_8_sig")


def plot_confusion_matrix(y_true, y_score, class_names, task='multi-class', save_path='confusion_matrix.png'):
    if task != 'multi-class':
        print("[warning] Confusion matrix is only supported for multi-class tasks.")
        return

    y_pred = np.argmax(y_score, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()