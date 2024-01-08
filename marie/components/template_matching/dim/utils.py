from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np


def IoU(r1, r2):
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2
    x12 = x11 + w1 - 1
    y12 = y11 + h1 - 1
    x22 = x21 + w2 - 1
    y22 = y21 + h2 - 1
    x_overlap = max(0, min(x12, x22) - max(x11, x21))
    y_overlap = max(0, min(y12, y22) - max(y11, y21))
    I = 1.0 * x_overlap * y_overlap
    U = (y12 - y11) * (x12 - x11) + (y22 - y21) * (x22 - x21) - I
    J = I / U
    return J


def score2curve(score, thres_delta=0.01):
    thres = np.linspace(0, 1, int(1.0 / thres_delta) + 1)
    success_num = []
    for th in thres:
        success_num.append(np.sum(score >= (th + 1e-6)))
    success_rate = np.array(success_num) / len(score)
    return thres, success_rate


def all_sample_iou(gt_list, pd_list):
    num_samples = len(gt_list)
    iou_list = []
    for idx in range(num_samples):
        image_gt, image_pd = gt_list[idx], pd_list[idx]
        iou = IoU(image_gt, image_pd)
        iou_list.append(iou)
    return iou_list


def plot_success_curve(iou_score, method, title=''):
    imageroot = 'results' + '/{m}/{n}-{auc}.png'
    thres, success_rate = score2curve(iou_score, thres_delta=0.05)
    auc_ = np.mean(
        success_rate[:-1]
    )  # this is same auc protocol as used in previous template matching papers #auc_ = auc( thres, success_rate ) # this is the actual auc
    plt.figure()
    plt.grid(True)
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylim(0, 1)
    plt.title(title + 'auc={}'.format(auc_))
    plt.plot(thres, success_rate)
    plt.savefig(imageroot.format(n='AUC', m=method, auc=auc_))
    plt.show()
