import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_iou_per_class(confusion_mat):
    ious = []
    for i in range(confusion_mat.shape[0]):
        tp = confusion_mat[i, i]
        fp = confusion_mat[:, i].sum() - tp
        fn = confusion_mat[i, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom != 0 else 0
        ious.append(iou)
    return np.array(ious)

def calculate_metrics(preds, targets, num_classes):
    # preds, targets: (N,) after argmax
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    ious = calculate_iou_per_class(cm)
    oa = np.diag(cm).sum() / cm.sum()
    mIoU = np.nanmean(ious)
    return oa, mIoU, ious, cm