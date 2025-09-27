# src/copd/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)

def binary_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_hat  = (y_prob >= thr).astype(int)

    acc  = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)     # = sensitivity
    f1   = f1_score(y_true, y_hat, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    spec = tn / (tn + fp + 1e-8)
    sens = rec
    bal  = balanced_accuracy_score(y_true, y_hat)
    mcc  = matthews_corrcoef(y_true, y_hat) if len(np.unique(y_hat)) > 1 else 0.0
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    aupr  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "specificity": spec, "sensitivity": sens, "balanced_accuracy": bal,
        "mcc": mcc, "auc_roc": auroc, "aupr": aupr, "brier": brier,
        "confusion": (tn, fp, fn, tp)
    }

def roc_pr_points(y_true, y_prob):
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
            "thr_roc": thr_roc, "thr_pr": thr_pr}
