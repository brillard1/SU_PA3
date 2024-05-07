import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import sys

FILE = sys.argv[1]

def compute_metrics():
    labels = []
    scores = []

    with open(FILE, 'r') as file:
        for line in file:
            parts = line.split('/')
            score = float(parts[-1].split()[-1])
            label = 1 if parts[0].strip().lower() == 'fake' else 0
            labels.append(label)
            scores.append(score)

    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[idx] + fnr[idx])/2

    print(f"EER:, {eer:.4f}")
    print(f"ROC-AUC:, {roc_auc_score(labels, scores):.4f}")

compute_metrics()