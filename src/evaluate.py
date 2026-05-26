import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
)
import os

os.makedirs("reports/figures", exist_ok=True)


def evaluate(y_true, y_pred, y_prob, model_name):
    print(f"\n  {'='*45}")
    print(f"  {model_name}")
    print(f"  {'='*45}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1-Score  : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Safe", "Abusive"]))

    return {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted"), 4),
        "Recall"   : round(recall_score(y_true, y_pred, average="weighted"), 4),
        "F1-Score" : round(f1_score(y_true, y_pred, average="weighted"), 4),
    }
