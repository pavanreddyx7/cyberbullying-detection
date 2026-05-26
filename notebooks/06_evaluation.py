"""
Stage 7 — Full Model Evaluation
Run: python notebooks/06_evaluation.py
Outputs: reports/figures/25 - 31  +  reports/final_evaluation.csv
"""

import sys, os, pickle
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("  STAGE 7 — MODEL EVALUATION")
print("=" * 55)

# ── Load test data ────────────────────────────────────────────────────────────
X_test  = scipy.sparse.load_npz("data/processed/X_test.npz")
y_test  = np.load("data/processed/y_test.npy")
df      = pd.read_csv("data/processed/cleaned_data.csv")

from sklearn.model_selection import train_test_split
_, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_texts  = val_df["tweet"].tolist()
val_labels = np.array(val_df["label"].tolist())

print(f"\nTest samples  : {len(y_test)}")
print(f"Val samples   : {len(val_labels)}")

all_results  = []
model_preds  = {}   # store (y_true, y_pred, y_prob) per model

# ═══════════════════════════════════════════════════════
#  Model 1 — XGBoost (GPU)
# ═══════════════════════════════════════════════════════
print("\n--- Loading XGBoost (GPU) ---")
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_gpu.json")
dtest     = xgb.DMatrix(X_test.toarray().astype(np.float32))
xgb_prob  = xgb_model.predict(dtest)
xgb_pred  = (xgb_prob > 0.5).astype(int)

all_results.append(evaluate(y_test, xgb_pred, xgb_prob, "XGBoost (GPU)"))
model_preds["XGBoost (GPU)"] = (y_test, xgb_pred, xgb_prob)

# ═══════════════════════════════════════════════════════
#  Model 2 — DistilBERT (GPU)
# ═══════════════════════════════════════════════════════
print("\n--- Loading DistilBERT (GPU) ---")
tokenizer  = DistilBertTokenizerFast.from_pretrained("models/distilbert_abuse")
bert_model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_abuse")
bert_model.eval().to(DEVICE)

bert_probs_list, bert_preds_list = [], []
for i in range(0, len(val_texts), 64):
    enc = tokenizer(
        val_texts[i:i+64],
        truncation=True, padding="max_length",
        max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        out   = bert_model(
            input_ids      = enc["input_ids"].to(DEVICE),
            attention_mask = enc["attention_mask"].to(DEVICE),
        )
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        bert_probs_list.extend(probs[:, 1])
        bert_preds_list.extend(preds)

bert_prob = np.array(bert_probs_list)
bert_pred = np.array(bert_preds_list)

all_results.append(evaluate(val_labels, bert_pred, bert_prob, "DistilBERT (GPU)"))
model_preds["DistilBERT (GPU)"] = (val_labels, bert_pred, bert_prob)

# ── Save results table ────────────────────────────────────────────────────────
df_results = pd.DataFrame(all_results).sort_values("F1-Score", ascending=False)
os.makedirs("reports", exist_ok=True)
df_results.to_csv("reports/final_evaluation.csv", index=False)
print("\n[Saved] reports/final_evaluation.csv")

# ═══════════════════════════════════════════════════════
#  Chart 25 — Final metrics comparison bar chart
# ═══════════════════════════════════════════════════════
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x       = np.arange(len(df_results))
width   = 0.18

fig, ax = plt.subplots(figsize=(10, 5))
colors  = ["#5c7cfa", "#20c997", "#ff6b6b", "#ffd43b"]
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals   = df_results[metric].values
    offset = (i - 1.5) * width
    bars   = ax.bar(x + offset, vals, width, label=metric,
                    color=color, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(df_results["Model"].values, rotation=10, ha="right", fontsize=11)
ax.set_ylabel("Score")
ax.set_ylim(0.85, 1.02)
ax.set_title("Final Model Evaluation — All Metrics", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/25_final_metrics.png", dpi=150)
plt.close()
print("[Saved] 25_final_metrics.png")

# ═══════════════════════════════════════════════════════
#  Chart 26 — Side-by-side confusion matrices
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
palette   = {"XGBoost (GPU)": "Oranges", "DistilBERT (GPU)": "Blues"}

for ax, (name, (y_true, y_pred, _)) in zip(axes, model_preds.items()):
    cm  = confusion_matrix(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap=palette[name],
                ax=ax, xticklabels=["Safe","Abusive"],
                yticklabels=["Safe","Abusive"],
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"{name}\nAcc={acc:.4f}  F1={f1:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("reports/figures/26_confusion_matrices_final.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] 26_confusion_matrices_final.png")

# ═══════════════════════════════════════════════════════
#  Chart 27 — ROC curves
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
colors_roc = {"XGBoost (GPU)": "#fb8c00", "DistilBERT (GPU)": "#1565c0"}

for name, (y_true, _, y_prob) in model_preds.items():
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2.5,
            color=colors_roc[name],
            label=f"{name}  (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — GPU Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/27_roc_curves.png", dpi=150)
plt.close()
print("[Saved] 27_roc_curves.png")

# ═══════════════════════════════════════════════════════
#  Chart 28 — Precision-Recall curves
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

for name, (y_true, _, y_prob) in model_preds.items():
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap           = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, linewidth=2.5,
            color=colors_roc[name],
            label=f"{name}  (AP = {ap:.4f})")

baseline = val_labels.mean()
ax.axhline(baseline, color="gray", linestyle="--",
           linewidth=1.2, label=f"Baseline (AP={baseline:.2f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — GPU Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower left", fontsize=10)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/28_precision_recall_curves.png", dpi=150)
plt.close()
print("[Saved] 28_precision_recall_curves.png")

# ═══════════════════════════════════════════════════════
#  Chart 29 — Confidence distribution (DistilBERT)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (label_val, title, color) in zip(axes, [
    (0, "Safe Messages — Confidence", "#43a047"),
    (1, "Abusive Messages — Confidence", "#e53935"),
]):
    mask  = val_labels == label_val
    probs = bert_prob[mask]
    ax.hist(probs, bins=40, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold=0.5")
    ax.set_title(f"DistilBERT — {title}", fontsize=11, fontweight="bold")
    ax.set_xlabel("P(Abusive)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("DistilBERT Prediction Confidence Distribution", fontsize=12,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("reports/figures/29_confidence_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] 29_confidence_distribution.png")

# ═══════════════════════════════════════════════════════
#  Chart 30 — Error analysis: misclassified message lengths
# ═══════════════════════════════════════════════════════
bert_correct   = (bert_pred == val_labels)
val_df_copy    = val_df.copy().reset_index(drop=True)
val_df_copy["correct"]    = bert_correct
val_df_copy["char_length"] = val_df_copy["tweet"].apply(len)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(val_df_copy[val_df_copy["correct"]]["char_length"],
        bins=40, alpha=0.7, color="#5c7cfa", label="Correct", edgecolor="white")
ax.hist(val_df_copy[~val_df_copy["correct"]]["char_length"],
        bins=40, alpha=0.7, color="#ff6b6b", label="Misclassified", edgecolor="white")
ax.set_title("DistilBERT — Message Length: Correct vs Misclassified",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Character Length")
ax.set_ylabel("Count")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/30_error_analysis_length.png", dpi=150)
plt.close()
print("[Saved] 30_error_analysis_length.png")

# ═══════════════════════════════════════════════════════
#  Chart 31 — Final leaderboard radar chart
# ═══════════════════════════════════════════════════════
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
N          = len(categories)
angles     = [n / float(N) * 2 * np.pi for n in range(N)]
angles    += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0.85, 1.0)

plot_colors = {"XGBoost (GPU)": "#fb8c00", "DistilBERT (GPU)": "#1565c0"}
for _, row in df_results.iterrows():
    vals    = [row[c] for c in categories]
    vals   += vals[:1]
    name    = row["Model"]
    ax.plot(angles, vals, "o-", linewidth=2,
            color=plot_colors[name], label=name)
    ax.fill(angles, vals, alpha=0.12, color=plot_colors[name])

ax.set_title("Model Performance Radar", fontsize=13,
             fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig("reports/figures/31_radar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] 31_radar_chart.png")

# ═══════════════════════════════════════════════════════
#  Print final leaderboard
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  FINAL LEADERBOARD")
print("=" * 55)
print(df_results.to_string(index=False))

# Error analysis summary
n_errors = (~bert_correct).sum()
print(f"\nDistilBERT Errors : {n_errors}/{len(val_labels)} ({n_errors/len(val_labels)*100:.2f}%)")
fp = ((bert_pred == 1) & (val_labels == 0)).sum()
fn = ((bert_pred == 0) & (val_labels == 1)).sum()
print(f"  False Positives (Safe -> Abusive) : {fp}")
print(f"  False Negatives (Abusive -> Safe) : {fn}")

print("\n" + "=" * 55)
print("  Stage 7 COMPLETE")
print("  Saved : reports/final_evaluation.csv")
print("  Charts: reports/figures/25 - 31")
print("  Proceed to Stage 8: Prediction Pipeline")
print("=" * 55)
