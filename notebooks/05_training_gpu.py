"""
Stage 6 — GPU-Accelerated Training
Run: python notebooks/05_training_gpu.py
Outputs:
  models/xgboost_gpu.json
  models/distilbert_abuse/
  reports/figures/20 - 24
"""

import sys, os, time, pickle
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_gpu import train_distilbert
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split

print("=" * 55)
print("  STAGE 6 — GPU-ACCELERATED TRAINING")
print("=" * 55)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Load data ─────────────────────────────────────────────────────────────────
X_train = scipy.sparse.load_npz("data/processed/X_train.npz")
X_test  = scipy.sparse.load_npz("data/processed/X_test.npz")
y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

df = pd.read_csv("data/processed/cleaned_data.csv")

print(f"\nTrain : {X_train.shape[0]} samples")
print(f"Test  : {X_test.shape[0]} samples")

# ═══════════════════════════════════════════════════════
#  OPTION A — XGBoost GPU
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  Option A: XGBoost with GPU")
print("=" * 55)

import xgboost as xgb

X_train_arr = X_train.toarray().astype(np.float32)
X_test_arr  = X_test.toarray().astype(np.float32)

dtrain = xgb.DMatrix(X_train_arr, label=y_train)
dtest  = xgb.DMatrix(X_test_arr,  label=y_test)

params = {
    "objective"        : "binary:logistic",
    "eval_metric"      : ["logloss", "error"],
    "tree_method"      : "hist",
    "device"           : DEVICE,
    "max_depth"        : 6,
    "eta"              : 0.1,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "scale_pos_weight" : (y_train == 0).sum() / (y_train == 1).sum(),
    "seed"             : 42,
}

print(f"\nTraining XGBoost on {DEVICE}...")
xgb_start = time.time()
evals_result = {}
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dtest, "val")],
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=50,
)
xgb_time = time.time() - xgb_start
print(f"XGBoost training time : {xgb_time:.1f}s")

xgb_model.save_model("models/xgboost_gpu.json")
print("Saved : models/xgboost_gpu.json")

# XGBoost evaluation
y_pred_xgb  = (xgb_model.predict(dtest) > 0.5).astype(int)
xgb_acc     = accuracy_score(y_test, y_pred_xgb)
xgb_f1      = f1_score(y_test, y_pred_xgb, average="weighted")
print(f"\nXGBoost  Accuracy: {xgb_acc:.4f}   F1: {xgb_f1:.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=["Safe","Abusive"]))

# ── Chart 20: XGBoost learning curve ─────────────────────────────────────────
train_logloss = evals_result["train"]["logloss"]
val_logloss   = evals_result["val"]["logloss"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_logloss, label="Train", color="#5c7cfa")
axes[0].plot(val_logloss,   label="Val",   color="#ff6b6b")
axes[0].set_title("XGBoost — Log Loss", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Boosting Round")
axes[0].set_ylabel("Log Loss")
axes[0].legend()
axes[0].spines[["top","right"]].set_visible(False)

train_err = evals_result["train"]["error"]
val_err   = evals_result["val"]["error"]
axes[1].plot(train_err, label="Train Error", color="#5c7cfa")
axes[1].plot(val_err,   label="Val Error",   color="#ff6b6b")
axes[1].set_title("XGBoost — Classification Error", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Boosting Round")
axes[1].set_ylabel("Error Rate")
axes[1].legend()
axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("reports/figures/20_xgboost_learning_curve.png", dpi=150)
plt.close()
print("[Saved] 20_xgboost_learning_curve.png")

# ── Chart 21: XGBoost feature importance (top 20) ────────────────────────────
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
feature_names = vectorizer.get_feature_names_out()

scores_dict = xgb_model.get_fscore()
if scores_dict:
    feat_df = (pd.DataFrame.from_dict(scores_dict, orient="index", columns=["score"])
               .reset_index().rename(columns={"index":"feature"})
               .sort_values("score", ascending=False).head(20))

    feat_df["feature_name"] = feat_df["feature"].apply(
        lambda f: feature_names[int(f[1:])] if f.startswith("f") and f[1:].isdigit() else f
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_df["feature_name"][::-1], feat_df["score"][::-1],
            color="#fb8c00", alpha=0.85, edgecolor="white")
    ax.set_title("XGBoost — Top 20 Feature Importances", fontsize=12, fontweight="bold")
    ax.set_xlabel("F-Score")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("reports/figures/21_xgboost_feature_importance.png", dpi=150)
    plt.close()
    print("[Saved] 21_xgboost_feature_importance.png")

# ── Chart 22: XGBoost confusion matrix ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Oranges", ax=ax,
            xticklabels=["Safe","Abusive"],
            yticklabels=["Safe","Abusive"],
            linewidths=0.5, linecolor="white")
ax.set_title(f"XGBoost (GPU) — Confusion Matrix\nF1={xgb_f1:.4f}", fontweight="bold")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("reports/figures/22_xgboost_confusion_matrix.png", dpi=150)
plt.close()
print("[Saved] 22_xgboost_confusion_matrix.png")

# ═══════════════════════════════════════════════════════
#  OPTION B — DistilBERT Fine-Tuning (GPU)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  Option B: DistilBERT Fine-Tuning (GPU)")
print("=" * 55)

# Use raw tweet text — BERT has its own tokenizer
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"\nTrain texts : {len(train_df)}")
print(f"Val texts   : {len(val_df)}\n")

torch.cuda.empty_cache()

bert_start = time.time()
bert_model, tokenizer, history = train_distilbert(
    train_df["tweet"].tolist(),
    train_df["label"].tolist(),
    val_df["tweet"].tolist(),
    val_df["label"].tolist(),
)
bert_time = time.time() - bert_start
print(f"\nDistilBERT training time : {bert_time/60:.1f} min")

# ── Evaluate DistilBERT on val set ────────────────────────────────────────────
print("\n--- DistilBERT Evaluation ---")
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

bert_model.eval()
all_preds = []
val_texts  = val_df["tweet"].tolist()
val_labels = val_df["label"].tolist()

for i in range(0, len(val_texts), 64):
    batch_enc = tokenizer(
        val_texts[i:i+64],
        truncation=True, padding="max_length",
        max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        out   = bert_model(
            input_ids      = batch_enc["input_ids"].to("cuda"),
            attention_mask = batch_enc["attention_mask"].to("cuda"),
        )
        preds = torch.argmax(out.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

bert_acc = accuracy_score(val_labels, all_preds)
bert_f1  = f1_score(val_labels, all_preds, average="weighted")
print(f"DistilBERT  Accuracy: {bert_acc:.4f}   F1: {bert_f1:.4f}")
print(classification_report(val_labels, all_preds, target_names=["Safe","Abusive"]))

# ── Chart 23: DistilBERT training history ────────────────────────────────────
epochs     = [h["epoch"]   for h in history]
val_accs   = [h["val_acc"] for h in history]
val_f1s    = [h["val_f1"]  for h in history]
train_loss = [h["loss"]    for h in history]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, train_loss, "o-", color="#5c7cfa", linewidth=2, markersize=7, label="Train Loss")
axes[0].set_title("DistilBERT — Training Loss", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_xticks(epochs)
axes[0].legend()
axes[0].spines[["top","right"]].set_visible(False)

axes[1].plot(epochs, val_accs, "o-", color="#20c997", linewidth=2, markersize=7, label="Val Accuracy")
axes[1].plot(epochs, val_f1s,  "s-", color="#ff6b6b", linewidth=2, markersize=7, label="Val F1")
axes[1].set_title("DistilBERT — Validation Metrics", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].set_xticks(epochs)
axes[1].set_ylim(0.8, 1.0)
axes[1].legend()
axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("reports/figures/23_distilbert_training.png", dpi=150)
plt.close()
print("[Saved] 23_distilbert_training.png")

# ── Chart 24: DistilBERT confusion matrix ────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm_bert = confusion_matrix(val_labels, all_preds)
sns.heatmap(cm_bert, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Safe","Abusive"],
            yticklabels=["Safe","Abusive"],
            linewidths=0.5, linecolor="white")
ax.set_title(f"DistilBERT (GPU) — Confusion Matrix\nF1={bert_f1:.4f}", fontweight="bold")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("reports/figures/24_distilbert_confusion_matrix.png", dpi=150)
plt.close()
print("[Saved] 24_distilbert_confusion_matrix.png")

# ── Final GPU comparison summary ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  GPU MODEL COMPARISON")
print("=" * 55)
print(f"  {'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Time'}")
print(f"  {'-'*55}")
print(f"  {'XGBoost (GPU)':<25} {xgb_acc:<12.4f} {xgb_f1:<12.4f} {xgb_time:.1f}s")
print(f"  {'DistilBERT (GPU)':<25} {bert_acc:<12.4f} {bert_f1:<12.4f} {bert_time/60:.1f}min")

# Save comparison
pd.DataFrame([
    {"Model":"XGBoost (GPU)",    "Accuracy":xgb_acc,  "F1":xgb_f1,  "Time":f"{xgb_time:.1f}s"},
    {"Model":"DistilBERT (GPU)", "Accuracy":bert_acc, "F1":bert_f1, "Time":f"{bert_time/60:.1f}min"},
]).to_csv("reports/gpu_model_comparison.csv", index=False)

print("\n" + "=" * 55)
print("  Stage 6 COMPLETE")
print("  Saved : models/xgboost_gpu.json")
print("  Saved : models/distilbert_abuse/")
print("  CSV   : reports/gpu_model_comparison.csv")
print("  Charts: reports/figures/20 - 24")
print("  Proceed to Stage 7: Model Evaluation")
print("=" * 55)
