"""
Stage 4 — Feature Engineering
Run: python notebooks/03_features.py
Output: models/tfidf_vectorizer.pkl, data/processed/splits.npz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse
import sys, os, pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_features, split_data

print("=" * 55)
print("  STAGE 4 — FEATURE ENGINEERING")
print("=" * 55)

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/cleaned_data.csv")
print(f"\nLoaded  : {len(df)} rows")

# ── Build TF-IDF features ─────────────────────────────────────────────────────
print("Building TF-IDF matrix...")
X, y, vectorizer = build_features(df, max_features=15000, ngram_range=(1, 2))

print(f"\nFeature matrix shape : {X.shape}")
print(f"  Rows (samples)     : {X.shape[0]}")
print(f"  Cols (features)    : {X.shape[1]}")
print(f"Vocabulary size      : {len(vectorizer.vocabulary_)}")
print(f"Matrix sparsity      : {(1 - X.nnz / (X.shape[0]*X.shape[1]))*100:.2f}%")
print(f"Vectorizer saved     : models/tfidf_vectorizer.pkl")

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = split_data(X, y)

print(f"\nTrain set : {X_train.shape[0]} samples")
print(f"Test  set : {X_test.shape[0]} samples")
print(f"Train label balance — Safe: {(y_train==0).sum()}  Abusive: {(y_train==1).sum()}")
print(f"Test  label balance — Safe: {(y_test==0).sum()}   Abusive: {(y_test==1).sum()}")

# ── Save splits ───────────────────────────────────────────────────────────────
scipy.sparse.save_npz("data/processed/X_train.npz", X_train)
scipy.sparse.save_npz("data/processed/X_test.npz",  X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy",  y_test)
print("\nSplits saved to data/processed/")

# ── Chart 1: Top 20 unigrams by TF-IDF score ─────────────────────────────────
feature_names = vectorizer.get_feature_names_out()
mean_tfidf    = X.mean(axis=0).A1
top20_idx     = mean_tfidf.argsort()[::-1][:20]
top20_words   = [feature_names[i] for i in top20_idx]
top20_scores  = [mean_tfidf[i] for i in top20_idx]

fig, ax = plt.subplots(figsize=(9, 5))
colors = cm.YlOrRd(np.linspace(0.4, 0.9, 20))
bars = ax.barh(list(reversed(top20_words)), list(reversed(top20_scores)),
               color=list(reversed(colors)), edgecolor="white")
ax.set_title("Top 20 Features by Mean TF-IDF Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean TF-IDF Score")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/12_top_tfidf_features.png", dpi=150)
plt.close()
print("[Saved] 12_top_tfidf_features.png")

# ── Chart 2: Top features per class (abusive vs safe) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, label_val, title, color_map in [
    (axes[0], 1, "Top 15 TF-IDF Features — Abusive", "Reds"),
    (axes[1], 0, "Top 15 TF-IDF Features — Safe",    "Greens"),
]:
    X_class    = X[y == label_val]
    mean_class = X_class.mean(axis=0).A1
    top15_idx  = mean_class.argsort()[::-1][:15]
    words      = [feature_names[i] for i in top15_idx]
    scores     = [mean_class[i] for i in top15_idx]
    colors_cls = getattr(cm, color_map)(np.linspace(0.4, 0.85, 15))

    ax.barh(list(reversed(words)), list(reversed(scores)),
            color=list(reversed(colors_cls)), edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean TF-IDF Score")
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("reports/figures/13_tfidf_per_class.png", dpi=150)
plt.close()
print("[Saved] 13_tfidf_per_class.png")

# ── Chart 3: Feature count distribution (tokens per doc in TF-IDF) ───────────
nonzero_per_doc = X.getnnz(axis=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(nonzero_per_doc, bins=40, color="#5c7cfa", alpha=0.85, edgecolor="white")
ax.axvline(nonzero_per_doc.mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"Mean = {nonzero_per_doc.mean():.1f}")
ax.set_title("Non-zero Features per Document", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Active TF-IDF Features")
ax.set_ylabel("Number of Documents")
ax.legend()
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/14_features_per_doc.png", dpi=150)
plt.close()
print("[Saved] 14_features_per_doc.png")

# ── Chart 4: Train / Test split pie ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, title, safe, abusive in [
    (axes[0], "Train Set", (y_train==0).sum(), (y_train==1).sum()),
    (axes[1], "Test Set",  (y_test==0).sum(),  (y_test==1).sum()),
]:
    ax.pie([safe, abusive],
           labels=[f"Safe\n{safe:,}", f"Abusive\n{abusive:,}"],
           colors=["#43a047","#e53935"],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title(title, fontsize=12, fontweight="bold")
plt.suptitle("Train / Test Label Distribution", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("reports/figures/15_train_test_split.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] 15_train_test_split.png")

# ── Verify vectorizer loads and transforms correctly ──────────────────────────
print("\n--- Vectorizer Smoke Test ---")
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    loaded_vec = pickle.load(f)

test_sentences = [
    "you are a terrible person i hate you",
    "have a wonderful day hope you are well",
]
for s in test_sentences:
    vec = loaded_vec.transform([s])
    print(f"  '{s[:50]}'")
    print(f"  -> shape {vec.shape}, non-zero features: {vec.nnz}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Stage 4 COMPLETE")
print(f"  Vectorizer  : models/tfidf_vectorizer.pkl")
print(f"  Features    : {X.shape[1]:,}")
print(f"  Train rows  : {X_train.shape[0]:,}")
print(f"  Test rows   : {X_test.shape[0]:,}")
print(f"  Charts      : reports/figures/12 – 15")
print("  Proceed to Stage 5: Classical Model Training")
print("=" * 55)
