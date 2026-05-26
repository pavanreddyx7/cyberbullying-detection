"""
Stage 3 — Data Preprocessing
Run: python notebooks/02_preprocessing.py
Output: data/processed/cleaned_data.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text

print("=" * 55)
print("  STAGE 3 — DATA PREPROCESSING")
print("=" * 55)

# ── Load raw data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/labeled_data.csv")
df = df.drop(columns=["Unnamed: 0"])
print(f"\nLoaded  : {len(df)} rows")

# ── Apply cleaning ────────────────────────────────────────────────────────────
print("Cleaning text... (this may take ~10 seconds)")
df["cleaned_text"] = df["tweet"].apply(clean_text)

# ── Drop rows that became empty after cleaning ────────────────────────────────
before = len(df)
df = df[df["cleaned_text"].str.strip() != ""]
df = df.dropna(subset=["cleaned_text"])
after  = len(df)
print(f"Dropped : {before - after} empty rows after cleaning")
print(f"Kept    : {after} rows")

# ── Binary label: 0 = safe, 1 = abusive (hate + offensive combined) ───────────
df["label"] = df["class"].apply(lambda x: 0 if x == 2 else 1)

label_counts = df["label"].value_counts()
print(f"\nBinary label distribution:")
print(f"  Safe    (0) : {label_counts[0]:>5}  ({label_counts[0]/len(df)*100:.1f}%)")
print(f"  Abusive (1) : {label_counts[1]:>5}  ({label_counts[1]/len(df)*100:.1f}%)")

# ── Show before/after for 5 samples ──────────────────────────────────────────
print("\n--- Sample Before vs After Cleaning ---")
samples = df.sample(5, random_state=42)[["tweet", "cleaned_text", "label"]]
for _, row in samples.iterrows():
    print(f"  ORIGINAL : {row['tweet'][:90]}")
    print(f"  CLEANED  : {row['cleaned_text'][:90]}")
    print(f"  LABEL    : {'Abusive' if row['label'] == 1 else 'Safe'}")
    print()

# ── Token length statistics ───────────────────────────────────────────────────
df["token_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))

print("--- Token Count After Cleaning ---")
print(f"  Avg tokens  : {df['token_count'].mean():.1f}")
print(f"  Max tokens  : {df['token_count'].max()}")
print(f"  Min tokens  : {df['token_count'].min()}")

# ── Chart: token count distribution before vs after ──────────────────────────
df["orig_tokens"] = df["tweet"].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["orig_tokens"],    bins=40, color="#5c7cfa", alpha=0.8, edgecolor="white")
axes[0].set_title("Word Count — BEFORE Cleaning", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Frequency")
axes[0].spines[["top","right"]].set_visible(False)

axes[1].hist(df["token_count"], bins=40, color="#20c997", alpha=0.8, edgecolor="white")
axes[1].set_title("Word Count — AFTER Cleaning", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Token Count")
axes[1].set_ylabel("Frequency")
axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("reports/figures/09_token_count_before_after.png", dpi=150)
plt.close()
print("[Saved] 09_token_count_before_after.png")

# ── Chart: binary label distribution ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Safe (0)", "Abusive (1)"],
       [label_counts[0], label_counts[1]],
       color=["#43a047", "#e53935"], edgecolor="white", width=0.4)
for i, val in enumerate([label_counts[0], label_counts[1]]):
    ax.text(i, val + 150, f"{val:,}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Binary Label Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, max(label_counts) * 1.12)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/10_binary_labels.png", dpi=150)
plt.close()
print("[Saved] 10_binary_labels.png")

# ── Chart: avg token count per original class ─────────────────────────────────
avg_tokens = df.groupby("class")["token_count"].mean()
class_names = {0: "Hate Speech", 1: "Offensive", 2: "Safe"}
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    [class_names[c] for c in avg_tokens.index],
    avg_tokens.values,
    color=["#e53935","#fb8c00","#43a047"], edgecolor="white", width=0.4
)
for bar, val in zip(bars, avg_tokens.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Avg Token Count After Cleaning", fontsize=12, fontweight="bold")
ax.set_ylabel("Avg Tokens")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/11_avg_tokens_per_class.png", dpi=150)
plt.close()
print("[Saved] 11_avg_tokens_per_class.png")

# ── Save cleaned dataset ──────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
save_cols = ["tweet", "cleaned_text", "class", "label", "token_count"]
df[save_cols].to_csv("data/processed/cleaned_data.csv", index=False)
print(f"\n[Saved] data/processed/cleaned_data.csv  ({len(df)} rows)")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Stage 3 COMPLETE")
print(f"  Output : data/processed/cleaned_data.csv")
print(f"  Rows   : {len(df)}")
print(f"  Charts : reports/figures/09, 10, 11")
print("  Proceed to Stage 4: Feature Engineering")
print("=" * 55)
