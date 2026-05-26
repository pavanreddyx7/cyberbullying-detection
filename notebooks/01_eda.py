"""
Stage 2 — Data Collection & Exploratory Data Analysis
Run: python notebooks/01_eda.py
Outputs saved to: reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
import os

os.makedirs("reports/figures", exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/labeled_data.csv")
df = df.drop(columns=["Unnamed: 0"])   # drop index column

print("=" * 55)
print("  STAGE 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

print(f"\nDataset shape   : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns         : {df.columns.tolist()}")
print(f"Missing values  : {df.isnull().sum().sum()}")
print(f"Duplicate rows  : {df.duplicated().sum()}")

print("\nLabel distribution:")
label_map = {0: "Hate Speech", 1: "Offensive", 2: "Safe (Neither)"}
counts = df["class"].value_counts().sort_index()
for cls, cnt in counts.items():
    pct = cnt / len(df) * 100
    print(f"  Class {cls} ({label_map[cls]:<15}) : {cnt:>5} rows  ({pct:.1f}%)")

# ── Chart 1: Class distribution bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#e53935", "#fb8c00", "#43a047"]
bars = ax.bar(
    [label_map[i] for i in counts.index],
    counts.values,
    color=colors,
    edgecolor="white",
    width=0.5
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Label Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Number of Messages")
ax.set_ylim(0, counts.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/01_class_distribution.png", dpi=150)
plt.close()
print("\n[Saved] 01_class_distribution.png")

# ── Chart 2: Message length distribution ─────────────────────────────────────
df["char_length"] = df["tweet"].apply(len)
df["word_count"]  = df["tweet"].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for idx, (cls_id, cls_name, color) in enumerate(
        [(0, "Hate Speech", "#e53935"),
         (1, "Offensive",   "#fb8c00"),
         (2, "Safe",        "#43a047")]):
    subset = df[df["class"] == cls_id]["char_length"]
    axes[0].hist(subset, bins=40, alpha=0.6, color=color, label=cls_name)

axes[0].set_title("Character Length by Class", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Character Count")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].spines[["top", "right"]].set_visible(False)

for cls_id, cls_name, color in [
        (0, "Hate Speech", "#e53935"),
        (1, "Offensive",   "#fb8c00"),
        (2, "Safe",        "#43a047")]:
    subset = df[df["class"] == cls_id]["word_count"]
    axes[1].hist(subset, bins=30, alpha=0.6, color=color, label=cls_name)

axes[1].set_title("Word Count by Class", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("reports/figures/02_length_distribution.png", dpi=150)
plt.close()
print("[Saved] 02_length_distribution.png")

# ── Chart 3: Boxplot — message length per class ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
data_by_class = [df[df["class"] == c]["char_length"].values for c in [0, 1, 2]]
bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], ["#e53935", "#fb8c00", "#43a047"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(["Hate Speech", "Offensive", "Safe"])
ax.set_title("Message Length Spread per Class", fontsize=12, fontweight="bold")
ax.set_ylabel("Character Count")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/03_length_boxplot.png", dpi=150)
plt.close()
print("[Saved] 03_length_boxplot.png")

# ── Chart 4: Voting agreement heatmap ────────────────────────────────────────
vote_cols = ["hate_speech", "offensive_language", "neither"]
corr = df[vote_cols].corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5, ax=ax,
            xticklabels=["Hate Speech", "Offensive", "Neither"],
            yticklabels=["Hate Speech", "Offensive", "Neither"])
ax.set_title("Annotator Vote Correlation", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/04_vote_correlation.png", dpi=150)
plt.close()
print("[Saved] 04_vote_correlation.png")

# ── Chart 5: Word clouds ──────────────────────────────────────────────────────
cloud_configs = [
    (df[df["class"] == 0]["tweet"], "Hate Speech Messages",   "black",  "Reds",   "05_wordcloud_hate.png"),
    (df[df["class"] == 1]["tweet"], "Offensive Messages",     "white",  "Oranges","06_wordcloud_offensive.png"),
    (df[df["class"] == 2]["tweet"], "Safe Messages",          "white",  "Greens", "07_wordcloud_safe.png"),
]

for texts, title, bg, colormap, filename in cloud_configs:
    text = " ".join(texts.dropna().tolist())
    wc = WordCloud(
        width=900, height=450,
        background_color=bg,
        colormap=colormap,
        max_words=120,
        collocations=False
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{filename}", dpi=150)
    plt.close()
    print(f"[Saved] {filename}")

# ── Chart 6: Top 15 most common words per class ───────────────────────────────
import re
from collections import Counter

def top_words(texts, n=15):
    words = []
    for t in texts.dropna():
        words += re.findall(r"\b[a-z]{3,}\b", t.lower())
    stopwords = {"the","and","for","you","that","this","with","are",
                 "was","but","have","not","from","they","will","what",
                 "your","just","been","had","its","our","can","out",
                 "get","got","one","all","his","her","him","she","he",
                 "about","also","would","amp","like","more","when",
                 "who","has","were","said","their","there","them"}
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(n)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
class_configs = [
    (0, "Hate Speech Top Words",   "#e53935"),
    (1, "Offensive Top Words",     "#fb8c00"),
    (2, "Safe Top Words",          "#43a047"),
]

for ax, (cls_id, title, color) in zip(axes, class_configs):
    top = top_words(df[df["class"] == cls_id]["tweet"])
    words_list, counts_list = zip(*top)
    ax.barh(list(reversed(words_list)), list(reversed(counts_list)),
            color=color, alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("reports/figures/08_top_words.png", dpi=150)
plt.close()
print("[Saved] 08_top_words.png")

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n--- Text Statistics ---")
print(f"Avg char length : {df['char_length'].mean():.0f}")
print(f"Max char length : {df['char_length'].max()}")
print(f"Avg word count  : {df['word_count'].mean():.1f}")
print(f"Max word count  : {df['word_count'].max()}")

print("\nPer-class averages:")
for cls_id, name in label_map.items():
    sub = df[df["class"] == cls_id]
    print(f"  {name:<18} avg chars: {sub['char_length'].mean():.0f}  avg words: {sub['word_count'].mean():.1f}")

print("\n" + "=" * 55)
print("  Stage 2 COMPLETE — 8 charts saved to reports/figures/")
print("  Proceed to Stage 3: Data Preprocessing")
print("=" * 55)
