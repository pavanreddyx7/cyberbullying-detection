# Abuse Detection Using AI — Final Project Guide

## Project Overview

A complete machine learning system that detects abusive, toxic, or harmful text content in real
time using NLP techniques, classical ML models, GPU-accelerated deep learning, voice input,
and a Flask web interface.

---

## Technology Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| ML / Classical | scikit-learn, XGBoost |
| Deep Learning | PyTorch (GPU), HuggingFace Transformers |
| NLP | NLTK, TF-IDF |
| Voice | SpeechRecognition, pyttsx3, Web Speech API |
| Web | Flask, HTML/CSS/JavaScript |
| Visualization | Matplotlib, Seaborn, WordCloud |
| GPU | NVIDIA CUDA 11.8 / 12.x |

---

## Stage Summary

| Stage | Goal | Key Output |
|-------|------|------------|
| 1 | Environment & GPU Setup | Working environment, CUDA verified |
| 2 | Data Collection & EDA | Dataset loaded, charts saved |
| 3 | Data Preprocessing | `cleaned_data.csv`, `preprocess.py` |
| 4 | Feature Engineering | TF-IDF matrix, saved vectorizer |
| 5 | Classical Model Training (CPU) | Trained `.pkl` models |
| 6 | GPU-Accelerated Training | XGBoost GPU + DistilBERT fine-tuned |
| 7 | Model Evaluation | Confusion matrices, comparison CSV |
| 8 | Prediction Pipeline | `predict.py` with confidence score |
| 9 | Web Application | Flask UI — text + voice input |
| 10 | Voice Integration | Browser mic + Python CLI voice demo |
| 11 | Report & Documentation | Final report + README |

---

## Development Order

```
Stage 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11
 Setup   Data Clean Feat Train GPU  Eval  Pred  Web Voice Report
```

---

---

# Stage 1 — Environment & GPU Setup

## 1.1 Project Folder Structure

Create this layout before writing any code:

```
ai-abuse-ml/
├── data/
│   ├── raw/                    # original downloaded datasets
│   └── processed/              # cleaned data after preprocessing
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_features.ipynb
│   ├── 04_training_cpu.ipynb
│   ├── 05_training_gpu.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── preprocess.py           # text cleaning functions
│   ├── features.py             # TF-IDF vectorization
│   ├── train_cpu.py            # classical models (CPU)
│   ├── train_gpu.py            # deep learning models (GPU)
│   ├── evaluate.py             # metrics and charts
│   ├── predict.py              # unified prediction function
│   └── voice.py                # speech-to-text functions
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── best_classical.pkl
│   └── distilbert_abuse/       # saved HuggingFace model folder
├── app/
│   ├── app.py                  # Flask backend
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── reports/
│   └── figures/                # all saved plots
├── requirements.txt
└── README.md
```

---

## 1.2 Install Python Dependencies

```bash
# Core data science
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud

# Web app and voice
pip install flask SpeechRecognition pyttsx3

# GPU-accelerated deep learning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# HuggingFace for transformer models
pip install transformers datasets accelerate

# XGBoost with GPU support
pip install xgboost

# Jupyter
pip install jupyter notebook

# pyaudio for microphone (Windows)
pip install pipwin
pipwin install pyaudio
```

> For Linux/Mac replace the pyaudio block with:
> `sudo apt-get install portaudio19-dev && pip install pyaudio`

---

## 1.3 GPU Verification

Run this immediately after installing PyTorch to confirm GPU is detected:

```python
import torch

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("GPU name        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("CUDA version    :", torch.version.cuda)

# Quick GPU tensor test
if torch.cuda.is_available():
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("Tensor on GPU   :", x)
```

Expected output example:
```
PyTorch version : 2.1.0+cu118
CUDA available  : True
GPU name        : NVIDIA GeForce RTX 3060
CUDA version    : 11.8
Tensor on GPU   : tensor([1., 2., 3.], device='cuda:0')
```

If `CUDA available: False`, install CUDA Toolkit 11.8 from the NVIDIA website and rerun.

---

## 1.4 Download NLTK Resources

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

**Deliverable:** All packages installed, `torch.cuda.is_available()` returns `True`.

---

---

# Stage 2 — Data Collection & EDA

## 2.1 Dataset Options

| Dataset | Size | Labels | Source |
|---------|------|--------|--------|
| Hate Speech & Offensive Language | ~25K rows | hate / offensive / neither | GitHub: t-davidson |
| Jigsaw Toxic Comments | ~160K rows | toxic, threat, insult, etc. | Kaggle |
| Twitter Sentiment140 | 1.6M rows | positive / negative | Kaggle |

**Recommended:** Start with the Hate Speech dataset (~25K rows). It is small, well-labelled,
and ideal for a first project.

Download link: https://github.com/t-davidson/hate-speech-and-offensive-language
File to use: `data/labeled_data.csv`

---

## 2.2 Load and Inspect

```python
import pandas as pd

df = pd.read_csv('data/raw/labeled_data.csv')

print("Shape   :", df.shape)
print("Columns :", df.columns.tolist())
print(df.head())
print("\nLabel distribution:")
print(df['class'].value_counts())
# class 0 = hate speech, 1 = offensive, 2 = neither (safe)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
```

---

## 2.3 Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

os.makedirs('reports/figures', exist_ok=True)

# 1. Class distribution
df['class'].value_counts().plot(kind='bar', color=['red','orange','green'])
plt.title('Label Distribution')
plt.xlabel('Class (0=Hate, 1=Offensive, 2=Safe)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('reports/figures/class_distribution.png')
plt.clf()

# 2. Message length distribution
df['text_length'] = df['tweet'].apply(len)
df['text_length'].hist(bins=50, color='steelblue')
plt.title('Message Length Distribution')
plt.xlabel('Character Count')
plt.savefig('reports/figures/text_length.png')
plt.clf()

# 3. Word cloud — abusive messages
abusive_text = ' '.join(df[df['class'] != 2]['tweet'])
wc = WordCloud(width=800, height=400, background_color='black').generate(abusive_text)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Abusive Messages Word Cloud')
plt.savefig('reports/figures/wordcloud_abusive.png')
plt.clf()

# 4. Word cloud — safe messages
safe_text = ' '.join(df[df['class'] == 2]['tweet'])
wc = WordCloud(width=800, height=400, background_color='white').generate(safe_text)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Safe Messages Word Cloud')
plt.savefig('reports/figures/wordcloud_safe.png')
plt.clf()

print("All EDA charts saved to reports/figures/")
```

**Deliverable:** EDA charts saved, dataset understood.

---

---

# Stage 3 — Data Preprocessing

## 3.1 Text Cleaning — `src/preprocess.py`

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'@\w+', '', text)                     # remove @mentions
    text = re.sub(r'#\w+', '', text)                     # remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)                 # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()             # collapse spaces
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(w) for w in tokens]   # lemmatize
    return ' '.join(tokens)
```

---

## 3.2 Apply Preprocessing & Save

```python
import pandas as pd
from src.preprocess import clean_text
import os

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/raw/labeled_data.csv')
df['cleaned_text'] = df['tweet'].apply(clean_text)

# Drop empty rows after cleaning
df = df[df['cleaned_text'].str.strip() != '']
df.dropna(subset=['cleaned_text'], inplace=True)

# Binary label: 0 = safe, 1 = abusive (hate or offensive)
df['label'] = df['class'].apply(lambda x: 0 if x == 2 else 1)

df.to_csv('data/processed/cleaned_data.csv', index=False)

print(f"Saved {len(df)} rows")
print(df['label'].value_counts())
```

---

## 3.3 Verify Cleaning Quality

```python
# Show before/after for 5 random rows
sample = df.sample(5)[['tweet', 'cleaned_text', 'label']]
for _, row in sample.iterrows():
    print("ORIGINAL :", row['tweet'])
    print("CLEANED  :", row['cleaned_text'])
    print("LABEL    :", row['label'])
    print()
```

**Deliverable:** `data/processed/cleaned_data.csv` with `cleaned_text` and `label` columns.

---

---

# Stage 4 — Feature Engineering

## 4.1 TF-IDF Vectorization — `src/features.py`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

os.makedirs('models', exist_ok=True)

def build_features(df):
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True,      # apply log normalization
        min_df=2                # ignore very rare terms
    )

    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label'].values

    # Save vectorizer
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"Feature matrix shape : {X.shape}")
    print(f"Vocabulary size      : {len(vectorizer.vocabulary_)}")
    return X, y, vectorizer
```

---

## 4.2 Train-Test Split

```python
import pandas as pd
from src.features import build_features
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/processed/cleaned_data.csv')
X, y, vectorizer = build_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {X_train.shape}")
print(f"Test  : {X_test.shape}")
```

**Deliverable:** TF-IDF matrix built, vectorizer saved to `models/tfidf_vectorizer.pkl`.

---

---

# Stage 5 — Classical Model Training (CPU)

## 5.1 Train Multiple Models — `src/train_cpu.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time

def train_all(X_train, y_train):
    models = {
        'Logistic Regression' : LogisticRegression(max_iter=1000, C=1.0),
        'Naive Bayes'         : MultinomialNB(alpha=0.1),
        'Linear SVM'          : LinearSVC(C=1.0, max_iter=2000),
        'Random Forest'       : RandomForestClassifier(n_estimators=200, n_jobs=-1),
    }

    trained = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        print(f"{name:<25} CV F1: {scores.mean():.4f} ± {scores.std():.4f}  ({elapsed:.1f}s)")

        trained[name] = model

    return trained

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")
```

---

## 5.2 Run Training

```python
from src.train_cpu import train_all, save_model

trained_models = train_all(X_train, y_train)

# Save the best performing model (typically Logistic Regression or SVM)
save_model(trained_models['Logistic Regression'], 'models/best_classical.pkl')
```

**Deliverable:** Classical models trained, best one saved as `models/best_classical.pkl`.

---

---

# Stage 6 — GPU-Accelerated Training

Two GPU options are provided. Use both for comparison.

- **Option A — XGBoost (GPU):** Gradient boosting on GPU. Easiest to set up on Windows.
- **Option B — DistilBERT (GPU):** Transformer fine-tuning. Most accurate.

---

## 6.1 Option A — XGBoost with GPU

```python
import xgboost as xgb
import torch
import time
import pickle
from scipy.sparse import issparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training XGBoost on: {device}")

# Convert sparse TF-IDF matrix for XGBoost
X_train_dense = X_train.toarray() if issparse(X_train) else X_train
X_test_dense  = X_test.toarray()  if issparse(X_test)  else X_test

dtrain = xgb.DMatrix(X_train_dense, label=y_train)
dtest  = xgb.DMatrix(X_test_dense,  label=y_test)

params = {
    'objective'        : 'binary:logistic',
    'eval_metric'      : 'logloss',
    'tree_method'      : 'hist',
    'device'           : device,
    'max_depth'        : 6,
    'eta'              : 0.1,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
}

start = time.time()
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtest, 'test')],
    early_stopping_rounds=20,
    verbose_eval=20
)
print(f"XGBoost GPU training time: {time.time()-start:.1f}s")

xgb_model.save_model('models/xgboost_gpu.json')
print("Saved: models/xgboost_gpu.json")
```

---

## 6.2 Option B — DistilBERT Fine-Tuning (GPU)

Create `src/train_gpu.py`:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pandas as pd
import numpy as np
import time
import os

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5

print(f"Training on: {DEVICE}")


class AbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids'      : self.encodings['input_ids'][idx],
            'attention_mask' : self.encodings['attention_mask'][idx],
            'labels'         : self.labels[idx]
        }


def train_distilbert(train_texts, train_labels, val_texts, val_labels):
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = AbuseDataset(train_texts, train_labels, tokenizer)
    val_dataset   = AbuseDataset(val_texts,   val_labels,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        total_loss = 0
        start = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels         = batch['labels'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}  Val Acc: {val_acc:.4f}  ({elapsed:.1f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models/distilbert_abuse', exist_ok=True)
            model.save_pretrained('models/distilbert_abuse')
            tokenizer.save_pretrained('models/distilbert_abuse')
            print(f"  => New best model saved (acc={val_acc:.4f})")

    print(f"\nFinal best validation accuracy: {best_val_acc:.4f}")
    return model, tokenizer
```

---

## 6.3 Run DistilBERT Training

```python
from src.train_gpu import train_distilbert
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data/processed/cleaned_data.csv')

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

model, tokenizer = train_distilbert(
    train_df['tweet'].tolist(),
    train_df['label'].tolist(),
    val_df['tweet'].tolist(),
    val_df['label'].tolist()
)
```

> Expected training time on a mid-range GPU (RTX 3060): ~4 minutes for 3 epochs on 25K rows.

**Deliverable:** `models/xgboost_gpu.json` and `models/distilbert_abuse/` saved.

---

---

# Stage 7 — Model Evaluation

## 7.1 Evaluate All Models — `src/evaluate.py`

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs('reports/figures', exist_ok=True)

def evaluate(y_test, y_pred, model_name):
    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=['Safe','Abusive']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe','Abusive'], yticklabels=['Safe','Abusive'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'reports/figures/cm_{model_name.replace(" ","_")}.png')
    plt.clf()

    return {
        'Model'    : model_name,
        'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'Recall'   : round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1-Score' : round(f1_score(y_test, y_pred, average='weighted'), 4),
    }
```

---

## 7.2 Compare All Models

```python
import pickle
import xgboost as xgb
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.evaluate import evaluate
import pandas as pd

results = []

# 1. Classical models
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    results.append(evaluate(y_test, y_pred, name))

# 2. XGBoost GPU
xgb_model = xgb.Booster()
xgb_model.load_model('models/xgboost_gpu.json')
dtest = xgb.DMatrix(X_test.toarray())
y_pred_xgb = (xgb_model.predict(dtest) > 0.5).astype(int)
results.append(evaluate(y_test, y_pred_xgb, 'XGBoost GPU'))

# 3. DistilBERT GPU
tokenizer = DistilBertTokenizerFast.from_pretrained('models/distilbert_abuse')
bert_model = DistilBertForSequenceClassification.from_pretrained('models/distilbert_abuse')
bert_model.eval().cuda()

# Run batch inference
all_preds = []
batch_texts = val_df['tweet'].tolist()
for i in range(0, len(batch_texts), 32):
    batch = tokenizer(batch_texts[i:i+32], truncation=True, padding='max_length',
                      max_length=128, return_tensors='pt')
    with torch.no_grad():
        out = bert_model(input_ids=batch['input_ids'].cuda(),
                         attention_mask=batch['attention_mask'].cuda())
        preds = torch.argmax(out.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

results.append(evaluate(val_df['label'].tolist(), all_preds, 'DistilBERT (GPU)'))

# Save comparison
df_results = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print("\n=== Model Comparison ===")
print(df_results.to_string(index=False))
df_results.to_csv('reports/model_comparison.csv', index=False)
```

---

## 7.3 Plot Model Comparison Bar Chart

```python
import matplotlib.pyplot as plt

df_results.set_index('Model')[['Accuracy','F1-Score']].plot(kind='bar', figsize=(10,5))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=30, ha='right')
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig('reports/figures/model_comparison.png')
print("Saved: reports/figures/model_comparison.png")
```

**Deliverable:** Confusion matrices + `reports/model_comparison.csv` + bar chart.

---

---

# Stage 8 — Prediction Pipeline

## 8.1 Unified Predict Function — `src/predict.py`

```python
import pickle
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.preprocess import clean_text

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_classical():
    with open('models/best_classical.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def load_bert():
    tokenizer = DistilBertTokenizerFast.from_pretrained('models/distilbert_abuse')
    model = DistilBertForSequenceClassification.from_pretrained('models/distilbert_abuse')
    model.eval().to(DEVICE)
    return model, tokenizer


def predict_classical(text, model, vectorizer):
    cleaned  = clean_text(text)
    features = vectorizer.transform([cleaned])
    label    = model.predict(features)[0]
    prob     = model.predict_proba(features)[0].max()
    return {
        'text'      : text,
        'prediction': 'Safe' if label == 0 else 'Abusive',
        'confidence': round(float(prob) * 100, 2),
        'model'     : 'Classical (TF-IDF)'
    }


def predict_bert(text, model, tokenizer):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids      = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        label   = probs.argmax()

    return {
        'text'      : text,
        'prediction': 'Safe' if label == 0 else 'Abusive',
        'confidence': round(float(probs[label]) * 100, 2),
        'model'     : 'DistilBERT (GPU)'
    }
```

---

## 8.2 Test Predictions

```python
from src.predict import load_classical, load_bert, predict_classical, predict_bert

classical_model, vectorizer = load_classical()
bert_model, tokenizer       = load_bert()

test_messages = [
    "You are such a wonderful person!",
    "I will destroy you, you worthless piece of garbage!",
    "Great job on the presentation today.",
    "I hate people like you so much.",
    "The weather is nice today.",
]

print(f"\n{'Text':<45} {'Classical':<12} {'DistilBERT'}")
print('-' * 80)
for msg in test_messages:
    r1 = predict_classical(msg, classical_model, vectorizer)
    r2 = predict_bert(msg, bert_model, tokenizer)
    print(f"{msg[:44]:<45} {r1['prediction']:<12} {r2['prediction']}  ({r2['confidence']}%)")
```

**Deliverable:** Both models predict correctly on the test messages.

---

---

# Stage 9 — Web Application (Flask)

## 9.1 Backend — `app/app.py`

```python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from src.predict import load_classical, load_bert, predict_classical, predict_bert
from src.voice import listen_from_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# Load both models at startup
classical_model, vectorizer = load_classical()
bert_model, tokenizer       = load_bert()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    text = data.get('text', '').strip()
    mode = data.get('mode', 'bert')       # 'classical' or 'bert'

    if not text:
        return jsonify({'error': 'Empty input'}), 400

    if mode == 'classical':
        result = predict_classical(text, classical_model, vectorizer)
    else:
        result = predict_bert(text, bert_model, tokenizer)

    return jsonify(result)


@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file     = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = listen_from_file(filepath)
    if not text:
        return jsonify({'error': 'Could not transcribe audio'}), 400

    result = predict_bert(text, bert_model, tokenizer)
    result['transcribed_text'] = text
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## 9.2 Frontend — `app/templates/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Abuse Detection System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 100vh;
            padding: 40px 16px;
        }
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            padding: 36px;
            width: 100%;
            max-width: 640px;
        }
        h1 { font-size: 1.6rem; color: #1a1a2e; margin-bottom: 6px; }
        .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 24px; }
        label { font-weight: 600; color: #333; display: block; margin-bottom: 6px; }
        textarea {
            width: 100%; height: 110px; padding: 10px;
            border: 1px solid #ccc; border-radius: 8px;
            font-size: 15px; resize: vertical;
        }
        textarea:focus { outline: none; border-color: #4CAF50; }

        .model-row { display: flex; gap: 16px; margin: 12px 0; }
        .model-row label { font-weight: normal; display: flex; align-items: center; gap: 6px; }

        .btn-row { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
        button {
            padding: 10px 22px; font-size: 15px;
            border: none; border-radius: 6px; cursor: pointer; font-weight: 600;
        }
        #analyzeBtn { background: #4CAF50; color: white; }
        #analyzeBtn:hover { background: #43a047; }
        #micBtn { background: #2196F3; color: white; }
        #micBtn:hover { background: #1e88e5; }
        #micBtn.recording { background: #e53935; }

        #transcribed {
            margin-top: 14px; font-size: 13px;
            color: #555; font-style: italic;
        }
        #result {
            margin-top: 18px; padding: 16px;
            border-radius: 8px; font-size: 17px;
            font-weight: 700; display: none;
        }
        #result.safe    { background: #e8f5e9; color: #2e7d32; border-left: 5px solid #43a047; }
        #result.abusive { background: #ffebee; color: #c62828; border-left: 5px solid #e53935; }
        .confidence { font-size: 13px; font-weight: normal; margin-top: 4px; color: #555; }
        .spinner { display: none; color: #999; margin-top: 14px; font-size: 14px; }
    </style>
</head>
<body>
<div class="card">
    <h1>Abuse Detection System</h1>
    <p class="subtitle">Detects abusive and harmful language using AI</p>

    <label for="inputText">Enter a message:</label>
    <textarea id="inputText" placeholder="Type or speak a message to analyze..."></textarea>

    <div class="model-row">
        <label><input type="radio" name="model" value="bert" checked> DistilBERT (GPU — recommended)</label>
        <label><input type="radio" name="model" value="classical"> Classical (TF-IDF)</label>
    </div>

    <div class="btn-row">
        <button id="analyzeBtn" onclick="analyzeText()">Analyze Text</button>
        <button id="micBtn"     onclick="toggleMic()">&#127908; Start Voice</button>
    </div>

    <p class="spinner" id="spinner">Analyzing...</p>
    <div id="transcribed"></div>
    <div id="result"></div>
</div>

<script>
    // ── Text analysis ──────────────────────────────────────────
    async function analyzeText() {
        const text = document.getElementById('inputText').value.trim();
        if (!text) { alert('Please enter a message.'); return; }
        const mode = document.querySelector('input[name="model"]:checked').value;
        showSpinner(true);
        const data = await callPredict(text, mode);
        showSpinner(false);
        showResult(data, null);
    }

    async function callPredict(text, mode) {
        const res = await fetch('/predict', {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ text, mode })
        });
        return res.json();
    }

    function showResult(data, transcribed) {
        const el = document.getElementById('result');
        if (data.error) {
            el.className = '';
            el.style.display = 'block';
            el.innerHTML = `Error: ${data.error}`;
            return;
        }
        const cls = data.prediction === 'Safe' ? 'safe' : 'abusive';
        const icon = cls === 'safe' ? '✔' : '✘';
        el.className = cls;
        el.style.display = 'block';
        el.innerHTML = `${icon} ${data.prediction}
            <div class="confidence">Confidence: ${data.confidence}% &nbsp;|&nbsp; Model: ${data.model}</div>`;
        if (transcribed) {
            document.getElementById('transcribed').textContent = `Transcribed: "${transcribed}"`;
            document.getElementById('inputText').value = transcribed;
        }
    }

    function showSpinner(show) {
        document.getElementById('spinner').style.display = show ? 'block' : 'none';
    }

    // ── Voice input (Web Speech API) ───────────────────────────
    let recognition = null;
    let isRecording = false;

    function toggleMic() {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) { alert('Voice input requires Google Chrome.'); return; }
        isRecording ? stopMic() : startMic(SR);
    }

    function startMic(SR) {
        recognition = new SR();
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = () => {
            isRecording = true;
            const btn = document.getElementById('micBtn');
            btn.textContent = '⏹ Stop';
            btn.classList.add('recording');
            document.getElementById('result').style.display = 'none';
            document.getElementById('transcribed').textContent = 'Listening...';
        };

        recognition.onresult = async (e) => {
            const text = e.results[0][0].transcript;
            const mode = document.querySelector('input[name="model"]:checked').value;
            showSpinner(true);
            const data = await callPredict(text, mode);
            showSpinner(false);
            showResult(data, text);
        };

        recognition.onerror = (e) => {
            document.getElementById('transcribed').textContent = `Mic error: ${e.error}`;
            resetMic();
        };

        recognition.onend = resetMic;
        recognition.start();
    }

    function stopMic() { if (recognition) recognition.stop(); }

    function resetMic() {
        isRecording = false;
        const btn = document.getElementById('micBtn');
        btn.textContent = '🎤 Start Voice';
        btn.classList.remove('recording');
    }
</script>
</body>
</html>
```

---

## 9.3 Run the Web App

```bash
cd app
python app.py
# Open http://127.0.0.1:5000 in Google Chrome
```

**Features:**
- Type any message and click **Analyze Text**
- Select between DistilBERT (GPU) or Classical model
- Click **Start Voice** to speak — Chrome transcribes and analyzes instantly
- Result shows Safe/Abusive with confidence % and which model was used

**Deliverable:** Fully working web app with both text and voice input.

---

---

# Stage 10 — Python Voice CLI

## 10.1 Voice Module — `src/voice.py`

```python
import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()
speaker    = pyttsx3.init()


def speak(text):
    speaker.say(text)
    speaker.runAndWait()


def listen_from_microphone(timeout=5):
    with sr.Microphone() as source:
        print("Calibrating... please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening — speak now.")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Google API error: {e}")
        return None


def listen_from_file(path):
    with sr.AudioFile(path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except (sr.UnknownValueError, sr.RequestError):
        return None
```

---

## 10.2 CLI Voice Demo — `src/voice_demo.py`

```python
from src.voice import listen_from_microphone, speak
from src.predict import load_bert, predict_bert

print("Loading model...")
model, tokenizer = load_bert()
print("=== Voice Abuse Detector — speak to analyze. Ctrl+C to quit. ===\n")

while True:
    try:
        text = listen_from_microphone()
        if text:
            result = predict_bert(text, model, tokenizer)
            output = f"Result: {result['prediction']} — {result['confidence']}% confidence."
            print(output)
            speak(output)
    except KeyboardInterrupt:
        print("\nExiting.")
        break
```

Run it:
```bash
python -m src.voice_demo
```

**Deliverable:** Voice CLI that listens, classifies, and speaks the result aloud.

---

---

# Stage 11 — Report & Documentation

## 11.1 Final Report Structure

Write your report with these sections:

| # | Section | Content |
|---|---------|---------|
| 1 | Abstract | 150-word summary of the project |
| 2 | Introduction | Problem statement, objectives, scope |
| 3 | Literature Review | 3-5 existing approaches cited |
| 4 | System Architecture | Diagram: Data → Preprocessing → Model → Web App |
| 5 | Dataset | Source, size, class distribution chart |
| 6 | Methodology | Preprocessing steps, TF-IDF, model choices |
| 7 | GPU Training | How DistilBERT was fine-tuned, hardware used |
| 8 | Results | Accuracy/F1 table, confusion matrices, comparison chart |
| 9 | Web App & Voice | Screenshots of the UI, how voice works |
| 10 | Discussion | Limitations, bias risks, future improvements |
| 11 | Conclusion | Summary, impact, what was learned |
| 12 | References | Dataset links, papers, libraries |

---

## 11.2 Screenshots to Include in Report

- `reports/figures/class_distribution.png`
- `reports/figures/wordcloud_abusive.png`
- `reports/figures/wordcloud_safe.png`
- `reports/figures/model_comparison.png`
- `reports/figures/cm_DistilBERT_(GPU).png`
- Screenshot of web app showing Safe result
- Screenshot of web app showing Abusive result

---

## 11.3 README.md

```markdown
# Abuse Detection Using AI

Detects abusive text using NLP and deep learning (DistilBERT on GPU).

## Setup
pip install -r requirements.txt

## Dataset
Download from: https://github.com/t-davidson/hate-speech-and-offensive-language
Place as: data/raw/labeled_data.csv

## Run Training
python src/train_cpu.py   # classical models
python src/train_gpu.py   # DistilBERT on GPU

## Run Web App
cd app && python app.py
Open http://127.0.0.1:5000 in Chrome

## Run Voice CLI
python -m src.voice_demo
```

---

## 11.4 Generate requirements.txt

```bash
pip freeze > requirements.txt
```

**Deliverable:** Complete report, README, and requirements.txt.

---

---

# GPU Quick Reference

| Task | Command / Setting |
|------|-------------------|
| Check GPU | `torch.cuda.is_available()` |
| Move model to GPU | `model.to('cuda')` |
| Move tensor to GPU | `tensor.cuda()` |
| XGBoost GPU | `params = {'device': 'cuda', 'tree_method': 'hist'}` |
| DistilBERT batch size | 32 (RTX 3060 8GB), 64 (RTX 3080 10GB+) |
| Monitor GPU usage | `nvidia-smi` in terminal |
| Free GPU memory | `torch.cuda.empty_cache()` |

---

# Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `BATCH_SIZE` from 32 to 16 |
| `pyaudio not found` | Use `pipwin install pyaudio` on Windows |
| `No speech detected` | Check microphone permissions in Windows Settings |
| `predict_proba not found` on SVM | Use `LinearSVC` with `CalibratedClassifierCV` wrapper |
| `ModuleNotFoundError: src` | Run scripts from project root, not inside `src/` |
| DistilBERT slow on CPU | Make sure `torch.cuda.is_available()` is `True` before training |
