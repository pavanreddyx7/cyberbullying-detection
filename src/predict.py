import pickle, torch
import numpy as np
import xgboost as xgb
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.preprocess import clean_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_classical_cache = {}
_bert_cache      = {}


def load_classical():
    if not _classical_cache:
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            _classical_cache["vectorizer"] = pickle.load(f)
        model = xgb.Booster()
        model.load_model("models/xgboost_gpu.json")
        _classical_cache["model"] = model
    return _classical_cache["model"], _classical_cache["vectorizer"]


def load_bert():
    if not _bert_cache:
        _bert_cache["tokenizer"] = DistilBertTokenizerFast.from_pretrained(
            "models/distilbert_abuse"
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            "models/distilbert_abuse"
        )
        model.eval().to(DEVICE)
        _bert_cache["model"] = model
    return _bert_cache["model"], _bert_cache["tokenizer"]


def predict_classical(text, model, vectorizer):
    cleaned  = clean_text(text)
    features = vectorizer.transform([cleaned])
    dmat     = xgb.DMatrix(features.toarray().astype(np.float32))
    prob     = float(model.predict(dmat)[0])
    label    = 1 if prob > 0.5 else 0
    return {
        "text"      : text,
        "prediction": "Abusive" if label == 1 else "Safe",
        "confidence": round((prob if label == 1 else 1 - prob) * 100, 2),
        "model"     : "XGBoost (GPU)",
    }


def predict_bert(text, model, tokenizer):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        out   = model(
            input_ids      = enc["input_ids"].to(DEVICE),
            attention_mask = enc["attention_mask"].to(DEVICE),
        )
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        label = int(probs.argmax())
    return {
        "text"      : text,
        "prediction": "Abusive" if label == 1 else "Safe",
        "confidence": round(float(probs[label]) * 100, 2),
        "model"     : "DistilBERT (GPU)",
    }


def predict(text, mode="bert"):
    """
    mode: 'bert' (default) or 'classical'
    """
    if mode == "classical":
        model, vectorizer = load_classical()
        return predict_classical(text, model, vectorizer)
    model, tokenizer = load_bert()
    return predict_bert(text, model, tokenizer)
