"""
Stage 8 — Prediction Pipeline Test
Run: python notebooks/07_prediction_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_classical, load_bert, predict_classical, predict_bert, predict

print("=" * 60)
print("  STAGE 8 — PREDICTION PIPELINE")
print("=" * 60)

# ── Load both models once ──────────────────────────────────────────────────────
print("\nLoading XGBoost...")
xgb_model, vectorizer = load_classical()

print("Loading DistilBERT...")
bert_model, tokenizer = load_bert()
print("Both models loaded.\n")

# ── Test messages ──────────────────────────────────────────────────────────────
test_cases = [
    ("You are such a wonderful and kind person!",                   "Safe"),
    ("I will destroy you, you worthless piece of garbage!",         "Abusive"),
    ("Great job on the presentation today, really impressive.",     "Safe"),
    ("I hate people like you so much, get out of here.",            "Abusive"),
    ("The weather is really nice today, perfect for a walk.",       "Safe"),
    ("Shut up you idiot, nobody asked for your stupid opinion.",    "Abusive"),
    ("Happy birthday! Hope you have an amazing day!",               "Safe"),
    ("You are the dumbest person I have ever seen in my life.",     "Abusive"),
    ("Let me know if you need any help with the project.",          "Safe"),
    ("Go kill yourself, the world would be better without you.",    "Abusive"),
]

# ── Side-by-side comparison table ─────────────────────────────────────────────
print(f"{'#':<3} {'Message':<55} {'Expected':<10} {'XGBoost':<12} {'DistilBERT':<12} {'XGB%':<8} {'BERT%'}")
print("-" * 115)

correct_xgb  = 0
correct_bert = 0

for i, (msg, expected) in enumerate(test_cases, 1):
    r_xgb  = predict_classical(msg, xgb_model, vectorizer)
    r_bert = predict_bert(msg, bert_model, tokenizer)

    xgb_ok  = "OK" if r_xgb["prediction"]  == expected else "FAIL"
    bert_ok = "OK" if r_bert["prediction"] == expected else "FAIL"

    if r_xgb["prediction"]  == expected: correct_xgb  += 1
    if r_bert["prediction"] == expected: correct_bert += 1

    print(f"{i:<3} {msg[:54]:<55} {expected:<10} "
          f"{r_xgb['prediction']:<8}({xgb_ok}) "
          f"{r_bert['prediction']:<8}({bert_ok}) "
          f"{r_xgb['confidence']:<8} {r_bert['confidence']}")

print("-" * 115)
print(f"{'Accuracy':<69} {correct_xgb}/{len(test_cases)}           {correct_bert}/{len(test_cases)}")

# ── Edge cases ─────────────────────────────────────────────────────────────────
print("\n--- Edge Cases ---")
edge_cases = [
    "lol you're such a clown haha",
    "I want to kill this project deadline",
    "this is absolutely terrible work",
    "that game was sick bro",
    "",
    "ok",
]

for msg in edge_cases:
    if not msg.strip():
        print(f"  [empty string] -> skipped")
        continue
    r = predict_bert(msg, bert_model, tokenizer)
    print(f"  '{msg}'")
    print(f"   -> {r['prediction']}  ({r['confidence']}% confidence)\n")

# ── Batch prediction demo ──────────────────────────────────────────────────────
print("--- Batch Prediction Demo ---")
batch = [
    "You are amazing, keep it up!",
    "I will make your life miserable",
    "Thanks for your help today",
    "You stupid fool, learn to code",
    "Have a great weekend everyone",
]

print(f"\n{'Message':<50} {'Label':<10} {'Confidence'}")
print("-" * 70)
for msg in batch:
    r = predict(msg, mode="bert")
    print(f"{msg[:49]:<50} {r['prediction']:<10} {r['confidence']}%")

# ── Single predict() wrapper demo ──────────────────────────────────────────────
print("\n--- predict() Wrapper (default = DistilBERT) ---")
samples = [
    "You are a terrible human being",
    "Have a wonderful day!",
]
for s in samples:
    r = predict(s)
    print(f"  Input      : {s}")
    print(f"  Prediction : {r['prediction']}  ({r['confidence']}%)  [{r['model']}]")
    print()

print("=" * 60)
print("  Stage 8 COMPLETE")
print("  predict() is ready for use in the web app and voice CLI")
print("  Proceed to Stage 9: Web Application")
print("=" * 60)
