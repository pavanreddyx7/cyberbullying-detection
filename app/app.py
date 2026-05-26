import os, sys

# Force UTF-8 on Windows (fixes EasyOCR / emoji progress bar encoding)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

from src.predict      import load_classical, load_bert, predict_classical, predict_bert
from src.image_model  import analyze_image, load_clip
from src.emoji_model  import combined_predict
from src.voice_model  import load_whisper, transcribe as whisper_transcribe

app = Flask(__name__)
UPLOAD = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD, exist_ok=True)

# ── Load models at startup ────────────────────────────────────────────────────
print("Loading XGBoost ...")
xgb_model, vectorizer = load_classical()

print("Loading DistilBERT ...")
bert_model, tokenizer = load_bert()

print("Loading CLIP (image model) ...")
load_clip()

print("Loading Whisper (voice model) ...")
load_whisper()

# EasyOCR loads lazily on first /predict-image call (~50 MB, one-time download)
print("\nAll models ready.  Starting server ...\n")
print("  http://127.0.0.1:5000\n")


# ── Helper ────────────────────────────────────────────────────────────────────
def _bert(text):
    return predict_bert(text, bert_model, tokenizer)


# ═══════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("instagram.html")

@app.route("/basic")
def basic():
    return render_template("index.html")


# ── 1. Text prediction (+ emoji combined) ────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    mode = data.get("mode", "bert")

    if not text:
        return jsonify({"error": "Please enter a message."}), 400

    if mode == "classical":
        result = predict_classical(text, xgb_model, vectorizer)
    elif mode == "emoji":
        result = combined_predict(text, _bert)
        # Normalise keys so frontend works the same way
        result["prediction"] = result["final_label"]
        result["confidence"] = result["final_confidence"]
    else:
        result = _bert(text)

    return jsonify(result)


# ── 2. Emoji-only analysis ────────────────────────────────────────────────────
@app.route("/predict-emoji", methods=["POST"])
def predict_emoji():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    result = combined_predict(text, _bert)
    return jsonify(result)


# ── 3. Voice / audio prediction (Whisper GPU) ────────────────────────────────
@app.route("/predict-voice", methods=["POST"])
def predict_voice():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    f      = request.files["audio"]
    suffix = os.path.splitext(f.filename or "rec.webm")[1] or ".webm"

    try:
        audio_bytes = f.read()
        transcribed = whisper_transcribe(audio_bytes, ext=suffix)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {e}"}), 500

    result                = combined_predict(transcribed, _bert)
    result["transcribed"] = transcribed
    result["prediction"]  = result["final_label"]
    result["confidence"]  = result["final_confidence"]
    return jsonify(result)


# ── 4. Image bullying prediction ──────────────────────────────────────────────
@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    f        = request.files["image"]
    filename = secure_filename(f.filename or "upload.jpg")
    filepath = os.path.join(UPLOAD, filename)
    f.save(filepath)

    try:
        image  = Image.open(filepath).convert("RGB")
        result = analyze_image(image, text_predictor=_bert)
    except Exception as e:
        return jsonify({"error": f"Image analysis failed: {e}"}), 400
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
