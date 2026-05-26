"""
Image Bullying Detection — two-stage pipeline:
  Stage 1: CLIP visual classification  (is the image harmful/bullying?)
  Stage 2: EasyOCR + DistilBERT        (is any text inside the image abusive?)
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CLIP labels ───────────────────────────────────────────────────────────────
SAFE_LABELS = [
    "a friendly and positive social media post",
    "people having fun and enjoying themselves",
    "a beautiful landscape or scenery",
    "a food photo or recipe",
    "a happy family or friends photo",
]

HARMFUL_LABELS = [
    "cyberbullying or online harassment content",
    "violent or threatening image",
    "hateful or offensive meme",
    "humiliating or degrading content",
    "graphic or disturbing imagery",
]

ALL_LABELS = SAFE_LABELS + HARMFUL_LABELS

_clip_cache = {}


def load_clip():
    if not _clip_cache:
        print("  Loading CLIP model on GPU...")
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        _clip_cache["model"]     = model
        _clip_cache["processor"] = processor
    return _clip_cache["model"], _clip_cache["processor"]


def classify_image_clip(image: Image.Image):
    """
    Returns dict with:
      visual_label     : 'Bullying' or 'Safe'
      visual_confidence: float (0-100)
      top_match        : best matching CLIP label string
      all_scores       : list of (label, score)
    """
    model, processor = load_clip()

    inputs = processor(
        text   = ALL_LABELS,
        images = image,
        return_tensors = "pt",
        padding = True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    label_scores = list(zip(ALL_LABELS, probs))
    label_scores.sort(key=lambda x: x[1], reverse=True)

    # sum probabilities for safe vs harmful groups
    safe_score    = float(sum(probs[:len(SAFE_LABELS)]))
    harmful_score = float(sum(probs[len(SAFE_LABELS):]))

    top_label   = label_scores[0][0]
    is_harmful  = harmful_score > safe_score
    confidence  = round((harmful_score if is_harmful else safe_score) * 100, 2)

    return {
        "visual_label"     : "Bullying" if is_harmful else "Safe",
        "visual_confidence": confidence,
        "top_match"        : top_label,
        "safe_score"       : round(safe_score * 100, 2),
        "harmful_score"    : round(harmful_score * 100, 2),
        "all_scores"       : [(lbl, round(float(sc)*100,2)) for lbl, sc in label_scores[:6]],
    }


_ocr_cache = {}
_ocr_available = None   # None = untested, True = ok, False = unavailable


def load_ocr():
    global _ocr_available
    if _ocr_available is False:
        return None
    if not _ocr_cache:
        try:
            import easyocr
            print("  Loading EasyOCR...")
            reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            _ocr_cache["reader"] = reader
            _ocr_available = True
        except Exception as e:
            print(f"  EasyOCR unavailable ({e.__class__.__name__}): image text extraction disabled.")
            _ocr_available = False
            return None
    return _ocr_cache.get("reader")


def extract_text_from_image(image: Image.Image) -> str:
    """Extract all text from image using EasyOCR. Returns '' if OCR unavailable."""
    reader = load_ocr()
    if reader is None:
        return ""
    try:
        results = reader.readtext(np.array(image), detail=0, paragraph=True)
        return " ".join(results).strip()
    except Exception:
        return ""


def analyze_image(image_path_or_pil, text_predictor=None):
    """
    Full pipeline: visual CLIP check + OCR text abuse check.
    text_predictor: callable that takes text string and returns prediction dict
                    (uses predict_bert from src.predict by default)
    Returns unified result dict.
    """
    # Load image
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")

    result = {"image_width": image.width, "image_height": image.height}

    # ── Stage 1: Visual classification ───────────────────────────────────────
    clip_result = classify_image_clip(image)
    result.update(clip_result)

    # ── Stage 2: OCR + text abuse ─────────────────────────────────────────────
    extracted_text = extract_text_from_image(image)
    result["extracted_text"] = extracted_text

    if extracted_text and text_predictor:
        text_result = text_predictor(extracted_text)
        result["text_label"]      = text_result["prediction"]
        result["text_confidence"] = text_result["confidence"]
    else:
        result["text_label"]      = None
        result["text_confidence"] = None

    # ── Combined verdict ──────────────────────────────────────────────────────
    visual_harmful = (result["visual_label"] == "Bullying")
    text_harmful   = (result["text_label"]   == "Abusive") if result["text_label"] else False

    if visual_harmful and text_harmful:
        result["final_label"]      = "Bullying"
        result["final_confidence"] = round(
            (result["visual_confidence"] + result["text_confidence"]) / 2, 2)
        result["reason"] = "Both visual content and embedded text indicate bullying."
    elif visual_harmful:
        result["final_label"]      = "Bullying"
        result["final_confidence"] = result["visual_confidence"]
        result["reason"] = "Visual content appears harmful or bullying."
    elif text_harmful:
        result["final_label"]      = "Bullying"
        result["final_confidence"] = result["text_confidence"]
        result["reason"] = f'Embedded text is abusive: "{extracted_text[:80]}"'
    else:
        result["final_label"]      = "Safe"
        result["final_confidence"] = round(
            (result["safe_score"] + (100 - (result["text_confidence"] or 0))) / 2, 2
        )
        result["reason"] = "No bullying content detected in image or text."

    return result
