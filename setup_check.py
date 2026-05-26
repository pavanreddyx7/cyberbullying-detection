"""
Stage 1 — Setup Verification Script
Run this after installing all dependencies to confirm everything is ready.
"""

import sys
import importlib

# ── 1. Python version ─────────────────────────────────────────────────────────
print("=" * 55)
print("  STAGE 1 — ENVIRONMENT CHECK")
print("=" * 55)

py = sys.version_info
print(f"\nPython version : {py.major}.{py.minor}.{py.micro}", end="")
if py.major == 3 and py.minor >= 10:
    print("  [OK]")
else:
    print("  [WARNING] Python 3.10+ recommended")

# ── 2. Required packages ──────────────────────────────────────────────────────
packages = {
    "pandas"          : "pandas",
    "numpy"           : "numpy",
    "sklearn"         : "scikit-learn",
    "nltk"            : "nltk",
    "matplotlib"      : "matplotlib",
    "seaborn"         : "seaborn",
    "wordcloud"       : "wordcloud",
    "flask"           : "flask",
    "speech_recognition": "SpeechRecognition",
    "pyttsx3"         : "pyttsx3",
    "torch"           : "torch",
    "transformers"    : "transformers",
    "xgboost"         : "xgboost",
}

print("\nPackage checks:")
all_ok = True
for import_name, pip_name in packages.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {pip_name:<22} {version:<12} [OK]")
    except ImportError:
        print(f"  {pip_name:<22} {'MISSING':<12} [FAIL] -> pip install {pip_name}")
        all_ok = False

# ── 3. CUDA / GPU ─────────────────────────────────────────────────────────────
print("\nGPU / CUDA check:")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available  : {cuda_available}")
    if cuda_available:
        print(f"  GPU name        : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Quick GPU tensor test
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        assert x.device.type == "cuda"
        print(f"  Tensor on GPU   : {x}  [OK]")
    else:
        print("  [WARNING] No GPU detected — deep learning will run on CPU (slow)")
        print("  Install CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 4. NLTK resources ─────────────────────────────────────────────────────────
print("\nNLTK resource check:")
import nltk
import os

nltk_resources = ["stopwords", "punkt", "wordnet"]
for resource in nltk_resources:
    try:
        if resource == "stopwords":
            from nltk.corpus import stopwords
            _ = stopwords.words("english")
        elif resource == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif resource == "wordnet":
            from nltk.corpus import wordnet
            _ = wordnet.synsets("test")
        print(f"  {resource:<20} [OK]")
    except LookupError:
        print(f"  {resource:<20} [MISSING] -> downloading...")
        nltk.download(resource, quiet=True)
        print(f"  {resource:<20} [OK] (just downloaded)")

# ── 5. Folder structure ───────────────────────────────────────────────────────
print("\nFolder structure check:")
required_dirs = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "app/templates",
    "app/static",
    "reports/figures",
    "uploads",
]
for d in required_dirs:
    exists = os.path.isdir(d)
    print(f"  {d:<30} {'[OK]' if exists else '[MISSING]'}")
    if not exists:
        os.makedirs(d, exist_ok=True)
        print(f"  {d:<30} [CREATED]")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if all_ok:
    print("  Stage 1 COMPLETE — ready to proceed to Stage 2")
else:
    print("  Stage 1 INCOMPLETE — fix the [FAIL] items above")
print("=" * 55)
