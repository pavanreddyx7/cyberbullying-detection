# Cyberbullying & Abuse Detection Using AI

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2011.8%2B-orange)
![Flask](https://img.shields.io/badge/Framework-Flask-green)

A comprehensive machine learning system designed to detect abusive, toxic, or harmful text content in real-time. This project combines Natural Language Processing (NLP) techniques, classical Machine Learning models, and GPU-accelerated Deep Learning (Transformers) to classify text and voice inputs accurately. It features a fully functional Flask web interface for interactive use.

## ✨ Features

- **Real-Time Text Detection:** Classify input text as either 'Safe' or 'Abusive' in milliseconds.
- **Voice Input Support:** Includes Speech-to-Text capabilities allowing users to provide audio input via microphone for abuse detection.
- **Multiple AI Models:** 
  - Classical Models: Logistic Regression, Naive Bayes, Linear SVM, Random Forest (TF-IDF vectorization).
  - Advanced Models: **XGBoost (GPU)** and **DistilBERT (GPU)** for high-accuracy contextual understanding.
- **Confidence Scoring:** Outputs not just the classification label but also the model's confidence percentage.
- **Interactive Web UI:** A beautiful and responsive web application built with Flask and Vanilla CSS for a seamless user experience.

## 🛠️ Technology Stack

- **Core & Data Processing:** Python 3.10+, Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Deep Learning (GPU):** PyTorch, HuggingFace Transformers (DistilBERT)
- **Natural Language Processing:** NLTK, TF-IDF
- **Voice Integration:** SpeechRecognition, pyttsx3, Web Speech API
- **Web Application:** Flask, HTML5, CSS3, JavaScript
- **Visualization:** Matplotlib, Seaborn, WordCloud

## 📂 Project Structure

```
ai-abuse-ml/
├── app/                  # Flask web application & UI templates
├── data/                 # Raw and processed datasets
├── models/               # Saved trained models (.pkl, .json, DistilBERT)
├── notebooks/            # Jupyter notebooks for EDA and model training
├── reports/              # Project reports, generated charts, and metrics
├── src/                  # Core Python modules (training, prep, predict)
├── requirements.txt      # Project dependencies
└── PROJECT_GUIDE.md      # Detailed step-by-step project development guide
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed. For GPU acceleration (highly recommended for DistilBERT and XGBoost), you will need an NVIDIA GPU and CUDA Toolkit 11.8+ installed.

### 1. Clone the repository

```bash
git clone https://github.com/pavanreddyx7/cyberbullying-detection.git
cd cyberbullying-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*(Note: Depending on your OS, you may need to install PyAudio specifically for the voice features. See `PROJECT_GUIDE.md` for details).*

### 3. Run the Web Application

Launch the Flask server:

```bash
python app/app.py
```

The web application will be accessible at `http://localhost:5000` or `http://127.0.0.1:5000` in your web browser.

## 📊 Dataset & Training

The system was initially built and tested using the [Hate Speech & Offensive Language dataset](https://github.com/t-davidson/hate-speech-and-offensive-language). The data goes through rigorous preprocessing (URL removal, mention stripping, lemmatization, stopword removal) before being passed into the models. For a detailed breakdown of the Exploratory Data Analysis (EDA) and training pipeline, refer to the `PROJECT_GUIDE.md` and the notebooks provided in the `notebooks/` directory.

## 📜 License

This project is open-source and available under the MIT License.
