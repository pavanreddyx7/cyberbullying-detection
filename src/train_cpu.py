import pickle, time, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

os.makedirs("models", exist_ok=True)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42
        ),
    }


def train_all(X_train, y_train, cv_folds=5):
    models = get_models()
    trained, results = {}, []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="f1_weighted", n_jobs=-1
        )
        trained[name] = model
        results.append({
            "name": name,
            "cv_f1_mean": scores.mean(),
            "cv_f1_std": scores.std(),
            "train_time": elapsed,
        })
        print(f"  {name:<22} CV F1: {scores.mean():.4f} +/- {scores.std():.4f}  ({elapsed:.1f}s)")

    return trained, results


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
