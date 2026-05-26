import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)


def build_features(df, max_features=15000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
    )
    X = vectorizer.fit_transform(df["cleaned_text"])
    y = df["label"].values

    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return X, y, vectorizer


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)
