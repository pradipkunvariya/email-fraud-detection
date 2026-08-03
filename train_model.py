"""
train_model.py
----------------
Trains an email fraud (spam/scam) detection model on the fraud_email_.csv
dataset and saves the fitted TF-IDF vectorizer and classifier to the
model/ directory so the Streamlit app (app.py) can load them at runtime.

Usage:
    python train_model.py
"""

import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/fraud_email_.csv"
MODEL_DIR = "model"
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip URLs/emails/numbers/punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"\d+", " ", text)                        # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Text"]).drop_duplicates(subset=["Text"])
    df["clean_text"] = df["Text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Dataset shape after cleaning: {df.shape}")
    print(df["Class"].value_counts())

    X = df["clean_text"]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Candidate models
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "MultinomialNB": MultinomialNB(),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")
    final_preds = best_model.predict(X_test_vec)
    print("\nClassification Report:\n", classification_report(y_test, final_preds, target_names=["Legit (0)", "Fraud (1)"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))

    print("\nSaving model artifacts...")
    joblib.dump(best_model, f"{MODEL_DIR}/fraud_model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")

    metrics_summary = pd.DataFrame(results).T
    metrics_summary.loc["BEST_MODEL"] = best_name
    metrics_summary.to_csv(f"{MODEL_DIR}/metrics.csv")

    print(f"Saved: {MODEL_DIR}/fraud_model.pkl, {MODEL_DIR}/vectorizer.pkl, {MODEL_DIR}/metrics.csv")
    print("Done.")


if __name__ == "__main__":
    main()
