"""
train_model.py
--------------
Run this script ONCE to train the model and save the pkl files.

    python train_model.py

Outputs:
    model/fraud_model.pkl
    model/vectorizer.pkl
"""

import re
import os
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import joblib

# ── 1. Load dataset ───────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("email_text_dataset.csv")
print(f"  Rows : {len(df)}")
print(f"  Fraud: {df['label'].sum()} | Legit: {(df['label']==0).sum()}")

# ── 2. Clean text ─────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalise raw email text before vectorising:
      - lowercase
      - remove URLs
      - remove email addresses
      - remove digits
      - remove punctuation
      - collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\nCleaning text...")
df["cleaned"] = df["Text"].apply(clean_text)

# ── 3. Train / test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── 4. TF-IDF vectorisation ───────────────────────────────────────
print("\nVectorising with TF-IDF (max_features=5000, ngrams 1-2)...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── 5. Train Logistic Regression ──────────────────────────────────
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X_train_vec, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]

print("\n" + "="*45)
print("       MODEL EVALUATION RESULTS")
print("="*45)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
print("="*45)
print()
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 7. Save artifacts ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,      "model/fraud_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n✅ Saved:")
print(f"   model/fraud_model.pkl  ({os.path.getsize('model/fraud_model.pkl'):,} bytes)")
print(f"   model/vectorizer.pkl   ({os.path.getsize('model/vectorizer.pkl'):,} bytes)")
print("\nYou can now run:  streamlit run app.py")
