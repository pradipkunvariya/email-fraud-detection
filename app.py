"""
app.py
------
Email Fraud Detection — Streamlit Web App
Author : Pradip Kunvariya

Run locally:
    python train_model.py   # only needed once
    streamlit run app.py
"""

import re
import os
import string
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import MultinomialNB

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Fraud Detector",
    page_icon="📧",
    layout="wide",
)

MODEL_PATH      = "model/fraud_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
DATASET_PATH    = "email_text_dataset.csv"

# ── Text cleaner (must match train_model.py exactly) ─────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Load pre-trained model + vectorizer ──────────────────────────
@st.cache_resource
def load_artifacts():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# ── Load dataset ─────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)

# ── Train all 4 models for comparison tab ────────────────────────
@st.cache_resource
def train_all_models():
    df = load_dataset()
    df["cleaned"] = df["Text"].apply(clean_text)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["cleaned"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr_v = vec.fit_transform(X_tr)
    X_te_v = vec.transform(X_te)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
        "Naive Bayes":         MultinomialNB(),
    }
    trained, results = {}, []
    for name, m in models.items():
        m.fit(X_tr_v, y_tr)
        p  = m.predict(X_te_v)
        pb = m.predict_proba(X_te_v)[:, 1]
        trained[name] = m
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, p),  4),
            "Precision": round(precision_score(y_te, p), 4),
            "Recall":    round(recall_score(y_te, p),    4),
            "F1 Score":  round(f1_score(y_te, p),        4),
            "ROC-AUC":   round(roc_auc_score(y_te, pb),  4),
            "_preds":    p,
            "_probs":    pb,
        })
    return trained, pd.DataFrame(results), vec, X_te_v, y_te

# ── Check files exist ─────────────────────────────────────────────
model_loaded   = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
dataset_exists = os.path.exists(DATASET_PATH)

if model_loaded:
    main_model, main_vec = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📧 Email Fraud Detector")
    st.markdown("---")
    st.markdown("### About")
    st.write(
        "This app uses a **TF-IDF + Machine Learning** pipeline "
        "trained on 1,800 email samples to detect phishing, scam, "
        "and fraudulent emails."
    )
    st.markdown("**Models:** Logistic Regression · Random Forest · Decision Tree · Naive Bayes")
    st.markdown("**Labels:** `0` = Legitimate &nbsp; `1` = Fraud")
    st.markdown("---")

    if not model_loaded:
        st.error(
            "⚠️ Model files not found.\n\n"
            "Run `python train_model.py` first to generate:\n"
            "- `model/fraud_model.pkl`\n"
            "- `model/vectorizer.pkl`"
        )
    else:
        st.success("✅ Model loaded and ready")

    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · TF-IDF")
    st.caption("Author: Pradip Kunvariya · GIT, B.Tech CE")

# ── Title ─────────────────────────────────────────────────────────
st.title("📧 Email Fraud Detection System")
st.markdown(
    "A Machine Learning web app that detects **phishing, scam, and fraudulent emails** "
    "using TF-IDF text vectorisation and multiple classifiers."
)
st.divider()

# ── Dataset KPI row (always visible) ─────────────────────────────
if dataset_exists:
    _df = load_dataset()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Emails",   f"{len(_df)}")
    k2.metric("Fraud Emails",   f"{_df['label'].sum()}")
    k3.metric("Legit Emails",   f"{((_df['label']==0).sum())}")
    k4.metric("Fraud Rate",     f"{_df['label'].mean()*100:.1f}%")
    st.divider()

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Single Email Check",
    "📁 Batch CSV Check",
    "📊 EDA",
    "🤖 Model Comparison",
    "📋 Dataset Preview",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE EMAIL CHECK
# ════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔎 Single Email Check")
    st.markdown("Paste any email content below and click **Analyse Email**.")

    # Example buttons
    ex1, ex2 = st.columns(2)
    with ex1:
        if st.button("📌 Load Scam Example"):
            st.session_state["email_text"] = (
                "CONGRATULATIONS!!! You have been selected as our lucky winner "
                "of ONE MILLION DOLLARS in the international lottery!!! "
                "To claim your prize IMMEDIATELY send your full name, bank account "
                "number and a processing fee of $100 to our claims office. "
                "ACT NOW this offer expires in 24 hours!!! Click here to verify!!!"
            )
    with ex2:
        if st.button("📌 Load Legit Example"):
            st.session_state["email_text"] = (
                "Hi team, just a reminder that the sprint review is scheduled "
                "for tomorrow at 10am. Please make sure your updates are added "
                "to the Jira board before the meeting. If you have any blockers "
                "please let me know in advance. Thanks and see you all tomorrow."
            )

    email_text = st.text_area(
        "Email content",
        height=220,
        key="email_text",
        placeholder="Paste the full email text here (subject + body)...",
    )

    if st.button("🚀 Analyse Email", type="primary", disabled=not model_loaded):
        if not email_text.strip():
            st.warning("Please paste some email text first.")
        else:
            cleaned   = clean_text(email_text)
            vec_input = main_vec.transform([cleaned])
            pred      = main_model.predict(vec_input)[0]
            prob      = main_model.predict_proba(vec_input)[0]

            st.divider()
            r1, r2 = st.columns([1, 2])

            with r1:
                if pred == 1:
                    st.error("🚨 **FRAUDULENT EMAIL**")
                    st.metric("Fraud Probability",      f"{prob[1]*100:.1f}%")
                    st.metric("Legitimate Probability", f"{prob[0]*100:.1f}%")
                    st.progress(float(prob[1]))
                else:
                    st.success("✅ **LEGITIMATE EMAIL**")
                    st.metric("Legitimate Probability", f"{prob[0]*100:.1f}%")
                    st.metric("Fraud Probability",      f"{prob[1]*100:.1f}%")
                    st.progress(float(prob[0]))

            with r2:
                st.markdown("#### Risk Gauge")
                fig, ax = plt.subplots(figsize=(5, 2.5))
                bar_color = "#e74c3c" if pred == 1 else "#2ecc71"
                ax.barh(["Fraud Risk"], [prob[1] * 100],
                        color=bar_color, edgecolor="white", height=0.4)
                ax.barh(["Fraud Risk"], [100 - prob[1] * 100],
                        left=[prob[1] * 100],
                        color="#ecf0f1", edgecolor="white", height=0.4)
                ax.set_xlim(0, 100)
                ax.set_xlabel("Fraud Probability (%)")
                ax.axvline(50, color="gray", linestyle="--", linewidth=1)
                ax.text(prob[1]*100/2, 0, f"{prob[1]*100:.1f}%",
                        ha="center", va="center", fontweight="bold",
                        color="white", fontsize=13)
                ax.set_title("Fraud Probability Gauge", fontsize=12)
                st.pyplot(fig)

            with st.expander("🔍 Cleaned text used for prediction"):
                st.code(cleaned)

# ════════════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV CHECK
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("📁 Batch Email Check")
    st.markdown(
        "Upload a CSV file with a column named **`Text`** "
        "(one email per row). The app will predict a label for every row."
    )

    # Download sample CSV
    if dataset_exists:
        sample = load_dataset()[["Text","label"]].head(10).copy()
        sample_no_label = sample.drop("label", axis=1)
        st.download_button(
            "⬇️ Download sample CSV (10 rows)",
            data=sample_no_label.to_csv(index=False).encode(),
            file_name="sample_emails.csv",
            mime="text/csv",
        )
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None and model_loaded:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            batch_df = None

        if batch_df is not None:
            if "Text" not in batch_df.columns:
                st.error("❌ CSV must have a column named **`Text`**.")
            else:
                with st.spinner("Analysing emails..."):
                    cleaned_batch = batch_df["Text"].apply(clean_text)
                    vecs          = main_vec.transform(cleaned_batch)
                    preds         = main_model.predict(vecs)
                    probs         = main_model.predict_proba(vecs)[:, 1]

                batch_df["Predicted_Label"]   = preds
                batch_df["Result"]            = batch_df["Predicted_Label"].map(
                    {0: "✅ Legit", 1: "🚨 Fraud"}
                )
                batch_df["Fraud_Probability"] = (probs * 100).round(1)

                fraud_n = int(preds.sum())
                legit_n = len(preds) - fraud_n

                b1, b2, b3 = st.columns(3)
                b1.metric("Total Scanned",   len(batch_df))
                b2.metric("Flagged Fraud",   fraud_n)
                b3.metric("Flagged Legit",   legit_n)

                st.success(f"✅ Analysed **{len(batch_df)}** emails — "
                           f"**{fraud_n}** flagged as fraud.")

                st.dataframe(
                    batch_df[["Text", "Result", "Fraud_Probability"]],
                    use_container_width=True,
                )

                st.download_button(
                    "⬇️ Download Results CSV",
                    data=batch_df.to_csv(index=False).encode(),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Exploratory Data Analysis")

    if not dataset_exists:
        st.warning("Dataset not found. Add email_text_dataset.csv to the repo.")
    else:
        df_eda = load_dataset()

        # Class distribution
        st.subheader("Class Distribution")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = df_eda["label"].value_counts()
            ax.bar(["Legitimate", "Fraud"], counts.values,
                   color=["#2ecc71","#e74c3c"], edgecolor="white", width=0.5)
            ax.set_title("Email Class Count", fontsize=13)
            ax.set_ylabel("Count")
            for i, v in enumerate(counts.values):
                ax.text(i, v + 5, str(v), ha="center",
                        fontsize=12, fontweight="bold")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(counts.values, labels=["Legitimate","Fraud"],
                   colors=["#2ecc71","#e74c3c"],
                   autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize":12})
            ax.set_title("Fraud vs Legitimate Split", fontsize=13)
            st.pyplot(fig)

        # Text length distribution
        st.subheader("Email Text Length Distribution")
        df_eda["text_len"] = df_eda["Text"].apply(lambda x: len(str(x)))
        fig, ax = plt.subplots(figsize=(10, 4))
        df_eda[df_eda["label"]==0]["text_len"].hist(
            ax=ax, bins=30, alpha=0.6, color="#2ecc71", label="Legitimate")
        df_eda[df_eda["label"]==1]["text_len"].hist(
            ax=ax, bins=30, alpha=0.6, color="#e74c3c", label="Fraud")
        ax.set_title("Email Text Length — Fraud vs Legitimate", fontsize=13)
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

        # Word count
        st.subheader("Word Count Distribution")
        df_eda["word_count"] = df_eda["Text"].apply(
            lambda x: len(str(x).split()))
        fig, ax = plt.subplots(figsize=(10, 4))
        df_eda[df_eda["label"]==0]["word_count"].hist(
            ax=ax, bins=25, alpha=0.6, color="#2ecc71", label="Legitimate")
        df_eda[df_eda["label"]==1]["word_count"].hist(
            ax=ax, bins=25, alpha=0.6, color="#e74c3c", label="Fraud")
        ax.set_title("Word Count per Email", fontsize=13)
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

        # Stats table
        st.subheader("Text Length Stats by Class")
        stats = df_eda.groupby("label")["text_len"].agg(
            ["min","max","mean","median"]).round(1)
        stats.index = ["Legitimate","Fraud"]
        st.dataframe(stats, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("🤖 Model Comparison")

    if not dataset_exists:
        st.warning("Dataset not found. Cannot train comparison models.")
    else:
        with st.spinner("Training all 4 models..."):
            all_models, res_df, cmp_vec, X_te_v, y_te = train_all_models()

        # Metrics table
        st.subheader("All Models — Metrics Summary")
        st.dataframe(
            res_df[["Model","Accuracy","Precision","Recall","F1 Score","ROC-AUC"]],
            use_container_width=True, hide_index=True
        )

        best = res_df.loc[res_df["F1 Score"].idxmax()]
        st.success(
            f"🏆 **Best Model: {best['Model']}** — "
            f"Accuracy: {best['Accuracy']} | "
            f"F1: {best['F1 Score']} | "
            f"AUC: {best['ROC-AUC']}"
        )

        # Bar comparison
        st.subheader("Metric Comparison Chart")
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        bar_colors = ["#3498db","#e67e22","#2ecc71","#9b59b6"]
        for ax, metric in zip(axes, ["Accuracy","F1 Score","ROC-AUC"]):
            ax.bar(res_df["Model"], res_df[metric],
                   color=bar_colors, edgecolor="white")
            ax.set_title(metric, fontsize=12)
            ax.set_ylim(0.7, 1.05)
            ax.set_xticklabels(res_df["Model"], rotation=20,
                               ha="right", fontsize=9)
            for i, v in enumerate(res_df[metric]):
                ax.text(i, v + 0.008, f"{v:.3f}", ha="center",
                        fontsize=10, fontweight="bold")
        plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        # ROC curves
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#3498db","#e67e22","#2ecc71","#9b59b6"]
        for (_, row), col in zip(res_df.iterrows(), colors):
            fpr, tpr, _ = roc_curve(y_te, row["_probs"])
            ax.plot(fpr, tpr, label=f"{row['Model']} (AUC={row['ROC-AUC']:.3f})",
                    linewidth=2, color=col)
        ax.plot([0,1],[0,1],"k--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models", fontsize=13)
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Confusion matrix per model
        st.subheader("Confusion Matrix")
        sel = st.selectbox("Select model", res_df["Model"].tolist(), key="cm_sel")
        sel_row = res_df[res_df["Model"] == sel].iloc[0]
        cm = confusion_matrix(y_te, sel_row["_preds"])
        fig, ax = plt.subplots(figsize=(5, 4))
        cmap = {"Logistic Regression":"Blues","Random Forest":"Greens",
                "Decision Tree":"Oranges","Naive Bayes":"Purples"}[sel]
        ConfusionMatrixDisplay(cm, display_labels=["Legitimate","Fraud"]).plot(
            ax=ax, colorbar=False, cmap=cmap)
        ax.set_title(f"{sel} — Confusion Matrix", fontsize=12)
        st.pyplot(fig)

# ════════════════════════════════════════════════════════════════
# TAB 5 — DATASET PREVIEW
# ════════════════════════════════════════════════════════════════
with tab5:
    st.header("📋 Dataset Preview")

    if not dataset_exists:
        st.warning("email_text_dataset.csv not found in the repository.")
    else:
        df_prev = load_dataset()

        st.subheader("df.head()")
        st.dataframe(df_prev.head(), use_container_width=True)

        st.subheader("df.tail()")
        st.dataframe(df_prev.tail(), use_container_width=True)

        st.subheader("df.describe()")
        st.dataframe(df_prev.describe(include="all").fillna("").round(2),
                     use_container_width=True)

        st.subheader("Missing Values — df.isnull().sum()")
        null_df = df_prev.isnull().sum().reset_index()
        null_df.columns = ["Column", "Missing Count"]
        st.dataframe(null_df, use_container_width=True, hide_index=True)

        st.subheader("Fraud Samples")
        st.dataframe(
            df_prev[df_prev["label"]==1][["Text","label"]].head(5),
            use_container_width=True, hide_index=True
        )

        st.subheader("Legitimate Samples")
        st.dataframe(
            df_prev[df_prev["label"]==0][["Text","label"]].head(5),
            use_container_width=True, hide_index=True
        )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📧 Email Fraud Detection System · Pradip Kunvariya · "
    "B.Tech Computer Engineering · Gandhinagar University · "
    "Built with Streamlit & scikit-learn"
)
