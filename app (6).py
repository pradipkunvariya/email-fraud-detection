"""
app.py
------
Streamlit web app for the Email Fraud Detection model.

Run locally with:
    streamlit run app.py

Requires model/fraud_model.pkl and model/vectorizer.pkl to exist.
Generate them by running `python train_model.py` first.
"""

import re
import string
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Email Fraud Detector",
    page_icon="📧",
    layout="centered",
)

MODEL_PATH = "model/fraud_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Same cleaning logic used during training — must stay in sync."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0][1]  # probability of class "1" (fraud)
    return pred, prob


# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
try:
    model, vectorizer = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a machine learning model trained on a labeled "
        "dataset of fraudulent and legitimate emails to flag suspicious "
        "messages such as phishing attempts and scam emails."
    )
    st.write("**Model:** TF-IDF + scikit-learn classifier")
    st.write("**Labels:** `0` = Legitimate, `1` = Fraud")

    if not model_loaded:
        st.error(
            "Model files not found. Run `python train_model.py` "
            "first to generate `model/fraud_model.pkl` and "
            "`model/vectorizer.pkl`."
        )

    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn")


# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("📧 Email Fraud Detection")
st.write(
    "Paste the content of an email below and the model will predict "
    "whether it looks **legitimate** or **fraudulent** (phishing / scam)."
)

tab1, tab2 = st.tabs(["🔎 Single Email Check", "📁 Batch Check (CSV Upload)"])

# ---- Tab 1: Single email ----
with tab1:
    example_col1, example_col2 = st.columns(2)
    with example_col1:
        if st.button("Load a scam example"):
            st.session_state["email_text"] = (
                "Congratulations! You have won $1,000,000 in the international "
                "lottery. To claim your prize, please send your full name, "
                "address and bank account details to our claims department "
                "immediately. This offer expires in 24 hours."
            )
    with example_col2:
        if st.button("Load a legit example"):
            st.session_state["email_text"] = (
                "Hi team, just a reminder that the sprint review is tomorrow "
                "at 10am. Please make sure your slides are ready and shared "
                "in the drive folder beforehand. Thanks!"
            )

    email_text = st.text_area(
        "Email content",
        height=220,
        key="email_text",
        placeholder="Paste the email text here...",
    )

    if st.button("Analyze Email", type="primary", disabled=not model_loaded):
        if not email_text.strip():
            st.warning("Please paste some email text first.")
        else:
            pred, prob = predict(email_text, model, vectorizer)
            st.markdown("---")
            if pred == 1:
                st.error("🚨 **This email looks FRAUDULENT.**")
            else:
                st.success("✅ **This email looks LEGITIMATE.**")

            if prob is not None:
                st.metric("Estimated fraud probability", f"{prob:.1%}")
                st.progress(min(max(prob, 0.0), 1.0))

            with st.expander("See cleaned text used for prediction"):
                st.code(clean_text(email_text))

# ---- Tab 2: Batch CSV ----
with tab2:
    st.write(
        "Upload a CSV file with a column named **`Text`** containing one "
        "email per row. The app will predict a label for every row."
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None and model_loaded:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            batch_df = None

        if batch_df is not None:
            if "Text" not in batch_df.columns:
                st.error("The uploaded CSV must contain a column named 'Text'.")
            else:
                with st.spinner("Analyzing emails..."):
                    cleaned = batch_df["Text"].apply(clean_text)
                    vecs = vectorizer.transform(cleaned)
                    preds = model.predict(vecs)
                    probs = (
                        model.predict_proba(vecs)[:, 1]
                        if hasattr(model, "predict_proba")
                        else [None] * len(preds)
                    )

                batch_df["Predicted_Class"] = preds
                batch_df["Predicted_Label"] = batch_df["Predicted_Class"].map(
                    {0: "Legit", 1: "Fraud"}
                )
                batch_df["Fraud_Probability"] = probs

                st.success(f"Analyzed {len(batch_df)} emails.")
                st.dataframe(
                    batch_df[["Text", "Predicted_Label", "Fraud_Probability"]],
                    use_container_width=True,
                )

                fraud_count = int((batch_df["Predicted_Class"] == 1).sum())
                st.write(
                    f"**{fraud_count} / {len(batch_df)}** emails were flagged as fraud."
                )

                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_out,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )
