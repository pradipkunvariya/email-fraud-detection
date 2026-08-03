import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Fraud Detection",
    page_icon="📧",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────
st.title("📧 Email Fraud Detection System")
st.markdown(
    "A Machine Learning web application to detect fraudulent emails "
    "using **Logistic Regression, Decision Tree, Random Forest** and **Naive Bayes**."
)
st.divider()

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_data(source):
    return pd.read_csv(source)

if os.path.exists("email_fraud_dataset.csv"):
    df = load_data("email_fraud_dataset.csv")
else:
    st.warning("⚠️  **email_fraud_dataset.csv** not found in the repository.")
    up = st.file_uploader("Upload email_fraud_dataset.csv", type=["csv"])
    if up is None:
        st.info("👆 Upload the dataset to continue, or add it to your GitHub repo.")
        st.stop()
    df = load_data(up)

# ── Preprocessing ─────────────────────────────────────────────────
@st.cache_data
def preprocess(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc  = sc.transform(X_te)
    return X, y, X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, sc

X, y, X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, scaler = preprocess(df)

# ── Train all models ──────────────────────────────────────────────
@st.cache_resource
def train_all(X_tr, y_tr, X_tr_sc):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes":         GaussianNB(),
    }
    trained = {}
    for name, m in models.items():
        if name == "Logistic Regression":
            m.fit(X_tr_sc, y_tr)
        else:
            m.fit(X_tr, y_tr)
        trained[name] = m
    return trained

trained_models = train_all(X_tr, y_tr, X_tr_sc)

# ── Compute metrics for all models ───────────────────────────────
@st.cache_data
def get_results(_trained, X_te, y_te, X_te_sc):
    rows = []
    for name, m in _trained.items():
        Xp = X_te_sc if name == "Logistic Regression" else X_te
        p  = m.predict(Xp)
        pb = m.predict_proba(Xp)[:, 1]
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, p), 4),
            "Precision": round(precision_score(y_te, p), 4),
            "Recall":    round(recall_score(y_te, p), 4),
            "F1 Score":  round(f1_score(y_te, p), 4),
            "ROC-AUC":   round(roc_auc_score(y_te, pb), 4),
            "_preds":    p,
            "_probs":    pb,
        })
    return pd.DataFrame(rows)

results_df = get_results(trained_models, X_te, y_te, X_te_sc)

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dataset",
    "📊 EDA",
    "🤖 Model Results",
    "📈 ROC & Comparison",
    "🔮 Predict Email",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ════════════════════════════════════════════════════════════════
with tab1:
    st.header("📋 Dataset Overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Emails",    f"{len(df)}")
    k2.metric("Fraud Emails",    f"{df['label'].sum()}")
    k3.metric("Legit Emails",    f"{(df['label']==0).sum()}")
    k4.metric("Fraud Rate",      f"{df['label'].mean()*100:.1f}%")

    st.subheader("df.head()")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("df.tail()")
    st.dataframe(df.tail(), use_container_width=True)

    st.subheader("df.describe()")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.subheader("Missing Values — df.isnull().sum()")
    null_df = df.isnull().sum().reset_index()
    null_df.columns = ["Feature", "Missing Count"]
    st.dataframe(null_df, use_container_width=True, hide_index=True)

    st.subheader("Column Data Types — df.info()")
    info_df = pd.DataFrame({
        "Column":   df.columns,
        "Dtype":    [str(df[c].dtype) for c in df.columns],
        "Non-Null": [df[c].notna().sum() for c in df.columns],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Exploratory Data Analysis")

    # Class distribution
    st.subheader("Class Distribution")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["label"].value_counts()
        ax.bar(["Legitimate", "Fraud"], counts.values,
               color=["#2ecc71","#e74c3c"], edgecolor="white", width=0.5)
        ax.set_title("Email Class Distribution", fontsize=13)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha="center", fontsize=12, fontweight="bold")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts.values, labels=["Legitimate","Fraud"],
               colors=["#2ecc71","#e74c3c"],
               autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 12})
        ax.set_title("Fraud vs Legitimate Split", fontsize=13)
        st.pyplot(fig)

    # Feature distributions
    st.subheader("Feature Distributions — Fraud vs Legitimate")
    feat = st.selectbox("Select feature", [c for c in df.columns if c != "label"])
    fig, ax = plt.subplots(figsize=(9, 4))
    df[df["label"]==0][feat].hist(ax=ax, alpha=0.6, color="#2ecc71",
                                   label="Legitimate", bins=25)
    df[df["label"]==1][feat].hist(ax=ax, alpha=0.6, color="#e74c3c",
                                   label="Fraud", bins=25)
    ax.set_title(f"{feat.replace('_',' ').title()} Distribution", fontsize=13)
    ax.legend()
    st.pyplot(fig)

    # Boxplots
    st.subheader("Boxplots — Key Features vs Label")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col in zip(axes, ["num_exclamations","char_freq_dollar","capital_run_avg"]):
        df.boxplot(column=col, by="label", ax=ax,
                   patch_artist=True,
                   medianprops={"color":"black","linewidth":2})
        ax.set_title(col.replace("_"," ").title(), fontsize=11)
        ax.set_xlabel("0 = Legit  |  1 = Fraud")
    plt.suptitle("Boxplots by Label", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 9))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr().round(2), mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=13)
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 Model Performance")

    # Metrics table
    st.subheader("All Models — Metrics Summary")
    st.dataframe(
        results_df[["Model","Accuracy","Precision","Recall","F1 Score","ROC-AUC"]],
        use_container_width=True, hide_index=True
    )

    # Best model
    best_row = results_df.loc[results_df["F1 Score"].idxmax()]
    st.success(
        f"🏆 **Best Model: {best_row['Model']}** — "
        f"Accuracy: {best_row['Accuracy']}  |  "
        f"F1: {best_row['F1 Score']}  |  "
        f"AUC: {best_row['ROC-AUC']}"
    )

    # Per-model detail
    st.subheader("Confusion Matrix — Select Model")
    sel = st.selectbox("Model", results_df["Model"].tolist(), key="cm_sel")
    row = results_df[results_df["Model"] == sel].iloc[0]

    Xp = X_te_sc if sel == "Logistic Regression" else X_te
    cm = confusion_matrix(y_te, row["_preds"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{row['Accuracy']:.4f}")
    c2.metric("Precision", f"{row['Precision']:.4f}")
    c3.metric("Recall",    f"{row['Recall']:.4f}")
    c4.metric("F1 Score",  f"{row['F1 Score']:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    cmaps = {
        "Logistic Regression": "Blues",
        "Decision Tree":       "Oranges",
        "Random Forest":       "Greens",
        "Naive Bayes":         "Purples",
    }
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate","Fraud"]).plot(
        ax=ax, colorbar=False, cmap=cmaps[sel]
    )
    ax.set_title(f"{sel} — Confusion Matrix", fontsize=13)
    st.pyplot(fig)

    # Decision Tree diagram
    if sel == "Decision Tree":
        st.subheader("Decision Tree Structure")
        fig_tree, ax_tree = plt.subplots(figsize=(20, 8))
        plot_tree(
            trained_models["Decision Tree"],
            feature_names=X.columns,
            class_names=["Legit","Fraud"],
            filled=True, rounded=True, max_depth=3,
            fontsize=9, ax=ax_tree
        )
        ax_tree.set_title("Decision Tree (depth shown = 3)", fontsize=13)
        st.pyplot(fig_tree)

    # Random Forest feature importance
    if sel == "Random Forest":
        st.subheader("Feature Importance")
        imp = pd.Series(
            trained_models["Random Forest"].feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
        imp.plot(kind="bar", color="#3498db", edgecolor="white", ax=ax_imp)
        ax_imp.set_title("Random Forest — Feature Importance", fontsize=13)
        ax_imp.set_ylabel("Importance")
        ax_imp.set_xticklabels(imp.index, rotation=45, ha="right")
        st.pyplot(fig_imp)

# ════════════════════════════════════════════════════════════════
# TAB 4 — ROC & COMPARISON
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("📈 ROC Curves & Model Comparison")

    # ROC curves
    st.subheader("ROC Curves — All Models")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db","#e67e22","#2ecc71","#9b59b6"]
    for (_, row), color in zip(results_df.iterrows(), colors):
        fpr, tpr, _ = roc_curve(y_te, row["_probs"])
        ax.plot(fpr, tpr, label=f"{row['Model']} (AUC={row['ROC-AUC']:.3f})",
                linewidth=2, color=color)
    ax.plot([0,1],[0,1],"k--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Bar comparison
    st.subheader("Metric Comparison — All Models")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors_bar = ["#3498db","#e67e22","#2ecc71","#9b59b6"]
    for ax, metric in zip(axes, ["Accuracy","F1 Score","ROC-AUC"]):
        ax.bar(results_df["Model"], results_df[metric],
               color=colors_bar, edgecolor="white")
        ax.set_title(metric, fontsize=13)
        ax.set_ylim(0.75, 1.05)
        ax.set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=9)
        for i, v in enumerate(results_df[metric]):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.suptitle("Model Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Full Results Table")
    st.dataframe(
        results_df[["Model","Accuracy","Precision","Recall","F1 Score","ROC-AUC"]],
        use_container_width=True, hide_index=True
    )

# ════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔮 Predict a Single Email")
    st.markdown(
        "Fill in the email features below and click **Predict** "
        "to check if it is Fraud or Legitimate."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("📝 Email Content")
        subject_length  = st.slider("Subject Length (chars)", 5, 100, 30)
        body_length     = st.slider("Body Length (chars)",    30, 1500, 300)
        num_links       = st.slider("Number of Links",        0, 20, 1)
        num_excl        = st.slider("Number of Exclamation Marks", 0, 25, 1)
        misspellings    = st.slider("Misspellings Count",     0, 20, 0)
        html_ratio      = st.slider("HTML Ratio (0–1)",       0.0, 1.0, 0.2, step=0.05)

    with c2:
        st.subheader("🔑 Keyword Flags")
        contains_urgent = st.radio("Contains Urgent?",         ["No","Yes"], horizontal=True)
        contains_money  = st.radio("Contains Money?",          ["No","Yes"], horizontal=True)
        contains_prize  = st.radio("Contains Prize/Win?",      ["No","Yes"], horizontal=True)
        contains_verify = st.radio("Contains Verify/Account?", ["No","Yes"], horizontal=True)
        contains_click  = st.radio("Contains Click Here?",     ["No","Yes"], horizontal=True)
        has_attachment  = st.radio("Has Attachment?",          ["No","Yes"], horizontal=True)

    with c3:
        st.subheader("📬 Sender Info")
        sender_domain_legit  = st.radio("Sender Domain Looks Legit?",      ["No","Yes"], horizontal=True)
        reply_to_diff_domain = st.radio("Reply-To is Different Domain?",   ["No","Yes"], horizontal=True)
        char_freq_dollar     = st.slider("$ Sign Frequency (0–1)",          0.0, 1.0, 0.01, step=0.01)
        capital_run_avg      = st.slider("Avg Capital Run Length",          1.0, 18.0, 2.0, step=0.5)
        capital_run_longest  = st.slider("Longest Capital Run",            1, 150, 5)

    pred_model = st.selectbox("Choose Model for Prediction",
                              results_df["Model"].tolist(), key="pred_model_sel")

    if st.button("🚀 Predict This Email", use_container_width=True):
        def yn(v): return 1 if v == "Yes" else 0

        input_vec = pd.DataFrame([{
            "subject_length":            subject_length,
            "body_length":               body_length,
            "num_links":                 num_links,
            "num_exclamations":          num_excl,
            "has_attachment":            yn(has_attachment),
            "sender_domain_legit":       yn(sender_domain_legit),
            "contains_urgent":           yn(contains_urgent),
            "contains_money":            yn(contains_money),
            "contains_prize":            yn(contains_prize),
            "contains_verify":           yn(contains_verify),
            "contains_click":            yn(contains_click),
            "misspellings":              misspellings,
            "reply_to_different_domain": yn(reply_to_diff_domain),
            "char_freq_dollar":          char_freq_dollar,
            "capital_run_avg":           capital_run_avg,
            "capital_run_longest":       capital_run_longest,
            "html_ratio":                html_ratio,
        }])

        model = trained_models[pred_model]
        if pred_model == "Logistic Regression":
            input_scaled = scaler.transform(input_vec)
            pred  = model.predict(input_scaled)[0]
            prob  = model.predict_proba(input_scaled)[0]
        else:
            pred  = model.predict(input_vec)[0]
            prob  = model.predict_proba(input_vec)[0]

        st.divider()
        r1, r2 = st.columns([1, 2])

        with r1:
            if pred == 1:
                st.error("🚨 **FRAUD EMAIL DETECTED**")
                st.metric("Fraud Probability",     f"{prob[1]*100:.1f}%")
                st.metric("Legitimate Probability",f"{prob[0]*100:.1f}%")
            else:
                st.success("✅ **LEGITIMATE EMAIL**")
                st.metric("Legitimate Probability",f"{prob[0]*100:.1f}%")
                st.metric("Fraud Probability",     f"{prob[1]*100:.1f}%")

        with r2:
            st.markdown("#### 📋 Your Input Summary")
            summary = pd.DataFrame({
                "Feature": input_vec.columns,
                "Value":   input_vec.iloc[0].values
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

    # Bulk scanner
    st.divider()
    st.subheader("📂 Bulk Email Scanner")
    st.markdown("Upload a CSV with the same columns as the dataset (without `label`) to scan many emails at once.")

    bulk_file = st.file_uploader("Upload CSV for bulk prediction", type=["csv"], key="bulk")
    if bulk_file:
        bulk_df = pd.read_csv(bulk_file)
        required = [c for c in df.columns if c != "label"]
        missing  = [c for c in required if c not in bulk_df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            bulk_model = trained_models[pred_model]
            if pred_model == "Logistic Regression":
                bulk_preds = bulk_model.predict(scaler.transform(bulk_df[required]))
                bulk_probs = bulk_model.predict_proba(scaler.transform(bulk_df[required]))[:, 1]
            else:
                bulk_preds = bulk_model.predict(bulk_df[required])
                bulk_probs = bulk_model.predict_proba(bulk_df[required])[:, 1]

            bulk_df["Prediction"]        = bulk_preds
            bulk_df["Fraud_Probability"] = (bulk_probs * 100).round(1)
            bulk_df["Result"]            = bulk_df["Prediction"].map(
                {1: "🚨 FRAUD", 0: "✅ LEGIT"}
            )

            b1, b2, b3 = st.columns(3)
            b1.metric("Total Scanned",      len(bulk_df))
            b2.metric("Flagged as Fraud",   int(bulk_preds.sum()))
            b3.metric("Flagged as Legit",   int((bulk_preds == 0).sum()))

            st.dataframe(bulk_df[["Result","Fraud_Probability"] + required[:5]],
                         use_container_width=True, hide_index=True)

            csv_out = bulk_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "fraud_predictions.csv", "text/csv",
                               use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("📧 Email Fraud Detection System · Pradip Kunvariya · Built with Streamlit & Scikit-learn")
