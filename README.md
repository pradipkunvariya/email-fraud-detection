# 📧 Email Fraud Detection System

A Machine Learning web application that detects fraudulent emails using multiple classification models — deployed live on **Streamlit Cloud**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live App

👉 **[Open Live App on Streamlit Cloud](https://your-app-link.streamlit.app)**

---

## 📌 Project Overview

Email fraud (phishing, spam, scam) is one of the most common forms of cybercrime. This project builds a **multi-model ML classification system** that analyses email features to determine whether an email is **Fraudulent** or **Legitimate**.

| Feature | Detail |
|---|---|
| Dataset | 1,200 email records, 17 features |
| Fraud Rate | 25% (300 fraud / 900 legit) |
| Models | Logistic Regression, Decision Tree, Random Forest, Naive Bayes |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
email-fraud-detection/
│
├── app.py                      # Streamlit web application
├── notebook.py                 # Jupyter Notebook (as Python script)
├── email_fraud_dataset.csv     # Dataset (1,200 emails, 17 features)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📊 Dataset Description

The dataset contains **1,200 email records** with **17 features** and 1 binary label.

| Feature | Type | Description |
|---|---|---|
| `subject_length` | int | Number of characters in email subject |
| `body_length` | int | Number of characters in email body |
| `num_links` | int | Number of hyperlinks in the email |
| `num_exclamations` | int | Count of exclamation marks |
| `has_attachment` | binary | Whether email has an attachment (0/1) |
| `sender_domain_legit` | binary | Whether sender's domain looks legitimate (0/1) |
| `contains_urgent` | binary | Whether email contains urgent/act-now language (0/1) |
| `contains_money` | binary | Whether email mentions money/payment (0/1) |
| `contains_prize` | binary | Whether email mentions prize/winner/free (0/1) |
| `contains_verify` | binary | Whether email asks to verify account (0/1) |
| `contains_click` | binary | Whether email contains "click here" (0/1) |
| `misspellings` | int | Number of misspelled words |
| `reply_to_different_domain` | binary | Whether reply-to domain differs from sender (0/1) |
| `char_freq_dollar` | float | Frequency of $ character in email |
| `capital_run_avg` | float | Average length of consecutive capital letter runs |
| `capital_run_longest` | int | Longest run of consecutive capital letters |
| `html_ratio` | float | Ratio of HTML tags to plain text |
| `label` | binary | **Target** — 0 = Legitimate, 1 = Fraud |

---

## 🤖 Models Used

| Model | Description |
|---|---|
| **Logistic Regression** | Linear classifier with L2 regularisation, uses scaled features |
| **Decision Tree** | Tree-based classifier with max_depth=5 |
| **Random Forest** | Ensemble of 100 Decision Trees |
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem |

---

## 📈 App Features

### 📋 Tab 1 — Dataset
- View `df.head()`, `df.tail()`, `df.describe()`
- Check missing values and data types
- KPI summary cards

### 📊 Tab 2 — EDA
- Class distribution bar and pie charts
- Feature distributions (Fraud vs Legitimate)
- Boxplots for key features
- Full correlation heatmap

### 🤖 Tab 3 — Model Results
- Accuracy, Precision, Recall, F1, ROC-AUC for all models
- Confusion matrix per model
- Decision Tree diagram
- Random Forest feature importance chart

### 📈 Tab 4 — ROC & Comparison
- ROC curves for all 4 models on one plot
- Metric comparison bar charts

### 🔮 Tab 5 — Predict Email
- Enter 17 email features manually
- Choose any model for prediction
- Get Fraud/Legit result with probability score
- **Bulk Scanner** — upload CSV and scan all rows at once

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/email-fraud-detection.git
cd email-fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push all files to your GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select your repo and set **Main file** to `app.py`
4. Click **Deploy** — done!

> Make sure `email_fraud_dataset.csv` is in the repo root.

---

## 📓 Jupyter Notebook

The `notebook.py` file contains all notebook cells as a Python script. To convert it to a `.ipynb` file:

```bash
pip install jupytext
jupytext --to notebook notebook.py
```

Then open `notebook.ipynb` in Jupyter:
```bash
jupyter notebook notebook.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualisation |
| Scikit-learn | ML models and evaluation |
| Streamlit | Web application framework |
| GitHub | Version control and deployment |

---

## 📋 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 👤 Author

**Pradip Kunvariya**
- B.Tech Computer Engineering — Gandhinagar University (2022–2026)
- 📧 pradipkunvariya80@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/Pradip-Kunvariya)
- 💻 [GitHub](https://github.com/Pradip-Kunvariya)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
