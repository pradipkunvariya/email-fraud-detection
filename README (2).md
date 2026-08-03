# 📧 Email Fraud Detection

A machine learning project that detects fraudulent (phishing / scam) emails from legitimate ones using **TF-IDF text features** and a **scikit-learn classifier**, wrapped in an interactive **Streamlit** web app.

> Paste an email into the app and instantly get a prediction: **Fraud** or **Legit**, along with a confidence score. You can also upload a CSV of emails for batch scoring.

---

## ✨ Features

- 🧹 Text cleaning & preprocessing pipeline (URLs, emails, numbers, punctuation removal)
- 🔢 TF-IDF vectorization (unigrams + bigrams)
- 🤖 Trains and compares three models — Logistic Regression, Multinomial Naive Bayes, Random Forest — and automatically keeps the best one
- 📊 Jupyter notebook with full EDA, training, and evaluation walkthrough
- 🌐 Streamlit app for single-email and batch (CSV) predictions
- 💾 Downloadable batch prediction results

---

## 📈 Model Performance

Trained on **10,181** cleaned/deduplicated emails (80/20 train-test split, stratified). Results on the held-out test set:

| Model               | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.978    | 0.998     | 0.951  | 0.974    |
| **Multinomial NB (best)** | **0.983**    | **0.985**     | **0.976**  | **0.980**    |
| Random Forest        | 0.982    | 0.991     | 0.966  | 0.978    |

The best-performing model (**Multinomial Naive Bayes**) is the one saved and used by the Streamlit app by default. Re-running `train_model.py` will re-select whichever model scores highest on F1.

---

## 🗂 Project Structure

```
email-fraud-detector/
├── app.py                     # Streamlit web app
├── train_model.py             # Script to train & save the model
├── requirements.txt           # Python dependencies
├── README.md
├── .gitignore
├── data/
│   └── fraud_email_.csv       # Dataset (Text, Class columns)
├── model/
│   ├── fraud_model.pkl        # Trained classifier (generated)
│   ├── vectorizer.pkl         # Fitted TF-IDF vectorizer (generated)
│   └── metrics.csv            # Model comparison metrics (generated)
└── notebooks/
    └── email_fraud_detection.ipynb   # EDA + training walkthrough
```

---

## 📊 Dataset

The dataset (`data/fraud_email_.csv`) contains two columns:

| Column  | Description                                  |
|---------|-----------------------------------------------|
| `Text`  | Raw email content (subject + body combined)   |
| `Class` | `1` = fraudulent / scam email, `0` = legitimate |

It contains **11,929** emails, roughly balanced between the two classes.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/email-fraud-detector.git
cd email-fraud-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

This reads `data/fraud_email_.csv`, trains and compares models, and saves the best one to `model/`.

```bash
python train_model.py
```

You should see output like:

```
Best model: MultinomialNB (F1 = 0.9803)
Saved: model/fraud_model.pkl, model/vectorizer.pkl, model/metrics.csv
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in your browser.

### 6. (Optional) Explore the notebook

```bash
jupyter notebook notebooks/email_fraud_detection.ipynb
```

This walks through EDA, cleaning, vectorization, model comparison, and evaluation — useful if you want to tweak features or try new models.

---

## 🖥 Using the App

**Single Email Check tab**
- Paste raw email text into the text box (or click "Load a scam example" / "Load a legit example")
- Click **Analyze Email**
- See the predicted label and estimated fraud probability

**Batch Check tab**
- Upload a CSV file with a `Text` column (one email per row)
- The app scores every row and lets you download the results as a CSV

---

## 🛠 How It Works

1. **Cleaning** — lowercase text, strip URLs, email addresses, numbers, and punctuation
2. **Vectorization** — TF-IDF with unigrams + bigrams, top 15,000 features, English stop words removed
3. **Classification** — Logistic Regression / Multinomial Naive Bayes / Random Forest are trained; the model with the best F1-score on the test set is kept
4. **Inference** — the same cleaning + vectorizer pipeline is applied to new text before prediction, ensuring train/serve consistency

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (see below).
2. Make sure `model/fraud_model.pkl` and `model/vectorizer.pkl` are committed (they're small, ~1 MB total) — the app loads them directly, so you don't need to train on the server.
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
4. Select your repo, branch (`main`), and set the main file path to `app.py`.
5. Click **Deploy**.

---

## 📤 Pushing This Project to GitHub

```bash
git init
git add .
git commit -m "Initial commit: email fraud detection project"
git branch -M main
git remote add origin https://github.com/<your-username>/email-fraud-detector.git
git push -u origin main
```

---

## 🔮 Possible Improvements

- Add more advanced NLP features (word embeddings, sentence transformers)
- Try deep learning models (LSTM, BERT-based classifiers)
- Add explainability (e.g. highlight words that most influenced the prediction, using LIME/SHAP)
- Add authentication and logging for production use
- Expand the dataset with more recent phishing examples

---

## 📄 License

This project is released under the MIT License — feel free to use, modify, and share it.

---

## 🙏 Acknowledgements

Built as a learning project for email fraud / phishing detection using classic NLP + machine learning techniques.
