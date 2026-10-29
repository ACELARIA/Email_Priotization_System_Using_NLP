"""
model/train.py
--------------
Training script for the Email Priority Classification System.

Usage (from project root):
    python model/train.py

What this script does
---------------------
1. Loads the synthetic email dataset via data.preprocess.load_dataset()
2. Preprocesses text using the clean_text pipeline
3. Splits data 80 / 20 (stratified)
4. Trains two classifiers:
   - Multinomial Naive Bayes
   - Logistic Regression
5. Evaluates both on the test split and selects the better one
6. Saves artefacts to the model/ directory:
   - email_model.pkl       — best trained classifier
   - tfidf_vectorizer.pkl  — fitted TF-IDF vectoriser
   - label_encoder.pkl     — label class list (numpy array)
"""

import sys
import os

# Ensure the project root is on the Python path so cross-directory imports work
sys.path.insert(0, r"d:\PROGRAMMING\ML_Projects\Email_Priotization_System_Using_NLP")

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from data.preprocess import clean_text, load_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = str(Path(__file__).resolve().parent.parent)
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "email_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")


def main() -> None:
    """End-to-end training pipeline."""

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Email Priority Classifier — Training Script")
    print("=" * 60)

    print("\n[1/6] Loading dataset...")
    df = load_dataset()
    print(f"      Dataset shape: {df.shape}")
    print(f"      Class distribution:\n{df['label'].value_counts().to_string()}")

    # ------------------------------------------------------------------
    # 2. Preprocess text
    # ------------------------------------------------------------------
    print("\n[2/6] Preprocessing text (this may take a moment)...")
    df["clean_text"] = df["text"].apply(clean_text)
    print("      Preprocessing complete.")

    # ------------------------------------------------------------------
    # 3. Encode labels
    # ------------------------------------------------------------------
    print("\n[3/6] Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"      Classes: {list(le.classes_)}")

    # ------------------------------------------------------------------
    # 4. Train / test split (stratified)
    # ------------------------------------------------------------------
    print("\n[4/6] Splitting data 80/20 (stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_text"],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"      Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")

    # ------------------------------------------------------------------
    # 5. TF-IDF vectorisation
    # ------------------------------------------------------------------
    print("\n[5/6] Fitting TF-IDF vectoriser...")
    vectorizer = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),     # unigrams and bigrams
        sublinear_tf=True,       # apply log normalisation to term frequencies
        min_df=2,                # ignore terms appearing in fewer than 2 docs
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")

    # ------------------------------------------------------------------
    # 6. Train and evaluate both models
    # ------------------------------------------------------------------
    print("\n[6/6] Training and evaluating classifiers...")

    # --- Multinomial Naive Bayes ---
    print("\n  Multinomial Naive Bayes")
    print("  " + "-" * 40)
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    nb_preds   = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_preds)
    print(f"  Accuracy: {nb_accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, nb_preds, target_names=le.classes_))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, nb_preds))

    # --- Logistic Regression ---
    print("\n  Logistic Regression")
    print("  " + "-" * 40)
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver="lbfgs",
    )
    lr_model.fit(X_train, y_train)
    lr_preds    = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    print(f"  Accuracy: {lr_accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, lr_preds, target_names=le.classes_))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, lr_preds))

    # ------------------------------------------------------------------
    # 7. Select the best model
    # ------------------------------------------------------------------
    if lr_accuracy >= nb_accuracy:
        best_model      = lr_model
        best_model_name = "Logistic Regression"
        best_accuracy   = lr_accuracy
    else:
        best_model      = nb_model
        best_model_name = "Multinomial Naive Bayes"
        best_accuracy   = nb_accuracy

    print("\n" + "=" * 60)
    print(f"  Selected model: {best_model_name}")
    print(f"  Test accuracy : {best_accuracy:.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 8. Persist artefacts
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model,  MODEL_PATH)
    joblib.dump(vectorizer,  VECTORIZER_PATH)
    joblib.dump(le.classes_, ENCODER_PATH)

    print(f"\nModel saved      : {MODEL_PATH}")
    print(f"Vectoriser saved : {VECTORIZER_PATH}")
    print(f"Label encoder saved: {ENCODER_PATH}")
    print("\nTraining complete. Run `python app/app.py` to start the web server.")


if __name__ == "__main__":
    main()
