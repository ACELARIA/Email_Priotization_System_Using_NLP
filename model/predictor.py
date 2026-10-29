"""
model/predictor.py
------------------
EmailPredictor — inference helper for the Email Priority Classification System.

Loads the persisted TF-IDF vectoriser and trained classifier from disk and
exposes a clean API for single and batch prediction.

Typical usage
-------------
    from model.predictor import EmailPredictor

    predictor = EmailPredictor(
        model_path="model/email_model.pkl",
        vectorizer_path="model/tfidf_vectorizer.pkl",
    )
    result = predictor.predict(subject="URGENT: server down", body="Please fix now.")
    print(result)
    # {'priority': 'High Priority', 'confidence': 0.97, 'confidence_scores': {...}}
"""

import re
import sys
import os

# Ensure project root is importable when this module is used standalone
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import nltk


# NLTK resource bootstrap
nltk.data.path.insert(0, os.path.expanduser("~/nltk_data"))

_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet.zip", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

for _path, _id in _NLTK_RESOURCES:
    try:
        nltk.data.find(_path)
    except LookupError:
        print(f"NLTK resource missing: {_id}")
        nltk.download(_id, download_dir=os.path.expanduser("~/nltk_data"))


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Priority sort order used in batch prediction (lower = higher urgency)
_PRIORITY_ORDER = {
    "High Priority":   0,
    "Medium Priority": 1,
    "Low Priority":    2,
    "Spam":            3,
}

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


class EmailPredictor:
    """
    Wrapper around the trained scikit-learn model and TF-IDF vectoriser.

    Parameters
    ----------
    model_path : str
        Path to the serialised classifier (joblib .pkl file).
    vectorizer_path : str
        Path to the serialised TF-IDF vectoriser (joblib .pkl file).
    """

    def __init__(self, model_path: str, vectorizer_path: str) -> None:
        # Load classifier
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                "Run `python model/train.py` from the project root to generate it."
            )
        # Load vectoriser
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectoriser file not found at: {vectorizer_path}\n"
                "Run `python model/train.py` from the project root to generate it."
            )

        self.model      = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        # Load label encoder (string class names) from the same directory as the model.
        # The encoder file is saved by train.py alongside the model pkl.
        encoder_path = os.path.join(os.path.dirname(model_path), "label_encoder.pkl")
        if os.path.exists(encoder_path):
            # label_encoder.pkl stores a numpy array of class name strings
            self._classes = list(joblib.load(encoder_path))
        elif hasattr(self.model, "classes_"):
            # Fallback: use whatever sklearn attached to the model
            self._classes = [str(c) for c in self.model.classes_]
        else:
            # Last-resort alphabetical order (matches LabelEncoder default)
            self._classes = ["High Priority", "Low Priority", "Medium Priority", "Spam"]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalise raw email text using the same pipeline as training.

        Steps: lowercase → strip HTML → remove non-alpha → tokenise →
        remove stopwords → lemmatise.

        Parameters
        ----------
        text : str
            Combined subject and body text.

        Returns
        -------
        str
            Space-joined token string.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Keep only letters and spaces
        text = re.sub(r"[^a-z\s]", " ", text)
        # Tokenise
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatise
        tokens = [
            _lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in _stop_words and len(tok) > 1
        ]
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict(self, subject: str, body: str) -> dict:
        """
        Classify a single email.

        Parameters
        ----------
        subject : str
            Email subject line.
        body : str
            Email body text.

        Returns
        -------
        dict
            {
                'priority':          str,         # predicted class label
                'confidence':        float,       # probability of top class
                'confidence_scores': dict[str, float],  # all class probabilities
            }
        """
        combined   = f"{subject} {body}"
        clean      = self.preprocess_text(combined)
        features   = self.vectorizer.transform([clean])

        # Use predict_proba when available (LR), else use decision_function fallback
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
        else:
            # For models without predict_proba, create a uniform distribution
            raw   = self.model.decision_function(features)[0]
            # Softmax normalisation
            exp   = [2.718281828 ** v for v in raw]
            total = sum(exp)
            proba = [e / total for e in exp]

        predicted_idx   = int(proba.argmax())
        predicted_label = self._classes[predicted_idx]
        confidence      = float(proba[predicted_idx])

        # Map numeric class indices back to readable label names
        confidence_scores = {
            self._classes[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return {
            "priority":          predicted_label,
            "confidence":        round(confidence, 4),
            "confidence_scores": confidence_scores,
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, emails_list: list) -> list:
        """
        Classify a list of emails and return results sorted by priority.

        Sort order: High Priority → Medium Priority → Low Priority → Spam.

        Parameters
        ----------
        emails_list : list[dict]
            Each dict must have keys ``subject`` and ``body``.

        Returns
        -------
        list[dict]
            Each result dict contains the original subject and body plus the
            fields returned by :meth:`predict`.
        """
        results = []
        for email in emails_list:
            subject = email.get("subject", "")
            body    = email.get("body", "")
            result  = self.predict(subject, body)
            results.append(
                {
                    "subject":          subject,
                    "body":             body,
                    "priority":         result["priority"],
                    "confidence":       result["confidence"],
                    "confidence_scores": result["confidence_scores"],
                }
            )

        # Sort by priority (High first)
        results.sort(key=lambda r: _PRIORITY_ORDER.get(r["priority"], 99))
        return results
