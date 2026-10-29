"""
app/app.py
----------
Flask web application for the Email Priority Classification System.

Routes
------
GET  /               — Serve the single-page UI
POST /predict        — Classify a single email {subject, body}
POST /batch_predict  — Classify a list of emails {emails: [{subject, body}]}
GET  /health         — Health check endpoint

Run (from project root):
    python app/app.py
Server starts at http://localhost:5000
"""

import sys
import os

# Ensure the project root is on the Python path for cross-directory imports
from pathlib import Path
import sys
import os

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from model.predictor import EmailPredictor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = str(Path(__file__).resolve().parent.parent)

MODEL_PATH = os.path.join(BASE_DIR, "model", "email_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
CORS(app)  # Allow cross-origin requests (useful during development)

# Attempt to load the predictor at startup.
# If model files are not yet present, predictor is set to None and every
# prediction route returns a 503 with a helpful message.
predictor: EmailPredictor | None = None

try:
    predictor = EmailPredictor(
        model_path=MODEL_PATH,
        vectorizer_path=VECTORIZER_PATH,
    )
    print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError as exc:
    print(f"[WARNING] {exc}")
    print("[WARNING] Run `python model/train.py` to train and save the model first.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _model_not_loaded_response():
    """Return a standardised 503 response when the model is unavailable."""
    return (
        jsonify(
            {
                "error":   "Model not loaded.",
                "message": (
                    "The classification model has not been trained yet. "
                    "Run `python model/train.py` from the project root directory, "
                    "then restart this server."
                ),
            }
        ),
        503,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the main single-page UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a single email.

    Request JSON
    ------------
    {
        "subject": "...",
        "body":    "..."
    }

    Response JSON
    -------------
    {
        "priority":          "High Priority" | "Medium Priority" | "Low Priority" | "Spam",
        "confidence":        0.0 – 1.0,
        "confidence_scores": {"High Priority": 0.0, ...}
    }
    """
    if predictor is None:
        return _model_not_loaded_response()

    try:
        data    = request.get_json(force=True)
        subject = data.get("subject", "")
        body    = data.get("body", "")

        if not subject and not body:
            return jsonify({"error": "Both subject and body are empty."}), 400

        result = predictor.predict(subject=subject, body=body)
        return jsonify(result)

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Classify a list of emails.

    Request JSON
    ------------
    {
        "emails": [
            {"subject": "...", "body": "..."},
            ...
        ]
    }

    Response JSON
    -------------
    {
        "results": [
            {
                "subject":          "...",
                "body":             "...",
                "priority":         "...",
                "confidence":       0.0,
                "confidence_scores": {...}
            },
            ...  (sorted: High → Medium → Low → Spam)
        ]
    }
    """
    if predictor is None:
        return _model_not_loaded_response()

    try:
        data   = request.get_json(force=True)
        emails = data.get("emails", [])

        if not isinstance(emails, list) or len(emails) == 0:
            return jsonify({"error": "Provide a non-empty 'emails' list."}), 400

        results = predictor.predict_batch(emails)
        return jsonify({"results": results})

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "Batch prediction failed.", "details": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    """
    Health check.

    Response JSON
    -------------
    {"status": "ok", "model_loaded": true | false}
    """
    return jsonify({"status": "ok", "model_loaded": predictor is not None})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
