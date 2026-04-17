from pathlib import Path

import joblib

from feature_engineering import combine_text_and_spelling_features, spelling_summary
from preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "spam_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "model" / "vectorizer.pkl"

_model = None
_vectorizer = None


def load_artifacts():
    """Load the trained model and vectorizer once per Python process."""
    global _model, _vectorizer

    if _model is None or _vectorizer is None:
        missing = [
            str(path.relative_to(PROJECT_ROOT))
            for path in (MODEL_PATH, VECTORIZER_PATH)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing model artifact(s): "
                + ", ".join(missing)
                + ". Run `python train_model.py` first."
            )

        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)

    return _model, _vectorizer


def predict_email_details(text):
    """Return label, confidence, spam probability, ham probability, and cleaned text."""
    model, vectorizer = load_artifacts()
    cleaned = clean_text(text)
    features = combine_text_and_spelling_features(vectorizer, [text], [cleaned])
    prediction = int(model.predict(features)[0])
    spelling = spelling_summary(text)

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    spam_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    ham_probability = float(probabilities[classes.index(0)]) if 0 in classes else 0.0

    label = "Spam" if prediction == 1 else "Not Spam"
    confidence = round(max(spam_probability, ham_probability) * 100, 2)

    return {
        "label": label,
        "confidence": confidence,
        "spam_probability": round(spam_probability * 100, 2),
        "ham_probability": round(ham_probability * 100, 2),
        "cleaned_text": cleaned,
        "misspelling_count": spelling["misspelling_count"],
        "misspelling_ratio": round(spelling["misspelling_ratio"] * 100, 2),
        "possible_misspellings": spelling["possible_misspellings"][:8],
        "scam_typo_count": spelling["scam_typo_count"],
    }


def predict_email(text):
    """Backward-compatible helper used by simple scripts."""
    details = predict_email_details(text)
    return details["label"], details["confidence"]
