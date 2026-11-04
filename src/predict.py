"""Load the saved models and classify new text without retraining (Try it).

Owner: Nawaf
"""

import joblib

from src.models import EMB_MODEL_NAME, EmbeddingSpamClassifier, load_encoder

LABELS = {0: "Not Spam", 1: "Spam"}
TFIDF_PATH = "models/tfidf_lr.joblib"
EMB_CLF_PATH = "models/embed_lr.joblib"


def load_models(tfidf_path=TFIDF_PATH, emb_clf_path=EMB_CLF_PATH):
    """Returns {"TF-IDF + LR": model, "MiniLM + LR": model}."""
    tfidf = joblib.load(tfidf_path)
    emb = EmbeddingSpamClassifier(load_encoder(EMB_MODEL_NAME), joblib.load(emb_clf_path))
    return {"TF-IDF + LR": tfidf, "MiniLM + LR": emb}


def classify(message, models):
    """Return a readable result for one message, or a helpful note on empty input."""
    if message is None or not str(message).strip():
        return "Please type a message to classify (the input was empty)."
    text = str(message).strip()
    lines = [f'Message: "{text}"']
    for name, model in models.items():
        p_spam = float(model.predict_proba([text])[0][1])
        label = LABELS[int(p_spam > 0.5)]  # same rule as model.predict
        lines.append(f"  {name:<12} -> {label:<8} (P(spam) = {p_spam:.2f})")
    return "\n".join(lines)
