"""Method 1 (TF-IDF + LR) and Method 2 (sentence embeddings + LR).

Both methods: learn on TRAIN, choose settings on DEV by Macro F1,
never look at TEST while tuning.

Method 1 owner: Saad · Method 2 owner: Nawaf
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

SEED = 42
C_GRID = [0.1, 0.3, 1, 3, 10, 30, 100]
NGRAM_GRID = [(1, 1), (1, 2)]
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------- Method 1
def build_tfidf_model(C=1.0, ngram_range=(1, 1)):
    """TF-IDF word features -> class-weighted Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=ngram_range,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, class_weight="balanced",
                                   max_iter=2000, random_state=SEED)),
    ])


def tune_tfidf(train, dev, c_grid=C_GRID, ngram_grid=NGRAM_GRID):
    """Grid search on DEV Macro F1. Returns (best_model, table_of_all_settings).
    The returned model is trained on TRAIN only."""
    rows, best, best_f1 = [], None, -1.0
    for C, ngram in itertools.product(c_grid, ngram_grid):
        model = build_tfidf_model(C, ngram).fit(train["text"], train["label"])
        f1 = f1_score(dev["label"], model.predict(dev["text"]), average="macro")
        rows.append({"C": C, "ngram_range": str(ngram), "dev_macro_f1": f1})
        if f1 > best_f1:
            best, best_f1 = model, f1
    return best, _grid_table(rows)


def _grid_table(rows):
    """Sort settings by the unrounded dev score with a stable sort, so ties keep
    grid order and row 0 is the setting of the returned (first best) model."""
    table = pd.DataFrame(rows).sort_values("dev_macro_f1", ascending=False,
                                           kind="mergesort")
    table["dev_macro_f1"] = table["dev_macro_f1"].round(4)
    return table.reset_index(drop=True)


def tfidf_top_words(model, text, k=5):
    """Words that pushed this message towards spam the most
    (coefficient x TF-IDF value). Used in error analysis."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    x = vec.transform([text])
    contrib = x.multiply(clf.coef_[0]).tocsr()
    names = vec.get_feature_names_out()
    pairs = sorted(zip(contrib.indices, contrib.data), key=lambda p: -p[1])
    return [(names[i], round(float(v), 3)) for i, v in pairs[:k]]


# ---------------------------------------------------------------- Method 2
def load_encoder(name=EMB_MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed(encoder, texts, batch_size=64):
    """Frozen pretrained sentence embeddings (L2-normalised)."""
    return encoder.encode(list(texts), batch_size=batch_size,
                          show_progress_bar=False, normalize_embeddings=True)


def build_embedding_classifier(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                              random_state=SEED)


def tune_embedding(X_train, y_train, X_dev, y_dev, c_grid=C_GRID):
    """Grid search over C on DEV Macro F1, using pre-computed embeddings.
    Returns (best_classifier, table). Trained on TRAIN only."""
    rows, best, best_f1 = [], None, -1.0
    for C in c_grid:
        clf = build_embedding_classifier(C).fit(X_train, y_train)
        f1 = f1_score(y_dev, clf.predict(X_dev), average="macro")
        rows.append({"C": C, "dev_macro_f1": f1})
        if f1 > best_f1:
            best, best_f1 = clf, f1
    return best, _grid_table(rows)


class EmbeddingSpamClassifier:
    """Wraps encoder + classifier so it predicts directly from raw text."""

    def __init__(self, encoder, clf):
        self.encoder, self.clf = encoder, clf

    def predict(self, texts):
        return self.clf.predict(embed(self.encoder, texts))

    def predict_proba(self, texts):
        return self.clf.predict_proba(embed(self.encoder, texts))
