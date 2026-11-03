"""Spam-associated word list (built from TRAINING data only) and test subsets.

Owner: Faisal (Team Leader)
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z']{3,}")  # words of 3+ letters


def tokens(text):
    """Lowercase word tokens (letters only, 3+ characters), unique per message."""
    return set(_TOKEN_RE.findall(text.lower()))


def build_spam_word_list(train, top_k=20, min_spam_docs=10, min_ham_docs=10, alpha=1.0):
    """Words most frequent in spam relative to ham, by smoothed log-ratio
    of document frequencies. Uses only the training split.

    min_ham_docs keeps only words that also occur in legitimate messages:
    words that never appear in ham (e.g. "claim", "prize") cannot cause false
    alarms on ham, so they would leave test subset (a) almost empty.

    Returns a DataFrame: word, spam_docs, ham_docs, log_ratio.
    """
    spam_df = Counter()
    ham_df = Counter()
    for text, label in zip(train["text"], train["label"]):
        (spam_df if label == 1 else ham_df).update(tokens(text))
    n_spam = int((train["label"] == 1).sum())
    n_ham = int((train["label"] == 0).sum())

    rows = []
    for w, s in spam_df.items():
        if s < min_spam_docs:
            continue
        h = ham_df.get(w, 0)
        if h < min_ham_docs:
            continue
        ratio = math.log(((s + alpha) / (n_spam + alpha)) /
                         ((h + alpha) / (n_ham + alpha)))
        rows.append((w, s, h, ratio))

    import pandas as pd
    out = pd.DataFrame(rows, columns=["word", "spam_docs", "ham_docs", "log_ratio"])
    return out.sort_values("log_ratio", ascending=False).head(top_k).reset_index(drop=True)


def contains_any(text, words):
    return bool(tokens(text) & set(words))


def ham_subsets(test, words):
    """Split test HAM into (a) with spam-associated words and (b) without."""
    ham = test[test["label"] == 0]
    mask = ham["text"].map(lambda t: contains_any(t, words))
    return ham[mask].reset_index(drop=True), ham[~mask].reset_index(drop=True)
