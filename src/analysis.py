"""Experiment helpers: evaluate both models on the test set and the three
ham subsets, and collect wrong predictions for error analysis.
"""

import pandas as pd

from src.evaluation import false_positive_rate, main_scores


def evaluate_all(models, test, ham_with, ham_without, challenge):
    """One row per model: Accuracy, Macro F1, and FPR on (a), (b), (c)."""
    rows = {}
    for name, model in models.items():
        r = main_scores(test["label"], model.predict(test["text"]))
        r["FPR (a) test ham WITH spam words"] = false_positive_rate(model.predict(ham_with["text"]))
        r["FPR (b) test ham WITHOUT spam words"] = false_positive_rate(model.predict(ham_without["text"]))
        r["FPR (c) challenge set"] = (false_positive_rate(model.predict(challenge["text"]))
                                      if len(challenge) else float("nan"))
        rows[name] = r
    table = pd.DataFrame(rows).T
    table.loc["n (examples)"] = [len(test), len(test), len(ham_with), len(ham_without), len(challenge)]
    return table


def errors(model, data):
    """Wrong predictions with the model's spam probability."""
    pred = model.predict(data["text"])
    proba = model.predict_proba(data["text"])[:, 1]
    out = data.assign(pred=pred, p_spam=proba.round(3))
    out = out[out["pred"] != out["label"]]
    return out[["text", "label", "pred", "p_spam"]].reset_index(drop=True)


def predictions_side_by_side(models, data):
    """Both models' predictions on the same messages (e.g., challenge set)."""
    out = data[["text"] + (["label"] if "label" in data else [])].copy()
    for name, model in models.items():
        out[name] = model.predict(data["text"])
        out[name + " P(spam)"] = model.predict_proba(data["text"])[:, 1].round(3)
    return out
