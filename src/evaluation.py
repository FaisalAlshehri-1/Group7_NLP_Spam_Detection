"""Shared evaluation helpers so both methods are scored the same way.

Owner: Saad (evaluation of both methods)
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def main_scores(y_true, y_pred):
    """Accuracy and Macro F1 on the unchanged test set."""
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def false_positive_rate(y_pred_on_ham):
    """Share of legitimate (ham) messages predicted as spam.
    Input: predictions for a set that contains ham only."""
    y = list(y_pred_on_ham)
    return sum(1 for p in y if p == 1) / len(y) if y else float("nan")


def confusion(y_true, y_pred):
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                        index=["true ham", "true spam"],
                        columns=["pred ham", "pred spam"])


def comparison_table(results):
    """results: {method_name: {metric: value}} -> tidy DataFrame."""
    return pd.DataFrame(results).T.round(4)
