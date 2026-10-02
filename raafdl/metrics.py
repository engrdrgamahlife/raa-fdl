"""Evaluation metrics with explicit label sets.

Review issue addressed: the 5GAD macro-F1 values in the original manuscript were below the
mathematical floor for a binary task. The usual cause is computing macro-F1 over a label set
that is larger than the dataset's real classes (e.g. passing the 15 CSE-CIC-IDS2018 labels
when evaluating 5GAD). Every function here takes the dataset's own label list explicitly.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def macro_f1(y_true, y_pred, labels) -> float:
    """Macro-F1 over exactly `labels` (the classes the dataset defines)."""
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def binary_macro_f1_floor(accuracy: float) -> float:
    """Lowest macro-F1 attainable in a 2-class problem at a given accuracy: a / (1 + a).

    Derivation: with errors e = 1 - a fixed, F1_c = 2TP_c / (2TP_c + e) is concave in TP_c,
    so F1_0 + F1_1 is minimised at TP_0 = 0, TP_1 = a, giving 2a / (1 + a).
    """
    return accuracy / (1.0 + accuracy)


def evaluate(y_true, y_pred, labels) -> dict:
    labels = list(labels)
    acc = float(accuracy_score(y_true, y_pred))
    mf1 = macro_f1(y_true, y_pred, labels)
    if len(labels) == 2:
        floor = binary_macro_f1_floor(acc)
        assert mf1 >= floor - 1e-9, (
            f"macro-F1 {mf1:.4f} is below the binary floor {floor:.4f} at accuracy {acc:.4f}: "
            "the label set passed to the metric is wrong")
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": mf1,
        "accuracy": acc,
        "per_class": {int(c): {"precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(si)}
                      for c, pi, ri, fi, si in zip(labels, p, r, f, s)},
    }


def majority_baseline(y_train, y_test, labels) -> dict:
    """Scores of a classifier that always predicts the training majority class."""
    vals, counts = np.unique(y_train, return_counts=True)
    maj = vals[np.argmax(counts)]
    return evaluate(y_test, np.full_like(y_test, maj), labels)
