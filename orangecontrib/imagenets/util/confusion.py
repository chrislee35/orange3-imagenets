"""Confusion counts for an image table that Classify Images has annotated with a ``Prediction`` column."""
import numpy as np


def true_and_predicted(data):
    """(true labels, predicted labels) as lists of str, or None if the table lacks a class or a Prediction column."""
    domain = data.domain
    if domain.class_var is None:
        return None
    pred_var = next((v for v in domain.metas if v.name == "Prediction"), None)
    if pred_var is None:
        return None
    col = list(domain.metas).index(pred_var)
    truth = [domain.class_var.values[int(v)] for v in data.Y]
    # Classify Images stores the predicted class name itself in the meta column; a table saved and reloaded may hold
    # the index of the name instead.
    pred = [v if isinstance(v, str) else pred_var.values[int(v)] for v in data.metas[:, col]]
    return truth, pred


def confusion_matrix(truth, pred):
    """(labels, counts) with counts[i, j] = images of true class labels[i] predicted as labels[j]."""
    labels = sorted(set(truth) | set(pred))
    index = {name: i for i, name in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(truth, pred):
        counts[index[t], index[p]] += 1
    return labels, counts
