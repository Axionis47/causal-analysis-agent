"""Evaluation metrics for synthetic datasets."""

from __future__ import annotations

from typing import Iterable


def precision_recall(true_edges: Iterable[tuple[str, str]], pred_edges: Iterable[tuple[str, str]]):
    true_set = set(true_edges)
    pred_set = set(pred_edges)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def ate_bias(true_ate: float, estimated_ate: float) -> float:
    return estimated_ate - true_ate
