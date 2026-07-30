from __future__ import annotations

import numpy as np

from .tracker import Box


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def center_error(a: Box, b: Box) -> float:
    return float(np.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2,
                          a[1] + a[3] / 2 - b[1] - b[3] / 2))


def ope_metrics(predictions: list[Box], truth: list[Box]) -> dict[str, float]:
    if len(predictions) != len(truth) or not truth:
        raise ValueError("Predictions and truth must have equal non-zero length")
    overlaps = np.array([iou(a, b) for a, b in zip(predictions, truth)])
    errors = np.array([center_error(a, b) for a, b in zip(predictions, truth)])
    thresholds = np.linspace(0, 1, 101)
    success = np.array([(overlaps >= threshold).mean() for threshold in thresholds])
    return {
        "precision_at_20": float((errors <= 20).mean()),
        "success_auc": float(np.trapezoid(success, thresholds)),
        "mean_iou": float(overlaps.mean()),
    }

