from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Box = tuple[int, int, int, int]


def _clip(box: Box, shape: tuple[int, ...]) -> Box:
    x, y, w, h = box
    height, width = shape[:2]
    w, h = min(w, width), min(h, height)
    return max(0, min(x, width - w)), max(0, min(y, height - h)), w, h


def _signature(frame: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = _clip(box, frame.shape)
    patch = frame[y:y + h, x:x + w].astype(np.float32)
    vector = np.median(patch.reshape(-1, patch.shape[-1]), axis=0)
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-8)


def _spectral_cost(frame: np.ndarray, box: Box, template: np.ndarray) -> float:
    x, y, w, h = _clip(box, frame.shape)
    pixels = frame[y:y + h, x:x + w].astype(np.float32).reshape(-1, frame.shape[-1])
    pixels /= np.maximum(np.linalg.norm(pixels, axis=1, keepdims=True), 1e-8)
    # Mean pixel similarity penalizes candidates that only partially overlap target.
    return 1.0 - float(np.mean(pixels @ template))


@dataclass
class SpectralSignatureTracker:
    search_radius: int = 18
    stride: int = 2
    update_rate: float = 0.04

    def initialize(self, frame: np.ndarray, box: Box) -> None:
        if frame.ndim != 3:
            raise ValueError("Expected H×W×bands hyperspectral cube")
        self.box = _clip(box, frame.shape)
        self.template = _signature(frame, self.box)

    def update(self, frame: np.ndarray) -> Box:
        x, y, w, h = self.box
        best_box, best_cost = self.box, float("inf")
        for dy in range(-self.search_radius, self.search_radius + 1, self.stride):
            for dx in range(-self.search_radius, self.search_radius + 1, self.stride):
                candidate = _clip((x + dx, y + dy, w, h), frame.shape)
                cost = _spectral_cost(frame, candidate, self.template)
                # A tiny motion prior breaks spectral ties in uniform regions.
                cost += 0.00005 * (dx * dx + dy * dy)
                if cost < best_cost:
                    best_box, best_cost = candidate, cost
        self.box = best_box
        observed = _signature(frame, best_box)
        updated = (1 - self.update_rate) * self.template + self.update_rate * observed
        self.template = updated / max(np.linalg.norm(updated), 1e-8)
        return best_box
