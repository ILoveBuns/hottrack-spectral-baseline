import tempfile
import unittest
from pathlib import Path

import numpy as np

from hottrack.metrics import ope_metrics
from hottrack.submission import validate_against_template, validate_submission, write_submission
from hottrack.tracker import SpectralSignatureTracker


def synthetic_sequence():
    rng = np.random.default_rng(7)
    frames, truth = [], []
    target_spectrum = np.linspace(0.15, 0.95, 16)
    for t in range(8):
        frame = rng.normal(0.25, 0.015, (72, 96, 16)).astype(np.float32)
        box = (12 + 3 * t, 20 + 2 * t, 12, 10)
        x, y, w, h = box
        frame[y:y + h, x:x + w] = target_spectrum + rng.normal(0, 0.005, (h, w, 16))
        frames.append(frame)
        truth.append(box)
    return frames, truth


class BaselineTest(unittest.TestCase):
    def test_tracks_spectral_target(self):
        frames, truth = synthetic_sequence()
        tracker = SpectralSignatureTracker(search_radius=8, stride=1)
        tracker.initialize(frames[0], truth[0])
        predictions = [truth[0]] + [tracker.update(frame) for frame in frames[1:]]
        scores = ope_metrics(predictions, truth)
        self.assertGreater(scores["success_auc"], 0.8)
        self.assertEqual(1.0, scores["precision_at_20"])

    def test_submission_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            write_submission(path, [("demo_1", (1, 2, 3, 4))])
            self.assertEqual(1, validate_submission(path))
            self.assertEqual(1, validate_against_template(path, path))


if __name__ == "__main__":
    unittest.main()
