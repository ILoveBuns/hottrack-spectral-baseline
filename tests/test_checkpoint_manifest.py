import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_hybrid_initialization import organizer_center
from scripts.run_csrt_from_listing import checkpoint_manifest, ensure_checkpoint_manifest, valid_content_type


class CheckpointManifestTest(unittest.TestCase):
    def test_organizer_initialization_is_explicit(self):
        self.assertEqual(
            checkpoint_manifest("kcf", None),
            {"schema": "hotc-checkpoint-v2", "tracker": "kcf", "initialization": "organizer_init_rect"},
        )

    def test_manifest_rejects_configuration_change(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            ensure_checkpoint_manifest(work, checkpoint_manifest("kcf", None))
            with self.assertRaises(ValueError):
                ensure_checkpoint_manifest(work, checkpoint_manifest("csrt", None))

    def test_legacy_checkpoint_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "checkpoints").mkdir()
            (work / "checkpoints/example.csv").write_text("ID,x,y,width,height\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ensure_checkpoint_manifest(work, checkpoint_manifest("kcf", None))

    def test_content_types_distinguish_frames_metadata_and_html(self):
        self.assertTrue(valid_content_type(Path("frame.jpg"), "image/jpeg"))
        self.assertFalse(valid_content_type(Path("frame.jpg"), "text/html"))
        self.assertTrue(valid_content_type(Path("init.txt"), "text/plain; charset=utf-8"))
        self.assertTrue(valid_content_type(Path("init.txt"), "application/octet-stream"))
        self.assertFalse(valid_content_type(Path("init.txt"), "text/html"))

    def test_organizer_box_converts_top_left_to_center(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.txt"
            path.write_text("10, 20, 6, 8\n", encoding="utf-8")
            self.assertEqual(organizer_center(path), [13.0, 24.0, 6.0, 8.0])


if __name__ == "__main__":
    unittest.main()
