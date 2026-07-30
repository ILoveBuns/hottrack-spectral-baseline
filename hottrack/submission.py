from __future__ import annotations

import csv
from pathlib import Path

from .tracker import Box


def write_submission(path: str | Path, rows: list[tuple[str, int, Box]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "frame", "x", "y", "width", "height"])
        for video, frame, box in rows:
            writer.writerow([video, frame, *box])


def validate_submission(path: str | Path) -> int:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video", "frame", "x", "y", "width", "height"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Expected columns: {sorted(required)}")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = row["video"], int(row["frame"])
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise ValueError(f"Invalid box size: {key}")
    return len(rows)

