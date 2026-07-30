from __future__ import annotations

import csv
from pathlib import Path

from .tracker import Box


def write_submission(path: str | Path, rows: list[tuple[str, Box]]) -> None:
    """Write the official HOTC 2026 columns: ID,x,y,width,height."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "x", "y", "width", "height"])
        for frame_id, box in rows:
            writer.writerow([frame_id, *box])


def validate_submission(path: str | Path) -> int:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ID", "x", "y", "width", "height"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Expected columns: {sorted(required)}")
    seen: set[str] = set()
    for row in rows:
        key = row["ID"]
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise ValueError(f"Invalid box size: {key}")
    return len(rows)


def validate_against_template(path: str | Path, template: str | Path) -> int:
    count = validate_submission(path)
    with Path(path).open(newline="") as predicted, Path(template).open(newline="") as expected:
        predicted_ids = [row["ID"] for row in csv.DictReader(predicted)]
        expected_ids = [row["ID"] for row in csv.DictReader(expected)]
    if predicted_ids != expected_ids:
        raise ValueError("Prediction IDs or ordering do not match the official template")
    return count
