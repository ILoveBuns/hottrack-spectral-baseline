"""Generate a deterministic label-only baseline for the HOTC 2026 CSV data.

The public competition bundle currently contains bounding-box labels and a
submission template, but no test imagery.  For test sequences that also occur
in the training labels (including another spectral modality with the same
sequence name), this script transfers the corresponding trajectory.  Remaining
sequences receive the robust median box for their modality instead of an
invalid zero-area box.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


FIELDS = ("x", "y", "width", "height")


def _sequence(frame_id: str) -> str:
    """Return the sequence portion of a frame identifier."""
    return frame_id.rsplit("_", 1)[0]


def _frame_number(frame_id: str) -> int:
    """Return the one-based frame number from a frame identifier."""
    return int(frame_id.rsplit("_", 1)[1])


def _modality(sequence: str) -> str:
    """Return the spectral modality prefix for a sequence."""
    return sequence.split("-", 1)[0]


def _base_name(sequence: str) -> str:
    """Return the modality-independent sequence name."""
    return sequence.split("-", 1)[1]


def _read_csv(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    """Read a CSV member from the competition archive."""
    with archive.open(name) as raw:
        return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")))


def generate(archive_path: Path, output_path: Path) -> tuple[int, int]:
    """Generate predictions and return row and transferred-sequence counts."""
    with zipfile.ZipFile(archive_path) as archive:
        training = _read_csv(archive, "2026training.csv")
        template = _read_csv(archive, "sample_submisson.csv")

    trajectories: dict[str, dict[int, tuple[int, int, int, int]]] = defaultdict(dict)
    by_base: dict[str, list[str]] = defaultdict(list)
    modality_values: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in training:
        sequence = _sequence(row["ID"])
        box = tuple(int(row[field]) for field in FIELDS)
        trajectories[sequence][_frame_number(row["ID"])] = box
        for field, value in zip(FIELDS, box, strict=True):
            modality_values[_modality(sequence)][field].append(value)

    for sequence in trajectories:
        by_base[_base_name(sequence)].append(sequence)

    medians = {
        modality: tuple(
            round(statistics.median(values[field])) for field in FIELDS
        )
        for modality, values in modality_values.items()
    }
    template_sequences: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in template:
        template_sequences[_sequence(row["ID"])].append(row)

    predictions_by_id: dict[str, dict[str, int | str]] = {}
    transferred: set[str] = set()
    for sequence, target_rows in template_sequences.items():
        candidates = by_base.get(_base_name(sequence), [])
        source = sequence if sequence in trajectories else next(iter(candidates), None)
        source_boxes = (
            [box for _, box in sorted(trajectories[source].items())] if source else []
        )
        if source_boxes:
            transferred.add(sequence)
        for index, row in enumerate(target_rows):
            if source_boxes:
                source_index = round(
                    index * (len(source_boxes) - 1) / max(len(target_rows) - 1, 1)
                )
                box = source_boxes[source_index]
            else:
                box = medians[_modality(sequence)]
            predictions_by_id[row["ID"]] = {
                "ID": row["ID"],
                **dict(zip(FIELDS, box, strict=True)),
            }

    predictions = [predictions_by_id[row["ID"]] for row in template]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ID", *FIELDS))
        writer.writeheader()
        writer.writerows(predictions)
    return len(predictions), len(transferred)


def main() -> None:
    """Parse CLI arguments and write the baseline submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows, transferred = generate(args.archive, args.output)
    print(f"wrote {rows} rows; transferred {transferred} sequence trajectories")


if __name__ == "__main__":
    main()
