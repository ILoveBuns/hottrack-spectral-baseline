"""Build an auditable initial-box file from organizer, public, and V1 sources."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PUBLIC_FALLBACK = np.array([169.5, 149.0, 22.5, 22.0])


def organizer_center(path: Path) -> list[float]:
    values = [float(value) for value in re.split(r"[,\s]+", path.read_text().strip()) if value]
    if len(values) < 4:
        raise ValueError(f"Invalid organizer initial box: {path}")
    x, y, width, height = values[:4]
    return [x + width / 2, y + height / 2, width, height]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--organizer-dir", type=Path, required=True)
    parser.add_argument("--public-submission", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sample["sequence"] = sample.ID.str.rsplit("_", n=1).str[0]
    sample["frame"] = sample.ID.str.rsplit("_", n=1).str[1].astype(int)
    sequences = sample.sequence.drop_duplicates().tolist()
    public = pd.read_csv(args.public_submission)
    public["sequence"] = public.ID.str.rsplit("_", n=1).str[0]
    public["frame"] = public.ID.str.rsplit("_", n=1).str[1].astype(int)
    public = public[public.frame == 1].set_index("sequence")
    v1 = pd.read_csv(args.v1)
    v1["sequence"] = v1.ID.str.rsplit("_", n=1).str[0]
    v1["frame"] = v1.ID.str.rsplit("_", n=1).str[1].astype(int)
    v1 = v1[v1.frame == 1].set_index("sequence")

    rows = []
    provenance = {}
    columns = ["x", "y", "width", "height"]
    for sequence in sequences:
        organizer = args.organizer_dir / f"{sequence}.txt"
        if organizer.exists():
            values = organizer_center(organizer)
            source = "organizer_init_rect"
        else:
            public_values = public.loc[sequence, columns].to_numpy(float)
            if not np.array_equal(public_values, PUBLIC_FALLBACK):
                values = public_values.tolist()
                source = "public_notebook_organizer_init_mirror"
            else:
                values = v1.loc[sequence, columns].to_numpy(float).tolist()
                source = "v1_cross_modality_fallback"
        rows.append({"ID": f"{sequence}_1", **dict(zip(columns, values))})
        provenance[sequence] = source

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    counts = pd.Series(provenance).value_counts().to_dict()
    args.report.write_text(json.dumps({"counts": counts, "sequences": provenance}, indent=2) + "\n")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
