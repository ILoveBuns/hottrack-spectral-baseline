"""Build an independent HOTC submission from a published Drive file index.

The file index is an organizer-data lookup aid exported by Tejaswi's public
Kaggle notebook. Predictions are produced locally by OpenCV CSRT and are
checkpointed per sequence so a long run can resume safely.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import gdown
import pandas as pd


SENSOR_PREFIX = {
    "HSI-NIR": "nir",
    "HSI-NIR-FalseColor": "nir",
    "HSI-RedNIR": "rednir",
    "HSI-RedNIR-FalseColor": "rednir",
    "HSI-VIS": "vis",
    "HSI-VIS-FalseColor": "vis",
    "HSI-VIS-FalseColor_25": "vis",
}


def frame_number(path: str) -> int:
    matches = re.findall(r"(\d+)", Path(path).stem)
    return int(matches[-1]) if matches else 0


def sequence_from_path(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    try:
        index = parts.index("validation")
        prefix = SENSOR_PREFIX.get(parts[index + 1])
        return f"{prefix}-{parts[index + 2]}" if prefix else None
    except (ValueError, IndexError):
        return None


def download(file_id: str, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(2):
        try:
            result = gdown.download(id=file_id, output=str(temporary), quiet=True, use_cookies=False)
            if result and temporary.stat().st_size > 0:
                temporary.replace(destination)
                return True
        except Exception:
            pass
        temporary.unlink(missing_ok=True)
        if attempt == 0:
            time.sleep(1)
    return False


def make_tracker(kind: str):
    constructor = f"Tracker{kind.upper()}_create"
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, constructor):
        return getattr(cv2.legacy, constructor)()
    return getattr(cv2, constructor)()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--initial-boxes", type=Path)
    parser.add_argument("--work", type=Path, default=Path("work/csrt"))
    parser.add_argument("--output", type=Path, default=Path("submissions/csrt_v1.csv"))
    parser.add_argument("--tracker", choices=("kcf", "csrt"), default="kcf")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sample["sequence"] = sample.ID.str.rsplit("_", n=1).str[0]
    sample["frame"] = sample.ID.str.rsplit("_", n=1).str[1].astype(int)
    sequences = sample.sequence.drop_duplicates().tolist()
    if args.limit:
        sequences = sequences[: args.limit]

    listing = pd.read_csv(args.listing)
    listing["sequence"] = listing.path.map(sequence_from_path)
    listing["frame"] = listing.path.map(frame_number)
    listing = listing[listing.sequence.isin(sequences)]
    initial_boxes = pd.read_csv(args.initial_boxes) if args.initial_boxes else None
    if initial_boxes is not None:
        initial_boxes["sequence"] = initial_boxes.ID.str.rsplit("_", n=1).str[0]
        initial_boxes["frame"] = initial_boxes.ID.str.rsplit("_", n=1).str[1].astype(int)
        initial_boxes = initial_boxes[initial_boxes.frame == 1].set_index("sequence")

    args.work.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.work / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    for index, sequence in enumerate(sequences, start=1):
        checkpoint = checkpoint_dir / f"{sequence}.csv"
        target = sample[sample.sequence == sequence].sort_values("frame")
        if checkpoint.exists() and len(pd.read_csv(checkpoint)) == len(target):
            completed += 1
            print(f"[{index}/{len(sequences)}] {sequence}: checkpoint", flush=True)
            continue

        rows = listing[listing.sequence == sequence]
        init_rows = rows[rows.path.str.endswith("init_rect.txt")]
        image_rows = rows[rows.path.str.lower().str.endswith((".jpg", ".jpeg"))]
        image_rows = image_rows[image_rows.frame.isin(target.frame)]
        if init_rows.empty or image_rows.empty:
            print(f"[{index}/{len(sequences)}] {sequence}: missing index entries", flush=True)
            continue

        sequence_dir = args.work / "frames" / sequence
        if initial_boxes is not None and sequence in initial_boxes.index:
            box = initial_boxes.loc[sequence]
            current = (
                float(box.x - box.width / 2),
                float(box.y - box.height / 2),
                float(box.width),
                float(box.height),
            )
        else:
            init_path = args.work / "initial" / f"{sequence}.txt"
            if init_rows.empty or not download(str(init_rows.iloc[0].id), init_path):
                print(f"[{index}/{len(sequences)}] {sequence}: initial box failed", flush=True)
                continue
            values = [float(value) for value in re.split(r"[,\s]+", init_path.read_text().strip()) if value]
            if len(values) < 4:
                continue
            current = tuple(values[:4])

        paths = {int(row.frame): sequence_dir / f"{int(row.frame):06d}.jpg" for row in image_rows.itertuples()}
        ids = {int(row.frame): str(row.id) for row in image_rows.itertuples()}
        probe_frame = min(paths)
        if not download(ids[probe_frame], paths[probe_frame]):
            checkpoint.unlink(missing_ok=True)
            for path in paths.values():
                path.unlink(missing_ok=True)
            print(
                f"[{index}/{len(sequences)}] {sequence}: probe frame unavailable; retryable",
                flush=True,
            )
            continue

        available = {probe_frame}
        with ThreadPoolExecutor(max_workers=12) as executor:
            jobs = {
                executor.submit(download, ids[number], path): number
                for number, path in paths.items()
                if number != probe_frame
            }
            available.update(jobs[job] for job in as_completed(jobs) if job.result())

        # A zero-frame sequence is not a tracking result.  Persisting only the
        # initial box would make later runs treat a failed download as a valid
        # checkpoint and could silently contaminate the final submission.
        if not available:
            checkpoint.unlink(missing_ok=True)
            for path in paths.values():
                path.unlink(missing_ok=True)
            print(
                f"[{index}/{len(sequences)}] {sequence}: 0/{len(target)} frames; retryable",
                flush=True,
            )
            continue

        tracker = None
        predictions = []
        for row in target.itertuples(index=False):
            image = cv2.imread(str(paths[row.frame])) if row.frame in available else None
            if image is not None and tracker is None:
                tracker = make_tracker(args.tracker)
                tracker.init(image, current)
            elif image is not None:
                ok, updated = tracker.update(image)
                if ok and updated[2] > 1 and updated[3] > 1:
                    current = tuple(float(value) for value in updated)
            x, y, width, height = current
            predictions.append(
                {"ID": row.ID, "x": x + width / 2, "y": y + height / 2, "width": width, "height": height}
            )
        pd.DataFrame(predictions).to_csv(checkpoint, index=False)
        completed += 1
        for path in paths.values():
            path.unlink(missing_ok=True)
        print(f"[{index}/{len(sequences)}] {sequence}: {len(available)}/{len(target)} frames", flush=True)

    checkpoints = []
    for sequence in sample.sequence.drop_duplicates():
        path = checkpoint_dir / f"{sequence}.csv"
        if path.exists():
            checkpoints.append(pd.read_csv(path))
    if len(checkpoints) == sample.sequence.nunique():
        submission = sample[["ID"]].merge(pd.concat(checkpoints), on="ID", how="left", sort=False)
        assert submission.ID.tolist() == sample.ID.tolist()
        assert not submission.isna().any().any()
        assert (submission[["width", "height"]] > 0).all().all()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.output, index=False)
        print(f"Submission ready: {args.output} ({len(submission)} rows)")
    else:
        print(f"Partial run: {completed}/{sample.sequence.nunique()} sequences checkpointed")


if __name__ == "__main__":
    main()
