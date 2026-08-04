"""Build an independent HOTC submission from a published Drive file index.

The file index is an organizer-data lookup aid exported by Tejaswi's public
Kaggle notebook. Predictions are produced locally by OpenCV CSRT and are
checkpointed per sequence so a long run can resume safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd
import requests


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


def valid_content_type(destination: Path, content_type: str) -> bool:
    """Reject Drive HTML/error pages while allowing organizer text metadata."""
    media_type = content_type.lower().split(";", 1)[0].strip()
    if destination.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return media_type.startswith("image/")
    return media_type in {"text/plain", "application/octet-stream"}


def download(file_id: str, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    endpoints = (
        (
            "https://drive.usercontent.google.com/download",
            {"id": file_id, "export": "download", "confirm": "t"},
        ),
        (
            "https://drive.google.com/uc",
            {"id": file_id, "export": "download", "confirm": "t"},
        ),
    )
    for attempt in range(2):
        for url, params in endpoints:
            try:
                # Both official endpoints occasionally throttle independently.
                # Keep every request bounded and reject HTML/error responses.
                with requests.get(url, params=params, stream=True, timeout=(8, 20)) as response:
                    response.raise_for_status()
                    if not valid_content_type(destination, response.headers.get("content-type", "")):
                        raise ValueError("Drive response has an unexpected content type")
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                output.write(chunk)
                if temporary.stat().st_size > 0:
                    temporary.replace(destination)
                    return True
            except Exception:
                temporary.unlink(missing_ok=True)
        if attempt == 0:
            time.sleep(1)
    return False


def make_tracker(kind: str):
    constructor = f"Tracker{kind.upper()}_create"
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, constructor):
        return getattr(cv2.legacy, constructor)()
    return getattr(cv2, constructor)()


def checkpoint_manifest(tracker: str, initial_boxes: Path | None) -> dict[str, str]:
    """Bind reusable checkpoints to every prediction-affecting input."""
    if initial_boxes is None:
        source = "organizer_init_rect"
    else:
        source = f"file_sha256:{hashlib.sha256(initial_boxes.read_bytes()).hexdigest()}"
    return {"schema": "hotc-checkpoint-v2", "tracker": tracker, "initialization": source}


def ensure_checkpoint_manifest(work: Path, expected: dict[str, str]) -> None:
    path = work / "checkpoint_manifest.json"
    checkpoint_dir = work / "checkpoints"
    existing_checkpoints = list(checkpoint_dir.glob("*.csv")) if checkpoint_dir.exists() else []
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Checkpoint configuration mismatch: {actual} != {expected}")
    elif existing_checkpoints:
        raise ValueError("Legacy checkpoints have no configuration manifest; use a new work directory")
    else:
        work.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--initial-boxes", type=Path)
    parser.add_argument("--work", type=Path, default=Path("work/csrt"))
    parser.add_argument("--output", type=Path, default=Path("submissions/csrt_v1.csv"))
    parser.add_argument("--tracker", choices=("kcf", "csrt"), default="kcf")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sequences-file", type=Path, help="Optional newline-delimited sequence subset"
    )
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sample["sequence"] = sample.ID.str.rsplit("_", n=1).str[0]
    sample["frame"] = sample.ID.str.rsplit("_", n=1).str[1].astype(int)
    sequences = sample.sequence.drop_duplicates().tolist()
    if args.sequences_file:
        requested = {
            line.strip() for line in args.sequences_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        unknown = sorted(requested.difference(sequences))
        if unknown:
            raise ValueError(f"Unknown requested sequences: {unknown}")
        sequences = [sequence for sequence in sequences if sequence in requested]
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

    ensure_checkpoint_manifest(args.work, checkpoint_manifest(args.tracker, args.initial_boxes))
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

        # A full-length CSV with carried-forward boxes is still not a complete
        # tracking result. Require every target image before checkpointing.
        if len(available) != len(target):
            checkpoint.unlink(missing_ok=True)
            for path in paths.values():
                path.unlink(missing_ok=True)
            print(
                f"[{index}/{len(sequences)}] {sequence}: {len(available)}/{len(target)} frames; retryable",
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
    for sequence in sequences:
        path = checkpoint_dir / f"{sequence}.csv"
        if path.exists():
            checkpoints.append(pd.read_csv(path))
    if len(checkpoints) == len(sequences):
        selected_sample = sample[sample.sequence.isin(sequences)]
        submission = selected_sample[["ID"]].merge(pd.concat(checkpoints), on="ID", how="left", sort=False)
        assert submission.ID.tolist() == selected_sample.ID.tolist()
        assert not submission.isna().any().any()
        assert (submission[["width", "height"]] > 0).all().all()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.output, index=False)
        print(f"Submission ready: {args.output} ({len(submission)} rows)")
    else:
        print(f"Partial run: {completed}/{len(sequences)} sequences checkpointed")


if __name__ == "__main__":
    main()
