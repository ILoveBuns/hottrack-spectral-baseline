"""Run a reproducible image-based HOTC 2026 baseline on Kaggle.

The organizers publish validation frames and first-frame boxes in a Google
Drive folder.  This notebook uses only false-colour frames, initializes from
the supplied box, and tracks sequentially with OpenCV CSRT.  The Drive folder
discovery pattern is credited to the public HOTC notebook by Tejaswi:
https://www.kaggle.com/code/tejaswi/hotc-2026-static-baseline-tracker
The tracking and submission implementation below is independent.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)

import cv2
import gdown
import numpy as np
import pandas as pd
import requests


DRIVE_FOLDER_ID = "1heJqpBgq42GpbH8y5TFF5IreE5lceqiW"
INPUT = Path("/kaggle/input/competitions/hyperspectral-object-tracking-challenge-2026")
WORK = Path("/kaggle/working/hotc_csrt")
WORK.mkdir(parents=True, exist_ok=True)
SENSOR_PREFIX = {
    "HSI-NIR-FalseColor": "nir",
    "HSI-RedNIR-FalseColor": "rednir",
    "HSI-VIS-FalseColor": "vis",
    "HSI-VIS-FalseColor_25": "vis",
}


def parse_validation_path(path: str):
    parts = path.replace("\\", "/").split("/")
    try:
        index = parts.index("validation")
        return parts[index + 1], parts[index + 2], "/".join(parts[index + 3 :])
    except (ValueError, IndexError):
        return None


def frame_number(path: str) -> int:
    matches = re.findall(r"(\d+)", Path(path).stem)
    return int(matches[-1]) if matches else 0


def download(file_id: str, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size > 100:
        return True
    try:
        response = requests.get(
            f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t",
            timeout=45,
            stream=True,
        )
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(1 << 16):
                if chunk:
                    handle.write(chunk)
        return destination.stat().st_size > 100
    except Exception:
        destination.unlink(missing_ok=True)
        return False


def tracker_factory():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
        return cv2.legacy.TrackerMIL_create()
    return cv2.TrackerMIL_create()


sample = pd.read_csv(INPUT / "sample_submisson.csv")
training = pd.read_csv(INPUT / "2026training.csv")
sample["sequence"] = sample.ID.str.rsplit("_", n=1).str[0]
sample["frame"] = sample.ID.str.rsplit("_", n=1).str[1].astype(int)
sequences = set(sample.sequence.unique())
fallback = tuple(
    float(training.loc[training.ID.str.endswith("_1"), column].median())
    for column in ("x", "y", "width", "height")
)

print("Indexing the organizers' validation folder…")
files = []
for attempt in range(4):
    try:
        files = gdown.download_folder(
            id=DRIVE_FOLDER_ID,
            output=str(WORK),
            skip_download=True,
            quiet=True,
            use_cookies=False,
        ) or []
        if files:
            break
    except Exception as error:
        print(f"Drive index attempt {attempt + 1}/4 failed: {type(error).__name__}")
    if attempt < 3:
        time.sleep(15 * (attempt + 1))
print(f"Indexed {len(files)} Drive entries")

initial_files = {}
image_files = defaultdict(list)
for item in files:
    if not item.id:
        continue
    parsed = parse_validation_path(item.path)
    if not parsed:
        continue
    sensor, short_name, rest = parsed
    prefix = SENSOR_PREFIX.get(sensor)
    if not prefix:
        continue
    sequence = f"{prefix}-{short_name}"
    if sequence not in sequences:
        continue
    suffix = Path(rest).suffix.lower()
    if rest.endswith("init_rect.txt"):
        initial_files.setdefault(sequence, item.id)
    elif suffix in {".jpg", ".jpeg"}:
        image_files[sequence].append((frame_number(rest), item.id))

predictions = {}
for sequence_index, sequence in enumerate(sorted(sequences), start=1):
    target_frames = sorted(sample.loc[sample.sequence == sequence, "frame"].tolist())
    init_file = WORK / f"{sequence}-init.txt"
    if sequence not in initial_files or not download(initial_files[sequence], init_file):
        print(f"[{sequence_index}/{len(sequences)}] {sequence}: no initial box")
        continue
    values = [float(value) for value in re.split(r"[,\s]+", init_file.read_text().strip()) if value]
    if len(values) < 4:
        continue
    current = tuple(values[:4])
    wanted = [(number, file_id) for number, file_id in image_files[sequence] if number in target_frames]
    local_files = {number: WORK / f"{sequence}-{number:06d}.jpg" for number, _ in wanted}
    with ThreadPoolExecutor(max_workers=16) as executor:
        jobs = {executor.submit(download, file_id, local_files[number]): number for number, file_id in wanted}
        available = {jobs[job] for job in as_completed(jobs) if job.result()}
    tracker = None
    last_image = None
    for number in target_frames:
        image = cv2.imread(str(local_files[number])) if number in available else None
        if image is not None and tracker is None:
            tracker = tracker_factory()
            tracker.init(image, current)
        elif image is not None and tracker is not None:
            ok, updated = tracker.update(image)
            if ok and updated[2] > 1 and updated[3] > 1:
                current = tuple(float(value) for value in updated)
        predictions[f"{sequence}_{number}"] = current
        last_image = image if image is not None else last_image
    for path in local_files.values():
        path.unlink(missing_ok=True)
    print(f"[{sequence_index}/{len(sequences)}] {sequence}: {len(available)}/{len(target_frames)} frames")

rows = []
for row in sample.itertuples(index=False):
    box = predictions.get(row.ID, fallback)
    rows.append({"ID": row.ID, "x": box[0], "y": box[1], "width": box[2], "height": box[3]})

submission = pd.DataFrame(rows)
submission.to_csv("/kaggle/working/submission.csv", index=False)
assert submission.ID.tolist() == sample.ID.tolist()
assert (submission[["width", "height"]].to_numpy() > 0).all()
print(f"Wrote {len(submission)} rows; tracked {len(predictions)} frames")
