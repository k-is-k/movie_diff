from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .models import ROI
from .video import probe


def _roi_slice(gray: np.ndarray, roi: ROI) -> np.ndarray:
    x, y, w, h = roi.x, roi.y, roi.width, roi.height
    return gray[y : y + h, x : x + w]


def analyze(
    input_path: str,
    rois: List[ROI],
    stride: int = 1,
    output_csv: Optional[str] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if not rois:
        raise ValueError("No ROIs provided")

    meta = probe(input_path)
    if meta is None:
        logger.warning("Failed to probe metadata; proceeding with OpenCV defaults")
        fps = 0.0
    else:
        fps = meta.fps or 0.0

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    pbar = None
    # Use decoder-reported frame count for progress, which better matches
    # actual readable frames on this machine than ffprobe's nb_frames.
    cap_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = (cap_frame_count // stride) if (cap_frame_count > 0 and stride > 0) else None
    if show_progress:
        pbar = tqdm(total=total, desc="Analyzing", unit="f")

    prev_gray = None
    rows = []
    frame_idx = -1
    out_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        values = []
        if prev_gray is None:
            # first sampled frame
            for _ in rois:
                values.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            for roi in rois:
                sl = _roi_slice(diff, roi)
                if sl.size == 0:
                    values.append(0.0)
                else:
                    mean_abs = float(np.mean(sl))
                    values.append(mean_abs / 255.0)

        timestamp = (frame_idx / fps) if fps else None
        row = {
            "frame_index": frame_idx,
            "timestamp_sec": timestamp if timestamp is not None else np.nan,
        }
        for i, v in enumerate(values, start=1):
            name = rois[i - 1].name or f"roi_{i}"
            row[name] = v
        rows.append(row)
        prev_gray = gray
        out_idx += 1
        if pbar is not None:
            pbar.update(1)

    cap.release()
    if pbar is not None:
        pbar.close()

    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info(f"CSV written: {output_csv}")
    return df

