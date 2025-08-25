from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import cv2
from loguru import logger


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    n_frames: Optional[int]
    duration: Optional[float]


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def probe_with_ffprobe(path: str) -> Optional[VideoMeta]:
    if not has_ffprobe():
        return None
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if not streams:
            return None
        s0 = streams[0]
        width = int(s0.get("width"))
        height = int(s0.get("height"))
        r = s0.get("r_frame_rate") or s0.get("avg_frame_rate") or "0/1"
        num, den = r.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
        nb_frames = s0.get("nb_frames")
        n_frames = int(nb_frames) if nb_frames and nb_frames.isdigit() else None
        duration = float(s0.get("duration")) if s0.get("duration") else None
        return VideoMeta(width=width, height=height, fps=fps, n_frames=n_frames, duration=duration)
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}")
        return None


def probe_with_cv2(path: str) -> Optional[VideoMeta]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = frame_count if frame_count > 0 else None
    duration = (n_frames / fps) if (n_frames and fps) else None
    cap.release()
    return VideoMeta(width=width, height=height, fps=fps, n_frames=n_frames, duration=duration)


def probe(path: str) -> Optional[VideoMeta]:
    meta = probe_with_ffprobe(path)
    if meta is None:
        meta = probe_with_cv2(path)
    return meta


def read_first_frame(path: str):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read first frame")
    return frame

