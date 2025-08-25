from __future__ import annotations

import argparse
import json
from typing import List

from loguru import logger

from .analyzer import analyze
from .models import ROI, ROISet
from .logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROI-based change rate analyzer")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--rois", required=True, help="ROI JSON file")
    p.add_argument("--stride", type=int, default=1, help="Frame stride (>=1)")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    try:
        with open(args.rois, "r", encoding="utf-8") as f:
            data = json.load(f)
        rs = ROISet(**data)
        rois: List[ROI] = list(rs.rois)
        if not rois:
            raise ValueError("ROIが空です")
        analyze(args.input, rois, stride=max(1, args.stride), output_csv=args.output, show_progress=not args.no_progress)
    except Exception as e:
        logger.exception(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

