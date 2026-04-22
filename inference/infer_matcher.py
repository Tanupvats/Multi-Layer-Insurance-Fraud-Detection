

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap  

bootstrap()

import cv2  

from src.car_segmenter import CarSegmenter          
from src.feature_matcher import FeatureMatcher     
from src.image_io import ImageLoadError, load_bgr  
from src.logging_config import configure_logging    


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SuperGlue keypoint matcher.")
    p.add_argument("--a", required=True, help="Path to first image")
    p.add_argument("--b", required=True, help="Path to second image")
    p.add_argument("--flip-a", action="store_true", help="Horizontally flip image A before matching")
    p.add_argument("--decouple", action="store_true",
                   help="Segment out the car and match background-only (mirror-fraud setup)")
    p.add_argument("--viz", default=None, help="Write match visualization here")
    return p


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    try:
        a = load_bgr(args.a)
        b = load_bgr(args.b)
    except ImageLoadError as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 2

    if args.flip_a:
        a = cv2.flip(a, 1)

    used_decouple = False
    if args.decouple:
        seg = CarSegmenter()
        _, bg_a = seg.segment(a)
        _, bg_b = seg.segment(b)
        if bg_a is None or bg_b is None:
            print("Car not detected in one or both images — cannot decouple. Exiting.",
                  file=sys.stderr)
            return 3
        a, b = bg_a, bg_b
        used_decouple = True

    matcher = FeatureMatcher()
    if args.viz:
        Path(args.viz).parent.mkdir(parents=True, exist_ok=True)
    count, viz_path = matcher.match(a, b, output_path=args.viz)

    print("-" * 50)
    print(f"Image A    : {args.a}{'  (flipped)' if args.flip_a else ''}")
    print(f"Image B    : {args.b}")
    print(f"Mode       : {'background-only (decoupled)' if used_decouple else 'full image'}")
    print(f"Model      : {matcher.version}")
    print(f"Matches    : {count}")
    if viz_path:
        print(f"Viz        : {viz_path}")
    print("-" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
