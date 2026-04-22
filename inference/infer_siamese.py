

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import bootstrap  

bootstrap()

import cv2  

from src.config import get_model_config              
from src.image_io import ImageLoadError, load_bgr    
from src.logging_config import configure_logging      
from src.parts_segmenter import PartsSegmenter       
from src.siamese_inferencer import WindshieldIdentifier  


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Siamese windshield-identity similarity.")
    p.add_argument("--a", required=True, help="First image (full claim photo, or a windshield crop)")
    p.add_argument("--b", required=True, help="Second image")
    p.add_argument("--already-cropped", action="store_true",
                   help="Treat inputs as tight windshield crops; skip the parts segmenter")
    p.add_argument("--save-crops", default=None,
                   help="Directory to write the cropped windshields to (only with segmenter)")
    p.add_argument("--weights", default=None, help="Override AUTOSHIELD_SIAMESE_WEIGHTS path")
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    try:
        img_a = load_bgr(args.a)
        img_b = load_bgr(args.b)
    except ImageLoadError as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 2

    if args.already_cropped:
        crop_a, crop_b = img_a, img_b
        used_segmenter = False
    else:
        seg = PartsSegmenter()  # raises if weights missing
        crop_a = seg.crop_windshield(img_a)
        crop_b = seg.crop_windshield(img_b)
        used_segmenter = True
        if crop_a is None or crop_b is None:
            print(
                f"Windshield not detected (a={'yes' if crop_a is not None else 'no'}, "
                f"b={'yes' if crop_b is not None else 'no'}).",
                file=sys.stderr,
            )
            return 3
        if args.save_crops:
            out = Path(args.save_crops)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "ws_a.jpg"), crop_a)
            cv2.imwrite(str(out / "ws_b.jpg"), crop_b)

    ident = WindshieldIdentifier(weights_path=args.weights)
    sim = ident.similarity(crop_a, crop_b)

    cfg = get_model_config().siamese
    if sim >= cfg.fraud_threshold:
        verdict = "DUPLICATE"
    elif sim >= cfg.review_threshold:
        verdict = "SUSPICIOUS"
    else:
        verdict = "DIFFERENT"

    if args.json:
        print(json.dumps({
            "image_a": args.a, "image_b": args.b,
            "used_segmenter": used_segmenter,
            "model": ident.version,
            "similarity": sim,
            "fraud_threshold": cfg.fraud_threshold,
            "review_threshold": cfg.review_threshold,
            "verdict": verdict,
        }, indent=2))
    else:
        print("-" * 50)
        print(f"Image A    : {args.a}")
        print(f"Image B    : {args.b}")
        print(f"Segmenter  : {'used' if used_segmenter else 'skipped (--already-cropped)'}")
        print(f"Model      : {ident.version}")
        print(f"Similarity : {sim:.4f}")
        print(f"Thresholds : fraud>={cfg.fraud_threshold}  review>={cfg.review_threshold}")
        print(f"Verdict    : {verdict}")
        print("-" * 50)

    # Exit codes: 0 different, 1 suspicious, 2 duplicate, 3 failed
    return {"DIFFERENT": 0, "SUSPICIOUS": 1, "DUPLICATE": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
