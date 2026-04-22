

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import bootstrap  # noqa: E402

bootstrap()

import cv2  # noqa: E402

from src.config import get_model_config            # noqa: E402
from src.image_io import ImageLoadError, load_bgr  # noqa: E402
from src.logging_config import configure_logging    # noqa: E402
from src.parts_segmenter import PartsSegmenter     # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Custom parts segmenter on a single image.")
    p.add_argument("--image", required=True)
    p.add_argument("--save-crop", default=None, help="Write detected windshield crop here")
    p.add_argument("--save-overlay", default=None, help="Write an annotated visualization here")
    p.add_argument("--json", action="store_true")
    return p


def _run_detailed(model, frame_bgr):
    """Return every detected part (not just windshield) for diagnostics."""
    cfg = get_model_config().parts_seg
    class_names = cfg.class_names
    results = model(frame_bgr, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for i, cls in enumerate(result.boxes.cls):
            cid = int(cls)
            conf = float(result.boxes.conf[i]) if result.boxes.conf is not None else 0.0
            box = result.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            detections.append({
                "class_id": cid,
                "class_name": class_names.get(cid, f"class_{cid}"),
                "confidence": round(conf, 4),
                "bbox_xyxy": box,
            })
    return detections


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    try:
        img = load_bgr(args.image)
    except ImageLoadError as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 2

    seg = PartsSegmenter()  # will raise if weights are missing
    detections = _run_detailed(seg.model, img)

    ws_crop = seg.crop_windshield(img)
    if args.save_crop and ws_crop is not None:
        Path(args.save_crop).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save_crop, ws_crop)

    if args.save_overlay:
        overlay = img.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d['class_name']} {d['confidence']:.2f}"
            cv2.putText(overlay, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        Path(args.save_overlay).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save_overlay, overlay)

    payload = {
        "image": args.image,
        "model": seg.version,
        "detections": detections,
        "windshield_detected": ws_crop is not None,
        "saved_crop": args.save_crop if ws_crop is not None else None,
        "saved_overlay": args.save_overlay,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("-" * 50)
        print(f"Image      : {args.image}")
        print(f"Model      : {seg.version}")
        print(f"Detections : {len(detections)}")
        for d in detections:
            print(f"  • {d['class_name']:14s} conf={d['confidence']:.3f}  bbox={d['bbox_xyxy']}")
        print(f"Windshield : {'found' if ws_crop is not None else 'not found'}")
        if args.save_crop and ws_crop is not None:
            print(f"Crop saved : {args.save_crop}")
        if args.save_overlay:
            print(f"Overlay    : {args.save_overlay}")
        print("-" * 50)
    return 0 if ws_crop is not None else 1


if __name__ == "__main__":
    sys.exit(main())
