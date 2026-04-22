

from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import bootstrap  # noqa: E402

bootstrap()

from src.image_io import ImageLoadError, load_bgr  # noqa: E402
from src.logging_config import configure_logging    # noqa: E402
from src.pose_inferencer import PoseInferencer      # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pose classifier on a single image.")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--weights", default=None, help="Override AUTOSHIELD_POSE_WEIGHTS path")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human summary")
    p.add_argument("--topk", type=int, default=3, help="How many top predictions to show")
    return p


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    try:
        img = load_bgr(args.image)
    except ImageLoadError as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 2

    inferencer = PoseInferencer(weights_path=args.weights)
    result = inferencer.predict(img)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print("-" * 50)
        print(f"Image        : {args.image}")
        print(f"Model        : {inferencer.version}")
        print(f"Top pose     : {result.pose.value}  (confidence {result.confidence:.3f})")
        print(f"Top {args.topk} probabilities:")
        top = sorted(result.all_probabilities.items(), key=lambda kv: -kv[1])[: args.topk]
        for cls, p in top:
            bar = "█" * int(p * 40)
            print(f"  {cls:3s}  {p:.3f}  {bar}")
        print("-" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
