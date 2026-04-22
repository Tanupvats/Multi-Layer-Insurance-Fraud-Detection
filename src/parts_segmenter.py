

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import get_model_config, get_settings
from .logging_config import get_logger

log = get_logger(__name__)


class PartsSegmenter:
    def __init__(self):
        from ultralytics import YOLO

        self.settings = get_settings()
        self.cfg = get_model_config().parts_seg

        weights = self.settings.resolve_weight(self.settings.PARTS_SEG_WEIGHTS)
        if not weights.exists():
            raise RuntimeError(
                f"Parts segmenter weights not found at {weights}. "
                "Train via training/train_segmentation.py or set "
                "AUTOSHIELD_PARTS_SEG_WEIGHTS to a valid path. "
                "Refusing to start to prevent the COCO-vs-custom class-confusion "
                "bug present in the original pipeline."
            )
        self.model = YOLO(str(weights))
        self._weights_path = weights
        log.info("parts_segmenter_loaded path=%s", str(weights))

    @property
    def version(self) -> str:
        return f"parts_seg:{self._weights_path.name}"

    def crop_windshield(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        target = self.cfg.windshield_class_id
        results = self.model(frame_bgr, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) == target:
                    box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    h, w = frame_bgr.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    return frame_bgr[y1:y2, x1:x2].copy()
        return None
