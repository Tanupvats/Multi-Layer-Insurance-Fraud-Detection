

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import get_model_config, get_settings
from .logging_config import get_logger

log = get_logger(__name__)


class CarSegmenter:
    def __init__(self):
        from ultralytics import YOLO

        self.settings = get_settings()
        self.cfg = get_model_config().car_seg

        weights = self.settings.resolve_weight(self.settings.CAR_SEG_WEIGHTS)
        if weights.exists():
            self.model = YOLO(str(weights))
            self._weights_path = weights
        else:
            if not self.settings.ALLOW_MODEL_DOWNLOAD:
                raise RuntimeError(
                    f"CarSegmenter weights {weights} not found and downloads are disabled."
                )
            log.info("car_segmenter_downloading name=%s", self.settings.CAR_SEG_WEIGHTS)
            self.model = YOLO(self.settings.CAR_SEG_WEIGHTS)
            self._weights_path = None

    @property
    def version(self) -> str:
        return f"car_seg:{self._weights_path.name if self._weights_path else self.settings.CAR_SEG_WEIGHTS}"

    def segment(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        results = self.model(frame_bgr, classes=[self.cfg.coco_car_class_id], verbose=False)
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            log.info("car_not_detected")
            return None, None

        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]))
        mask_binary = (mask > self.cfg.mask_threshold).astype(np.uint8)

        car = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_binary)
        bg = cv2.bitwise_and(frame_bgr, frame_bgr, mask=(1 - mask_binary))
        return car, bg
