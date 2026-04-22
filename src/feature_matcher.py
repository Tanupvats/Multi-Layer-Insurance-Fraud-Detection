

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .config import get_model_config, get_settings
from .device import resolve_device
from .logging_config import get_logger

log = get_logger(__name__)


class FeatureMatcher:
    def __init__(self):
        from transformers import AutoImageProcessor, SuperGlueForKeypointMatching

        self.device = resolve_device()
        self.cfg = get_model_config().mirror
        self.settings = get_settings()

        name = self.cfg.superglue_model
        log.info("superglue_loading model=%s", name)
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = SuperGlueForKeypointMatching.from_pretrained(name).to(self.device).eval()

    @property
    def version(self) -> str:
        return f"superglue:{self.cfg.superglue_model.split('/')[-1]}"

    def match(
        self,
        img1_bgr: np.ndarray,
        img2_bgr: np.ndarray,
        output_path: Optional[str | Path] = None,
    ) -> Tuple[int, Optional[str]]:
        i1 = Image.fromarray(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB))
        i2 = Image.fromarray(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=[[i1, i2]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        sizes = [[(i1.height, i1.width), (i2.height, i2.width)]]
        raw = self.processor.post_process_keypoint_matching(outputs, sizes)[0]
        matches = raw["matches"]
        match_count = int((matches != -1).sum().item())

        viz_path: Optional[str] = None
        if output_path is not None and self.settings.SAVE_VISUALIZATIONS:
            viz_path = str(output_path)
            os.makedirs(os.path.dirname(viz_path) or ".", exist_ok=True)
            self._visualize(
                img1_bgr, img2_bgr, raw["keypoints0"], raw["keypoints1"], matches, viz_path
            )
        return match_count, viz_path

    @staticmethod
    def _visualize(img1, img2, kp0, kp1, matches, path: str) -> None:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1 : w1 + w2] = img2
        for i, m in enumerate(matches):
            m_i = int(m)
            if m_i != -1:
                p1 = (int(kp0[i][0]), int(kp0[i][1]))
                p2 = (int(kp1[m_i][0] + w1), int(kp1[m_i][1]))
                cv2.line(canvas, p1, p2, (0, 255, 0), 1)
        cv2.imwrite(path, canvas)
