

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .config import get_model_config, get_settings
from .device import resolve_device
from .logging_config import get_logger
from .nets import CarPoseModel
from .schema import Pose, PoseResult

log = get_logger(__name__)


class PoseInferencer:
    def __init__(self, weights_path: str | Path | None = None):
        self.device = resolve_device()
        self.cfg = get_model_config().pose
        self.settings = get_settings()

        weights = Path(weights_path) if weights_path else self.settings.resolve_weight(
            self.settings.POSE_WEIGHTS
        )

        if weights.exists():
            ckpt = torch.load(str(weights), map_location=self.device)
            if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                raise RuntimeError(
                    f"Pose checkpoint {weights} missing 'model_state_dict'. "
                    "Expected dict produced by training/train_pose.py."
                )
            self.classes = ckpt.get("classes") or list(self.cfg.classes)
            self._validate_classes(self.classes)
            self.model = CarPoseModel(num_classes=len(self.classes), pretrained=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self._weights_path: Path | None = weights
            log.info("pose_model_loaded path=%s classes=%s", str(weights), self.classes)
        else:
            log.warning(
                "pose_weights_missing path=%s — using ImageNet backbone (predictions WILL be wrong)",
                str(weights),
            )
            self.classes = list(self.cfg.classes)
            self.model = CarPoseModel(num_classes=len(self.classes), pretrained=True)
            self._weights_path = None

        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.cfg.input_size, self.cfg.input_size), antialias=True),
            transforms.Normalize(self.cfg.normalize_mean, self.cfg.normalize_std),
        ])

    @staticmethod
    def _validate_classes(classes: list[str]) -> None:
        valid = {p.value for p in Pose}
        for c in classes:
            if c not in valid:
                raise ValueError(f"Pose class {c!r} is not a known Pose: {sorted(valid)}")

    @property
    def version(self) -> str:
        if self._weights_path is None:
            return "pose:imagenet-pretrained-UNCALIBRATED"
        return f"pose:{self._weights_path.name}"

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb).unsqueeze(0).to(self.device)

    def predict(self, img_bgr: np.ndarray) -> PoseResult:
        x = self._preprocess(img_bgr)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return PoseResult(
            pose=Pose(self.classes[idx]),
            confidence=float(probs[idx]),
            all_probabilities={c: float(p) for c, p in zip(self.classes, probs)},
        )
