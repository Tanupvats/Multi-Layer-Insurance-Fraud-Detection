

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import get_model_config, get_settings
from .device import resolve_device
from .logging_config import get_logger
from .nets import SiameseNetwork

log = get_logger(__name__)


class WindshieldIdentifier:
    def __init__(self, weights_path: str | Path | None = None):
        self.device = resolve_device()
        self.cfg = get_model_config().siamese
        self.settings = get_settings()

        self.model = SiameseNetwork(embed_dim=self.cfg.embed_dim, pretrained=True)

        weights = Path(weights_path) if weights_path else self.settings.resolve_weight(
            self.settings.SIAMESE_WEIGHTS
        )

        if weights.exists():
            state = torch.load(str(weights), map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)
            self._weights_path: Path | None = weights
            log.info("siamese_loaded path=%s", str(weights))
        else:
            log.warning(
                "siamese_weights_missing path=%s — using ImageNet backbone "
                "(similarity is NOT meaningful for identity)",
                str(weights),
            )
            self._weights_path = None

        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.cfg.input_size, self.cfg.input_size), antialias=True),
            transforms.Normalize(self.cfg.normalize_mean, self.cfg.normalize_std),
        ])

    @property
    def version(self) -> str:
        return f"siamese:{self._weights_path.name if self._weights_path else 'imagenet-pretrained-UNCALIBRATED'}"

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb).unsqueeze(0).to(self.device)

    def embed(self, crop_bgr: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.model(self._preprocess(crop_bgr))

    def similarity(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        emb_a = self.embed(crop_a)
        emb_b = self.embed(crop_b)
        return float(SiameseNetwork.similarity(emb_a, emb_b).item())
