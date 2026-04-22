

from __future__ import annotations

import logging

import torch

from .config import get_settings

log = logging.getLogger(__name__)


def resolve_device() -> torch.device:
    pref = get_settings().DEVICE
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            # Silent CPU fallback on a supposed-GPU deployment is how
            # production incidents get diagnosed three days late.
            raise RuntimeError(
                "AUTOSHIELD_DEVICE=cuda but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    log.info("CUDA not available; using CPU.")
    return torch.device("cpu")
