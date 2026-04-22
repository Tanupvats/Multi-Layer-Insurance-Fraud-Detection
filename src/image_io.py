

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

PathLike = Union[str, Path]


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded."""


def load_bgr(path: PathLike) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"File not found: {p}")
    if not p.is_file():
        raise ImageLoadError(f"Not a regular file: {p}")

    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr_rgb = np.array(im)
    except UnidentifiedImageError as e:
        raise ImageLoadError(f"Unsupported or corrupt image: {p}") from e
    except Exception as e:
        raise ImageLoadError(f"Failed to read {p}: {e}") from e

    if arr_rgb.size == 0:
        raise ImageLoadError(f"Empty image: {p}")

    return arr_rgb[:, :, ::-1].copy()  # RGB -> BGR for OpenCV consumers


def load_bgr_from_bytes(data: bytes) -> np.ndarray:
    """Decode an in-memory image payload (e.g. from a FastAPI upload)."""
    import io
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr_rgb = np.array(im)
    except UnidentifiedImageError as e:
        raise ImageLoadError("Unsupported or corrupt image payload") from e
    except Exception as e:
        raise ImageLoadError(f"Failed to decode image: {e}") from e

    if arr_rgb.size == 0:
        raise ImageLoadError("Empty image payload")
    return arr_rgb[:, :, ::-1].copy()


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
