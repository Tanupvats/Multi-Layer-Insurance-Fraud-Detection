"""
Tests for src.image_io — EXIF handling, corrupt inputs, hashing.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from src.image_io import (
    ImageLoadError,
    load_bgr,
    load_bgr_from_bytes,
    sha256_bytes,
    sha256_file,
)


def test_load_bgr_returns_hw3_bgr(fixtures_dir):
    img = load_bgr(fixtures_dir / "solid_red.png")
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    # Red in RGB becomes (20, 20, 200) in BGR (approx — PNG is exact here)
    r, g, b = img[32, 32]
    assert b == 20 and g == 20 and r == 200


def test_load_bgr_missing_file_raises():
    with pytest.raises(ImageLoadError):
        load_bgr("/no/such/file.png")


def test_load_bgr_corrupt_raises(fixtures_dir):
    with pytest.raises(ImageLoadError):
        load_bgr(fixtures_dir / "corrupt.png")


def test_load_bgr_from_bytes_roundtrip(fixtures_dir):
    data = (fixtures_dir / "solid_blue.png").read_bytes()
    img = load_bgr_from_bytes(data)
    assert img.shape == (64, 64, 3)
    b, g, r = img[0, 0]
    assert b == 200 and g == 20 and r == 20


def test_load_bgr_from_bytes_garbage_raises():
    with pytest.raises(ImageLoadError):
        load_bgr_from_bytes(b"not an image at all")


def test_sha256_file_matches_stdlib(fixtures_dir):
    data = (fixtures_dir / "solid_green.png").read_bytes()
    expected = hashlib.sha256(data).hexdigest()
    got = sha256_file(fixtures_dir / "solid_green.png")
    assert got == expected


def test_sha256_bytes_matches_stdlib():
    data = b"hello world"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_exif_orientation_is_respected(tmp_path):
    """
    Create a portrait-oriented image flagged as landscape via EXIF.
    After loading, pixel dimensions should reflect the rotated view.
    """
    # Create a 40x80 (portrait) image and tag it with EXIF orientation 6
    # (which means "rotate 90° CW for display"). PIL applies this via
    # ImageOps.exif_transpose — so load_bgr should return 80x40.
    im = Image.new("RGB", (40, 80), color=(100, 150, 200))
    # Build an EXIF block with orientation=6
    exif = im.getexif()
    exif[0x0112] = 6  # Orientation tag
    path = tmp_path / "rotated.jpg"
    im.save(path, "JPEG", exif=exif.tobytes())

    out = load_bgr(path)
    # After EXIF transpose, the 40-wide portrait becomes 80-wide landscape.
    # Exact dims depend on orientation direction, but it should differ
    # from the on-disk pixel layout (40, 80).
    assert out.shape[:2] != (80, 40)  # not the naked shape
    # The corrected dims should be either (40, 80) or (80, 40) depending
    # on JPEG save/load, but what we really want to assert is that
    # exif_transpose was honored (non-identity transform applied).
    # Easier check: dimensions are 80x40 or 40x80 ignoring order.
    assert set(out.shape[:2]) == {40, 80}
