

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    """Create a session-scoped directory of tiny synthetic test images."""
    d = tmp_path_factory.mktemp("autoshield_fixtures")

    # A few distinct small PNGs — enough to exercise load + hash + decode paths.
    rng = np.random.default_rng(0)
    for name, color in [
        ("solid_red.png", (200, 20, 20)),
        ("solid_blue.png", (20, 20, 200)),
        ("solid_green.png", (20, 200, 20)),
    ]:
        arr = np.full((64, 64, 3), color, dtype=np.uint8)
        Image.fromarray(arr).save(d / name)

    # A noisy image that still has structure
    noise = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(noise).save(d / "noise.png")

    # A corrupt / not-an-image file to test error paths
    (d / "corrupt.png").write_bytes(b"not actually a PNG")

    return d
