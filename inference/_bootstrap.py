

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        "Could not locate project root (no ancestor directory contains 'src/')."
    )
