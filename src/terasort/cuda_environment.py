"""Runtime setup shared by TeraSort's optional CUDA backends."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def configure_cupy_cache() -> Path:
    """Choose a local kernel-cache directory before CuPy is imported.

    CuPy's default Windows cache may sit under a redirected or protected user
    profile. Kernel cache writes there can stall before any CUDA work starts.
    An explicit CUPY_CACHE_DIR always takes precedence.
    """
    configured = os.environ.get("CUPY_CACHE_DIR")
    cache_dir = (Path(configured).expanduser() if configured else
                 Path(tempfile.gettempdir()) / "terasort-cupy-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not configured:
        os.environ["CUPY_CACHE_DIR"] = str(cache_dir)
    return cache_dir
