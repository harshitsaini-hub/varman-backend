"""GPU capability probing and caching for Varman Engine.

Probes CUDA device availability once at startup or on first call,
caching the result in memory so that request-time routes (such as
GET /capabilities or POST /images/protect) never import torch or
query the CUDA driver synchronously during request handling.
"""

from typing import Optional

_GPU_AVAILABLE: Optional[bool] = None


def probe_gpu_available(force_refresh: bool = False) -> bool:
    """Check and cache whether CUDA GPU acceleration is available.

    This function caches the result globally. Subsequent calls return the
    cached boolean immediately without touching torch or the driver.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None or force_refresh:
        try:
            import torch
            _GPU_AVAILABLE = bool(torch.cuda.is_available())
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def get_cached_gpu_available() -> bool:
    """Return the cached GPU availability status without importing torch."""
    if _GPU_AVAILABLE is None:
        return probe_gpu_available()
    return _GPU_AVAILABLE


def set_cached_gpu_available(value: Optional[bool]) -> None:
    """Explicitly set or reset the cached GPU availability (used in unit tests)."""
    global _GPU_AVAILABLE
    _GPU_AVAILABLE = value
