"""
Shared device detection utility for Speech Understanding PA-2.

Provides a single function that picks the best available torch device,
respecting the TORCH_DEVICE environment variable and supporting
CUDA, MPS (Apple Silicon), and CPU fallback.
"""

import os

import torch


def get_device(override: str = None) -> str:
    """Return the best available torch device string.

    Priority:
        1. Explicit *override* argument (e.g. from CLI --device)
        2. TORCH_DEVICE environment variable
        3. Auto-detect: cuda > mps > cpu

    Returns one of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    requested = override or os.environ.get("TORCH_DEVICE", "").strip()

    if requested:
        requested = requested.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cpu":
            return "cpu"
        if requested not in ("cuda", "mps", "cpu"):
            return requested

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
