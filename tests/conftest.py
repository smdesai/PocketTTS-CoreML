"""Global test configuration.

Forces single-threaded torch to match the reference's own discipline
(`pocket_tts/__init__.py` sets `torch.set_num_threads(1)` at import time).
Without this, golden comparisons drift due to reduction-order variation
across CPU threads.
"""

from __future__ import annotations

import os

import torch

# Run before any test imports the reference repo.
torch.set_num_threads(1)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
