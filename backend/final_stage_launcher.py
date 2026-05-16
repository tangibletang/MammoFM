"""Run LLaVA `final_stage.py` after disabling flash/mem-efficient SDPA on CUDA.

Older GPUs (e.g. V100) can hit: RuntimeError: cutlassF: no kernel found to launch!
from PyTorch's scaled_dot_product_attention backends. Stage 2 has no sitecustomize
patch, so we fix it here before any model forward.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass

runpy.run_path(cfg.FINAL_STAGE_SCRIPT, run_name="__main__")
