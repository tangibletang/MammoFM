#!/usr/bin/env python3
"""GPU job: 8-bit load + LoRA merge only (no generate). Catches merge hook / bnb .to regressions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACK = Path(__file__).resolve().parent
_PATCH = _BACK / "patch_site"
if str(_BACK) not in sys.path:
    sys.path.insert(0, str(_BACK))

import config as cfg  # noqa: E402

for p in reversed((str(_PATCH), cfg.LLAVA_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import sitecustomize  # noqa: E402, F401
from llava.mm_utils import get_model_name_from_path  # noqa: E402
from llava.model.builder import load_pretrained_model_llama  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=cfg.CHECKPOINT_DIR)
    ap.add_argument("--model-base", default=cfg.MODEL_BASE)
    ap.add_argument("--load-4bit", action="store_true", help="Use 4-bit instead of 8-bit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    disable_torch_init()
    name = get_model_name_from_path(args.model_path)
    load_pretrained_model_llama(
        args.model_path,
        args.model_base,
        name,
        load_8bit=not args.load_4bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    print("STAGE1_LOAD_8BIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
