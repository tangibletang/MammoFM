#!/usr/bin/env python3
"""Fail fast if MammoFM backend is misconfigured (e.g. wrong embedding script path)."""
from __future__ import annotations

import sys
from pathlib import Path

# Same resolution as production: config lives in this directory
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg  # noqa: E402


def main() -> int:
    enc = Path(cfg.EMBEDDING_SCRIPT)
    if enc.name != "save_img_embedding_mammofm.py":
        print(
            "ERROR: EMBEDDING_SCRIPT must be save_img_embedding_mammofm.py, got:",
            cfg.EMBEDDING_SCRIPT,
            file=sys.stderr,
        )
        return 1
    if not enc.is_file():
        print("ERROR: encoder script missing:", enc, file=sys.stderr)
        return 1
    cfg.verify_paths()
    print("OK:", enc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
