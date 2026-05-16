#!/usr/bin/env python3
"""Fail fast if MammoFM backend is misconfigured (e.g. wrong embedding script path)."""
from __future__ import annotations

import os
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

    if os.environ.get("MAMMOFM_SKIP_LLAMA_CACHE_CHECK") != "1":
        hf = cfg.effective_hf_home()
        ref_b = os.environ.get("MAMMOFM_MODEL_BASE", cfg.MODEL_BASE)
        ref_i = os.environ.get("MAMMOFM_MODEL_ID", cfg.MODEL_ID)
        for label, ref in (("MODEL_BASE", ref_b), ("MODEL_ID", ref_i)):
            resolved = cfg.resolve_hf_cached_repo_with_fallback(ref, hf_home=hf)
            snap = Path(resolved)
            if not snap.is_dir() or not (snap / "config.json").is_file():
                print(
                    "ERROR: Llama must resolve to a local HF hub snapshot for offline Stage 1/2 "
                    f"({label}={ref!r}).",
                    file=sys.stderr,
                )
                print("  HF_HOME:", hf, file=sys.stderr)
                print("  Resolved:", resolved, file=sys.stderr)
                print(
                    "  Fix: seed hub cache (see README / local_hf_env.inc.sh) or set MAMMOFM_SKIP_LLAMA_CACHE_CHECK=1.",
                    file=sys.stderr,
                )
                return 1
        print("OK: Llama offline snapshots under HF_HOME (Stage 1/2)")

    print("OK:", enc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
