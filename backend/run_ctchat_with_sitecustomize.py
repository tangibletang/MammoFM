"""Stage 1 driver: import sitecustomize *before* ctchat pulls builder (needed for BNB skip + other hooks)."""
from __future__ import annotations

import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: run_ctchat_with_sitecustomize.py PATH/ctchat_validation_llama.py [args...]", file=sys.stderr)
        sys.exit(2)
    import sitecustomize  # noqa: F401

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
