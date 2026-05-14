#!/usr/bin/env python3
"""Confirm Stage 1 PYTHONPATH patches are active (no GPU; runs on login node).

Fails fast if sitecustomize did not monkeypatch LoRA merge / Llava loaders.
Catches regressions like shipping unpatchable sitecustomize or wrong PYTHONPATH order.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACK = Path(__file__).resolve().parent
_PATCH = _BACK / "patch_site"
if str(_BACK) not in sys.path:
    sys.path.insert(0, str(_BACK))

import config as cfg  # noqa: E402

if not Path(cfg.LLAVA_SRC).is_dir():
    print("ERROR: LLAVA_SRC missing:", cfg.LLAVA_SRC, file=sys.stderr)
    sys.exit(1)
if not (_PATCH / "sitecustomize.py").is_file():
    print("ERROR: patch_site/sitecustomize.py missing:", _PATCH, file=sys.stderr)
    sys.exit(1)

# Production order: patch_site first (must precede LLAVA_SRC for sitecustomize).
for p in reversed((str(_PATCH), cfg.LLAVA_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import sitecustomize  # noqa: E402 — applies hooks; must load after path setup

import transformers.modeling_utils as modeling_utils  # noqa: E402
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM  # noqa: E402
from peft.tuners.lora.model import LoraModel  # noqa: E402


def main() -> int:
    sc = Path(_PATCH / "sitecustomize.py").read_text(encoding="utf-8")
    if "_merged_model_is_quantized" not in sc:
        print("ERROR: sitecustomize missing _merged_model_is_quantized (stale tree?)", file=sys.stderr)
        return 1

    mean_fn = modeling_utils.PreTrainedModel._init_added_embeddings_weights_with_mean
    if mean_fn.__name__ != "_low_gpu_mean_init":
        print("ERROR: mean-resize patch not applied:", mean_fn.__name__, file=sys.stderr)
        return 1

    if LlavaLlamaForCausalLM.from_pretrained.__name__ != "_llava_from_pretrained_full_materialize":
        print("ERROR: LlavaLlama from_pretrained not patched:", LlavaLlamaForCausalLM.from_pretrained.__name__, file=sys.stderr)
        return 1

    if LoraModel.merge_and_unload.__name__ != "_lora_merge_and_unload_on_cpu":
        print("ERROR: LoraModel.merge_and_unload not patched:", LoraModel.merge_and_unload.__name__, file=sys.stderr)
        return 1

    print("OK: sitecustomize Stage 1 patches active (import check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
