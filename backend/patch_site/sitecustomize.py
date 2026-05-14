"""Loaded first on PYTHONPATH (see pipeline.py Stage 1).

HF resize_token_embeddings uses _init_added_embeddings_weights_with_mean, which does
`old_embeddings.weight.data.to(float32)` on GPU — ~2GiB peak on 16GB cards → OOM.

Setting mean_resizing=False avoids OOM but breaks meta / Accelerate paths (Peft load fails).

Fix:
1. `_init_added_embeddings_weights_with_mean` — compute mean stats on CPU; skip HF meta+eig path for
   lm_head by only normal-initing new rows when `weight` is still meta.
2. `nn.Module.load_state_dict` — default `assign=True` for PyTorch 2.x meta materialization.
3. `LlavaLlamaForCausalLM.from_pretrained` — for **full fp16** loads: force `low_cpu_mem_usage=False`
   and drop `device_map` / `max_memory` so the base model is fully materialized before Peft.
   For **bitsandbytes** loads (`load_in_8bit` / `load_in_4bit` / `quantization_config`): pass kwargs
   through unchanged so `device_map` is preserved.
5. Optional: `LlavaLlamaForCausalLM.generate` — if `MAMMOFM_NO_KV_CACHE=1`, force `use_cache=False`
   to lower VRAM during decoding on 16GB cards. Image tensors are cast to the model's device/dtype.
6. Optional: `MAMMOFM_CPU_GENERATE=1` — after **fp16** LoRA merge, keep the merged model on **CPU float32**
   (reliable smoke on V100 where GPU decode can OOM). Slow (~minutes/sample). Incompatible with 8/4-bit loads.
7. Quantized (8/4-bit) bases: LoRA **merge is skipped** (bitsandbytes merged weights break inference); adapters stay
   attached and Stage 1 runs on **CUDA** (`MAMMOFM_CPU_GENERATE` must be off).
"""
from __future__ import annotations

try:
    import os
    from contextlib import suppress

    import torch
    import torch.nn as nn
    import transformers.modeling_utils as _mu

    _orig_load_state_dict = nn.Module.load_state_dict

    def _load_state_dict_assign_meta(self, state_dict, strict=True, assign=True):
        return _orig_load_state_dict(self, state_dict, strict=strict, assign=assign)

    nn.Module.load_state_dict = _load_state_dict_assign_meta  # type: ignore[method-assign]

    from transformers.models.llama import modeling_llama as _llama_mod

    _orig_llama_init_weights = _llama_mod.LlamaPreTrainedModel._init_weights

    def _llama_init_weights_skip_quant_stored(module_self, module):
        """bitsandbytes 8-bit Linear weights are int8; HF init must not call normal_ on them."""
        w = getattr(module, "weight", None)
        if w is not None and w.dtype in (torch.int8, torch.uint8):
            return
        if w is not None and not w.is_floating_point() and not w.is_complex():
            return
        return _orig_llama_init_weights(module_self, module)

    _llama_mod.LlamaPreTrainedModel._init_weights = _llama_init_weights_skip_quant_stored  # type: ignore[method-assign]

    def _low_gpu_mean_init(
        self,
        old_embeddings,
        new_embeddings,
        old_embedding_dim,
        old_num_tokens,
        added_num_tokens,
    ):
        w = old_embeddings.weight.data
        if w.is_meta:
            nw = new_embeddings.weight.data
            if not nw.is_meta and added_num_tokens > 0:
                with torch.no_grad():
                    torch.nn.init.normal_(nw[-added_num_tokens:], mean=0.0, std=0.02)
            return

        device = new_embeddings.weight.device
        out_dtype = old_embeddings.weight.dtype
        old_cpu_fp32 = w.detach().cpu().float()
        mean_embeddings = torch.mean(old_cpu_fp32, dim=0)
        new_block = mean_embeddings.unsqueeze(0).repeat(added_num_tokens, 1).to(
            device=device, dtype=out_dtype
        )
        new_embeddings.weight.data[-added_num_tokens:, :] = new_block

    _mu.PreTrainedModel._init_added_embeddings_weights_with_mean = _low_gpu_mean_init  # type: ignore[method-assign]

    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as _LlavaCausalLM

    _orig_llava_from_pretrained = _LlavaCausalLM.from_pretrained.__func__

    @classmethod
    def _llava_from_pretrained_full_materialize(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs = dict(kwargs)
        quant = bool(
            kwargs.get("load_in_8bit")
            or kwargs.get("load_in_4bit")
            or kwargs.get("quantization_config") is not None
        )
        if quant:
            return _orig_llava_from_pretrained(
                cls, pretrained_model_name_or_path, *model_args, **kwargs
            )
        kwargs["low_cpu_mem_usage"] = False
        kwargs.pop("device_map", None)
        kwargs.pop("max_memory", None)
        return _orig_llava_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    _LlavaCausalLM.from_pretrained = _llava_from_pretrained_full_materialize  # type: ignore[assignment]

    _orig_llava_generate = _LlavaCausalLM.generate

    @torch.no_grad()
    def _llava_generate_optional_no_cache(self, *args, **kwargs):
        kwargs = dict(kwargs)
        if os.environ.get("MAMMOFM_NO_KV_CACHE") == "1":
            kwargs["use_cache"] = False
        imgs = kwargs.get("images")
        if imgs is not None:
            p0 = next(self.parameters())
            kwargs["images"] = imgs.to(device=p0.device, dtype=p0.dtype)
        return _orig_llava_generate(self, *args, **kwargs)

    _LlavaCausalLM.generate = _llava_generate_optional_no_cache  # type: ignore[assignment]

    from peft.tuners.lora.model import LoraModel as _LoraModel

    _orig_lora_merge_and_unload = _LoraModel.merge_and_unload

    def _merged_model_is_quantized(module):
        cfg = getattr(module, "config", None)
        if cfg is not None and getattr(cfg, "quantization_config", None) is not None:
            return True
        return bool(getattr(module, "is_quantized", False))

    def _lora_merge_and_unload_on_cpu(self, *args, **kwargs):
        inner = self.model
        # bitsandbytes: merging LoRA into 8/4-bit Linear breaks inference (`CB` / `.to` / `.float`
        # not supported). Keep LoRA wrappers attached (standard PEFT + quant inference).
        if _merged_model_is_quantized(inner):
            if os.environ.get("MAMMOFM_CPU_GENERATE") == "1":
                raise RuntimeError(
                    "8-bit/4-bit Stage 1 needs CUDA with unmerged LoRA. "
                    "Unset MAMMOFM_CPU_GENERATE for quantized loads."
                )
            out = inner
            if hasattr(out, "config"):
                out.config.use_cache = False
                with suppress(Exception):
                    if hasattr(out, "set_attn_implementation"):
                        out.set_attn_implementation("sdpa")
                    elif hasattr(out, "config"):
                        setattr(out.config, "attn_implementation", "sdpa")
            torch.cuda.empty_cache()
            return out

        if os.environ.get("MAMMOFM_SKIP_CPU_BEFORE_LORA_MERGE") != "1":
            self.to("cpu")
        out = _orig_lora_merge_and_unload(self, *args, **kwargs)
        if os.environ.get("MAMMOFM_CPU_GENERATE") == "1":
            out = out.cpu().float()
            if hasattr(out, "config"):
                out.config.use_cache = False
            return out
        if torch.cuda.is_available():
            dev = torch.device("cuda", torch.cuda.current_device())
            if not _merged_model_is_quantized(out):
                out = out.to(device=dev, dtype=torch.float16)
            if hasattr(out, "config"):
                out.config.use_cache = False
                with suppress(Exception):
                    if hasattr(out, "set_attn_implementation"):
                        out.set_attn_implementation("sdpa")
                    elif hasattr(out, "config"):
                        setattr(out.config, "attn_implementation", "sdpa")
            torch.cuda.empty_cache()
        return out

    _LoraModel.merge_and_unload = _lora_merge_and_unload_on_cpu  # type: ignore[method-assign]
except Exception:
    pass
