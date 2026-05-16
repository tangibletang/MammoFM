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
5. Optional: `LlavaLlamaForCausalLM.generate` — if `MAMMOFM_NO_KV_CACHE=1`, force `use_cache=False` (rarely
   needed; default leaves HF/ctchat `use_cache=True` for incremental decode on long multimodal prefixes).
   Re-apply `attn_implementation=sdpa` each call unless `MAMMOFM_DISABLE_SDPA=1`. Image tensors are cast to the model's device/dtype.
6. Optional: `MAMMOFM_CPU_GENERATE=1` — after **fp16** LoRA merge, keep the merged model on **CPU float32**
   (reliable smoke on V100 where GPU decode can OOM). Slow (~minutes/sample). Incompatible with 8/4-bit loads.
7. Quantized (8/4-bit) bases: LoRA **merge is skipped** (bitsandbytes merged weights break inference); adapters stay
   attached and Stage 1 runs on **CUDA** (`MAMMOFM_CPU_GENERATE` must be off).
8. **8-bit load (GPU Stage 1):** `BitsAndBytesConfig` with `llm_int8_skip_modules` (`mm_projector`, `vision_tower`,
   `lm_head`) and **`device_map={"": 0}`** by default so RoPE is not split across CPU/CUDA (set
   `MAMMOFM_STAGE1_BNB_DEVICE_MAP=auto` to opt into accelerate auto-map). Disable wrapper: `MAMMOFM_BNB_SKIP_PROJECTOR_PATCH=1`.
9. **`prepare_inputs_labels_for_multimodal`:** (optional) hub-id alias for local snapshot / checkpoint `_name_or_path`
   (4-view tokens). Disable alias only: `MAMMOFM_DISABLE_PREPARE_MM_NAME_ALIAS=1`.
   Always (unless `MAMMOFM_DISABLE_MULTIMODAL_CUDA_ALIGN=1`): move multimodal `inputs_embeds` / masks / `position_ids`
   to **current CUDA device** — bitsandbytes+HF can leave `self.device` as CPU while linear layers need GPU.
9b. **`GenerationMixin._prepare_special_tokens`:** when only **`inputs_embeds`** are on CUDA but **`model.device`** is
   CPU (typical bitsandbytes `get_parameter_device`), HF stores `_pad_token_tensor` on CUDA while the decode loop moves
   logits to **`input_ids.device`** (CPU) → `next_tokens * … + pad_token_id * …` mixes devices. We place BOS/EOS/pad
   tensors on **`model.device`** when they would otherwise land only on CUDA while the module reports CPU.
   Disable: `MAMMOFM_DISABLE_GEN_SPECIAL_TOKEN_DEVICE_FIX=1`.
9c. **`GenerationMixin.prepare_inputs_for_generation`:** if multimodal **`inputs_embeds`** were supplied but HF still
   took the **`input_ids` / `embed_tokens`** path on cache step 0, set **`input_ids=None`** and restore
   **`inputs_embeds`**. Disable: `MAMMOFM_DISABLE_PREPARE_INPUTS_EMBEDS_PREFILL_FIX=1`.
9d. **`prepare_inputs_for_generation`:** under bitsandbytes, **`model.device`** can disagree with
   **`embed_tokens.weight.device`** (ids on CUDA, table on CPU → `embedding` crash). We move **`input_ids`** to the
   embedding table device whenever they differ. Disable: `MAMMOFM_DISABLE_ALIGN_INPUT_IDS_TO_EMBED_DEVICE=1`.
9e. **`LlamaModel.forward` (bnb):** `embed_tokens` may stay on **CPU** while int8 layers are on **CUDA** — moving token
   ids to the embedding table then leaves **`inputs_embeds` on CPU** and `q_proj` fails (`int8_vectorwise_quant` needs
   GPU). We **`embed_tokens.to(decoder layer device)`** when the first attention proj is CUDA and the table is CPU,
   then align **`input_ids`** to the table device before lookup. Disable: `MAMMOFM_DISABLE_LLAMA_EMBED_DEVICE_FIX=1`.
9f. **`LlamaForCausalLM.forward`:** after promoting embeddings / bnb matmuls, **hidden states can stay float32**
   while **`lm_head` is fp16** → `F.linear` dtype error. We cast the **lm_head slice** to `lm_head.weight.dtype` before
   the projection. Disable: `MAMMOFM_DISABLE_LLAMA_LM_HEAD_DTYPE_FIX=1`.
10. **`LlamaRotaryEmbedding.forward`:** align `position_ids` and `inv_freq` to `x.device` when bitsandbytes + mixed
   placement leaves RoPE buffers on CPU (`cuda:0` vs `cpu` bmm in `modeling_llama`). Disable:
   `MAMMOFM_DISABLE_LLAMA_ROPE_DEVICE_FIX=1`.
11. **`LlamaRMSNorm.forward`:** align RMSNorm `weight` to `hidden_states.device` if a submodule stayed on CPU (8-bit path).
   Disable: `MAMMOFM_DISABLE_LLAMA_RMS_DEVICE_FIX=1`.
12. **8-bit `max_memory`:** `{0: "<GiB>", "cpu": 0}` during load to block Accelerate CPU offload (`MAMMOFM_STAGE1_BNB_MAX_MEMORY_GIB`,
   `MAMMOFM_STAGE1_BNB_ALLOW_CPU_OFFLOAD=1` to opt out).
"""
from __future__ import annotations

try:
    import gc
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

    _orig_llama_rotary_forward = _llama_mod.LlamaRotaryEmbedding.forward

    @torch.no_grad()
    def _llama_rotary_forward_same_device(self, x, position_ids):
        if os.environ.get("MAMMOFM_DISABLE_LLAMA_ROPE_DEVICE_FIX") == "1":
            return _orig_llama_rotary_forward(self, x, position_ids)
        dev = x.device
        if position_ids is not None and position_ids.device != dev:
            position_ids = position_ids.to(device=dev, non_blocking=True)
        if self.inv_freq.device != dev:
            inv = self.inv_freq.to(dev)
            self.register_buffer("inv_freq", inv, persistent=False)
            self.original_inv_freq = inv
        return _orig_llama_rotary_forward(self, x, position_ids)

    _llama_mod.LlamaRotaryEmbedding.forward = _llama_rotary_forward_same_device  # type: ignore[method-assign]

    _orig_llama_rms_forward = _llama_mod.LlamaRMSNorm.forward

    @torch.no_grad()
    def _llama_rms_forward_same_device(self, hidden_states):
        if os.environ.get("MAMMOFM_DISABLE_LLAMA_RMS_DEVICE_FIX") == "1":
            return _orig_llama_rms_forward(self, hidden_states)
        if self.weight.device != hidden_states.device:
            self.weight = nn.Parameter(self.weight.to(hidden_states.device))
        return _orig_llama_rms_forward(self, hidden_states)

    _llama_mod.LlamaRMSNorm.forward = _llama_rms_forward_same_device  # type: ignore[method-assign]

    _orig_llama_model_forward = _llama_mod.LlamaModel.forward

    def _mammofm_first_decoder_linear_device(llama_model):
        try:
            lay0 = llama_model.layers[0]
            q = getattr(lay0.self_attn, "q_proj", None)
            if q is None:
                return None
            base = getattr(q, "base_layer", q)
            w = getattr(base, "weight", None)
            return w.device if w is not None else None
        except Exception:
            return None

    def _llama_model_forward_bnb_embed_device(self, *args, **kwargs):
        if os.environ.get("MAMMOFM_DISABLE_LLAMA_EMBED_DEVICE_FIX") == "1":
            return _orig_llama_model_forward(self, *args, **kwargs)
        layer_dev = _mammofm_first_decoder_linear_device(self)
        from_kw_id = "input_ids" in kwargs
        arg_list = list(args)
        input_ids = kwargs.get("input_ids") if from_kw_id else None
        if input_ids is None and len(arg_list) >= 1:
            input_ids = arg_list[0]
        emb = getattr(self, "embed_tokens", None)
        ew = getattr(emb, "weight", None) if emb is not None else None
        if (
            ew is not None
            and layer_dev is not None
            and layer_dev.type == "cuda"
            and ew.device.type == "cpu"
        ):
            emb = emb.to(layer_dev)
            self.embed_tokens = emb
            ew = emb.weight
        if input_ids is not None and isinstance(input_ids, torch.Tensor) and ew is not None:
            if input_ids.device != ew.device:
                input_ids = input_ids.to(ew.device, non_blocking=True)
                if from_kw_id:
                    kw = dict(kwargs)
                    kw["input_ids"] = input_ids
                    return _orig_llama_model_forward(self, *args, **kw)
                if len(arg_list) >= 1:
                    arg_list[0] = input_ids
                    return _orig_llama_model_forward(self, *tuple(arg_list), **kwargs)
        ie = kwargs.get("inputs_embeds")
        if (
            ie is not None
            and isinstance(ie, torch.Tensor)
            and layer_dev is not None
            and layer_dev.type == "cuda"
            and ie.device.type == "cpu"
        ):
            kw = dict(kwargs)
            kw["inputs_embeds"] = ie.to(layer_dev, non_blocking=True)
            return _orig_llama_model_forward(self, *args, **kw)
        return _orig_llama_model_forward(self, *args, **kwargs)

    _llama_mod.LlamaModel.forward = _llama_model_forward_bnb_embed_device  # type: ignore[method-assign]

    from transformers.modeling_outputs import CausalLMOutputWithPast as _CausalLMOutputWithPast

    _orig_llama_causal_lm_forward = _llama_mod.LlamaForCausalLM.forward

    def _llama_causal_lm_forward_lm_head_dtype(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        num_logits_to_keep=0,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        h = hidden_states[:, -num_logits_to_keep:, :]
        if os.environ.get("MAMMOFM_DISABLE_LLAMA_LM_HEAD_DTYPE_FIX") != "1":
            lh = self.lm_head
            lw = getattr(lh, "weight", None)
            if lw is not None and h.dtype != lw.dtype:
                h = h.to(lw.dtype)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return _CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    _llama_mod.LlamaForCausalLM.forward = _llama_causal_lm_forward_lm_head_dtype  # type: ignore[method-assign]

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
        # HF reads config._attn_implementation; None → eager (huge VRAM). SDPA is required on ~16GB + fp16.
        if not kwargs.get("attn_implementation") and os.environ.get("MAMMOFM_DISABLE_SDPA") != "1":
            kwargs["attn_implementation"] = "sdpa"
        return _orig_llava_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    _LlavaCausalLM.from_pretrained = _llava_from_pretrained_full_materialize  # type: ignore[assignment]

    _orig_llava_generate = _LlavaCausalLM.generate

    def _force_llava_sdpa(module: nn.Module) -> None:
        """Transformers Llama uses config._attn_implementation; plain setattr(config, 'attn_implementation', …) is a no-op."""
        if os.environ.get("MAMMOFM_DISABLE_SDPA") == "1":
            return
        for cfg in (
            getattr(module, "config", None),
            getattr(getattr(module, "model", None), "config", None),
        ):
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "sdpa"
        with suppress(Exception):
            if hasattr(module, "set_attn_implementation"):
                module.set_attn_implementation("sdpa")

    @torch.no_grad()
    def _llava_generate_optional_no_cache(self, *args, **kwargs):
        kwargs = dict(kwargs)
        if os.environ.get("MAMMOFM_NO_KV_CACHE") == "1":
            kwargs["use_cache"] = False
        _force_llava_sdpa(self)
        gc.collect()
        torch.cuda.empty_cache()
        imgs = kwargs.get("images")
        if imgs is not None:
            p0 = next(self.parameters())
            kwargs["images"] = imgs.to(device=p0.device, dtype=p0.dtype)
        return _orig_llava_generate(self, *args, **kwargs)

    _LlavaCausalLM.generate = _llava_generate_optional_no_cache  # type: ignore[assignment]

    from transformers.generation.utils import GenerationMixin as _GenerationMixin

    _orig_prepare_special_tokens = _GenerationMixin._prepare_special_tokens

    def _prepare_special_tokens_mammofm(
        self,
        generation_config,
        kwargs_has_attention_mask=None,
        device=None,
    ):
        if os.environ.get("MAMMOFM_DISABLE_GEN_SPECIAL_TOKEN_DEVICE_FIX") != "1":
            mdev = getattr(self, "device", None)
            ddev = device
            if (
                mdev is not None
                and getattr(mdev, "type", None) == "cpu"
                and ddev is not None
                and getattr(ddev, "type", None) == "cuda"
            ):
                device = mdev
        return _orig_prepare_special_tokens(
            self, generation_config, kwargs_has_attention_mask=kwargs_has_attention_mask, device=device
        )

    _GenerationMixin._prepare_special_tokens = _prepare_special_tokens_mammofm  # type: ignore[method-assign]

    _orig_prepare_inputs_for_generation = _GenerationMixin.prepare_inputs_for_generation

    def _prepare_inputs_for_generation_mammofm(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        out = _orig_prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        out = dict(out)
        if os.environ.get("MAMMOFM_DISABLE_PREPARE_INPUTS_EMBEDS_PREFILL_FIX") != "1":
            if (
                not getattr(self.config, "is_encoder_decoder", False)
                and inputs_embeds is not None
                and cache_position is not None
                and cache_position.numel() > 0
            ):
                try:
                    c0v = int(cache_position.reshape(-1)[0].item())
                except Exception:
                    c0v = -1
                if c0v == 0 and out.get("inputs_embeds") is None:
                    out["input_ids"] = None
                    out["inputs_embeds"] = inputs_embeds
        if os.environ.get("MAMMOFM_DISABLE_ALIGN_INPUT_IDS_TO_EMBED_DEVICE") != "1":
            tid = out.get("input_ids")
            if tid is not None and isinstance(tid, torch.Tensor):
                with suppress(Exception):
                    emb = self.get_input_embeddings()
                    w = getattr(emb, "weight", None)
                    if w is not None and tid.device != w.device:
                        out["input_ids"] = tid.to(w.device, non_blocking=True)
        return out

    _GenerationMixin.prepare_inputs_for_generation = _prepare_inputs_for_generation_mammofm  # type: ignore[method-assign]

    # Multimodal prep uses `.to(self.device)`; under bnb/accelerate `self.device` is often CPU → bitsandbytes matmul error.
    from llava.model.llava_arch import LlavaMetaForCausalLM as _LlavaMetaForCausalLM

    _orig_prepare_mm = _LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal
    _LLAMA31_HUB_ALIAS = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def _needs_prepare_mm_hub_alias(cfg) -> bool:
        raw = getattr(cfg, "_name_or_path", None)
        mp = raw.replace("\\", "/").lower() if isinstance(raw, str) else ""
        if mp == "lmsys/vicuna-13b-v1.5":
            return False
        if mp == "meta-llama/meta-llama-3.1-8b-instruct":
            return False
        if "meta-llama-3.1-8b-instruct" in mp:
            return True
        if "models--meta-llama--meta-llama-3.1-8b-instruct" in mp.replace("/", "--"):
            return True
        rs = getattr(cfg, "rope_scaling", None)
        if getattr(cfg, "model_type", None) == "llava_llama" and isinstance(rs, dict) and rs.get("rope_type") == "llama3":
            return True
        return False

    def _align_multimodal_batch_cuda(self, out):
        if os.environ.get("MAMMOFM_DISABLE_MULTIMODAL_CUDA_ALIGN") == "1":
            return out
        if not torch.cuda.is_available():
            return out
        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = out
        target = torch.device("cuda", torch.cuda.current_device())
        lm = getattr(self, "lm_head", None)
        embed_dtype = None
        if lm is not None and getattr(lm, "weight", None) is not None:
            embed_dtype = lm.weight.dtype
        if inputs_embeds is not None:
            if inputs_embeds.device != target:
                inputs_embeds = inputs_embeds.to(target, non_blocking=True)
            if embed_dtype is not None and inputs_embeds.dtype != embed_dtype:
                inputs_embeds = inputs_embeds.to(embed_dtype)
        if position_ids is not None and position_ids.device != target:
            position_ids = position_ids.to(target, non_blocking=True)
        if attention_mask is not None and attention_mask.device != target:
            attention_mask = attention_mask.to(target, non_blocking=True)
        if input_ids is not None and input_ids.device != target:
            input_ids = input_ids.to(target, non_blocking=True)
        if labels is not None and getattr(labels, "device", None) is not None and labels.device != target:
            labels = labels.to(target, non_blocking=True)
        return (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels)

    def _prepare_mm_mammofm(self, *args, **kwargs):
        cfg = getattr(self, "config", None)
        use_alias = (
            os.environ.get("MAMMOFM_DISABLE_PREPARE_MM_NAME_ALIAS") != "1"
            and cfg is not None
            and _needs_prepare_mm_hub_alias(cfg)
        )
        if use_alias:
            old = cfg._name_or_path
            try:
                cfg._name_or_path = _LLAMA31_HUB_ALIAS
                out = _orig_prepare_mm(self, *args, **kwargs)
            finally:
                cfg._name_or_path = old
        else:
            out = _orig_prepare_mm(self, *args, **kwargs)
        return _align_multimodal_batch_cuda(self, out)

    _LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = _prepare_mm_mammofm  # type: ignore[method-assign]

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
                _force_llava_sdpa(out)
            torch.cuda.empty_cache()
            return out

        if os.environ.get("MAMMOFM_SKIP_CPU_BEFORE_LORA_MERGE") != "1":
            self.to("cpu")
        out = _orig_lora_merge_and_unload(self, *args, **kwargs)
        if os.environ.get("MAMMOFM_CPU_GENERATE") == "1":
            out = out.cpu().float()
            if hasattr(out, "config"):
                out.config.use_cache = True
            return out
        if torch.cuda.is_available():
            dev = torch.device("cuda", torch.cuda.current_device())
            if not _merged_model_is_quantized(out):
                out = out.to(device=dev, dtype=torch.float16)
            if hasattr(out, "config"):
                # Incremental decode (HF cache) matches ctchat_validation_llama.py use_cache=True.
                out.config.use_cache = True
                _force_llava_sdpa(out)
            torch.cuda.empty_cache()
        return out

    _LoraModel.merge_and_unload = _lora_merge_and_unload_on_cpu  # type: ignore[method-assign]
except Exception:
    pass

# --- bitsandbytes 8-bit (GPU Stage 1): skip fragile modules + pin one GPU (auto map → CPU layers → RoPE cuda/cpu crash) ---
try:
    import os as _os_site


    def _stage1_8bit_device_map(device: str, fallback_dm: str):
        """bnb + accelerate: device_map='auto' may offload to CPU and break RoPE (mat2 cuda vs cpu). Pin GPU0 by default."""
        if device != "cuda":
            return fallback_dm
        raw = (_os_site.environ.get("MAMMOFM_STAGE1_BNB_DEVICE_MAP") or "").strip().lower()
        if raw in ("auto", "balanced", "sequential"):
            return raw  # opt-in old behavior
        return {"": 0}

    def _stage1_8bit_max_memory():
        """Forbid Accelerate CPU offload for int8 loads (else LayerNorm / Linear split across cuda vs cpu)."""
        if _os_site.environ.get("MAMMOFM_STAGE1_BNB_ALLOW_CPU_OFFLOAD", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return None
        try:
            import torch
        except Exception:
            return None
        if not torch.cuda.is_available():
            return None
        g = (_os_site.environ.get("MAMMOFM_STAGE1_BNB_MAX_MEMORY_GIB") or "").strip()
        if g:
            try:
                return {0: f"{max(1, int(g))}GiB", "cpu": 0}
            except ValueError:
                pass
        props = torch.cuda.get_device_properties(0)
        gib = max(1, int(props.total_memory / (1024**3)) - 2)
        return {0: f"{gib}GiB", "cpu": 0}

    def _patch_llava_builder_int8_skip_multimodal() -> None:
        if _os_site.environ.get("MAMMOFM_BNB_SKIP_PROJECTOR_PATCH", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        try:
            from transformers import BitsAndBytesConfig

            import llava.model.builder as _builder
        except Exception:
            return
        if getattr(_builder.load_pretrained_model_llama, "_mammofm_int8_skip", False):
            return

        _orig_llama = _builder.load_pretrained_model_llama
        _orig_generic = _builder.load_pretrained_model

        def _wrap_llama(
            model_path,
            model_base,
            model_name,
            load_8bit=False,
            load_4bit=False,
            device_map="auto",
            device="cuda",
            use_flash_attn=False,
            **kwargs,
        ):
            if load_8bit:
                kwargs = dict(kwargs)
                kwargs.pop("device_map", None)
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=[
                        "mm_projector",
                        "vision_tower",
                        "lm_head",
                    ],
                )
                _dm = _stage1_8bit_device_map(device, device_map)
                _mm = _stage1_8bit_max_memory()
                if _mm is not None:
                    kwargs["max_memory"] = _mm
                return _orig_llama(
                    model_path,
                    model_base,
                    model_name,
                    False,
                    load_4bit,
                    _dm,
                    device,
                    use_flash_attn,
                    **kwargs,
                )
            return _orig_llama(
                model_path,
                model_base,
                model_name,
                load_8bit,
                load_4bit,
                device_map,
                device,
                use_flash_attn,
                **kwargs,
            )

        def _wrap_generic(
            model_path,
            model_base,
            model_name,
            load_8bit=False,
            load_4bit=False,
            device_map="auto",
            device="cuda",
            use_flash_attn=False,
            **kwargs,
        ):
            if load_8bit:
                kwargs = dict(kwargs)
                kwargs.pop("device_map", None)
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=[
                        "mm_projector",
                        "vision_tower",
                        "lm_head",
                    ],
                )
                _dm = _stage1_8bit_device_map(device, device_map)
                _mm = _stage1_8bit_max_memory()
                if _mm is not None:
                    kwargs["max_memory"] = _mm
                return _orig_generic(
                    model_path,
                    model_base,
                    model_name,
                    False,
                    load_4bit,
                    _dm,
                    device,
                    use_flash_attn,
                    **kwargs,
                )
            return _orig_generic(
                model_path,
                model_base,
                model_name,
                load_8bit,
                load_4bit,
                device_map,
                device,
                use_flash_attn,
                **kwargs,
            )

        _wrap_llama._mammofm_int8_skip = True  # type: ignore[attr-defined]
        _wrap_generic._mammofm_int8_skip = True  # type: ignore[attr-defined]
        _builder.load_pretrained_model_llama = _wrap_llama
        _builder.load_pretrained_model = _wrap_generic

    _patch_llava_builder_int8_skip_multimodal()
except Exception:
    pass
