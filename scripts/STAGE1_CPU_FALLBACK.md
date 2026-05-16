# Stage 1 CPU fallback (optional)

Use this when **Stage 1 LLaVA `generate` hits GPU OOM** on tight cards (~16 GB), or when you want a slower but predictable run.

## Enable (web server on SCC)

**Preferred:** from the MammoFM repo on the login node:

```bash
./scripts/mammofm submit-cpu
```

Optional: set max new tokens for the Stage 1 subprocess (passed through SGE `-v`):

```bash
MAMMOFM_STAGE1_MAX_NEW_TOKENS=64 ./scripts/mammofm submit-cpu
```

**Manual `qsub`:**

```bash
qsub -v MAMMOFM_STAGE1_CPU_GENERATE=1 /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/start_server.sh
```

With token cap:

```bash
qsub -v MAMMOFM_STAGE1_CPU_GENERATE=1,MAMMOFM_STAGE1_MAX_NEW_TOKENS=64 \
  /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/start_server.sh
```

**Confirm:** tail `server.log` after the job starts; the banner includes `STAGE1_CPU=1`.

## Environment chain

| Layer | Variable | Meaning |
|--------|----------|---------|
| Server / Uvicorn parent | `MAMMOFM_STAGE1_CPU_GENERATE=1` | Tells the backend to run Stage 1 text generation on CPU. |
| Stage 1 subprocess | `MAMMOFM_CPU_GENERATE=1` | Set by `pipeline.py`; consumed by `backend/patch_site/sitecustomize.py` (LoRA merge keeps merged fp16 model on CPU for decode). |

You normally only set **`MAMMOFM_STAGE1_CPU_GENERATE`** before `qsub` or via `submit-cpu`.

## Smoke test (e2e on its own job)

```bash
qsub -v SMOKE_CPU_STAGE1=1 /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/smoke_test_stage1_patch.sh
```

## Tradeoffs

- **Latency:** often **many minutes per heavy case** (CPU autoregressive decode vs GPU).
- **VRAM:** no Stage 1 GPU peak during decode; embedding and other steps may still use the GPU depending on your stack.

## `cutlassF: no kernel found to launch!` (not necessarily Stage 1)

That message comes from **PyTorch CUDA SDPA** (flash / memory-efficient attention routing into CUTLASS). It often appears on **V100-class** GPUs when Stage 2 (`final_stage.py`) or another HF `generate` path picks a backend that has no kernel.

**Important:** **`MAMMOFM_STAGE1_CPU_GENERATE=1` only affects Stage 1.** Stage 2 still runs **Llama on GPU** (4-bit in `final_stage.py`). If Stage 1 finishes and the job fails “during generation,” check **`logs/stage2.log`** first.

MammoFM mitigations (keep repo updated): `PYTORCH_ENABLE_MEM_EFFICIENT_SDPA=0` for encode / Stage 1 / Stage 2 env in `backend/pipeline.py`, and `backend/final_stage_launcher.py` disables flash + mem-efficient SDP before running upstream `final_stage.py`.

## 8-bit / 4-bit Stage 1 and CPU

**Not supported together:** with quantized bases, LoRA does not merge the same way; `sitecustomize.py` requires **CUDA** for that path and errors if `MAMMOFM_CPU_GENERATE=1`. If you use load-in-8bit/4bit for Stage 1, keep **`MAMMOFM_STAGE1_CPU_GENERATE` unset or `0`**.

## Related tunables

- `MAMMOFM_STAGE1_MAX_NEW_TOKENS` — shorter outputs use less time on CPU; may truncate reports if set too low.
- `MAMMOFM_NO_KV_CACHE=1` — forces `use_cache=False` in the patched `generate` wrapper; **debug / special cases only**; leave unset for normal incremental decode unless you know you need it.
