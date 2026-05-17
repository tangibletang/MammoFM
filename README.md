# MammoFM — Mammography Report Generator

A locally-hosted web UI that takes 4 mammogram PNGs, runs a two-stage AI pipeline, and returns preliminary + final radiology reports.

## Pipeline Overview

```
User uploads LCC, LMLO, RCC, RMLO PNGs
        ↓
[Stage 0] Image encoding (Mammo-CLIP / EfficientNet-B5)
        ↓
[Stage 1] LLaVA + Llama-3.1-8B (Hugging Face `generate`, optional 8-bit base) → preliminary report
        ↓
[Stage 2] LLaMA-3.1-8B reconciliation (`final_stage.py`) → final report
```

Model weights and upstream LLaVA code live in shared lab paths; this repo wires FastAPI, job orchestration, HF cache resolution, and small launcher patches around those subprocesses.

---

## Why Stage 1 defaults to 8-bit (bitsandbytes)

**I had to make an alteration to the workflow, defaulting Stage 1 to 8-bit**

- The default V100 GPUs do not reliably run full **fp16** LLaVA + a **7–8B** Llama backbone for multimodal generation: you can hit **OOM**, or PyTorch attention/back-end paths that error on some architectures (we disable a few brittle SDPA modes in the job environment).
- We could use a large-VRM nodes (e.g. A100-class), but on SCC they are always being used, so we wouldn't be able to start a node.
- 8-bit weights for the language-model backbone (via bitsandbytes) cut weight memory sharply so Stage 1 fits the GPUs we actually get under the default `qsub` resource request.

To force **fp16 Stage 1** when you have enough VRAM:

```bash
qsub -v MAMMOFM_STAGE1_LOAD_8BIT=0 /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/start_server.sh
```

(or set the same variable in your environment before `mammofm submit` / `start_server.sh`.)

### Do 8-bit and fp16 reports differ? How?

**They are not numerically identical**, but **free-text reports are often very hard to tell apart** in practice.

| Aspect | fp16 Stage 1 | 8-bit Stage 1 (default) |
|--------|----------------|-------------------------|
| Weights | fp16 tensors | int8 (+ scales) in quantized linear layers |
| Math | standard matmul | quantized matmul approximations |
| Decoding | argmax / sampling from logits | same API, logits differ slightly |

**Mechanically:** quantization changes hidden states and therefore **logits over the vocabulary**. With **greedy** or **low-temperature** decoding, the **argmax token is often unchanged** for many steps; when two tokens are **close in logit space**, a small shift can **flip** the winner → **local wording** changes, occasionally a different phrase for the same imaging finding.

**What we do *not* see systematically:** a consistent “8-bit always understates malignancy”-style bias has **not** been characterized here; overlap with fp16 is **high** for normal screening-style prose. If you need **strict reproducibility** (e.g. matching a published fp16 eval), run **fp16 Stage 1** on hardware that can hold it.

Stage 2 (final report) is a **separate** Llama pass over Stage 1 text + structured fields; it is **not** the 8-bit vs fp16 comparator for the *whole* product, but it will reflect whatever wording Stage 1 produced.


**Note:** To use the fp16 Stage 1, you can click the "generate with CPU" which will take longer but still use the default fp16 Stagen 1. 

---

## One-Time Setup

### 1. Unzip the checkpoint (already done)

```bash
unzip /restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb4views_new_loss/checkpoints_v1_bu_ve_old_loss.zip \
    -d /restricted/projectnb/batmanlab/atang4/data/
```

### 2. Set up the HF cache symlink (already done)

```bash
mkdir -p /restricted/projectnb/batmanlab/atang4/data/.hf_cache/hub
ln -sf /restricted/projectnb/batmanlab/shawn24/PhD/.hf_cache/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct \
       /restricted/projectnb/batmanlab/atang4/data/.hf_cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
```

### 3. Install dependencies into Shawn's conda env (already done)

```bash
/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/pip install \
    fastapi "uvicorn[standard]" python-multipart pandas aiofiles \
    simsimd opencv-python-headless "numpy<2" omegaconf
```

Optional sanity check (login node, no GPU):

```bash
cd /restricted/projectnb/batmanlab/atang4/MammoFM && ./scripts/mammofm verify
```

---

## Running the web UI (SCC)

### 1. Submit the server job

From an SCC login node (e.g. `scc4`):

```bash
cd /restricted/projectnb/batmanlab/atang4/MammoFM
./scripts/mammofm submit
```

Equivalent:

```bash
qsub /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/start_server.sh
```

### 2. Wait until the job is actually running

Poll the queue until your job leaves `qw` and shows state **`r`** (running):

```bash
./scripts/mammofm status
# or
qstat -u "$USER"
```

`start_server.sh` writes **`server_info.txt` on the shared filesystem** as soon as the job starts on a compute node, so you do **not** need to tail logs for routine use.

### 3. Get the SSH tunnel command (on SCC)

```bash
./scripts/mammofm tunnel
```

This prints a one-line `ssh ... -L ...:NODE:8000 ...` command and the **local URL** (default local port `25001`; override with `LOCAL_PORT=25002 ./scripts/mammofm tunnel` if busy).

If `tunnel` says `server_info.txt` is missing, the job has not started yet — go back to **step 2**.

### 4. Open the tunnel on your laptop

Paste and run the printed `ssh` command in a **terminal on your laptop** (not on SCC). Leave it open.

Then open the printed URL in your **laptop** browser, e.g. `http://127.0.0.1:25001` .

### 5. Optional troubleshooting

- **Server log (debug only):** `/restricted/projectnb/batmanlab/atang4/data/server.log`
- **Per-upload job logs:** `/restricted/projectnb/batmanlab/atang4/data/jobs/{job_id}/logs/` (`encode.log`, `stage1.log`, `stage2.log`, …)

The batch job runs up to **8 hours** (`h_rt=8:00:00`). Resubmit when it expires.

---

## Connecting from Your Laptop (manual SSH, if you prefer)

If you already know the compute node name (from `server_info.txt` or `mammofm status`):

```bash
ssh -J YOUR_USER@scc1.bu.edu -L 8000:NODE_NAME:8000 YOUR_USER@scc4.bu.edu
```

Then browse to `http://localhost:8000` (or match the local port you chose). The `mammofm tunnel` command avoids hunting for hostnames and uses a high local port to reduce collisions.

---

## Using the Web UI

1. Enter **Patient ID** and **Exam ID**
2. Upload 4 mammogram PNGs:
   - **LCC** — Left Craniocaudal
   - **LMLO** — Left Mediolateral Oblique
   - **RCC** — Right Craniocaudal
   - **RMLO** — Right Mediolateral Oblique
3. Optionally upload a **Classifier CSV** with `zero_shot_per_image` column (mass/asymmetry/calcification hints for Stage 2)
4. Click **Generate Report**
5. Wait on the order of **~5–10 minutes** for encoding + Stage 1 + Stage 2 (varies with GPU load and token length)
6. Preliminary and final reports appear side by side

---

## Key Paths

| Resource | Path |
|----------|------|
| Checkpoint (unzipped) | `/restricted/projectnb/batmanlab/atang4/data/checkpoints_v1_bu_ve_old_loss/` |
| Job outputs | `/restricted/projectnb/batmanlab/atang4/data/jobs/{job_id}/` |
| Server log | `/restricted/projectnb/batmanlab/atang4/data/server.log` |
| Server info (node + tunnel hints) | `/restricted/projectnb/batmanlab/atang4/data/server_info.txt` |
| Stage 1 script | `.../src_pos_emb4views_new_loss/llava/serve/ctchat_validation_llama.py` |
| Stage 2 script | `.../src_pos_emb4views_new_loss/final_stage.py` |
| Encoding script | `MammoFM/backend/save_img_embedding_mammofm.py` (patched; do not use upstream `save_img_embedding.py`) |

---

## Validation / smoke tests (`qsub`)

Artifacts under `/restricted/projectnb/batmanlab/atang4/data/validation/` unless noted.

```bash
# Full pipeline: encode → Stage 1 → Stage 2 (blocking until job finishes if you pass -sync y)
qsub -sync y /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/smoke_e2e_full_pipeline.sh

# Stage 1 only (8-bit by default; fp16: -v SMOKE_LOAD_8BIT=0)
qsub /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/smoke_test_stage1_patch.sh

# Stage 2 only
qsub scripts/test_stage2.sh

# Stage 1 harness (older comparison script)
qsub scripts/test_stage1.sh

# Image encoding only
qsub scripts/test_encoding.sh
```

---


