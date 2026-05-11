# MammoFM — Mammography Report Generator

A locally-hosted web UI that takes 4 mammogram PNGs, runs a two-stage AI pipeline, and returns preliminary + final radiology reports.

## Pipeline Overview

```
User uploads LCC, LMLO, RCC, RMLO PNGs
        ↓
[Stage 0] Image Encoding (Mammo-CLIP / EfficientNet-B5)
        ↓
[Stage 1] LLaVA Inference via DeepSpeed → Preliminary Report
        ↓
[Stage 2] LLaMA-3.1-8B Reconciliation → Final Report
```

All model weights and scripts live in Shawn's directories and are called read-only as subprocesses.

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

---

## Starting the Server

The server requires a GPU node. Submit it as a batch job:

```bash
qsub /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/start_server.sh
```

Once the job starts, the node hostname and SSH tunnel command are written to:
```
/restricted/projectnb/batmanlab/atang4/data/server_info.txt
```

Check it with:
```bash
cat /restricted/projectnb/batmanlab/atang4/data/server_info.txt
```

The server runs for 8 hours. Resubmit when needed.

---

## Connecting from Your Laptop

**From your local machine**, open a new terminal and run the SSH tunnel using a jump host:

```bash
ssh -J atang4@scc1.bu.edu -L 8000:scc-307:8000 atang4@scc4.bu.edu
```

Replace `scc-307` with the node name shown in `server_info.txt`. Keep this terminal open.

Then open your browser at:
```
http://localhost:8000
```

---

## Using the Web UI

1. Enter **Patient ID** and **Exam ID**
2. Upload 4 mammogram PNGs:
   - **LCC** — Left Craniocaudal
   - **LMLO** — Left Mediolateral Oblique
   - **RCC** — Right Craniocaudal
   - **RMLO** — Right Mediolateral Oblique
3. Optionally upload a **Classifier CSV** with `zero_shot_per_image` column (provides mass/asymmetry/calcification predictions to Stage 2)
4. Click **Generate Report**
5. Wait ~5–10 minutes for encoding + Stage 1 + Stage 2 to complete
6. Preliminary and Final reports appear side by side

---

## Key Paths

| Resource | Path |
|---|---|
| Checkpoint (unzipped) | `/restricted/projectnb/batmanlab/atang4/data/checkpoints_v1_bu_ve_old_loss/` |
| Job outputs | `/restricted/projectnb/batmanlab/atang4/data/jobs/{job_id}/` |
| Server log | `/restricted/projectnb/batmanlab/atang4/data/server.log` |
| Server info (node + tunnel) | `/restricted/projectnb/batmanlab/atang4/data/server_info.txt` |
| Stage 1 script | `.../src_pos_emb4views_new_loss/llava/serve/ctchat_validation_llama.py` |
| Stage 2 script | `.../src_pos_emb4views_new_loss/final_stage.py` |
| Encoding script | `.../src_pos_emb4views_new_loss/llava/model/multimodal_encoder/save_img_embedding.py` |

---

## Validation Scripts

```bash
# Test Stage 2 only (no GPU queue wait — runs on GPU node via qsub)
qsub scripts/test_stage2.sh

# Test Stage 1 (deepspeed inference)
qsub scripts/test_stage1.sh

# Test image encoding
qsub scripts/test_encoding.sh
```

Results saved to `/restricted/projectnb/batmanlab/atang4/data/validation/`.

---

## Notes

- **View order** must be LMLO, LCC, RMLO, RCC (confirmed from existing BU records)
- **MammoCLIP checkpoint**: using `epoch4.tar` — if reports seem wrong, try epoch3
- **DeepSpeed**: required for Stage 1 even for single-patient inference
- **LLaMA weights**: cached at `/restricted/projectnb/batmanlab/shawn24/PhD/.hf_cache/` — no download needed
- The server job runs for 8 hours max (SGE `h_rt=8:00:00`). Resubmit if it expires.
