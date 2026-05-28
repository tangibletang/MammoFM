# MammoFM — Project Guide for Claude Code

## What this repo is

MammoFM is a FastAPI web app that takes four mammogram views (LCC, LMLO, RCC, RMLO), runs a
two-stage report pipeline, and serves a browser UI. Deployed on BU SCC via qsub.

```
backend/
  app.py              — FastAPI routes (upload, run, status, results, PDF download)
  pipeline.py         — job lifecycle + Stage 1/2 orchestration
  pdf_utils.py        — reportlab PDF builder
  smoke_e2e_pipeline.py — headless E2E smoke test (no server needed)
frontend/
  index.html, app.js  — browser UI
scripts/
  mammofm             — ONE CLI for all SCC operations (submit, status, tunnel, verify-*, autopilot-*)
  start_server.sh     — qsub script that starts the FastAPI server on a compute node
  smoke_e2e_a100.sh   — 8-bit GPU verification job (A100/A40-class GPU)
  smoke_e2e_cpu.sh    — CPU fallback verification job
harness/
  autopilot.sh        — offline autopilot (submits, monitors, self-heals, validates PDF)
  autopilot-prompt.txt — system prompt for the claude -p repair agent
  integration_pdf_curl.sh — curl-based PDF integration test
  runs/               — per-run state.json, autopilot.log, report.md
```

## Two pipeline pathways

| Pathway | Env | When to use |
|---------|-----|-------------|
| **8-bit GPU** (default) | `MAMMOFM_STAGE1_LOAD_8BIT=1` | A100/A40 GPU — production |
| **CPU fallback** | `MAMMOFM_STAGE1_CPU_GENERATE=1`, `MAMMOFM_STAGE1_LOAD_8BIT=0` | No GPU / GPU OOM |

CPU pathway is slow — use `MAMMOFM_STAGE1_MAX_NEW_TOKENS=64` for quick smoke tests.

## Running things

```bash
# NEVER run training or the server on a login node — always qsub

./scripts/mammofm submit          # qsub webapp
./scripts/mammofm status          # check queue + server_info
./scripts/mammofm tunnel          # print SSH tunnel command for laptop

./scripts/mammofm verify-cpu      # qsub CPU verification
./scripts/mammofm verify-8bit     # qsub 8-bit GPU verification
./scripts/mammofm verify-all      # both at once

./scripts/mammofm autopilot-submit   # qsub the autopilot (offline-safe)
./scripts/mammofm autopilot-run      # foreground (tmux)
```

## Sentinels (verification output)

```
/restricted/projectnb/batmanlab/atang4/data/validation/verify_cpu.OK
/restricted/projectnb/batmanlab/atang4/data/validation/verify_8bit.OK
```

## PDF download

`GET /api/report.pdf/<job_id>` — requires job status `done`. Returns `application/pdf`.
Built by `backend/pdf_utils.py` using reportlab (no Chromium).

## Autopilot self-heal

See `harness/README.md`. The autopilot calls `claude -p` with `harness/autopilot-prompt.txt`
when a verification job fails. The agent is constrained to minimal fixes + resubmit only.
Claude binary: `/restricted/projectnb/batmanlab/atang4/bin/claude`.

## Rules

- **Never train on a login node** — always qsub.
- **Never commit large files** (checkpoints, PNGs) — `.gitignore` covers `*.pth`, `*.ckpt`.
- Verification sentinels live under `/restricted/projectnb/batmanlab/atang4/data/validation/`.
- Harness run dirs live under `harness/runs/<timestamp>/`.
