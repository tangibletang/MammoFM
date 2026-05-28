# MammoFM Harness — Offline Autopilot

The harness submits all three verification pathways (CPU+GPU, 16-bit A100, 8-bit GPU) as qsub jobs,
monitors them, invokes `claude -p` for minimal self-healing on failure, then validates the full
webapp + PDF endpoint.

## Three pathways

| Pathway | Stage 1 | Stage 2 | Resources | Script |
|---|---|---|---|---|
| **CPU + GPU** | fp16 on CPU | 4-bit on GPU | 1 CPU + 1 GPU (any) | `smoke_e2e_cpu.sh` |
| **16-bit A100** | fp16 on GPU | 4-bit on GPU | A100/A40-class (CC ≥ 8.0) | `smoke_e2e_fp16_a100.sh` |
| **8-bit A100** | 8-bit on GPU | 4-bit on GPU | A100/A40-class (CC ≥ 8.0) | `smoke_e2e_a100.sh` |

## Deployment table

| Method | Offline-safe | Use when |
|--------|-------------|----------|
| **qsub autopilot** (primary) | ✅ Fully | Signed off, want unattended run |
| tmux autopilot | ⚠️ Login-node fragile | Short runs you'll babysit |
| Cloud Claude Code | ❌ No qsub/SCC | Code edits + PRs only |

## Quickstart (primary path — sign off freely)

```bash
ssh scc4
cd /restricted/projectnb/batmanlab/atang4/MammoFM

./scripts/mammofm autopilot-submit
# Prints: Autopilot submitted: job <JID>
#         Run dir: harness/runs/YYYYMMDD_HHMMSS/
```

Disconnect. Come back any time:

```bash
qstat -j <JOB_ID>                              # is it still running?
tail -f harness/runs/<ts>/autopilot.log        # live progress
cat  harness/runs/<ts>/state.json              # structured status
cat  harness/runs/<ts>/report.md               # final summary when done
qdel <JOB_ID>                                  # stop early
```

## Quickstart (tmux, short runs)

```bash
tmux new -s mammofm-autopilot
./scripts/mammofm autopilot-run
# Ctrl+b d  to detach; tmux attach -t mammofm-autopilot to reattach
```

## Running individual steps

```bash
# Verification only
./scripts/mammofm verify-cpu          # qsub CPU+GPU pathway
./scripts/mammofm verify-fp16         # qsub 16-bit A100 pathway
./scripts/mammofm verify-8bit         # qsub 8-bit A100 pathway
./scripts/mammofm verify-all          # all three at once

# After jobs finish, check sentinels:
cat /restricted/projectnb/batmanlab/atang4/data/validation/verify_cpu.OK
cat /restricted/projectnb/batmanlab/atang4/data/validation/verify_fp16.OK
cat /restricted/projectnb/batmanlab/atang4/data/validation/verify_8bit.OK

# PDF integration test only (needs webapp already running)
SERVER_URL=http://<node>:8000 bash harness/integration_pdf_curl.sh
```

## Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `HARNESS_MAX_RETRIES` | `3` | Retries per pathway before marking `needs_human` |
| `HARNESS_POLL_INTERVAL` | `60` | Seconds between qstat polls |
| `HARNESS_SKIP_WEBAPP_TEST` | `0` | Set to `1` to skip webapp + PDF integration test |
| `MAMMOFM_CLAUDE_BIN` | `/restricted/projectnb/batmanlab/atang4/bin/claude` | Path to `claude` CLI |
| `MAMMOFM_STAGE1_MAX_NEW_TOKENS` | _(unset)_ | Cap tokens for CPU pathway (set to `64` for faster smoke) |

## Self-heal logic

1. When a verification job leaves the queue without `verify_<pathway>.OK`, the autopilot reads
   the sentinel + log tail and calls `claude -p --output-format json` with `autopilot-prompt.txt`.
2. Claude must print `NEW_JOB_ID=<id>` after resubmitting, or `NEEDS_HUMAN=<reason>` to escalate.
3. Same error signature on 3 consecutive retries → automatic `needs_human` (anti-infinite-loop).
4. If `claude` binary is unavailable, the autopilot falls back to a plain resubmit (pause-mode).

## State file format (`state.json`)

```json
{
  "run_dir": "harness/runs/20260528_120000",
  "started":  "2026-05-28T12:00:00Z",
  "cpu_status":   "ok | queued | repairing | needs_human | failed_final | submit_failed",
  "cpu_job_id":   "1234567",
  "cpu_retries":  "1",
  "fp16_status":  "queued",
  "fp16_job_id":  "1234568",
  "fp16_retries": "0",
  "8bit_status":  "queued",
  "8bit_job_id":  "1234569",
  "8bit_retries": "0",
  "webapp_status": "ok | skipped | failed | start_timeout | submit_failed"
}
```
