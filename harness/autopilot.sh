#!/usr/bin/env bash
# MammoFM autopilot — submits both verification pathways, monitors, self-heals, then validates
# the full webapp + PDF endpoint.
#
# Primary launch (offline-safe, survives SSH disconnect):
#   ./scripts/mammofm autopilot-submit
#
# Foreground launch (tmux):
#   ./scripts/mammofm autopilot-run
#
# Env-configurable:
#   HARNESS_MAX_RETRIES=3          retries per pathway before marking needs_human
#   HARNESS_POLL_INTERVAL=60       seconds between qstat polls
#   HARNESS_SKIP_WEBAPP_TEST=0     set to 1 to skip webapp + PDF integration test
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
HARNESS="$ROOT/harness"
SENTINEL_DIR=/restricted/projectnb/batmanlab/atang4/data/validation
CLAUDE_BIN="${MAMMOFM_CLAUDE_BIN:-/restricted/projectnb/batmanlab/atang4/bin/claude}"

MAX_RETRIES="${HARNESS_MAX_RETRIES:-3}"
POLL="${HARNESS_POLL_INTERVAL:-60}"
SKIP_WEBAPP="${HARNESS_SKIP_WEBAPP_TEST:-0}"

# --- run directory ---
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${HARNESS_RUN_DIR:-$HARNESS/runs/$TS}"
mkdir -p "$RUN_DIR"
STATE="$RUN_DIR/state.json"
REPORT="$RUN_DIR/report.md"
LOG="$RUN_DIR/autopilot.log"

exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "  MammoFM Autopilot"
echo "  Run dir: $RUN_DIR"
echo "  $(date)"
echo "============================================"

# Safety check: claude binary
CLAUDE_AVAILABLE=0
if command -v claude &>/dev/null || [[ -x "$CLAUDE_BIN" ]]; then
    CLAUDE_AVAILABLE=1
    CLAUDE_CMD="${CLAUDE_BIN}"
    command -v claude &>/dev/null && CLAUDE_CMD="claude"
    echo "Claude binary: $CLAUDE_CMD"
else
    echo "WARNING: claude binary not found at $CLAUDE_BIN — self-heal will be skipped (pause-mode)."
fi

# ---- helpers ----

_write_state() {
    python3 - "$STATE" "$@" <<'PYEOF'
import sys, json
path = sys.argv[1]
updates = {}
for kv in sys.argv[2:]:
    k, _, v = kv.partition('=')
    updates[k] = v
try:
    with open(path) as f: state = json.load(f)
except Exception:
    state = {}
state.update(updates)
with open(path, 'w') as f: json.dump(state, f, indent=2)
PYEOF
}

_read_state() {
    python3 -c "import sys,json; d=json.load(open('$STATE')) if __import__('os').path.exists('$STATE') else {}; print(d.get('$1',''))"
}

_error_signature() {
    # Produce a short fingerprint of the last ~10 lines of output for loop detection
    local fail_file="$1"
    tail -10 "$fail_file" 2>/dev/null | md5sum | awk '{print $1}'
}

# Initialize state
python3 -c "import json; json.dump({
  'run_dir': '$RUN_DIR',
  'started': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
  'cpu_status': 'pending',
  'cpu_job_id': '',
  'cpu_retries': '0',
  'cpu_error_sig': '',
  '8bit_status': 'pending',
  '8bit_job_id': '',
  '8bit_retries': '0',
  '8bit_error_sig': '',
  'webapp_status': 'pending',
}, open('$STATE','w'), indent=2)"

# ---- submit both verification jobs ----

echo ""
echo "--- Submitting verification jobs ---"
CPU_JID=$(qsub -terse "$SCRIPTS/smoke_e2e_cpu.sh" 2>/dev/null || echo "")
if [[ -z "$CPU_JID" ]]; then
    echo "ERROR: failed to submit CPU verification job" >&2
    _write_state "cpu_status=submit_failed"
else
    echo "CPU  pathway: job $CPU_JID"
    _write_state "cpu_job_id=$CPU_JID" "cpu_status=queued"
fi

GPU_JID=$(qsub -terse "$SCRIPTS/smoke_e2e_a100.sh" 2>/dev/null || echo "")
if [[ -z "$GPU_JID" ]]; then
    echo "ERROR: failed to submit 8-bit GPU verification job" >&2
    _write_state "8bit_status=submit_failed"
else
    echo "8bit pathway: job $GPU_JID"
    _write_state "8bit_job_id=$GPU_JID" "8bit_status=queued"
fi

# ---- poll loop ----

_poll_one() {
    local PATHWAY="$1"    # cpu or 8bit
    local JID_VAR="$2"    # shell var name holding the current job id

    local JID="${!JID_VAR}"
    [[ -z "$JID" ]] && return

    local OK_FILE="$SENTINEL_DIR/verify_${PATHWAY}.OK"
    local FAIL_FILE="$SENTINEL_DIR/verify_${PATHWAY}.FAIL"
    local STATUS_KEY="${PATHWAY}_status"
    local RETRIES_KEY="${PATHWAY}_retries"
    local SIG_KEY="${PATHWAY}_error_sig"

    local CURRENT_STATUS
    CURRENT_STATUS=$(_read_state "$STATUS_KEY")
    [[ "$CURRENT_STATUS" == "ok" || "$CURRENT_STATUS" == "needs_human" || "$CURRENT_STATUS" == "failed_final" ]] && return

    # Still in qstat?
    if qstat -j "$JID" &>/dev/null 2>&1; then
        echo "  [$PATHWAY] job $JID still running"
        return
    fi

    # Job left the queue — check sentinels
    if [[ -f "$OK_FILE" ]]; then
        echo "  [$PATHWAY] PASSED — $OK_FILE"
        _write_state "${STATUS_KEY}=ok"
        return
    fi

    # Failure path
    local RETRIES
    RETRIES=$(_read_state "$RETRIES_KEY")
    RETRIES="${RETRIES:-0}"
    local PREV_SIG
    PREV_SIG=$(_read_state "$SIG_KEY")

    if [[ ! -f "$FAIL_FILE" ]]; then
        # Sentinel missing — job may have crashed before writing it
        echo "  [$PATHWAY] job $JID finished with no sentinel — treating as failure"
        echo "exit_code=unknown" > "$FAIL_FILE"
    fi

    local NEW_SIG
    NEW_SIG=$(_error_signature "$FAIL_FILE")

    # Anti-infinite-loop: same error 3x → give up
    if [[ "$NEW_SIG" == "$PREV_SIG" && "$RETRIES" -ge 2 ]]; then
        echo "  [$PATHWAY] REPEATED failure (sig=$NEW_SIG) after $RETRIES retries → needs_human"
        _write_state "${STATUS_KEY}=needs_human" "${SIG_KEY}=$NEW_SIG"
        return
    fi

    if [[ "$RETRIES" -ge "$MAX_RETRIES" ]]; then
        echo "  [$PATHWAY] max retries ($MAX_RETRIES) reached → failed_final"
        _write_state "${STATUS_KEY}=failed_final"
        return
    fi

    # Attempt self-heal via claude -p
    echo "  [$PATHWAY] FAILED (retry $RETRIES/$MAX_RETRIES) — invoking claude for repair"
    _write_state "${STATUS_KEY}=repairing" "${RETRIES_KEY}=$((RETRIES+1))" "${SIG_KEY}=$NEW_SIG"

    local SCRIPT
    [[ "$PATHWAY" == "cpu" ]] && SCRIPT="$SCRIPTS/smoke_e2e_cpu.sh" || SCRIPT="$SCRIPTS/smoke_e2e_a100.sh"

    local QSUB_LOG
    QSUB_LOG=$(ls "$SENTINEL_DIR/smoke_e2e_${PATHWAY}."*.log 2>/dev/null | sort | tail -1 || echo "")

    local CLAUDE_OUTPUT=""
    if [[ "$CLAUDE_AVAILABLE" -eq 1 ]]; then
        CLAUDE_OUTPUT=$(
            HARNESS_RUN_DIR="$RUN_DIR" \
            HARNESS_PATHWAY="$PATHWAY" \
            HARNESS_RETRY="$RETRIES" \
            HARNESS_FAIL_FILE="$FAIL_FILE" \
            HARNESS_LOG_FILE="${QSUB_LOG:-$FAIL_FILE}" \
            "$CLAUDE_CMD" -p \
                --output-format json \
                "$(cat "$HARNESS/autopilot-prompt.txt")" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',''))" 2>/dev/null \
            || echo ""
        )
        echo "  [$PATHWAY] claude output: $CLAUDE_OUTPUT"
    else
        echo "  [$PATHWAY] claude unavailable — pause mode; will resubmit without fix"
    fi

    # Check for NEEDS_HUMAN
    if echo "$CLAUDE_OUTPUT" | grep -q "NEEDS_HUMAN"; then
        local REASON
        REASON=$(echo "$CLAUDE_OUTPUT" | grep "NEEDS_HUMAN" | head -1)
        echo "  [$PATHWAY] claude says $REASON → needs_human"
        _write_state "${STATUS_KEY}=needs_human"
        return
    fi

    # Try to parse NEW_JOB_ID from claude output; fallback to plain resubmit
    local NEW_JID
    NEW_JID=$(echo "$CLAUDE_OUTPUT" | grep "^NEW_JOB_ID=" | head -1 | cut -d= -f2 | tr -d '[:space:]')

    if [[ -z "$NEW_JID" ]]; then
        echo "  [$PATHWAY] no NEW_JOB_ID from claude — resubmitting ourselves"
        NEW_JID=$(qsub -terse "$SCRIPT" 2>/dev/null || echo "")
    fi

    if [[ -z "$NEW_JID" ]]; then
        echo "  [$PATHWAY] resubmit failed → needs_human"
        _write_state "${STATUS_KEY}=needs_human"
        return
    fi

    echo "  [$PATHWAY] resubmitted as job $NEW_JID"
    printf -v "$JID_VAR" '%s' "$NEW_JID"
    _write_state "${PATHWAY}_job_id=$NEW_JID" "${STATUS_KEY}=queued"
}

_all_resolved() {
    for P in cpu 8bit; do
        local S
        S=$(_read_state "${P}_status")
        case "$S" in
            ok|needs_human|failed_final|submit_failed) ;;
            *) return 1 ;;
        esac
    done
    return 0
}

echo ""
echo "--- Polling every ${POLL}s ---"
while ! _all_resolved; do
    sleep "$POLL"
    echo ""
    echo "$(date -u +%H:%M:%SZ) — polling"
    _poll_one cpu CPU_JID
    _poll_one 8bit GPU_JID
done

echo ""
echo "--- Both pathways resolved ---"
echo "  cpu  : $(_read_state cpu_status)"
echo "  8bit : $(_read_state 8bit_status)"

# ---- webapp + PDF integration test ----

WEBAPP_STATUS="skipped"
if [[ "$SKIP_WEBAPP" != "1" ]]; then
    echo ""
    echo "--- Starting webapp for integration test ---"
    WEBAPP_JID=$(qsub -terse "$SCRIPTS/start_server.sh" 2>/dev/null || echo "")
    if [[ -z "$WEBAPP_JID" ]]; then
        echo "WARNING: could not submit webapp job — skipping integration test"
        _write_state "webapp_status=submit_failed"
        WEBAPP_STATUS="submit_failed"
    else
        echo "Webapp job: $WEBAPP_JID"
        _write_state "webapp_job_id=$WEBAPP_JID" "webapp_status=starting"

        # Wait for server_info.txt
        DATA_DIR=/restricted/projectnb/batmanlab/atang4/data
        WAIT=0
        NODE=""
        while [[ $WAIT -lt 300 ]]; do
            sleep 15; WAIT=$((WAIT+15))
            if [[ -f "$DATA_DIR/server_info.txt" ]]; then
                NODE=$(grep '^Node:' "$DATA_DIR/server_info.txt" | awk '{print $2}' || true)
                PORT=$(grep '^Port:' "$DATA_DIR/server_info.txt" | awk '{print $2}' || echo "8000")
                [[ -n "$NODE" ]] && break
            fi
        done

        if [[ -z "$NODE" ]]; then
            echo "WARNING: webapp did not start within 5 min — skipping integration test"
            WEBAPP_STATUS="start_timeout"
        else
            echo "Webapp node: $NODE:$PORT"
            # The autopilot runs on a compute node with the same /restricted mount, so direct HTTP works
            SERVER_URL="http://${NODE}:${PORT}"
            if SERVER_URL="$SERVER_URL" bash "$HARNESS/integration_pdf_curl.sh" 2>&1; then
                WEBAPP_STATUS="ok"
            else
                WEBAPP_STATUS="failed"
            fi
            # Clean up webapp job
            qdel "$WEBAPP_JID" 2>/dev/null || true
        fi
        _write_state "webapp_status=$WEBAPP_STATUS"
    fi
fi

# ---- final report ----

CPU_S=$(_read_state cpu_status)
GPU_S=$(_read_state 8bit_status)
WEBAPP_S=$(_read_state webapp_status)
TS_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "$REPORT" <<MDEOF
# MammoFM Autopilot Report

**Run dir:** $RUN_DIR
**Started:** $(python3 -c "import json; d=json.load(open('$STATE')); print(d.get('started','?'))")
**Finished:** $TS_END

## Pathway Results

| Pathway | Status |
|---------|--------|
| CPU fallback | $CPU_S |
| 8-bit GPU (A100) | $GPU_S |
| Webapp + PDF | $WEBAPP_S |

## Details

\`\`\`json
$(cat "$STATE")
\`\`\`

## Sentinel files

- CPU:  $SENTINEL_DIR/verify_cpu.OK (or .FAIL)
- 8bit: $SENTINEL_DIR/verify_8bit.OK (or .FAIL)

## Log

$LOG
MDEOF

echo ""
echo "============================================"
echo "  Autopilot complete"
echo "  CPU  : $CPU_S"
echo "  8bit : $GPU_S"
echo "  Webapp: $WEBAPP_S"
echo "  Report: $REPORT"
echo "============================================"
