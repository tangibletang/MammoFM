#!/usr/bin/env bash
# Run this on your *laptop* (not on SCC). It does not start uvicorn; it only opens an SSH tunnel.
#
# After the job starts, set COMPUTE_HOST to the node in server_info.txt / qstat (e.g. scc-q33).
# Short name usually works if it resolves from scc4; if the tunnel fails to connect, try
# COMPUTE_HOST=scc-q33.bu.edu (full hostname).
#
# Then open: http://127.0.0.1:${LOCAL_PORT}/
#
# Usage:
#   COMPUTE_HOST=scc-q33 ./ssh_tunnel_from_laptop.sh
#   COMPUTE_HOST=scc-q33.bu.edu LOCAL_PORT=25002 ./ssh_tunnel_from_laptop.sh
#
# Default LOCAL_PORT is 25001 to avoid "address already in use" on 8000–8002 when IDEs forward ports.
#
# Open an interactive shell on scc4 *with* the same port forward (no -N):
#   INTERACTIVE=1 COMPUTE_HOST=scc-q33 ./ssh_tunnel_from_laptop.sh

set -euo pipefail

JUMP="${JUMP:-atang4@scc1.bu.edu}"
LOGIN="${LOGIN:-atang4@scc4.bu.edu}"
COMPUTE_HOST="${COMPUTE_HOST:-scc-q33}"
LOCAL_PORT="${LOCAL_PORT:-25001}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

SSH_BASE=(ssh -F /dev/null -S none -o ControlMaster=no)

if [[ "${INTERACTIVE:-0}" == "1" ]]; then
  exec "${SSH_BASE[@]}" -J "$JUMP" -L "${LOCAL_PORT}:${COMPUTE_HOST}:${REMOTE_PORT}" "$LOGIN"
else
  exec "${SSH_BASE[@]}" -N -J "$JUMP" -L "${LOCAL_PORT}:${COMPUTE_HOST}:${REMOTE_PORT}" "$LOGIN"
fi
