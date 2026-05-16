#!/usr/bin/env bash
# Quick VRAM / process snapshot on a GPU node (run on SCC compute node or job host).
set -euo pipefail

if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found — not a CUDA host or driver missing."
  exit 1
fi

echo "=== GPU summary ==="
nvidia-smi

echo ""
echo "=== Per-GPU memory (MiB) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo ""
echo "=== Compute processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
  || echo "(no compute apps or query unsupported)"
