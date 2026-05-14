# Source this file to set HF_HOME to node-local storage and seed Meta-Llama from shared cache.
# Used by scripts/start_server.sh and scripts/run_uvicorn_local_hf.sh
#
# Optional overrides:
#   MAMMOFM_HF_SOURCE  — hub parent directory (default: project .hf_cache)
#   MAMMOFM_HF_LOCAL_DIR — force local root (skips scratch/tmp heuristics)

MAMMOFM_HF_SOURCE="${MAMMOFM_HF_SOURCE:-/restricted/projectnb/batmanlab/atang4/data/.hf_cache}"
LLAMA_HUB_REPO="models--meta-llama--Meta-Llama-3.1-8B-Instruct"
if [[ -n "${MAMMOFM_HF_LOCAL_DIR:-}" ]]; then
  LOCAL_HF_ROOT="$MAMMOFM_HF_LOCAL_DIR"
elif [[ -d "/scratch/${USER}" ]]; then
  LOCAL_HF_ROOT="/scratch/${USER}/mammofm_hf_cache"
elif [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" ]]; then
  LOCAL_HF_ROOT="${TMPDIR}/mammofm_hf_${USER}"
else
  LOCAL_HF_ROOT="/tmp/mammofm_hf_${USER}"
fi
export HF_HOME="$LOCAL_HF_ROOT"
mkdir -p "${HF_HOME}/hub"
SRC_REPO="${MAMMOFM_HF_SOURCE}/hub/${LLAMA_HUB_REPO}"
DST_REPO="${HF_HOME}/hub/${LLAMA_HUB_REPO}"
if [[ -d "$SRC_REPO" ]]; then
  if [[ ! -d "$DST_REPO/snapshots" ]]; then
    echo "Seeding HF_HOME from shared cache (one copy per job node; avoids Llama mmap on NFS)..."
    echo "  src: $SRC_REPO"
    echo "  dst: $DST_REPO"
    rsync -a "$SRC_REPO" "${HF_HOME}/hub/"
  else
    echo "Using existing local Llama HF cache at $DST_REPO"
  fi
else
  echo "WARNING: Shared Llama cache not found at $SRC_REPO — set MAMMOFM_HF_SOURCE or download weights."
  echo "Falling back HF_HOME to shared path (may mmap-fail on NFS)."
  export HF_HOME="$MAMMOFM_HF_SOURCE"
fi
