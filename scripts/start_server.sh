#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=8:00:00
#$ -N mammofm_server
#$ -o /restricted/projectnb/batmanlab/atang4/data/server.log
#$ -j y

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TRITON_CACHE_DIR=/restricted/projectnb/batmanlab/atang4/data/.triton
export HF_HOME=/restricted/projectnb/batmanlab/atang4/data/.hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

NODE=$(hostname)
echo "=== MammoFM Server ==="
echo "Node: $NODE"
echo "Started: $(date)"
echo ""
echo "To connect, run this on your laptop:"
echo "  ssh -L 8000:${NODE}:8000 atang4@scc.bu.edu"
echo ""
echo "Then open: http://localhost:8000"
echo "========================"

# Write connection info to a fixed file so it's easy to find
cat > /restricted/projectnb/batmanlab/atang4/data/server_info.txt << INFO
Node: $NODE
Started: $(date)
SSH tunnel: ssh -L 8000:${NODE}:8000 atang4@scc.bu.edu
URL: http://localhost:8000
INFO

cd /restricted/projectnb/batmanlab/atang4/MammoFM/backend
/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10 \
    -m uvicorn app:app --host 0.0.0.0 --port 8000
