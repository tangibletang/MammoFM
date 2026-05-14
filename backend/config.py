import os
from pathlib import Path

WORKING_DIR     = "/restricted/projectnb/batmanlab/atang4/data"
REPO_DIR        = "/restricted/projectnb/batmanlab/atang4/MammoFM"
LLAVA_SRC       = "/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb4views_new_loss"

# Always resolve backend helper scripts next to this file (never Shawn’s upstream copy).
_BACKEND_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR  = f"{WORKING_DIR}/checkpoints_v1_bu_ve_old_loss/llava-llama3.1_8B_breast_clip-finetune_512-lora/checkpoint-5500"
MAMMO_CLIP_CHKPT = "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/mayo/MammoCLIP-MayoClinic-epoch4.tar"

VALIDATION_SCRIPT  = f"{LLAVA_SRC}/llava/serve/ctchat_validation_llama.py"
EMBEDDING_SCRIPT   = str(_BACKEND_DIR / "save_img_embedding_mammofm.py")
FINAL_STAGE_SCRIPT = f"{LLAVA_SRC}/final_stage.py"
JSON_TO_CSV_SCRIPT = str(_BACKEND_DIR / "json_to_csv.py")

MODEL_BASE   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_ID     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Prefer $HF_HOME when set (e.g. node-local cache from scripts/start_server.sh) to avoid mmap failures on NFS.
HF_HOME = os.environ.get(
    "HF_HOME",
    "/restricted/projectnb/batmanlab/atang4/data/.hf_cache",
)

JOBS_DIR     = f"{WORKING_DIR}/jobs"
PYTHON_BIN     = "/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10"
DEEPSPEED_BIN  = "/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/deepspeed"
DEEPSPEED_PORT = 12438

# view names in the order save_img_embedding / ctchat expect for BU records
VIEW_ORDER = ["lmlo", "lcc", "rmlo", "rcc"]

# fake path prefix that makes ctchat_validation_llama.py resolve embeddings
# via its "controls" branch: bu_path/controls/test_images_png/{exam_id}/{view}.png
FAKE_IMG_PREFIX = "/restricted/projectnb/pixel/hariri/MGdata_for_mirai/png/controls"


def verify_paths():
    if Path(EMBEDDING_SCRIPT).name != "save_img_embedding_mammofm.py":
        raise RuntimeError(
            "EMBEDDING_SCRIPT must be the patched MammoFM encoder "
            f"(save_img_embedding_mammofm.py), got: {EMBEDDING_SCRIPT}"
        )
    required = {
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "MAMMO_CLIP_CHKPT": MAMMO_CLIP_CHKPT,
        "VALIDATION_SCRIPT": VALIDATION_SCRIPT,
        "EMBEDDING_SCRIPT": EMBEDDING_SCRIPT,
        "FINAL_STAGE_SCRIPT": FINAL_STAGE_SCRIPT,
        "JSON_TO_CSV_SCRIPT": JSON_TO_CSV_SCRIPT,
        "LLAVA_SRC/zero3.json": f"{LLAVA_SRC}/zero3.json",
    }
    missing = [name for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Missing required paths: {missing}")
