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


# Shared project HF hub layout (Llama snapshot lives here for offline Stage 1/2). Node-local HF_HOME may be empty
# until rsync seeds from this tree (see scripts/local_hf_env.inc.sh).
SHARED_PROJECT_HF_HOME = f"{WORKING_DIR}/.hf_cache"


def effective_hf_home() -> str:
    """Runtime HF cache root (prefer current process env over import-time HF_HOME)."""
    return os.environ.get("HF_HOME", HF_HOME)


JOBS_DIR     = f"{WORKING_DIR}/jobs"
PYTHON_BIN     = "/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10"
DEEPSPEED_BIN  = "/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/deepspeed"
DEEPSPEED_PORT = 12438

# view names in the order save_img_embedding / ctchat expect for BU records
VIEW_ORDER = ["lmlo", "lcc", "rmlo", "rcc"]

# fake path prefix that makes ctchat_validation_llama.py resolve embeddings
# via its "controls" branch: bu_path/controls/test_images_png/{exam_id}/{view}.png
FAKE_IMG_PREFIX = "/restricted/projectnb/pixel/hariri/MGdata_for_mirai/png/controls"


_HF_HUB_DIR_PREFIX = "models--"


def resolve_hf_cached_repo(
    repo_ref: str,
    *,
    hf_home: str | Path | None = None,
) -> str:
    """If `repo_ref` is a Hub id, return a local snapshot directory under HF hub cache when present.

    Passing an absolute path to an existing `config.json` folder avoids any Hub/gated HTTP while
    `TRANSFORMERS_OFFLINE=1` — using the raw Hub id still triggers `cached_file` resolution that can
    401 on gated models when the cache layout does not satisfy the resolver.
    """
    ref = (repo_ref or "").strip()
    if not ref:
        return repo_ref
    path_try = Path(ref).expanduser()
    if path_try.is_dir() and (path_try / "config.json").is_file():
        return str(path_try.resolve())
    if "/" not in ref or ref.startswith(("/", ".")):
        return ref
    root = Path(hf_home if hf_home is not None else effective_hf_home())
    hub_leaf = f"{_HF_HUB_DIR_PREFIX}{ref.replace('/', '--')}"
    snapshots = root / "hub" / hub_leaf / "snapshots"
    if not snapshots.is_dir():
        return ref
    found: list[Path] = []
    for child in snapshots.iterdir():
        if child.is_dir() and (child / "config.json").is_file():
            found.append(child)
    if not found:
        return ref
    found.sort(key=lambda p: p.name)
    return str(found[0].resolve())


def resolve_hf_cached_repo_with_fallback(
    repo_ref: str,
    *,
    hf_home: str | Path | None = None,
) -> str:
    """Like `resolve_hf_cached_repo`, but try additional HF roots so offline jobs do not fall back to a Hub id.

    Order: `hf_home` (or `effective_hf_home()`), optional `MAMMOFM_HF_FALLBACK_HOME`, then `SHARED_PROJECT_HF_HOME`.
    Returns the first path that exists with `config.json`, otherwise the last resolver result (may still be a Hub id).
    """
    ref = (repo_ref or "").strip()
    path_try = Path(ref).expanduser()
    if path_try.is_dir() and (path_try / "config.json").is_file():
        return str(path_try.resolve())
    primary = hf_home if hf_home is not None else effective_hf_home()
    homes: list[str] = [str(Path(primary).expanduser().resolve())]
    fb = (os.environ.get("MAMMOFM_HF_FALLBACK_HOME") or "").strip()
    if fb:
        rp = str(Path(fb).expanduser().resolve())
        if rp not in homes:
            homes.append(rp)
    sp = str(Path(SHARED_PROJECT_HF_HOME).resolve())
    if sp not in homes:
        homes.append(sp)
    last = ref
    for h in homes:
        cand = resolve_hf_cached_repo(ref, hf_home=h)
        last = cand
        p = Path(cand)
        if p.is_dir() and (p / "config.json").is_file():
            return str(p.resolve())
    return last


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


if __name__ == "__main__":
    # Usage: python config.py [--id] [hub_id_or_path]
    #   (no args)  -> resolve MAMMOFM_MODEL_BASE or MODEL_BASE
    #   --id        -> resolve MAMMOFM_MODEL_ID or MODEL_ID
    #   one arg     -> resolve that ref
    # Exits 2 if snapshot dir + config.json not found (offline / gated Hub will break).
    import sys

    argv = sys.argv[1:]
    if argv == ["--id"]:
        ref = os.environ.get("MAMMOFM_MODEL_ID", MODEL_ID)
    elif argv:
        ref = argv[0]
    else:
        ref = os.environ.get("MAMMOFM_MODEL_BASE", MODEL_BASE)
    out = resolve_hf_cached_repo_with_fallback(ref, hf_home=effective_hf_home())
    print(out)
    p = Path(out)
    raise SystemExit(0 if p.is_dir() and (p / "config.json").is_file() else 2)
