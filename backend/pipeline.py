"""Orchestrates the two-stage mammography report pipeline for a single patient job."""
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# In-memory job store: job_id → {"status": str, "message": str, "error": str|None}
_jobs: dict[str, dict] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "message": "Queued", "error": None}
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


def _update(job_id: str, status: str, message: str):
    _jobs[job_id]["status"] = status
    _jobs[job_id]["message"] = message


async def _run(cmd: list[str], job_id: str, log_file: Path, cwd: Optional[str] = None, env: Optional[dict] = None):
    """Run a subprocess, stream stdout/stderr to log_file. Raises on non-zero exit."""
    merged_env = {**os.environ, **(env or {})}
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as lf:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=lf,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=merged_env,
        )
        await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\nSee {log_file}"
        )


async def run_pipeline(
    job_id: str,
    patient_id: str,
    exam_id: str,
    image_paths: dict,       # {"lcc": Path, "lmlo": Path, "rcc": Path, "rmlo": Path}
    classifier_csv_path: Optional[Path] = None,
):
    job_dir = Path(cfg.JOBS_DIR) / job_id
    log_dir = job_dir / "logs"
    embed_dir = job_dir / "embed_bu" / "controls" / "test_images_png" / exam_id
    embed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 1: copy uploaded PNGs into embedding dir ────────────────────
        _update(job_id, "encoding", "Copying images…")
        for view in cfg.VIEW_ORDER:
            src = image_paths[view]
            dst = embed_dir / f"{view}.png"
            dst.write_bytes(Path(src).read_bytes())

        # ── Step 2: build 4-row CSV for save_img_embedding.py ────────────────
        img_csv = job_dir / "images.csv"
        rows = []
        for view in cfg.VIEW_ORDER:
            rows.append({
                "dataset": "BU",
                "file_path": f"{cfg.FAKE_IMG_PREFIX}/{exam_id}/{view}.png",
            })
        pd.DataFrame(rows).to_csv(img_csv, index=False)

        # ── Step 3: run image encoder ─────────────────────────────────────────
        _update(job_id, "encoding", "Encoding images with Mammo-CLIP…")
        embed_bu_root = str(job_dir / "embed_bu")
        await _run(
            [
                cfg.PYTHON_BIN, cfg.EMBEDDING_SCRIPT,
                "--mammo-clip-chkpt", cfg.MAMMO_CLIP_CHKPT,
                "--data-csv", str(img_csv),
                "--bu_path", embed_bu_root,
                "--inference-mode", "y",
            ],
            job_id,
            log_dir / "encode.log",
            cwd=cfg.LLAVA_SRC,
            env={"PYTHONPATH": cfg.LLAVA_SRC},
        )

        # ── Step 4: build source JSON for ctchat ─────────────────────────────
        fake_images = [
            f"{cfg.FAKE_IMG_PREFIX}/{exam_id}/{view}.png"
            for view in cfg.VIEW_ORDER
        ]
        source_json = job_dir / "source.json"
        record = {
            "id": f"job_{job_id}",
            "dataset": "BU",
            "patient_id": patient_id,
            "exam_id": exam_id,
            "image": str(fake_images),
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWould you mind generating the radiology report for the specified 2D screening mammogram?<report_generation>",
                },
                {"from": "gpt", "value": ""},
            ],
        }
        source_json.write_text(json.dumps([record], indent=2))

        # ── Step 5: Stage 1 — LLaVA inference via deepspeed ──────────────────
        _update(job_id, "stage1", "Stage 1: generating preliminary report (LLaVA)…")
        val_results = job_dir / "val_results.json"
        await _run(
            [
                cfg.DEEPSPEED_BIN, "--master_port", str(cfg.DEEPSPEED_PORT),

                "llava/serve/ctchat_validation_llama.py",
                "--deepspeed", "./zero3.json",
                "--model-path", cfg.CHECKPOINT_DIR,
                "--model-base", cfg.MODEL_BASE,
                "--source_json", str(source_json),
                "--bu_path", embed_bu_root,
                "--out-path", str(val_results),
            ],
            job_id,
            log_dir / "stage1.log",
            cwd=cfg.LLAVA_SRC,
            env={
                "HF_HOME": cfg.HF_HOME,
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
                "PYTHONPATH": cfg.LLAVA_SRC,
            },
        )

        # ── Step 6: JSON → CSV bridge ─────────────────────────────────────────
        _update(job_id, "converting", "Converting Stage 1 output…")
        intermediate_csv = job_dir / "intermediate.csv"
        bridge_cmd = [
            cfg.PYTHON_BIN, cfg.JSON_TO_CSV_SCRIPT,
            "--val-results-json", str(val_results),
            "--output-csv", str(intermediate_csv),
            "--patient-id", patient_id,
            "--exam-id", exam_id,
        ]
        if classifier_csv_path:
            bridge_cmd += ["--classifier-csv", str(classifier_csv_path)]
        await _run(bridge_cmd, job_id, log_dir / "bridge.log")

        # ── Step 7: Stage 2 — LLaMA reconciliation ───────────────────────────
        _update(job_id, "stage2", "Stage 2: generating final report (LLaMA)…")
        final_csv = job_dir / "final.csv"
        await _run(
            [
                cfg.PYTHON_BIN, cfg.FINAL_STAGE_SCRIPT,
                "--input-csv", str(intermediate_csv),
                "--output-csv", str(final_csv),
                "--model-id", cfg.MODEL_ID,
                "--max-new-tokens", "320",
                "--temperature", "0.3",
                "--top-p", "0.9",
                "--patient-id", patient_id,
                "--exam-id", exam_id,
            ],
            job_id,
            log_dir / "stage2.log",
            env={"HF_HOME": cfg.HF_HOME, "PYTHONPATH": cfg.LLAVA_SRC},
        )

        _update(job_id, "done", "Complete")

    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
        raise


def read_results(job_id: str) -> dict:
    job_dir = Path(cfg.JOBS_DIR) / job_id

    val_results = job_dir / "val_results.json"
    preliminary = ""
    if val_results.exists():
        data = json.loads(val_results.read_text())
        if data:
            preliminary = data[0]["conversations_out"][0]["answer"].strip()

    final_csv = job_dir / "final.csv"
    final = ""
    if final_csv.exists():
        df = pd.read_csv(final_csv)
        if not df.empty and "final_generated_report_zs" in df.columns:
            final = str(df.iloc[0]["final_generated_report_zs"]).strip()

    return {"preliminary_report": preliminary, "final_report": final, "job_id": job_id}
