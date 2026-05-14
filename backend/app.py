"""FastAPI server for MammoFM web UI."""
import asyncio
import os
from pathlib import Path
from typing import List, Optional

_BACKEND = Path(__file__).resolve().parent
os.environ["MAMMOFM_EMBED_SCRIPT"] = str(_BACKEND / "save_img_embedding_mammofm.py")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

import config as cfg
import pipeline as pl
import upload_utils as uu

cfg.verify_paths()

app = FastAPI(title="MammoFM")

FRONTEND_DIR = Path(cfg.REPO_DIR) / "frontend"


@app.post("/api/preview")
async def api_preview(files: Optional[List[UploadFile]] = File(default=None)):
    if not files:
        return {
            "slots": {},
            "warnings": [],
            "rows": uu.checklist_rows({}, []),
        }
    names = [f.filename or "image" for f in files]
    slots, warnings = uu.slot_uploaded_files(names)
    return {
        "slots": slots,
        "warnings": warnings,
        "rows": uu.checklist_rows(slots, warnings),
    }


@app.post("/api/run")
async def api_run(
    files: List[UploadFile] = File(...),
    patient_id: Optional[str] = Form(default=None),
    exam_id: Optional[str] = Form(default=None),
    classifier_csv: Optional[UploadFile] = File(None),
):
    if not files or len(files) < 1:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    resolved: list[tuple[str, bytes]] = []
    for f in files:
        name = Path(f.filename or "image").name
        resolved.append((name, await f.read()))

    names = [n for n, _ in resolved]
    slots, warnings = uu.slot_uploaded_files(names)
    if len(slots) != len(uu.REQUIRED_VIEWS):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not identify all four required views from filenames "
                "(need LCC, LMLO, RCC, RMLO in names).",
                "warnings": warnings,
                "slots": slots,
            },
        )

    by_name: dict[str, bytes] = {}
    for name, blob in resolved:
        by_name[name] = blob

    job_id = pl.create_job()
    patient = (patient_id or "").strip() or f"P_{job_id[:8]}"
    exam = (exam_id or "").strip() or f"E_{job_id[:8]}"
    job_dir = Path(cfg.JOBS_DIR) / job_id
    upload_dir = job_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    image_paths = {}
    try:
        for view in uu.REQUIRED_VIEWS:
            client_name = slots[view]
            raw = by_name.get(client_name)
            if raw is None:
                raise ValueError(f"Missing bytes for resolved file name: {client_name}")
            dest = upload_dir / f"{view.lower()}.png"
            uu.normalize_to_png_path(raw, dest)
            image_paths[view.lower()] = dest

        cls_path = None
        if classifier_csv and classifier_csv.filename:
            cls_path = upload_dir / "classifier.csv"
            cls_path.write_bytes(await classifier_csv.read())
    except Exception as exc:
        pl.mark_job_failed(job_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    asyncio.create_task(pl.run_pipeline(job_id, patient, exam, image_paths, cls_path))
    return {"job_id": job_id, "patient_id": patient, "exam_id": exam, "warnings": warnings}


@app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    job = pl.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    progress = uu.STATUS_PROGRESS.get(job["status"], 0.05)
    return {**job, "progress": progress}


@app.get("/api/results/{job_id}")
async def api_results(job_id: str):
    job = pl.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")
    payload = pl.read_results(job_id)
    try:
        pl.validate_result_artifacts(job_id, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return payload


# Mount static files LAST so API routes take priority
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
