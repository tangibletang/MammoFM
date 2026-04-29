"""FastAPI server for MammoFM web UI."""
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config as cfg
import pipeline as pl

cfg.verify_paths()

app = FastAPI(title="MammoFM")

FRONTEND_DIR = Path(cfg.REPO_DIR) / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.post("/api/run")
async def api_run(
    lcc: UploadFile = File(...),
    lmlo: UploadFile = File(...),
    rcc: UploadFile = File(...),
    rmlo: UploadFile = File(...),
    patient_id: str = Form(...),
    exam_id: str = Form(...),
    classifier_csv: Optional[UploadFile] = File(None),
):
    job_id = pl.create_job()
    job_dir = Path(cfg.JOBS_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded PNGs
    image_paths = {}
    for view, upload in [("lcc", lcc), ("lmlo", lmlo), ("rcc", rcc), ("rmlo", rmlo)]:
        dest = job_dir / f"{view}.png"
        dest.write_bytes(await upload.read())
        image_paths[view] = dest

    # Save optional classifier CSV
    cls_path = None
    if classifier_csv and classifier_csv.filename:
        cls_path = job_dir / "classifier.csv"
        cls_path.write_bytes(await classifier_csv.read())

    asyncio.create_task(
        pl.run_pipeline(job_id, patient_id, exam_id, image_paths, cls_path)
    )
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    job = pl.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/results/{job_id}")
async def api_results(job_id: str):
    job = pl.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")
    return pl.read_results(job_id)
