"""FastAPI server for MammoFM web UI."""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_BACKEND = Path(__file__).resolve().parent
os.environ["MAMMOFM_EMBED_SCRIPT"] = str(_BACKEND / "save_img_embedding_mammofm.py")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

import config as cfg
import pdf_utils
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


def _optional_form_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


@app.post("/api/run")
async def api_run(
    files: List[UploadFile] = File(...),
    patient_id: Optional[str] = Form(default=None),
    exam_id: Optional[str] = Form(default=None),
    classifier_csv: Optional[UploadFile] = File(None),
    use_cpu_stage1: Optional[str] = Form(default=None),
    slot_map: Optional[str] = Form(default=None),
):
    if not files or len(files) < 1:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    resolved: list[tuple[str, bytes]] = []
    for f in files:
        name = Path(f.filename or "image").name
        resolved.append((name, await f.read()))

    names = [n for n, _ in resolved]
    warnings: List[str]
    raw_map = (slot_map or "").strip()
    if raw_map:
        try:
            parsed = json.loads(raw_map)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"slot_map must be valid JSON: {exc}",
            ) from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="slot_map JSON must be an object.")
        try:
            slots, warnings = uu.validate_slot_map(parsed, names)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
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

    asyncio.create_task(
        pl.run_pipeline(
            job_id,
            patient,
            exam,
            image_paths,
            cls_path,
            stage1_cpu_override=_optional_form_bool(use_cpu_stage1),
        )
    )
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


@app.get("/api/report.pdf/{job_id}")
async def api_report_pdf(job_id: str):
    job = pl.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not ready: {job['status']}")
    payload = pl.read_results(job_id)
    uploads_dir = pl.JOBS_DIR / job_id / "uploads"
    image_paths = {}
    if uploads_dir.exists():
        for f in uploads_dir.iterdir():
            view = f.stem.upper()
            if view in ("LCC", "LMLO", "RCC", "RMLO"):
                image_paths[view] = f
    created_at = datetime.fromisoformat(job["created_at"]) if job.get("created_at") else None
    pdf_bytes = pdf_utils.build_report_pdf(
        job_id=job_id,
        patient_id=job.get("patient_id"),
        exam_id=job.get("exam_id"),
        preliminary=payload.get("preliminary_report", ""),
        final=payload.get("final_report", ""),
        image_paths=image_paths,
        created_at=created_at,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="mammo_report_{job_id[:8]}.pdf"'},
    )


# Mount static files LAST so API routes take priority
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
