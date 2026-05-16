#!/usr/bin/env python3
"""Run encode → Stage 1 → bridge → Stage 2 on sample mammogram PNGs (GPU job)."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
import pipeline as pl  # noqa: E402
from gpu_preflight import enforce_idle_cuda_device_or_fail  # noqa: E402


def _tail(path: Path, n: int = 80) -> str:
    if not path.is_file():
        return f"(missing {path})"
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _log_fail(job_id: str, exc: BaseException) -> None:
    base = Path(cfg.JOBS_DIR) / job_id / "logs"
    for name in ("encode.log", "stage1.log", "bridge.log", "stage2.log"):
        p = base / name
        if p.is_file():
            print(f"\n----- {name} (tail) -----", file=sys.stderr)
            print(_tail(p), file=sys.stderr)


async def _main() -> None:
    enforce_idle_cuda_device_or_fail()

    seed = Path(
        os.environ.get(
            "MAMMOFM_SMOKE_JOB_DIR",
            "/restricted/projectnb/batmanlab/atang4/data/jobs"
            "/500f8458-82ba-455f-a6c7-43036a287d3b",
        )
    )
    patient_id = os.environ.get("MAMMOFM_SMOKE_PATIENT_ID", "36536418")
    exam_id = os.environ.get("MAMMOFM_SMOKE_EXAM_ID", "869643546")
    exam_dir = seed / "embed_bu" / "controls" / "test_images_png" / exam_id
    need = ("lcc.png", "lmlo.png", "rcc.png", "rmlo.png")
    for f in need:
        p = exam_dir / f
        if not p.is_file():
            raise FileNotFoundError(f"Smoke asset missing: {p}")

    job_id = pl.create_job()
    print("smoke job_id:", job_id)

    try:
        await pl.run_pipeline(
            job_id,
            patient_id=patient_id,
            exam_id=exam_id,
            image_paths={
                "lcc": exam_dir / "lcc.png",
                "lmlo": exam_dir / "lmlo.png",
                "rcc": exam_dir / "rcc.png",
                "rmlo": exam_dir / "rmlo.png",
            },
        )
    except BaseException as exc:
        _log_fail(job_id, exc)
        raise

    res = pl.read_results(job_id)
    pl.validate_result_artifacts(job_id, res)
    print("E2E_OK", job_id)
    print("preliminary chars:", len(res.get("preliminary_report") or ""))
    print("final chars:", len(res.get("final_report") or ""))


if __name__ == "__main__":
    asyncio.run(_main())
