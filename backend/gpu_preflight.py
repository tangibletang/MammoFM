"""Stage 1 GPU preflight: warn when VRAM is already heavily used or other jobs hold the device."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class _GpuMem:
    index: int
    used_mb: int
    total_mb: int


@dataclass
class _ComputeApp:
    pid: str
    name: str
    used_mem: str


def _query_gpu_mem() -> List[_GpuMem]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return []
    rows: List[_GpuMem] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append(
                _GpuMem(
                    index=int(parts[0]),
                    used_mb=int(parts[1]),
                    total_mb=int(parts[2]),
                )
            )
        except ValueError:
            continue
    return rows


def _query_compute_apps() -> List[_ComputeApp]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return []
    apps: List[_ComputeApp] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        apps.append(
            _ComputeApp(pid=parts[0], name=parts[1], used_mem=parts[2])
        )
    return apps


def collect_stage1_gpu_alerts() -> List[str]:
    """Return warning lines for stderr/log; empty if checks skipped or nothing suspicious."""
    if (os.environ.get("MAMMOFM_DISABLE_GPU_PREFLIGHT") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return []

    if not shutil.which("nvidia-smi"):
        return []

    warn_frac = _env_float("MAMMOFM_GPU_WARN_FRACTION", 0.25)
    warn_frac = max(0.0, min(warn_frac, 0.95))
    min_abs_mb = _env_int("MAMMOFM_GPU_WARN_MIN_MB", 1536)
    gpu_index = _env_int("MAMMOFM_GPU_PREFLIGHT_INDEX", 0)

    gpus = _query_gpu_mem()
    if not gpus:
        return []

    target = next((g for g in gpus if g.index == gpu_index), gpus[0])
    used, total = target.used_mb, target.total_mb
    if total <= 0:
        return []

    alerts: List[str] = []
    frac = used / float(total)
    if frac >= warn_frac and used >= min_abs_mb:
        alerts.append(
            f"GPU {target.index} VRAM already high before Stage 1: "
            f"{used} MiB / {total} MiB ({frac * 100:.1f}%). "
            f"Another process (or leftover context) may be using the card — Stage 1 can OOM. "
            f"Tune MAMMOFM_GPU_WARN_FRACTION / MAMMOFM_GPU_WARN_MIN_MB or run `scripts/check_gpu.sh` on the node."
        )

    apps = _query_compute_apps()
    if apps:
        lines = [
            f"  pid {a.pid}  {a.name}  {a.used_mem}"
            for a in apps[:16]
        ]
        rest = len(apps) - 16
        if rest > 0:
            lines.append(f"  … and {rest} more")
        alerts.append(
            "GPU compute processes (possible contention before Stage 1):\n" + "\n".join(lines)
        )

    return alerts


def emit_stage1_gpu_preflight(log_dir) -> None:
    """Write alerts to per-job file and stderr; no-op if nothing to say."""
    import sys
    from pathlib import Path

    alerts = collect_stage1_gpu_alerts()
    if not alerts:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    warn_path = log_dir / "stage1_gpu_preflight.warn"
    body = "\n".join(alerts) + "\n"
    warn_path.write_text(body)

    for line in alerts:
        print(f"[MammoFM][GPU preflight] {line}", file=sys.stderr, flush=True)


def enforce_idle_cuda_device_or_fail() -> None:
    """Fail fast when the visible CUDA device already has compute processes (shared-node safety).

    Opt-in for smoke / CI: set ``MAMMOFM_ENFORCE_IDLE_CUDA_DEVICE=1``.
    Uses ``CUDA_VISIBLE_DEVICES`` first index when set (physical GPU id for nvidia-smi -i).
    """
    raw = (os.environ.get("MAMMOFM_ENFORCE_IDLE_CUDA_DEVICE") or "").strip().lower()
    if raw not in ("1", "true", "yes", "y"):
        return
    if not shutil.which("nvidia-smi"):
        return

    idx_raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    try:
        smi_i = int(idx_raw)
    except ValueError:
        smi_i = 0

    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(smi_i),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return

    preview = "\n".join(lines[:12])
    more = "" if len(lines) <= 12 else f"\n … and {len(lines) - 12} more"
    raise RuntimeError(
        f"CUDA device {smi_i} is not idle ({len(lines)} compute process(es)). "
        "Refusing to run with MAMMOFM_ENFORCE_IDLE_CUDA_DEVICE=1 — use an exclusive GPU or resubmit.\n"
        f"{preview}{more}"
    )
