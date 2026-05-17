"""Shared mammography upload slotting and PNG normalization (FastAPI + optional Gradio)."""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

REQUIRED_VIEWS = ("LCC", "LMLO", "RCC", "RMLO")

STATUS_PROGRESS = {
    "pending": 0.05,
    "encoding": 0.20,
    "stage1": 0.45,
    "converting": 0.70,
    "stage2": 0.85,
    "done": 1.00,
    "failed": 1.00,
}


def view_from_filename(file_path: str) -> Optional[str]:
    stem = Path(file_path).stem.upper()
    compact = re.sub(r"[^A-Z0-9]", "", stem)

    explicit_patterns = {
        "LCC": [r"LCC", r"LEFTCC"],
        "LMLO": [r"LMLO", r"LEFTMLO"],
        "RCC": [r"RCC", r"RIGHTCC"],
        "RMLO": [r"RMLO", r"RIGHTMLO"],
    }
    for view, patterns in explicit_patterns.items():
        if any(re.search(pattern, compact) for pattern in patterns):
            return view

    side = None
    if re.search(r"(^|[^A-Z])L(EFT)?([^A-Z]|$)", stem) or "LEFT" in compact:
        side = "L"
    elif re.search(r"(^|[^A-Z])R(IGHT)?([^A-Z]|$)", stem) or "RIGHT" in compact:
        side = "R"

    view_type = None
    if "MLO" in compact:
        view_type = "MLO"
    elif "CC" in compact:
        view_type = "CC"

    if side and view_type:
        return f"{side}{view_type}"
    return None


def slot_uploaded_files(filenames: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Map REQUIRED_VIEWS → client-visible filename. ``filenames`` should be basenames (e.g. UploadFile.filename)."""
    slots: Dict[str, str] = {}
    warnings: List[str] = []
    for raw in filenames:
        name = Path(raw).name
        view = view_from_filename(name)
        if view is None:
            warnings.append(f"Could not identify the view for `{name}`.")
            continue
        if view in slots:
            warnings.append(
                f"Duplicate `{view}` upload: kept `{Path(slots[view]).name}`, skipped `{name}`."
            )
            continue
        slots[view] = name

    missing = [view for view in REQUIRED_VIEWS if view not in slots]
    if missing:
        warnings.append(f"Missing required view(s): {', '.join(missing)}.")
    return slots, warnings


def checklist_rows(slots: Dict[str, str], warnings: List[str]) -> List[List[str]]:
    """Table rows: [view, fname, status] plus extra rows for unknown / duplicate warnings (Gradio semantics)."""
    rows: List[List[str]] = []
    for view in REQUIRED_VIEWS:
        if view in slots:
            rows.append([view, slots[view], "Detected"])
        else:
            rows.append([view, "", "Missing"])
    for warning in warnings:
        if warning.startswith("Could not identify"):
            rows.append(["Unknown", warning.split("`")[1], "Needs rename"])
        elif warning.startswith("Duplicate"):
            rows.append(["Duplicate", warning.split("`")[-2], "Skipped"])
    return rows


def validate_slot_map(
    raw: Dict[str, object],
    upload_basenames: List[str],
) -> Tuple[Dict[str, str], List[str]]:
    """Validate client JSON slot map: all REQUIRED_VIEWS, values are upload basenames, four distinct files.

    Uploaded files must have distinct basenames when using manual assignment (same rule as multipart bytes map).
    """
    names_list = [Path(n).name for n in upload_basenames]
    names_set = set(names_list)
    if len(names_list) != len(names_set):
        raise ValueError(
            "Uploaded files must have distinct names when using manual view assignment."
        )

    slots: Dict[str, str] = {}
    for view in REQUIRED_VIEWS:
        if view not in raw:
            raise ValueError(f"slot_map is missing required key {view!r}.")
        val = raw[view]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"slot_map[{view!r}] must be a non-empty filename string.")
        bn = Path(val.strip()).name
        if bn not in names_set:
            raise ValueError(
                f"slot_map[{view!r}]={bn!r} is not among uploaded files {sorted(names_set)}."
            )
        slots[view] = bn

    assigned = list(slots.values())
    if len(set(assigned)) != len(assigned):
        raise ValueError(
            "slot_map assigns the same file to more than one view; each view needs a different file."
        )

    warnings = ["Manual view assignment."]
    return slots, warnings


def normalize_to_png_path(source: Union[str, Path, bytes], destination: Path) -> None:
    """Open image (path or raw bytes), convert to RGB, write PNG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        stream: Union[BytesIO, str, Path] = BytesIO(source)
    else:
        stream = source
    with Image.open(stream) as image:
        image.convert("RGB").save(destination, format="PNG")
