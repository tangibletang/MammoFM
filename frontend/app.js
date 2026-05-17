const POLL_INTERVAL = 3000;
const VIEW_ORDER = ["LCC", "LMLO", "RCC", "RMLO"];

const STATUS_PROGRESS_FALLBACK = {
  pending: 0.05,
  encoding: 0.2,
  stage1: 0.45,
  converting: 0.7,
  stage2: 0.85,
  done: 1,
  failed: 1,
};

const STATUS_PHASE_LABELS = {
  pending: "Starting…",
  encoding: "Encoding images (Mammo-CLIP)",
  stage1: "Stage 1: preliminary report (LLaVA)",
  converting: "Converting to intermediate format",
  stage2: "Stage 2: final report (LLaMA)",
  done: "Complete",
  failed: "Failed",
};

let uploadFiles = [];
let lastPreview = { slots: {}, warnings: [], rows: [] };
/** @type {Record<string, string> | null} view → basename; null = use auto lastPreview.slots only */
let manualSlots = null;
let galleryObjectUrls = [];
let elapsedTimerId = null;
let jobStartMs = null;

function revokeGalleryUrls() {
  galleryObjectUrls.forEach((u) => URL.revokeObjectURL(u));
  galleryObjectUrls = [];
}

function setProgressFromStatus(status, apiProgress) {
  const bar = document.getElementById("jobProgress");
  if (!bar) return;
  const v =
    typeof apiProgress === "number"
      ? apiProgress
      : STATUS_PROGRESS_FALLBACK[status] ?? 0.05;
  bar.value = v;
  const done = status === "done" || status === "failed";
  bar.classList.toggle("job-progress--active", !done && v < 1);
}

function setStatusPhaseAndDetail(status, message) {
  const phaseEl = document.getElementById("statusPhase");
  const hintEl = document.getElementById("statusStepHint");
  const label = STATUS_PHASE_LABELS[status] || status || "Working…";
  const msg = (message != null && String(message).trim()) || "";

  if (phaseEl) {
    phaseEl.textContent = msg || label;
  }
  if (hintEl) {
    if (msg && msg !== label) {
      hintEl.textContent = label;
      hintEl.classList.remove("hidden");
    } else {
      hintEl.textContent = "";
      hintEl.classList.add("hidden");
    }
  }
}

function showError(msg) {
  document.getElementById("errorMsg").textContent = msg;
  document.getElementById("errorBox").classList.remove("hidden");
}

function formatErrorDetail(detail) {
  if (detail == null) return "Request failed.";
  if (typeof detail === "string") return detail;
  if (detail.message && typeof detail.message === "string") {
    const w = detail.warnings?.length
      ? ` ${detail.warnings.join(" ")}`
      : "";
    return detail.message + w;
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

function fileByName(name) {
  if (!name) return undefined;
  return uploadFiles.find((f) => f.name === name);
}

function getAutoSlots() {
  return lastPreview.slots || {};
}

function ensureManualSlotsFromAuto() {
  manualSlots = {};
  const auto = getAutoSlots();
  for (const v of VIEW_ORDER) {
    manualSlots[v] = auto[v] || "";
  }
}

function getEffectiveSlots() {
  if (manualSlots != null) return manualSlots;
  return getAutoSlots();
}

function buildManualChecklistRows(slots) {
  return VIEW_ORDER.map((v) => {
    const fn = slots[v] || "";
    return [v, fn, fn ? "Detected" : "Missing"];
  });
}

function refreshPreviewUI() {
  const slots = getEffectiveSlots();
  if (manualSlots != null) {
    renderChecklist(buildManualChecklistRows(slots));
  } else {
    renderChecklist(lastPreview.rows || []);
  }
  renderGallery(slots);
  renderWarnings(lastPreview.warnings || []);
}

function assignFileToView(view, newFile) {
  ensureManualSlotsFromAuto();
  const s = { ...manualSlots };
  const prev = s[view] || "";
  if (newFile === prev) return;
  if (!newFile) {
    s[view] = "";
  } else {
    const other = VIEW_ORDER.find((v) => v !== view && (s[v] || "") === newFile);
    if (other) {
      s[other] = prev;
    }
    s[view] = newFile;
  }
  manualSlots = s;
  refreshPreviewUI();
}

function swapViews(a, b) {
  ensureManualSlotsFromAuto();
  const s = { ...manualSlots };
  const t = s[a] || "";
  s[a] = s[b] || "";
  s[b] = t;
  manualSlots = s;
  refreshPreviewUI();
}

function renderChecklist(rows) {
  const tbody = document.getElementById("checklistBody");
  tbody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = cell ?? "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderGallery(slots) {
  const host = document.getElementById("thumbGallery");
  host.innerHTML = "";
  revokeGalleryUrls();

  VIEW_ORDER.forEach((view) => {
    const slot = document.createElement("div");
    slot.className = "thumb-slot";
    slot.dataset.view = view;
    slot.draggable = true;
    slot.setAttribute("aria-label", `${view} thumbnail slot`);

    const label = document.createElement("span");
    label.className = "thumb-label";
    label.textContent = view;
    slot.appendChild(label);

    const name = slots[view];
    const file = name ? fileByName(name) : null;
    if (file) {
      const url = URL.createObjectURL(file);
      galleryObjectUrls.push(url);
      const img = document.createElement("img");
      img.src = url;
      img.alt = `${view}: ${file.name}`;
      img.draggable = false;
      slot.appendChild(img);
    } else {
      const ph = document.createElement("div");
      ph.className = "thumb-placeholder";
      ph.textContent = "—";
      slot.appendChild(ph);
    }

    const sel = document.createElement("select");
    sel.className = "thumb-assign";
    sel.setAttribute("aria-label", `Assign file to ${view}`);
    const optEmpty = document.createElement("option");
    optEmpty.value = "";
    optEmpty.textContent = "— Choose file —";
    sel.appendChild(optEmpty);
    uploadFiles.forEach((f) => {
      const o = document.createElement("option");
      o.value = f.name;
      o.textContent = f.name;
      sel.appendChild(o);
    });
    sel.value = name && fileByName(name) ? name : "";
    sel.addEventListener("change", () => {
      assignFileToView(view, sel.value);
    });
    slot.appendChild(sel);

    slot.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", view);
      e.dataTransfer.effectAllowed = "move";
    });
    slot.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      slot.classList.add("thumb-drag-over");
    });
    slot.addEventListener("dragleave", () => {
      slot.classList.remove("thumb-drag-over");
    });
    slot.addEventListener("drop", (e) => {
      e.preventDefault();
      slot.classList.remove("thumb-drag-over");
      const from = e.dataTransfer.getData("text/plain");
      const to = view;
      if (from && from !== to) swapViews(from, to);
    });

    host.appendChild(slot);
  });
}

function renderWarnings(warnings) {
  const box = document.getElementById("previewWarnings");
  if (!warnings.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  box.innerHTML =
    "<strong>Upload checks</strong><ul>" +
    warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("") +
    "</ul>";
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatElapsed(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  const sec = s % 60;
  const min = m % 60;
  if (h > 0) {
    return `${h}:${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  }
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function tickElapsed() {
  const el = document.getElementById("elapsedTime");
  if (!el || jobStartMs == null) return;
  el.textContent = formatElapsed(performance.now() - jobStartMs);
}

function startElapsedTimer() {
  stopElapsedTimer();
  jobStartMs = performance.now();
  tickElapsed();
  elapsedTimerId = setInterval(tickElapsed, 1000);
}

function stopElapsedTimer() {
  if (elapsedTimerId != null) {
    clearInterval(elapsedTimerId);
    elapsedTimerId = null;
  }
  jobStartMs = null;
}

function resetElapsedDisplay() {
  const el = document.getElementById("elapsedTime");
  if (el) el.textContent = "—";
}

async function runPreview() {
  const fd = new FormData();
  uploadFiles.forEach((f) => fd.append("files", f));

  if (!uploadFiles.length) {
    lastPreview = { slots: {}, warnings: [], rows: [] };
    manualSlots = null;
    renderChecklist(lastPreview.rows);
    renderGallery({});
    renderWarnings([]);
    return;
  }

  try {
    const res = await fetch("/api/preview", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      lastPreview = { slots: {}, warnings: [formatErrorDetail(data.detail)], rows: data.rows || [] };
    } else {
      lastPreview = data;
    }
  } catch (e) {
    lastPreview = { slots: {}, warnings: [e.message || String(e)], rows: [] };
  }

  manualSlots = null;
  refreshPreviewUI();
}

function syncFilesFromList(fileList) {
  uploadFiles = Array.from(fileList || []);
  runPreview();
}

async function pollStatus(jobId) {
  while (true) {
    tickElapsed();
    const res = await fetch(`/api/status/${jobId}`);
    const data = await res.json();

    setStatusPhaseAndDetail(data.status, data.message || data.status);
    setProgressFromStatus(data.status, data.progress);

    if (data.status === "done") {
      setProgressFromStatus("done", 1);
      setStatusPhaseAndDetail(
        "done",
        data.message || "All pipeline stages finished."
      );
      if (jobStartMs != null) {
        document.getElementById("elapsedTime").textContent = formatElapsed(
          performance.now() - jobStartMs
        );
      }
      stopElapsedTimer();
      const r = await fetch(`/api/results/${jobId}`);
      const reports = await r.json();
      if (!r.ok) {
        showError(formatErrorDetail(reports.detail) || "Could not load results.");
        return;
      }
      document.getElementById("prelimReport").textContent =
        reports.preliminary_report || "";
      document.getElementById("finalReport").textContent = reports.final_report || "";
      document.getElementById("results").classList.remove("hidden");
      return;
    }

    if (data.status === "failed") {
      setProgressFromStatus("failed", data.progress ?? 1);
      setStatusPhaseAndDetail("failed", data.error || data.message || "Pipeline failed.");
      if (jobStartMs != null) {
        document.getElementById("elapsedTime").textContent = formatElapsed(
          performance.now() - jobStartMs
        );
      }
      stopElapsedTimer();
      showError(data.error || "Pipeline failed.");
      return;
    }

    await new Promise((r) => setTimeout(r, POLL_INTERVAL));
  }
}

const fileInput = document.getElementById("fileInput");
fileInput.addEventListener("change", (e) => {
  syncFilesFromList(e.target.files);
});

const dropZone = document.getElementById("dropZone");
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  syncFilesFromList(e.dataTransfer.files);
});

dropZone.addEventListener("click", (e) => {
  if (e.target.closest(".btn-ghost")) return;
  fileInput.click();
});

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

document.getElementById("refreshDetection").addEventListener("click", () => runPreview());

function setRunButtonsDisabled(disabled) {
  document.getElementById("submitBtn").disabled = disabled;
  const cpuBtn = document.getElementById("submitCpuBtn");
  if (cpuBtn) cpuBtn.disabled = disabled;
}

function slotsReadyForRun() {
  const slots = getEffectiveSlots();
  return VIEW_ORDER.every((v) => {
    const fn = slots[v];
    return fn && fileByName(fn);
  });
}

/** @param {boolean} forceCpuStage1 */
async function submitJob(forceCpuStage1) {
  if (!slotsReadyForRun()) {
    showError(
      "All four views (LCC, LMLO, RCC, RMLO) must have an image assigned. Use filenames, drag-and-drop between slots, or the menus under each thumbnail."
    );
    return;
  }

  stopElapsedTimer();
  resetElapsedDisplay();

  document.getElementById("progress").classList.remove("hidden");
  document.getElementById("results").classList.add("hidden");
  document.getElementById("errorBox").classList.add("hidden");
  setRunButtonsDisabled(true);

  const submitMsg = forceCpuStage1
    ? "Submitting (Stage 1 will use CPU — slower)…"
    : "Submitting…";
  setStatusPhaseAndDetail("pending", submitMsg);
  setProgressFromStatus("pending", STATUS_PROGRESS_FALLBACK.pending);

  const fd = new FormData();
  uploadFiles.forEach((f) => fd.append("files", f));

  if (manualSlots != null) {
    fd.append("slot_map", JSON.stringify(manualSlots));
  }

  const pid = document.getElementById("patientId").value.trim();
  const eid = document.getElementById("examId").value.trim();
  if (pid) fd.append("patient_id", pid);
  if (eid) fd.append("exam_id", eid);
  const csv = document.getElementById("classifierCsv").files[0];
  if (csv) fd.append("classifier_csv", csv);
  const useCpu =
    forceCpuStage1 || document.getElementById("useCpuStage1").checked;
  if (useCpu) fd.append("use_cpu_stage1", "1");

  try {
    const res = await fetch("/api/run", { method: "POST", body: fd });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      showError(formatErrorDetail(body.detail));
      setRunButtonsDisabled(false);
      return;
    }
    document.getElementById("jobIdField").value = body.job_id || "";
    startElapsedTimer();
    setStatusPhaseAndDetail("pending", body.message || "Job queued. Waiting for worker…");
    await pollStatus(body.job_id);
  } catch (err) {
    stopElapsedTimer();
    resetElapsedDisplay();
    showError(err.message || String(err));
  }
  setRunButtonsDisabled(false);
}

document.getElementById("reportForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  await submitJob(false);
});

document.getElementById("submitCpuBtn").addEventListener("click", () => {
  submitJob(true);
});

document.querySelectorAll(".btn-copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const id = btn.getAttribute("data-copy-target");
    const el = document.getElementById(id);
    const text = el?.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
      setTimeout(() => {
        btn.textContent = "Copy";
      }, 1500);
    } catch {
      showError("Clipboard not available.");
    }
  });
});

renderChecklist([]);
